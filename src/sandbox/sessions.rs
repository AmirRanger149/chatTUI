//! Commands that outlive the tool call that started them.
//!
//! A one-shot `bash` call has to finish inside the shell timeout, which rules
//! out the three things an agent most often needs to run: a long build, a dev
//! server, and a REPL. A session removes that coupling — the command keeps
//! running, the tool call returns whatever has been printed so far, and the
//! model comes back for more when it wants it.
//!
//! Sessions are deliberately not a way around the sandbox: a session is
//! spawned through the same builder as a one-shot command, so it gets the same
//! filtered environment, the same `setsid` (no controlling terminal — a TUI
//! still cannot run), and the same kernel isolation.

use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::io::{Read, Write};
use std::process::{Child, ChildStdin};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Longest a single call will wait. Bounded so a model asking to "wait until
/// it is done" cannot park a round indefinitely.
pub const MAX_YIELD: Duration = Duration::from_secs(120);

/// Longest a session may live. A dev server the model forgot about should not
/// still be running next week.
pub const MAX_SESSION_LIFETIME: Duration = Duration::from_secs(30 * 60);

/// Most sessions alive at once.
pub const MAX_SESSIONS: usize = 8;

/// Everything chatTUI knows about one still-running command.
#[derive(Debug)]
struct Session {
    child: Child,
    stdin: ChildStdin,
    /// Everything the command has printed. The reader threads append; the tool
    /// call reads from `consumed` onwards, so no output is ever shown twice.
    output: Arc<Mutex<Vec<u8>>>,
    consumed: usize,
    command: String,
    started: Instant,
}

/// What one look at a session found.
#[derive(Debug, Default)]
pub struct SessionReport {
    /// Output printed since the last look.
    pub output: String,
    /// Exit status, when the command has finished.
    pub exited: Option<String>,
    /// How long it has been running.
    pub running_for: Duration,
    /// True when `output` had to be shortened.
    pub truncated: bool,
}

/// The session table.
///
/// Cheap to clone, and every clone sees the same table — which matters because
/// `Sandbox` is cloned for each tool round, and a session started in one round
/// has to be reachable from the next.
#[derive(Debug, Default, Clone)]
pub struct Sessions {
    inner: Arc<Mutex<HashMap<String, Session>>>,
    next: Arc<AtomicU64>,
}

impl Sessions {
    pub fn new() -> Self {
        Self::default()
    }

    /// Take ownership of a freshly spawned command. Returns its id.
    ///
    /// On rejection the command is killed rather than dropped: dropping a
    /// `Child` leaves the process running with nobody able to reach it.
    pub fn insert(
        &self,
        mut child: Child,
        stdin: ChildStdin,
        command: String,
        // The buffer the pipe readers are already writing into. It has to be
        // this one, not a fresh copy: a second buffer would collect nothing.
        output: Arc<Mutex<Vec<u8>>>,
    ) -> Result<String> {
        let mut table = self.lock()?;
        if table.len() >= MAX_SESSIONS {
            kill_tree(&mut child);
            return Err(anyhow!(
                "{MAX_SESSIONS} sessions are already running — kill_session one first"
            ));
        }
        let id = format!("s{}", self.next.fetch_add(1, Ordering::SeqCst) + 1);
        table.insert(
            id.clone(),
            Session {
                child,
                stdin,
                output,
                consumed: 0,
                command,
                started: Instant::now(),
            },
        );
        Ok(id)
    }

    /// Wait up to `wait` for new output, then report what there is.
    ///
    /// Returns early when the command exits, so a fast command does not make
    /// the model wait out the whole window.
    pub async fn wait_and_report(&self, id: &str, wait: Duration) -> Result<SessionReport> {
        let wait = wait.min(MAX_YIELD);
        let deadline = Instant::now() + wait;
        loop {
            let report = self.snapshot(id)?;
            if report.exited.is_some() || !report.output.is_empty() {
                return Ok(report);
            }
            if Instant::now() >= deadline {
                return Ok(report);
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }

    /// Read whatever is new, and check whether the command has finished.
    fn snapshot(&self, id: &str) -> Result<SessionReport> {
        let mut table = self.lock()?;
        let session = table
            .get_mut(id)
            .ok_or_else(|| anyhow!("no session '{id}' — it may have been killed already"))?;

        let exited = match session.child.try_wait() {
            Ok(Some(status)) => Some(status.to_string()),
            Ok(None) => None,
            Err(error) => Some(format!("unknown ({error})")),
        };
        let running_for = session.started.elapsed();

        let fresh = {
            let buffer = session
                .output
                .lock()
                .map_err(|_| anyhow!("session output lock poisoned"))?;
            let fresh = if buffer.len() > session.consumed {
                buffer[session.consumed..].to_vec()
            } else {
                Vec::new()
            };
            // Advanced under the same lock as the read, so output appended in
            // between is reported next time rather than skipped.
            session.consumed = buffer.len();
            fresh
        };

        let mut text = String::from_utf8_lossy(&fresh).into_owned();
        let mut truncated = false;
        let limit = crate::sandbox::MAX_TOOL_OUTPUT_BYTES;
        if text.len() > limit {
            text = crate::sandbox::keep_head_and_tail(&text, limit);
            truncated = true;
        }
        Ok(SessionReport {
            output: text,
            exited,
            running_for,
            truncated,
        })
    }

    /// Send input to a running command.
    pub fn write(&self, id: &str, input: &str) -> Result<()> {
        let mut table = self.lock()?;
        let session = table
            .get_mut(id)
            .ok_or_else(|| anyhow!("no session '{id}' — it may have been killed already"))?;
        // A newline is almost always what the model means: it is answering a
        // prompt, and a prompt that never sees a line ending never answers.
        let payload = if input.ends_with('\n') {
            input.to_string()
        } else {
            format!("{input}\n")
        };
        session
            .stdin
            .write_all(payload.as_bytes())
            .map_err(|error| anyhow!("writing to session '{id}': {error}"))?;
        session
            .stdin
            .flush()
            .map_err(|error| anyhow!("flushing session '{id}': {error}"))
    }

    /// Kill one session and its whole process tree.
    pub fn kill(&self, id: &str) -> Result<String> {
        let mut table = self.lock()?;
        let mut session = table
            .remove(id)
            .ok_or_else(|| anyhow!("no session '{id}' — it may have been killed already"))?;
        let command = session.command.clone();
        kill_tree(&mut session.child);
        Ok(format!("session {id} killed: {command}"))
    }

    /// Kill everything still running. Called on the way out, because a session
    /// is its own process session and would otherwise outlive chatTUI.
    pub fn kill_all(&self) {
        if let Ok(mut table) = self.inner.lock() {
            for (_, session) in table.iter_mut() {
                kill_tree(&mut session.child);
            }
            table.clear();
        }
    }

    /// Reap sessions that have run past their lifetime, returning what was
    /// killed so the transcript can say so.
    pub fn reap_expired(&self) -> Vec<String> {
        let expired: Vec<String> = match self.inner.lock() {
            Ok(table) => table
                .iter()
                .filter(|(_, session)| session.started.elapsed() > MAX_SESSION_LIFETIME)
                .map(|(id, _)| id.clone())
                .collect(),
            Err(_) => return Vec::new(),
        };
        let mut killed = Vec::new();
        for id in expired {
            match self.kill(&id) {
                Ok(message) => killed.push(message),
                Err(_) => {}
            }
        }
        killed
    }

    /// Ids and commands of everything still running, for the transcript.
    pub fn list(&self) -> Vec<String> {
        match self.inner.lock() {
            Ok(table) => table
                .iter()
                .map(|(id, session)| {
                    format!(
                        "{id} ({}s): {}",
                        session.started.elapsed().as_secs(),
                        session.command
                    )
                })
                .collect(),
            Err(_) => Vec::new(),
        }
    }

    fn lock(&self) -> Result<std::sync::MutexGuard<'_, HashMap<String, Session>>> {
        self.inner
            .lock()
            .map_err(|_| anyhow!("session table lock poisoned"))
    }
}

/// Kill the process tree, then make sure the pipes are released.
#[cfg(unix)]
fn kill_tree(child: &mut Child) {
    let pid = child.id();
    if pid > 0 {
        // Negative pid = the whole process group, which is what `setsid` made
        // this command lead. Killing only the `sh` would leave its children.
        unsafe {
            libc::kill(pid as i32, libc::SIGKILL);
            libc::kill(-(pid as i32), libc::SIGKILL);
        }
    }
    let _ = child.wait();
}

/// Without `setsid` there is no process group to target, so this can only
/// reach the `sh` process itself — documented, not pretended away.
#[cfg(not(unix))]
fn kill_tree(child: &mut Child) {
    let _ = child.kill();
    let _ = child.wait();
}

/// Spawn a thread that copies a pipe into a shared buffer until EOF.
///
/// Both pipes get one, and they share the buffer: a command that writes a lot
/// to stderr while stdout is idle must not fill an undrained pipe and
/// deadlock, and the interleaving is close enough to what a terminal shows.
pub fn spawn_pipe_reader<R>(pipe: R, sink: Arc<Mutex<Vec<u8>>>)
where
    R: Read + Send + 'static,
{
    std::thread::spawn(move || {
        let mut pipe = pipe;
        let mut chunk = [0u8; 4096];
        loop {
            match pipe.read(&mut chunk) {
                Ok(0) => break,
                Ok(count) => {
                    let Ok(mut buffer) = sink.lock() else {
                        break;
                    };
                    buffer.extend_from_slice(&chunk[..count]);
                }
                Err(_) => break,
            }
        }
    });
}
