//! Workspace-scoped tool execution for agent mode.
//!
//! ## Security model — read this before trusting any of it
//!
//! Two independent layers protect shell commands, and both are documented
//! here with their real limits:
//!
//! 1. **Application-level restrictions** (always active):
//!    - The file tools (`read_file`, `write_file`, `edit_file`, `list_files`)
//!      resolve every path against the workspace root and refuse paths that
//!      escape it (directly, via `..`, or via symlinks) as well as sensitive
//!      files. These checks live in this process: a path validated here can
//!      still be swapped for a symlink between validation and I/O (TOCTOU),
//!      which only OS primitives such as `openat2` could close portably.
//!    - The permission layer ([`permissions::PermissionMode`]) gates which
//!      tool *classes* may run at all.
//!    - A command never gets the user's terminal: stdin is `/dev/null`,
//!      stdout/stderr are pipes, and the child runs in its own **session**
//!      (`setsid`), so it has no controlling terminal and `/dev/tty` cannot
//!      be opened. Without this, a single `cargo run` of a TUI would put the
//!      real terminal into raw mode and the alternate screen, and being
//!      killed at the timeout would leave it that way. Captured output is
//!      also stripped of terminal control sequences (see [`ansi`]) before it
//!      reaches the transcript, so escape bytes cannot be replayed at the
//!      user's terminal either.
//!    - `bash` refuses commands that *name* sensitive files and keeps a tiny
//!      advisory denylist — both are string scans and both are trivially
//!      bypassable; they are foot-gun guards, never security controls.
//!
//! 2. **Kernel-level isolation for shell commands** (Linux, enabled by
//!    default via `os_isolation: auto`; see [`os_isolation`]): before the
//!    shell is spawned on a dedicated worker thread, the thread receives
//!    Landlock filesystem rules (reads everywhere, writes only under the
//!    workspace plus a small set of scratch roots) and a seccomp-bpf filter
//!    (network sockets, ptrace/process-injection, kernel-module loading,
//!    namespace/mount tricks, io_uring, the all-process `kill(-1)`, and
//!    more are denied). The two facilities are applied **independently**
//!    (one being unavailable on a kernel does not silently remove the
//!    other), the shell inherits whatever applied, and it cannot remove
//!    it. When a facility is unavailable, `auto` mode says exactly which
//!    one in the tool output and `require` mode refuses to run — the
//!    fallback is never silent.
//!
//!    Limits that remain even with isolation active, by design: reads
//!    outside the workspace are still possible (toolchains must read system
//!    files, so secret-*read* protection stays an application policy);
//!    `connect()` is denied even for Unix sockets; `/tmp`, `$TMPDIR`,
//!    `/dev/shm`, `$CARGO_HOME` and `$CARGO_TARGET_DIR` are writable;
//!    *targeted* signals (`kill <pid>`, `pkill`, `pidfd_send_signal`) can
//!    still reach same-uid processes including this session — an inherent
//!    limit of unprivileged same-uid sandboxing (only the all-process
//!    `kill(-1)` is denied, see below); and like every software sandbox
//!    this trusts the kernel.
//!
//! On platforms or configurations where kernel isolation is off, `bash` is
//! *not* confined at all: a command runs with the full privileges of the
//! chatTUI process and can read or write any file the user can, reach the
//! network, and spawn arbitrary child processes. Describe the system as
//! **workspace-restricted tool execution** — never as a sandbox — unless
//! kernel isolation is verified active.

pub(crate) mod ansi;
pub mod os_isolation;
pub(crate) mod patch;
pub(crate) mod sessions;
pub mod permissions;

use crate::api::types::ToolCall;
use crate::sandbox::os_isolation::OsIsolation;
use crate::sandbox::permissions::{authorize, Gate, PermissionMode, ToolKind};
use anyhow::{anyhow, Context, Result};
use serde_json::Value;
use std::ffi::OsStr;
use std::path::{Component, Path, PathBuf};
use std::process::Stdio;
#[cfg(unix)] // used by the sandboxed worker thread only
use std::sync::atomic::{AtomicI32, Ordering};
use std::time::Duration;
use tokio::fs;

pub const DEFAULT_SHELL_TIMEOUT_SECS: u64 = 30;
/// Lower bound for the enforced shell timeout so a tool call can never be
/// configured (or tricked) into blocking forever.
pub const MIN_SHELL_TIMEOUT_SECS: u64 = 1;
/// Tool output larger than this is truncated before it reaches the model.
/// Visible to [`crate::instructions`] so the prompt quotes the real budget
/// instead of a number that can drift away from it.
pub(crate) const MAX_TOOL_OUTPUT_BYTES: usize = 20_000;

/// Upper bound for a per-call shell timeout. A build longer than this wants a
/// session, not a longer timer.
pub const MAX_SHELL_TIMEOUT_SECS: u64 = 1800;

/// Default line window for ranged reads. Reading files window by window
/// keeps a 10,000-line file from flooding the conversation context — the
/// model pages through it instead of swallowing it whole.
pub const DEFAULT_READ_LIMIT: usize = 250;
/// Hard cap on a single read window.
pub const MAX_READ_LIMIT: usize = 1_000;
/// Display cap for one source line; minified files must not blow the
/// output budget on a single line.
const MAX_LINE_DISPLAY_CHARS: usize = 1_000;

#[derive(Debug, Clone)]
pub struct SandboxConfig {
    pub enabled: bool,
    pub workspace_root: PathBuf,
    pub auto_approve: bool,
    pub allow_shell: bool,
    pub max_file_size: usize, // bytes
    /// Hard timeout for the `bash` tool; enforced by killing the command's
    /// process group. Clamped to at least [`MIN_SHELL_TIMEOUT_SECS`].
    pub shell_timeout: Duration,
    /// Application-level permission mode (see [`permissions`]).
    pub permission_mode: PermissionMode,
    /// When to apply kernel-level isolation (Landlock filesystem
    /// confinement + seccomp syscall filter) to shell commands. See
    /// [`os_isolation`] for the exact guarantees and limits.
    pub os_isolation: OsIsolation,
    /// Extra sensitive file names/suffixes (from config.json) appended to
    /// the built-in sensitive-file policy. An entry starting with `.` is a
    /// suffix match, anything else an exact (case-insensitive) name match.
    pub extra_sensitive_names: Vec<String>,
}

impl Default for SandboxConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            // Empty means unset: never fall back to process cwd / this crate.
            workspace_root: PathBuf::new(),
            auto_approve: true,
            allow_shell: true,
            max_file_size: 1024 * 1024, // 1MB
            shell_timeout: Duration::from_secs(DEFAULT_SHELL_TIMEOUT_SECS),
            permission_mode: PermissionMode::FullAuto,
            os_isolation: OsIsolation::Auto,
            extra_sensitive_names: Vec::new(),
        }
    }
}

impl SandboxConfig {
    /// The timeout actually enforced by the `bash` tool — never zero.
    pub fn effective_shell_timeout(&self) -> Duration {
        if self.shell_timeout.is_zero() {
            Duration::from_secs(MIN_SHELL_TIMEOUT_SECS)
        } else {
            self.shell_timeout
        }
    }
}

#[derive(Debug, Clone)]
pub struct Sandbox {
    pub config: SandboxConfig,
    /// What the user has already approved this session: tool names for file
    /// tools, command prefixes for shell. Approving one `git push` covers the
    /// next, so an ask-mode stays usable without a prompt per command.
    pub allowances: Vec<String>,
    /// Commands still running from an earlier tool call. Shared behind an
    /// `Arc` so a session started in one round is reachable from the next.
    pub sessions: sessions::Sessions,
}

/// The permission layer's verdict on one call, plus the line to show the
/// user when the verdict is "ask".
#[derive(Debug, Clone)]
pub struct CallVerdict {
    pub gate: Gate,
    /// What the call would do, in one line.
    pub summary: String,
}

impl Sandbox {
    pub fn new(config: SandboxConfig) -> Self {
        Self {
            config,
            allowances: Vec::new(),
            sessions: sessions::Sessions::new(),
        }
    }

    /// Remember an approval for the rest of the session.
    pub fn allow(&mut self, key: String) {
        if !self.allowances.contains(&key) {
            self.allowances.push(key);
        }
    }

    /// A copy of this sandbox with kernel isolation switched off, for the one
    /// call the user has just agreed to run unconfined. Nothing else about it
    /// changes: the workspace restriction, the sensitive-file policy and the
    /// permission mode all still apply.
    pub fn without_isolation(&self) -> Sandbox {
        let mut unconfined = self.clone();
        unconfined.config.os_isolation = OsIsolation::Off;
        unconfined
    }

    /// Forget every approval — used when the mode changes.
    pub fn clear_allowances(&mut self) {
        self.allowances.clear();
    }

    /// The key an approval for `tool_call` would be remembered under.
    pub fn allowance_key_for(&self, tool_call: &ToolCall) -> String {
        let args: Value = serde_json::from_str(&tool_call.arguments).unwrap_or(Value::Null);
        crate::sandbox::permissions::allowance_key(&tool_call.name, args["command"].as_str())
    }

    /// Ask the permission layer what would happen to `tool_call`, without
    /// running it. `allow_once` is an approval granted for a single attempt,
    /// so "allow this one" does not have to be remembered anywhere.
    pub fn gate_call(&self, tool_call: &ToolCall, allow_once: Option<&str>) -> CallVerdict {
        let kind = match ToolKind::from_tool_name(&tool_call.name) {
            Some(kind) => kind,
            None => {
                return CallVerdict {
                    gate: Gate::Deny {
                        reason: format!("unknown tool: {}", tool_call.name),
                    },
                    summary: tool_call.name.clone(),
                }
            }
        };
        // Unparseable arguments are reported by `execute_tool`; the gate only
        // needs the command for its lookup, and treats missing arguments as
        // "no command" rather than guessing.
        let args: Value = serde_json::from_str(&tool_call.arguments).unwrap_or(Value::Null);
        let command = args["command"].as_str();
        let target = args["path"]
            .as_str()
            .or_else(|| args["session_id"].as_str());
        let allowed: Vec<String> = match allow_once {
            Some(key) => {
                let mut allowed = self.allowances.clone();
                allowed.push(key.to_string());
                allowed
            }
            None => self.allowances.clone(),
        };
        CallVerdict {
            gate: crate::sandbox::permissions::gate(
                self.config.permission_mode,
                kind,
                &tool_call.name,
                command,
                &allowed,
            ),
            summary: crate::sandbox::permissions::describe_call(&tool_call.name, command, target),
        }
    }

    #[allow(dead_code)] // test helper; not used by the app itself
    pub fn with_root(root: PathBuf) -> Self {
        let mut cfg = SandboxConfig::default();
        // Canonicalize so the stored root is the true workspace prefix
        // (symlinks resolved); fall back to the raw path if that fails.
        cfg.workspace_root = root.canonicalize().unwrap_or(root);
        Self::new(cfg)
    }

    pub fn has_target(&self) -> bool {
        !self.config.workspace_root.as_os_str().is_empty()
    }

    /// Set the agent target directory. Relative paths are resolved from cwd
    /// only for this explicit user choice — never used as an implicit default.
    pub fn set_target(&mut self, raw: &str) -> Result<PathBuf> {
        let path = expand_user_path(raw)?;
        if !path.exists() {
            return Err(anyhow!("target directory does not exist: {}", path.display()));
        }
        if !path.is_dir() {
            return Err(anyhow!("target is not a directory: {}", path.display()));
        }
        let canonical = path
            .canonicalize()
            .with_context(|| format!("resolving target {}", path.display()))?;
        self.config.workspace_root = canonical.clone();
        Ok(canonical)
    }

    /// Resolve a user-supplied path to an absolute path inside workspace_root.
    ///
    /// Guarantees about the returned path:
    /// - it is lexically normalized (no surviving `.` / `..` components), so
    ///   the kernel cannot later resolve a leftover `..` outside the root;
    /// - its deepest existing ancestor is symlink-resolved and still inside
    ///   the workspace, so a symlink cannot redirect I/O out of the root;
    /// - components that do not exist yet cannot be symlinks, so appending
    ///   them to the canonical ancestor is safe at resolution time.
    ///
    /// Remaining gap (documented, not solvable portably here): a symlink can
    /// appear *between* this check and the actual I/O (TOCTOU).
    pub fn resolve_path(&self, user_path: &str) -> Result<PathBuf> {
        if !self.has_target() {
            return Err(anyhow!(
                "no target directory set — use /sandbox <path> or --sandbox <path>"
            ));
        }
        let user_path = user_path.trim();
        if user_path.is_empty() {
            return Err(anyhow!("empty path"));
        }
        if user_path.contains('\0') {
            return Err(anyhow!("path contains NUL byte"));
        }

        // `set_target` / `with_root` store an already-canonical root, so
        // this is the true workspace prefix, not a stale look-alike.
        let root = self.config.workspace_root.clone();
        let requested = Path::new(user_path);
        let joined = if requested.is_absolute() {
            requested.to_path_buf()
        } else {
            root.join(requested)
        };

        // Normalize `..` and `.` lexically FIRST. Checking the raw path and
        // canonicalizing only when it exists is not enough: an absolute path
        // with `..` through a not-yet-existing directory would otherwise be
        // accepted unresolved and escape once the kernel resolves it.
        let normalized = normalize_lexical(&joined);
        if !normalized.starts_with(&root) {
            return Err(anyhow!("path escapes workspace: {}", user_path));
        }

        // Resolve symlinks through the deepest existing ancestor; a link
        // pointing outside the workspace is caught here.
        let resolved = canonicalize_deepest(&normalized);
        if !resolved.starts_with(&root) {
            return Err(anyhow!("path escapes workspace via symlink: {}", user_path));
        }
        Ok(resolved)
    }

    /// Enforce the sensitive-file policy for `path` (which must already be
    /// workspace-resolved). Every workspace-relative component is checked,
    /// so `secrets/.env` and `backup/.git/hooks/x` are caught, not just the
    /// final file name.
    pub fn check_access(&self, path: &Path, access: Access) -> Result<()> {
        let relative = path
            .strip_prefix(&self.config.workspace_root)
            .unwrap_or(path);
        let mut inside_git = false;
        for component in relative.components() {
            let name = component.as_os_str().to_string_lossy();
            if name == ".git" {
                inside_git = true;
            }
            if is_sensitive_name(&name, &self.config.extra_sensitive_names) {
                return Err(anyhow!("access to sensitive file denied: {}", name));
            }
            if inside_git {
                match access {
                    Access::Write => {
                        return Err(anyhow!(
                            "writes inside .git/ are denied: {}",
                            path.display()
                        ));
                    }
                    Access::Read if name == "config" || name == "credentials" => {
                        return Err(anyhow!("access to sensitive file denied: {name} (inside .git)"));
                    }
                    _ => {}
                }
            }
        }
        Ok(())
    }

    /// Read a window of a file: up to `limit` lines starting at the
    /// 1-based `offset`. The output is numbered lines with a header that
    /// says which range of the file was served, so large files are read in
    /// chunks instead of flooding the conversation context.
    pub async fn read_file(&self, path: &str, offset: usize, limit: usize) -> Result<String> {
        self.authorize(ToolKind::Read)?;
        let full = self.resolve_path(path)?;
        self.check_access(&full, Access::Read)?;
        let metadata = fs::metadata(&full).await.context("file not found")?;
        if !metadata.is_file() {
            return Err(anyhow!("not a file: {}", path));
        }
        if metadata.len() as usize > self.config.max_file_size {
            return Err(anyhow!(
                "file too large ({} bytes > {} limit)",
                metadata.len(),
                self.config.max_file_size
            ));
        }
        let content = fs::read_to_string(&full).await.context("reading file")?;
        let lines: Vec<&str> = content.lines().collect();
        let total = lines.len();

        if total == 0 {
            return Ok(format!("{path}: empty file"));
        }
        let offset = offset.max(1);
        if offset > total {
            return Ok(format!(
                "{path}: the file has {total} line(s); offset {offset} is past the end"
            ));
        }
        let limit = limit.clamp(1, MAX_READ_LIMIT);
        let start = offset - 1;
        let end = (start + limit).min(total);

        let mut out = format!("{path}: lines {}-{} of {total}\n", offset, end);
        for (index, line) in lines[start..end].iter().enumerate() {
            let display: String = if line.chars().count() > MAX_LINE_DISPLAY_CHARS {
                line.chars().take(MAX_LINE_DISPLAY_CHARS).collect::<String>() + " …"
            } else {
                (*line).to_string()
            };
            out.push_str(&format!("{:>6}│{}\n", start + index + 1, display));
        }
        if end < total {
            out.push_str(&format!(
                "… {} more line(s) — call read_file again with offset {}\n",
                total - end,
                end + 1
            ));
        }
        Ok(out)
    }

    pub async fn write_file(&self, path: &str, content: &str) -> Result<String> {
        self.authorize(ToolKind::Write)?;
        let full = self.resolve_path(path)?;
        self.check_access(&full, Access::Write)?;
        if let Some(parent) = full.parent() {
            fs::create_dir_all(parent).await.context("creating parent dirs")?;
            // Re-validate after creating the parents: create_dir_all follows
            // symlinks, so make sure the directory we are about to write
            // into still resolves inside the workspace. (A symlink swapped
            // in after this check remains a TOCTOU race — see module docs.)
            let canonical_parent = parent.canonicalize().context("resolving parent")?;
            if !canonical_parent.starts_with(&self.config.workspace_root) {
                return Err(anyhow!("path escapes workspace: {}", path));
            }
        }
        // Line stats for the report: how many lines the file had before
        // (if it existed) and how many it has now.
        let old_lines = fs::read_to_string(&full)
            .await
            .map(|old| old.lines().count())
            .unwrap_or(0);
        let new_lines = content.lines().count();
        fs::write(&full, content).await.context("writing file")?;
        if old_lines > 0 {
            Ok(format!("wrote {path} (+{new_lines} -{old_lines})"))
        } else {
            Ok(format!("wrote {path} (+{new_lines})"))
        }
    }

    /// Apply a patch document: several files, all or nothing.
    ///
    /// Unlike `edit_file`, which fails the moment one exact string does not
    /// match, a patch is *planned* in full against the current contents of
    /// every file it touches and only then written. A hunk that does not
    /// apply therefore changes nothing at all, instead of leaving the tree
    /// half-edited.
    ///
    /// Path resolution, the sensitive-file policy and the permission mode
    /// are checked for every path up front, so a patch cannot use one
    /// allowed file to smuggle a forbidden one past the first write.
    pub async fn apply_patch(&self, text: &str) -> Result<String> {
        self.authorize(ToolKind::Edit)?;
        let parsed = patch::parse(text)?;

        // Resolve and policy-check every path before reading anything.
        let mut targets: Vec<(String, PathBuf)> = Vec::new();
        for op in &parsed.ops {
            let raw = match op {
                patch::FileOp::Add { path, .. }
                | patch::FileOp::Update { path, .. }
                | patch::FileOp::Delete { path } => path,
            };
            let full = self.resolve_path(raw)?;
            self.check_access(&full, Access::Write)?;
            if !targets.iter().any(|(known, _)| known == raw) {
                targets.push((raw.clone(), full));
            }
        }

        // Snapshot what is there now; the planner is pure and reads through
        // this closure.
        let mut current: Vec<(String, Option<String>)> = Vec::with_capacity(targets.len());
        for (raw, full) in &targets {
            current.push((raw.clone(), fs::read_to_string(full).await.ok()));
        }
        let planned = patch::plan(&parsed, |path| {
            current
                .iter()
                .find(|(known, _)| known == path)
                .and_then(|(_, contents)| contents.clone())
        })?;

        // Everything is validated; now write.
        for file in &planned {
            let full = targets
                .iter()
                .find(|(known, _)| *known == file.path)
                .map(|(_, full)| full.clone())
                .ok_or_else(|| anyhow!("patch planned a path that was not checked: {}", file.path))?;
            match &file.content {
                Some(text) => {
                    if let Some(parent) = full.parent() {
                        fs::create_dir_all(parent).await.context("creating parent dirs")?;
                    }
                    fs::write(&full, text).await.context("writing file")?;
                }
                None => {
                    fs::remove_file(&full).await.context("deleting file")?;
                }
            }
        }
        Ok(patch::format_report(&planned))
    }

    pub async fn edit_file(&self, path: &str, old_string: &str, new_string: &str) -> Result<String> {
        self.authorize(ToolKind::Edit)?;
        let full = self.resolve_path(path)?;
        self.check_access(&full, Access::Write)?;
        let current = fs::read_to_string(&full).await.context("reading file for edit")?;
        if !current.contains(old_string) {
            return Err(anyhow!("old_string not found in file"));
        }
        // Ensure unique occurrence or allow first?
        let count = current.matches(old_string).count();
        if count > 1 {
            return Err(anyhow!(
                "old_string appears {} times, must be unique for safe edit",
                count
            ));
        }
        let new_content = current.replacen(old_string, new_string, 1);
        let removed = old_string.lines().count();
        let added = new_string.lines().count();
        fs::write(&full, &new_content).await.context("writing edited file")?;
        Ok(format!("edited {path} (+{added} -{removed})"))
    }

    pub async fn list_files(&self, path: &str) -> Result<String> {
        self.authorize(ToolKind::Read)?;
        let dir_path = if path.trim().is_empty() { "." } else { path };
        let full = self.resolve_path(dir_path)?;
        self.check_access(&full, Access::Read)?;
        let metadata = fs::metadata(&full).await.context("path not found")?;
        if !metadata.is_dir() {
            return Err(anyhow!("not a directory: {}", dir_path));
        }
        let mut entries = fs::read_dir(&full).await.context("reading dir")?;
        let mut lines = Vec::new();
        while let Some(entry) = entries.next_entry().await? {
            let name = entry.file_name().to_string_lossy().to_string();
            // Keep the sensitive-file inventory out of directory listings.
            if is_sensitive_name(&name, &self.config.extra_sensitive_names) {
                continue;
            }
            let is_dir = entry.file_type().await.map(|t| t.is_dir()).unwrap_or(false);
            let prefix = if is_dir { "dir " } else { "file" };
            lines.push(format!("{} {}", prefix, name));
        }
        lines.sort();
        if lines.is_empty() {
            Ok("(empty directory)".to_string())
        } else {
            Ok(lines.join("\n"))
        }
    }

    /// Run a shell command. **Read the module docs first.** What this method
    /// actually guarantees:
    /// - the workspace root is the command's current directory (cwd only);
    /// - the command runs in its own session and process group, with no
    ///   controlling terminal, and is killed — tree included — when the
    ///   configured timeout expires;
    /// - the environment is reduced to [`ENV_ALLOWLIST`], so API keys and
    ///   other credentials are not inherited;
    /// - commands that name sensitive files are refused (best effort);
    /// - **kernel-level isolation when available and enabled** (Linux,
    ///   `os_isolation` != off): the command cannot modify files outside the
    ///   workspace/scratch roots, open network connections, ptrace other
    ///   processes, load kernel modules, create namespaces or mounts, or
    ///   signal every process on the machine (`kill(-1)`) — enforced by the
    ///   kernel, not by command inspection. The restrictions are installed
    ///   in the forked child just before exec (see
    ///   [`os_isolation::confine_in_child`]), so the shell never runs
    ///   unconfined, and each facility is enforced independently of the
    ///   other. When the kernel lacks a facility, `auto` mode runs the
    ///   command with exactly the warning that names what is missing in the
    ///   output and `require` mode refuses to run; neither pretends the
    ///   command was confined, and neither can panic the worker.
    pub async fn bash(&self, command: &str) -> Result<String> {
        self.bash_timed(command, None).await
    }

    /// `bash` with a per-call timeout.
    ///
    /// A model that knows it is about to run a full release build should be
    /// able to say so, instead of the command being killed at the default and
    /// the model concluding the build is broken. Clamped, because an unbounded
    /// timeout is a hung round.
    pub async fn bash_timed(&self, command: &str, timeout_secs: Option<u64>) -> Result<String> {
        let cmd = self.prepare_shell(command)?;
        let timeout = match timeout_secs {
            Some(secs) => Duration::from_secs(secs.clamp(1, MAX_SHELL_TIMEOUT_SECS)),
            None => self.config.effective_shell_timeout(),
        };

        #[cfg(unix)]
        {
            let workspace = self.config.workspace_root.clone();
            let roots = os_isolation::writable_roots(&workspace);
            let mode = self.config.os_isolation;
            let cmd_owned = cmd.to_string();

            let (otx, orx) = tokio::sync::oneshot::channel::<Result<ShellOutcome, String>>();
            let pid_slot = std::sync::Arc::new(AtomicI32::new(-1));
            let pid_for_thread = pid_slot.clone();

            let worker = std::thread::Builder::new()
                .name("sandboxed-shell".into())
                .spawn(move || {
                    let outcome = run_shell_process(
                        &cmd_owned,
                        &workspace,
                        &pid_for_thread,
                        mode,
                        &roots,
                    );
                    let _ = otx.send(outcome);
                });

            if let Err(error) = worker {
                return Err(anyhow!("could not start the sandboxed shell worker: {error}"));
            }

            let outcome = match tokio::time::timeout(timeout, orx).await {
                Ok(Ok(inner)) => inner,
                Ok(Err(_worker_gone)) => Err("shell worker exited without reporting".to_string()),
                Err(_elapsed) => {
                    let pid = pid_slot.load(Ordering::SeqCst);
                    if pid > 0 {
                        kill_process_group(pid as u32).await;
                    }
                    return Err(anyhow!(
                        "command timed out after {}s and was killed: {}\n\nThere is no terminal here: stdin is /dev/null and the command has no controlling tty, so an interactive program (an editor, a pager, a TUI app, ssh, sudo, a dev server) cannot work and will only ever reach this timeout. Use a non-interactive equivalent — a build, a test run, `--help`, or the program's headless flags — or a command that exits on its own.",
                        timeout.as_secs(),
                        cmd
                    ));
                }
            };

            let outcome = outcome.map_err(|error| anyhow!("{error}"))?;
            let output =
                format_shell_parts(&outcome.stdout, &outcome.stderr, &outcome.status_text);
            // Deliberately a heuristic, and deliberately checked here rather
            // than by the caller: this is the one place that still knows both
            // the exit status and whether isolation was actually applied. A
            // false positive costs the user one question; a false negative
            // costs the three-round retry loop the stagnation guard exists to
            // kill.
            let failed = !outcome.status_text.contains("exit status: 0");
            if failed && mode != OsIsolation::Off && contains_denial_marker(&output) {
                return Ok(format!("{output}\n{SANDBOX_DENIAL_MARKER}"));
            }
            Ok(output)
        }

        #[cfg(not(unix))]
        {
            let output = tokio::process::Command::new("sh")
                .arg("-c")
                .arg(cmd)
                .current_dir(&self.config.workspace_root)
                .stdin(Stdio::null())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .kill_on_drop(true)
                .output()
                .await
                .context("executing command")?;
            let _ = timeout; // no timeout on this platform yet; documented gap
            Ok(format_shell_output(output))
        }
    }

    /// Start a command that keeps running after this call returns.
    ///
    /// The command is spawned exactly as a one-shot `bash` would be — filtered
    /// environment, own session, kernel isolation — so this buys time, not
    /// privilege.
    pub async fn bash_session(&self, command: &str, yield_ms: u64) -> Result<String> {
        let cmd = self.prepare_shell(command)?.to_string();
        let workspace = self.config.workspace_root.clone();
        #[cfg(unix)]
        let roots = os_isolation::writable_roots(&workspace);
        #[cfg(unix)]
        let mode = self.config.os_isolation;

        let (child, stdin, output) = {
            #[cfg(unix)]
            let mut builder =
                shell_builder(&cmd, &workspace, mode, &roots, std::process::Stdio::piped());
            // No kernel isolation on this platform; the workspace restriction
            // and the filtered environment still apply.
            #[cfg(not(unix))]
            let mut builder = {
                let mut builder = std::process::Command::new("sh");
                builder
                    .arg("-c")
                    .arg(&cmd)
                    .current_dir(&workspace)
                    .stdin(std::process::Stdio::piped())
                    .stdout(std::process::Stdio::piped())
                    .stderr(std::process::Stdio::piped())
                    .kill_on_drop(true);
                builder
            };
            let mut child = builder
                .spawn()
                .map_err(|error| anyhow!("spawning sh: {error}"))?;
            let stdin = child
                .stdin
                .take()
                .ok_or_else(|| anyhow!("stdin unavailable"))?;
            let stdout = child
                .stdout
                .take()
                .ok_or_else(|| anyhow!("stdout unavailable"))?;
            let stderr = child
                .stderr
                .take()
                .ok_or_else(|| anyhow!("stderr unavailable"))?;
            let output: std::sync::Arc<std::sync::Mutex<Vec<u8>>> =
                std::sync::Arc::default();
            sessions::spawn_pipe_reader(stdout, output.clone());
            sessions::spawn_pipe_reader(stderr, output.clone());
            (child, stdin, output)
        };
        // `insert` kills the command if it refuses it (too many sessions
        // alive), so a rejected start never leaves an unreachable process.
        let id = self.sessions.insert(child, stdin, cmd, output)?;
        let report = self
            .sessions
            .wait_and_report(&id, std::time::Duration::from_millis(yield_ms))
            .await?;
        Ok(format_session_report(&id, &report, true))
    }

    /// Send input to a running session and return whatever it printed next.
    pub async fn session_input(&self, id: &str, input: &str, yield_ms: u64) -> Result<String> {
        self.sessions.write(id, input)?;
        let report = self
            .sessions
            .wait_and_report(id, std::time::Duration::from_millis(yield_ms))
            .await?;
        Ok(format_session_report(id, &report, false))
    }

    /// Kill one running session.
    pub fn kill_session(&self, id: &str) -> Result<String> {
        self.sessions.kill(id)
    }

    /// Shared pre-flight checks for shell execution; returns the trimmed
    /// command on success.
    fn prepare_shell<'a>(&self, command: &'a str) -> Result<&'a str> {
        if !self.has_target() {
            return Err(anyhow!(
                "no target directory set — use /sandbox <path> or --sandbox <path>"
            ));
        }
        if !self.config.allow_shell {
            return Err(anyhow!(
                "shell execution disabled in config (sandbox.allow_shell = false)"
            ));
        }
        self.authorize(ToolKind::Shell)?;
        let cmd = command.trim();
        if cmd.is_empty() {
            return Err(anyhow!("empty command"));
        }
        // Advisory only — a foot-gun guard, NOT a security boundary.
        let lower = cmd.to_ascii_lowercase();
        for blocked in ADVISORY_BLOCKED {
            if lower.contains(blocked) {
                return Err(anyhow!("blocked dangerous command: {}", blocked));
            }
        }
        // Defense in depth only — see `first_sensitive_reference`.
        if let Some(reference) = self.first_sensitive_reference(cmd) {
            return Err(anyhow!(
                "command references sensitive file '{}' — denied by the sensitive-file policy (best-effort scan)",
                reference
            ));
        }
        Ok(cmd)
    }

    /// Best-effort scan: refuse commands that *name* a sensitive file.
    /// Quoting (`cat .e''nv`), command substitution and many other shell
    /// features defeat this trivially — it exists only so the common
    /// `cat .env` cannot quietly succeed where `read_file(".env")` fails.
    fn first_sensitive_reference(&self, command: &str) -> Option<String> {
        const DELIMITERS: &[char] = &[
            ' ', '\t', '\n', '\r', ';', '|', '&', '(', ')', '<', '>', '"', '\'', '`',
        ];
        for token in command.split(|c| DELIMITERS.contains(&c)) {
            if token.is_empty() {
                continue;
            }
            let lower = token.to_ascii_lowercase();
            if lower.contains(".git/config") || lower.contains(".git/credentials") {
                return Some(token.to_string());
            }
            for segment in token.split('/') {
                if is_sensitive_name(segment, &self.config.extra_sensitive_names) {
                    return Some(segment.to_string());
                }
            }
        }
        None
    }

    /// Refuse `kind` unless the configured permission mode allows it.
    pub fn authorize(&self, kind: ToolKind) -> Result<()> {
        authorize(self.config.permission_mode, kind)
    }

    /// Execute a ToolCall by name, after the permission layer approves the
    /// tool's capability class.
    /// `allow_once` is an approval granted for a single attempt. It has to be
    /// handed down rather than looked up: an "allow this one" answer lives in
    /// the caller, not in the session's remembered approvals, and re-gating
    /// without it denies the call the user just said yes to.
    pub async fn execute_tool(
        &self,
        tool_call: &ToolCall,
        allow_once: Option<&str>,
    ) -> Result<String> {
        // Never silently fall back to empty arguments: unparseable JSON
        // almost always means the call was truncated mid-stream, and the
        // model deserves an error that says so (and how to recover).
        let args: Value = match serde_json::from_str(&tool_call.arguments) {
            Ok(value) => value,
            Err(_) => {
                return Err(anyhow!(
                    "tool call arguments are not valid JSON — they most likely arrived truncated when the stream was cut mid-call; retry with smaller edits"
                ));
            }
        };
        // Decided here, once, with the session's approvals *and* any approval
        // granted for this one attempt — rather than inside each tool, where
        // an approval the user just gave would be denied a second time.
        match self.gate_call(tool_call, allow_once).gate {
            Gate::Allow => {}
            Gate::Deny { reason } => {
                return Err(anyhow!(
                    "permission denied: {reason}. {}",
                    self.config.permission_mode.remedy()
                ))
            }
            Gate::Ask { reason } => {
                // Spell out what the gate saw. "permission denied" alone tells
                // nobody whether the user was never asked, the answer was lost
                // on the way here, or the key simply did not match.
                let key = self.allowance_key_for(tool_call);
                return Err(anyhow!(
                    "permission denied: {reason}, and this call was not approved \
                     (mode {}, approval key '{key}', remembered approvals: [{}], \
                     one-time approval: {})",
                    self.config.permission_mode.as_str(),
                    if self.allowances.is_empty() {
                        "none".to_string()
                    } else {
                        self.allowances.join(", ")
                    },
                    allow_once.unwrap_or("none"),
                ));
            }
        }
        match tool_call.name.as_str() {
            "ask_user" => Err(anyhow!(
                "ask_user is answered by the client, not by the sandbox"
            )),
            "read_file" => {
                let path = args["path"].as_str().ok_or_else(|| anyhow!("missing path"))?;
                let offset = args["offset"].as_u64().map(|v| v as usize).unwrap_or(1);
                let limit = args["limit"]
                    .as_u64()
                    .map(|v| v as usize)
                    .unwrap_or(DEFAULT_READ_LIMIT);
                self.read_file(path, offset, limit).await
            }
            "write_file" => {
                let path = args["path"].as_str().ok_or_else(|| anyhow!("missing path"))?;
                let content = args["content"].as_str().ok_or_else(|| anyhow!("missing content"))?;
                self.write_file(path, content).await
            }
            "edit_file" => {
                let path = args["path"].as_str().ok_or_else(|| anyhow!("missing path"))?;
                let old = args["old_string"].as_str().ok_or_else(|| anyhow!("missing old_string"))?;
                let new = args["new_string"].as_str().ok_or_else(|| anyhow!("missing new_string"))?;
                self.edit_file(path, old, new).await
            }
            "list_files" => {
                let path = args["path"].as_str().unwrap_or(".");
                self.list_files(path).await
            }
            "apply_patch" => {
                let text = args["patch"].as_str().ok_or_else(|| anyhow!("missing patch"))?;
                self.apply_patch(text).await
            }
            "bash" => {
                let cmd = args["command"].as_str().ok_or_else(|| anyhow!("missing command"))?;
                if args["session"].as_bool().unwrap_or(false) {
                    let yield_ms = args["yield_time_ms"].as_u64().unwrap_or(10_000);
                    return self.bash_session(cmd, yield_ms).await;
                }
                let timeout_secs = args["timeout_secs"].as_u64();
                self.bash_timed(cmd, timeout_secs).await
            }
            "write_stdin" => {
                let id = args["session_id"]
                    .as_str()
                    .ok_or_else(|| anyhow!("missing session_id"))?;
                let input = args["input"].as_str().unwrap_or("");
                let yield_ms = args["yield_time_ms"].as_u64().unwrap_or(5_000);
                self.session_input(id, input, yield_ms).await
            }
            "kill_session" => {
                let id = args["session_id"]
                    .as_str()
                    .ok_or_else(|| anyhow!("missing session_id"))?;
                self.kill_session(id)
            }
            other => Err(anyhow!("unknown tool: {}", other)),
        }
    }
}

/// Which direction an access check is for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Access {
    Read,
    Write,
}

/// Commands that are refused outright. **Advisory only** — trivial to
/// bypass and explicitly not a security control; kept as a foot-gun guard
/// against typo'd destructive commands (see module docs).
const ADVISORY_BLOCKED: &[&str] = &["rm -rf /", "mkfs", ":(){:|:&};:", "dd if="];

/// Environment variables passed through to shell commands. Everything else
/// is stripped so credentials (API keys, tokens, cloud credentials) held by
/// the chatTUI process are not exposed to model-generated commands. This is
/// hygiene, not isolation: a command can still read files containing
/// secrets if it can reach them (see module docs).
const ENV_ALLOWLIST: &[&str] = &[
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "SHELL",
    "TERM",
    "TZ",
    "TMPDIR",
    "TMP",
    "LANG",
    "CARGO_HOME",
    "RUSTUP_HOME",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "PYTHONUNBUFFERED",
    "VIRTUAL_ENV",
    "GOROOT",
    "GOPATH",
    "JAVA_HOME",
    "MAKEFLAGS",
    "CARGO_TARGET_DIR",
];

fn env_var_allowed(name: &OsStr) -> bool {
    let name = name.to_string_lossy();
    ENV_ALLOWLIST.iter().any(|allowed| *allowed == name) || name.starts_with("LC_")
}

/// Built-in sensitive-file policy. Deliberately small and principled —
/// extend it through `SandboxConfig::extra_sensitive_names` instead of
/// growing this list ad hoc:
/// - dotenv-style files: `.env`, `.env.*` and `*.env`
/// - key material: `*.pem`, `*.key`, `*.p12`, `*.pfx`, `*.jks` and the
///   common SSH private-key names (`id_rsa`, `id_ed25519`, …)
/// - git internals: handled in [`Sandbox::check_access`] (all writes under
///   `.git/` are denied — hooks are code execution — plus reads of
///   `.git/config` / `.git/credentials`)
///
/// Accepted trade-off: `.env.example` is caught by the `.env.*` rule and
/// denied too, rather than maintaining an exception list in this phase.
fn is_sensitive_name(name: &str, extra: &[String]) -> bool {
    let lower = name.to_ascii_lowercase();
    if lower == ".env" || lower.starts_with(".env.") || lower.ends_with(".env") {
        return true;
    }
    const KEY_SUFFIXES: &[&str] = &[".pem", ".key", ".p12", ".pfx", ".jks"];
    if KEY_SUFFIXES.iter().any(|suffix| lower.ends_with(suffix)) {
        return true;
    }
    const KEY_NAMES: &[&str] = &["id_rsa", "id_dsa", "id_ecdsa", "id_ed25519"];
    if KEY_NAMES.contains(&lower.as_str()) {
        return true;
    }
    for entry in extra {
        let entry = entry.trim().to_ascii_lowercase();
        if entry.is_empty() {
            continue;
        }
        if entry.starts_with('.') {
            if lower.ends_with(&entry) {
                return true;
            }
        } else if lower == entry {
            return true;
        }
    }
    false
}

/// Lexically resolve `.` and `..` without touching the filesystem. The walk
/// never escapes the root component: `..` at the root is a no-op.
fn normalize_lexical(path: &Path) -> PathBuf {
    let mut result: Vec<Component<'_>> = Vec::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => match result.last() {
                Some(Component::RootDir) | None => {}
                _ => {
                    result.pop();
                }
            },
            other => result.push(other),
        }
    }
    result.into_iter().collect()
}

/// Best-effort canonicalization that also works when the tail of `path`
/// does not exist yet: canonicalize the deepest existing ancestor, then
/// append the remaining components (which, not existing, cannot be
/// symlinks — at resolution time).
fn canonicalize_deepest(path: &Path) -> PathBuf {
    if let Ok(canonical) = path.canonicalize() {
        return canonical;
    }
    let mut suffix: Vec<std::ffi::OsString> = Vec::new();
    let mut current: &Path = path;
    while let Some(parent) = current.parent() {
        if let Some(name) = current.file_name() {
            suffix.push(name.to_os_string());
        }
        if let Ok(canonical) = parent.canonicalize() {
            let mut resolved = canonical;
            for part in suffix.iter().rev() {
                resolved.push(part);
            }
            return resolved;
        }
        current = parent;
    }
    path.to_path_buf()
}

#[cfg(not(unix))] // the unix path renders through `ShellOutcome` instead
fn format_shell_output(output: std::process::Output) -> String {
    format_shell_parts(&output.stdout, &output.stderr, &output.status.to_string())
}

/// Raw result of one sandboxed shell run, before rendering.
#[cfg(unix)]
struct ShellOutcome {
    stdout: Vec<u8>,
    stderr: Vec<u8>,
    status_text: String,
}

/// Spawn `sh -c <cmd>` on the current (unrestricted) worker thread, in its
/// own process group, with the filtered environment, and collect the
/// output. Kernel isolation is installed **inside the forked child, just
/// before execve**, via a `pre_exec` hook — never on this thread — so the
/// process-spawn machinery runs normally and the shell still never executes
/// a single instruction unconfined. `require`-mode failures surface as
/// clean errors here (std plumbs `pre_exec` errors through its spawn-error
/// channel); auto-mode failures are announced by the child on stderr.
#[cfg(unix)]
/// Build the `sh -c` command with chatTUI's environment filtering, its own
/// session, and kernel isolation.
///
/// Shared by the one-shot path and by long-running sessions on purpose: the
/// `setsid` and `pre_exec` blocks below are the reason a command cannot take
/// over the user's terminal, and two copies of them would eventually disagree.
#[cfg(unix)]
fn shell_builder(
    cmd: &str,
    workspace: &Path,
    isolation: OsIsolation,
    writable_roots: &[PathBuf],
    stdin: Stdio,
) -> std::process::Command {
    use std::os::unix::process::CommandExt;

    let mut builder = std::process::Command::new("sh");
    builder
        .arg("-c")
        .arg(cmd)
        .current_dir(workspace)
        .stdin(stdin)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .env_clear();
    for (name, value) in std::env::vars_os() {
        if env_var_allowed(&name) {
            builder.env(name, value);
        }
    }
    // Own process group so a timeout can take down the whole tree
    // (background jobs, grandchildren), not just the `sh` process.
    //
    // DECISION FLAG — `kill -9 -1` from a tool command ends the whole
    // chatTUI session. Chosen approach: seccomp-deny `kill(-1, sig)` inside
    // the sandboxed child (see `seccomp::build_filter` in os_isolation) —
    // the one *broad* kill target, everything-at-once — plus this
    // documented note. Reasoning:
    // - The alternative wrapper/supervisor-as-group-leader idea does NOT
    //   actually help: `kill(-1)` on Linux signals every process the sender
    //   may signal except the sender itself, regardless of process-group
    //   membership, so a supervisor as pgid leader would still die — and so
    //   would chatTUI (same uid, same session). Re-shaping the group only
    //   moves the blast radius.
    // - Documentation alone was rejected because the footgun stays one
    //   accidental command away from ending the session; denying `kill(-1)`
    //   is one small BPF rule, cannot break the timeout kill (that runs
    //   from the unsandboxed parent via `kill_process_group`), and leaves
    //   targeted signals (`kill <pid>`, `kill -<pgid>`, `kill %job`) fully
    //   working for normal build/test flows.
    // - Residual, documented limits: targeted `kill <pid>`, `pkill` and
    //   `pidfd_send_signal` can still reach same-uid processes, chatTUI
    //   included — that is inherent to unprivileged same-uid sandboxing
    //   (the agent can always read /proc and find the pid) and is called
    //   out in the module security-model docs.
    // Own *session*, not merely an own process group.
    //
    // `setpgid` alone leaves the command attached to chatTUI's controlling
    // terminal, so anything the agent runs can open `/dev/tty` and take the
    // real terminal over: a TUI or an editor switches to the alternate
    // screen, hides the cursor and puts the terminal into raw mode, and when
    // the command is killed at the timeout it never restores any of it.
    // `setsid` detaches the controlling terminal, which makes that open fail
    // — an interactive program simply cannot run, instead of silently
    // destroying the session it was launched from.
    //
    // It also makes the child its own session and process-group leader
    // (pgid == pid), which is exactly what the timeout kill targets, so the
    // whole process tree is still taken down. `setpgid` stays as the
    // fallback for the rare kernel that refuses `setsid` here.
    unsafe {
        builder.pre_exec(|| {
            if libc::setsid() == -1 && libc::setpgid(0, 0) != 0 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
    if isolation != OsIsolation::Off {
        let roots = writable_roots.to_vec();
        let strict = isolation == OsIsolation::Require;
        unsafe {
            builder.pre_exec(move || {
                os_isolation::confine_in_child(&roots, os_isolation::WRITE_ONLY_DEVICES, strict)
            });
        }
    }
    builder
}

#[cfg(unix)]
fn run_shell_process(
    cmd: &str,
    workspace: &Path,
    pid_slot: &AtomicI32,
    isolation: OsIsolation,
    writable_roots: &[PathBuf],
) -> Result<ShellOutcome, String> {
    use std::io::Read;

    let mut builder = shell_builder(cmd, workspace, isolation, writable_roots, Stdio::null());

    let mut child = match builder.spawn() {
        Ok(child) => child,
        Err(error) => {
            if isolation == OsIsolation::Require
                && matches!(
                    error.kind(),
                    std::io::ErrorKind::PermissionDenied | std::io::ErrorKind::Unsupported
                )
            {
                return Err(
                    "OS-level isolation is required but unavailable on this system - refusing \
                     to run the command. Set sandbox.os_isolation = \"auto\" in config.json to \
                     allow an explicit unrestricted fallback."
                        .to_string(),
                );
            }
            return Err(format!("spawning sh: {error}"));
        }
    };
    pid_slot.store(child.id() as i32, Ordering::SeqCst);

    // Drain both pipes concurrently so a chatty child can never fill them
    // and deadlock while `wait()` runs.
    let mut stdout_pipe = child.stdout.take().ok_or_else(|| "stdout unavailable".to_string())?;
    let mut stderr_pipe = child.stderr.take().ok_or_else(|| "stderr unavailable".to_string())?;
    let stdout_reader = std::thread::spawn(move || {
        let mut buffer = Vec::new();
        let _ = stdout_pipe.read_to_end(&mut buffer);
        buffer
    });
    let stderr_reader = std::thread::spawn(move || {
        let mut buffer = Vec::new();
        let _ = stderr_pipe.read_to_end(&mut buffer);
        buffer
    });

    let status = child.wait().map_err(|error| format!("waiting on sh: {error}"))?;
    let stdout = stdout_reader.join().unwrap_or_default();
    let stderr = stderr_reader.join().unwrap_or_default();
    Ok(ShellOutcome { stdout, stderr, status_text: status.to_string() })
}

/// Appended to shell output when the kernel sandbox looks like the reason the
/// command failed. The caller strips it before the output is shown or stored;
/// it exists only to turn "the sandbox said no" into a question instead of an
/// error the model will retry three times. Control characters keep it from
/// colliding with anything a program would actually print.
pub const SANDBOX_DENIAL_MARKER: &str = "\u{1}sandbox-denial\u{1}";

/// Render one look at a session for the model.
///
/// The state matters more than the text: a model that cannot tell "still
/// running" from "finished with no output" will either wait forever or assume
/// success.
fn format_session_report(
    id: &str,
    report: &sessions::SessionReport,
    started: bool,
) -> String {
    let head = if started {
        format!(
            "session {id} started ({}s of output below). It is still running: use write_stdin to send it input, or ask for more output later.",
            report.running_for.as_secs()
        )
    } else {
        format!("session {id}, {}s in.", report.running_for.as_secs())
    };
    let state = match &report.exited {
        Some(status) => format!("It has now exited ({status})."),
        None => "It is still running.".to_string(),
    };
    let body = if report.output.trim().is_empty() {
        "(no output yet)".to_string()
    } else {
        report.output.clone()
    };
    let truncated = if report.truncated {
        "\n(output shortened — the beginning and the end were kept)"
    } else {
        ""
    };
    format!("{head}\n\n{body}{truncated}\n\n{state}")
}

/// Phrases that mean "the kernel refused this", as opposed to the program
/// failing on its own merits. The first two are filesystem denials (Landlock);
/// the rest are what a blocked `socket()` looks like from the programs that
/// hit it most often — cargo, npm, pip, curl, git.
const SANDBOX_DENIAL_MARKERS: [&str; 6] = [
    "permission denied",
    "operation not permitted",
    "network is unreachable",
    "address family not supported by protocol",
    "temporary failure in name resolution",
    "couldn't connect to server",
];

fn contains_denial_marker(output: &str) -> bool {
    let lower = output.to_ascii_lowercase();
    SANDBOX_DENIAL_MARKERS
        .iter()
        .any(|marker| lower.contains(marker))
}

/// Remove the denial marker, and the newline it was appended with.
pub fn strip_denial_marker(output: &str) -> String {
    output.replace(&format!("\n{SANDBOX_DENIAL_MARKER}"), "")
}

fn format_shell_parts(stdout: &[u8], stderr: &[u8], status_text: &str) -> String {
    // Strip terminal control sequences before anything else. These bytes were
    // written by a program that may believe it owns a terminal — a TUI, a
    // coloured build log, a progress bar — and displaying them verbatim in
    // the transcript would hand the user's terminal to whatever the agent
    // last ran. Stripping first also means the byte budget below is spent on
    // real content rather than on escapes.
    let stdout = ansi::strip(&String::from_utf8_lossy(stdout));
    let stderr = ansi::strip(&String::from_utf8_lossy(stderr));
    let mut result = String::new();
    if !stdout.is_empty() {
        result.push_str(&stdout);
    }
    if !stderr.is_empty() {
        if !result.is_empty() {
            result.push_str("\n--- stderr ---\n");
        }
        result.push_str(&stderr);
    }
    if result.is_empty() {
        result = format!("(command exited with status {status_text})");
    }
    if result.len() > MAX_TOOL_OUTPUT_BYTES {
        result = keep_head_and_tail(&result, MAX_TOOL_OUTPUT_BYTES);
    }
    result
}

/// Largest byte index at or before `index` that falls on a char boundary.
/// Slicing a `str` anywhere else panics, and tool output is arbitrary bytes
/// from an arbitrary program.
fn floor_char_boundary(text: &str, index: usize) -> usize {
    if index >= text.len() {
        return text.len();
    }
    let mut at = index;
    while at > 0 && !text.is_char_boundary(at) {
        at -= 1;
    }
    at
}

/// Smallest byte index at or after `index` that falls on a char boundary.
fn ceil_char_boundary(text: &str, index: usize) -> usize {
    let mut at = index.min(text.len());
    while at < text.len() && !text.is_char_boundary(at) {
        at += 1;
    }
    at
}

/// Keep the beginning and the end of a long output, dropping the middle.
///
/// Cutting only the end deletes exactly the part that matters: a build or a
/// test run reports its failure last, so head-only truncation hands the model
/// a log whose conclusion was removed, and it retries blind. Half the budget
/// goes to each end, cut on line boundaries where possible, with a marker
/// saying how much was dropped — so nobody has to guess whether what they are
/// reading is the whole output.
pub(crate) fn keep_head_and_tail(input: &str, budget: usize) -> String {
    const MARKER: &str = "output truncated: ";
    // Room for the marker, the dropped-byte count and the newlines.
    let usable = budget.saturating_sub(MARKER.len() + 40).max(256);
    let half = usable / 2;

    let mut head_end = floor_char_boundary(input, half);
    if let Some(newline) = input[..head_end].rfind('\n') {
        head_end = newline + 1;
    }
    let mut tail_from = ceil_char_boundary(input, input.len().saturating_sub(usable - half));
    if let Some(offset) = input[tail_from..].find('\n') {
        tail_from += offset + 1;
    }

    if tail_from <= head_end || tail_from >= input.len() {
        // Degenerate: the budget is too small to split, or the tail window
        // landed past the end. Keep the head and say so.
        let cut = floor_char_boundary(input, usable);
        let dropped = input.len() - cut;
        return format!("{}\n{MARKER}{dropped} bytes dropped", &input[..cut]);
    }

    let dropped = tail_from - head_end;
    format!(
        "{}\n{MARKER}{dropped} bytes dropped …\n{}",
        &input[..head_end],
        &input[tail_from..]
    )
}

/// Best-effort SIGKILL for a whole process group. Uses the shell builtin so
/// no extra crates are needed. Group members that escaped into their own
/// session (double-fork / setsid) survive — a documented limitation.
#[cfg(unix)]
async fn kill_process_group(pid: u32) {
    if pid == 0 {
        // `-0` would mean "my own process group" — never kill that.
        return;
    }
    let script = format!("kill -9 -{pid} 2>/dev/null || true");
    let kill = tokio::process::Command::new("sh")
        .arg("-c")
        .arg(&script)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .kill_on_drop(true)
        .output();
    let _ = tokio::time::timeout(Duration::from_secs(2), kill).await;
}

#[cfg(not(unix))]
async fn kill_process_group(_pid: u32) {}

fn expand_user_path(raw: &str) -> Result<PathBuf> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err(anyhow!("empty target path"));
    }
    if trimmed == "~" {
        let home = std::env::var_os("HOME").ok_or_else(|| anyhow!("HOME is not set"))?;
        return Ok(PathBuf::from(home));
    }
    if let Some(rest) = trimmed.strip_prefix("~/") {
        let home = std::env::var_os("HOME").ok_or_else(|| anyhow!("HOME is not set"))?;
        return Ok(PathBuf::from(home).join(rest));
    }
    Ok(PathBuf::from(trimmed))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(unix)]
    use std::time::Instant;

    fn unique_root(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("chatTUI_sec_{}_{}", name, std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[cfg(unix)]
    fn sandbox_with_timeout(root: PathBuf, secs: u64) -> Sandbox {
        let mut cfg = SandboxConfig::default();
        cfg.workspace_root = root.canonicalize().unwrap_or(root);
        cfg.shell_timeout = Duration::from_secs(secs);
        Sandbox::new(cfg)
    }

    #[test]
    fn sandbox_resolves_inside_root() {
        let dir = unique_root("resolve");
        let sandbox = Sandbox::with_root(dir.clone());
        let p = sandbox.resolve_path("src/main.rs").unwrap();
        assert!(p.starts_with(&dir));
    }

    #[test]
    fn sandbox_blocks_traversal() {
        let dir = unique_root("traversal");
        let sandbox = Sandbox::with_root(dir);
        assert!(sandbox.resolve_path("../../etc/passwd").is_err());
        assert!(sandbox.resolve_path("..").is_err());
        assert!(sandbox.resolve_path("a/../../..").is_err());
        assert!(sandbox.resolve_path("./sub/../../../x").is_err());
    }

    #[test]
    fn absolute_paths_outside_workspace_are_denied() {
        let dir = unique_root("abs-deny");
        let sandbox = Sandbox::with_root(dir);
        assert!(sandbox.resolve_path("/etc/passwd").is_err());
        assert!(sandbox.resolve_path("/usr/bin/env").is_err());
    }

    #[test]
    fn absolute_path_with_dotdot_through_missing_dir_cannot_escape() {
        // Regression: canonicalize() fails for the missing directory, and the
        // old fallback accepted the raw unresolved path — the kernel would
        // then resolve the `..` outside the workspace.
        let dir = unique_root("abs-dotdot");
        let sandbox = Sandbox::with_root(dir.clone());
        let outside = dir.join("missing-dir").join("..").join("..").join("etc");
        let escape = outside.join("passwd");
        assert!(escape.is_absolute());
        assert!(sandbox.resolve_path(&escape.to_string_lossy()).is_err());
    }

    #[test]
    fn absolute_path_inside_workspace_is_allowed_even_if_missing() {
        let dir = unique_root("abs-allow");
        let sandbox = Sandbox::with_root(dir.clone());
        let root = dir.canonicalize().unwrap();
        let target = root.join("sub").join("new-file.txt");
        let resolved = sandbox.resolve_path(&target.to_string_lossy()).unwrap();
        assert!(resolved.starts_with(&root));
        assert_eq!(resolved, target);
    }

    #[test]
    fn dotdot_that_stays_inside_the_workspace_is_allowed() {
        let dir = unique_root("dotdot-inside");
        std::fs::create_dir_all(dir.join("sub")).unwrap();
        let sandbox = Sandbox::with_root(dir.clone());
        let root = dir.canonicalize().unwrap();
        let resolved = sandbox.resolve_path("sub/../keep.txt").unwrap();
        assert_eq!(resolved, root.join("keep.txt"));
    }

    #[test]
    fn nul_bytes_and_empty_paths_are_rejected() {
        let dir = unique_root("nul");
        let sandbox = Sandbox::with_root(dir);
        assert!(sandbox.resolve_path("").is_err());
        assert!(sandbox.resolve_path("   ").is_err());
        assert!(sandbox.resolve_path("a\0b").is_err());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn symlink_escape_is_denied() {
        let dir = unique_root("symlink");
        let sandbox = Sandbox::with_root(dir.clone());
        std::os::unix::fs::symlink("/etc/hosts", dir.join("escape.txt")).unwrap();
        std::os::unix::fs::symlink("/", dir.join("out")).unwrap();
        // Symlinked file pointing outside.
        assert!(sandbox.read_file("escape.txt").await.is_err());
        // Directory symlink: read through it.
        assert!(sandbox.read_file("out/etc/hosts").await.is_err());
        // Directory symlink: write through it.
        assert!(sandbox
            .write_file("out/escape-write.txt", "x")
            .await
            .is_err());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn nested_symlink_escape_is_denied() {
        let dir = unique_root("nested-symlink");
        std::fs::create_dir_all(dir.join("a")).unwrap();
        std::os::unix::fs::symlink("/etc", dir.join("a").join("b")).unwrap();
        let sandbox = Sandbox::with_root(dir);
        assert!(sandbox.read_file("a/b/passwd").await.is_err());
    }

    #[tokio::test]
    async fn reading_nonexistent_paths_fails_cleanly() {
        let dir = unique_root("missing");
        let sandbox = Sandbox::with_root(dir);
        let err = sandbox.read_file("does/not/exist.txt").await.unwrap_err();
        assert!(!err.to_string().contains("panic"));
        // And writing into a non-existent nested directory still works.
        sandbox
            .write_file("does/new/file.txt", "created")
            .await
            .unwrap();
        assert_eq!(
            sandbox
                .read_file("does/new/file.txt")
                .await
                .unwrap(),
            "created"
        );
    }

    #[tokio::test]
    async fn sensitive_files_are_denied_for_every_file_tool() {
        let dir = unique_root("sensitive");
        let sandbox = Sandbox::with_root(dir.clone());
        // dotenv-style names
        assert!(sandbox.read_file(".env").await.is_err());
        assert!(sandbox.write_file(".env", "X=1").await.is_err());
        assert!(sandbox.edit_file(".env", "A", "B").await.is_err());
        assert!(sandbox.read_file("config/.env.local").await.is_err());
        assert!(sandbox.read_file("production.env").await.is_err());
        // key material
        assert!(sandbox.read_file("server.pem").await.is_err());
        assert!(sandbox.write_file("tls/host.key", "x").await.is_err());
        assert!(sandbox.read_file("id_ed25519").await.is_err());
        // git internals: reads of config/credentials, all writes
        assert!(sandbox.read_file(".git/config").await.is_err());
        assert!(sandbox.write_file(".git/hooks/pre-commit", "#!/bin/sh").await.is_err());
        // non-sensitive git internals stay readable
        std::fs::create_dir_all(dir.join(".git")).unwrap();
        std::fs::write(dir.join(".git").join("HEAD"), "ref: refs/heads/main").unwrap();
        assert!(sandbox.read_file(".git/HEAD").await.is_ok());
        // normal files keep working
        sandbox.write_file("normal.txt", "fine").await.unwrap();
        assert_eq!(sandbox.read_file("normal.txt").await.unwrap(), "fine");
    }

    #[tokio::test]
    async fn list_files_hides_sensitive_entries() {
        let dir = unique_root("list");
        std::fs::write(dir.join(".env"), "SECRET=1").unwrap();
        std::fs::write(dir.join("id_rsa"), "key").unwrap();
        std::fs::write(dir.join("readme.md"), "hi").unwrap();
        let sandbox = Sandbox::with_root(dir);
        let listing = sandbox.list_files(".").await.unwrap();
        assert!(listing.contains("readme.md"));
        assert!(!listing.contains(".env"));
        assert!(!listing.contains("id_rsa"));
    }

    #[tokio::test]
    async fn extra_sensitive_names_from_config_are_enforced() {
        let dir = unique_root("extra");
        std::fs::write(dir.join("secrets.json"), "{}").unwrap();
        std::fs::write(dir.join("token.txt"), "t").unwrap();
        std::fs::write(dir.join("anything-else.txt"), "ok").unwrap();
        let mut cfg = SandboxConfig::default();
        cfg.workspace_root = dir.canonicalize().unwrap();
        cfg.extra_sensitive_names = vec!["secrets.json".into(), ".token".into()];
        let sandbox = Sandbox::new(cfg);
        assert!(sandbox.read_file("secrets.json").await.is_err());
        assert!(sandbox.read_file("token.txt").await.is_err()); // suffix match
        assert!(sandbox.read_file("anything-else.txt").await.is_ok());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn write_through_parent_symlink_outside_workspace_is_denied() {
        let dir = unique_root("parent-symlink");
        let outside = unique_root("parent-symlink-outside");
        std::os::unix::fs::symlink(&outside, dir.join("link")).unwrap();
        let sandbox = Sandbox::with_root(dir);
        assert!(sandbox.write_file("link/child.txt", "x").await.is_err());
        assert!(!outside.join("child.txt").exists());
    }

    #[tokio::test]
    async fn sandbox_read_write() {
        let dir = unique_root("rw");
        let sandbox = Sandbox::with_root(dir.clone());
        sandbox.write_file("test.txt", "hello").await.unwrap();
        let content = sandbox.read_file("test.txt").await.unwrap();
        assert_eq!(content, "hello");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn default_sandbox_has_no_implicit_cwd_target() {
        let sandbox = Sandbox::new(SandboxConfig::default());
        assert!(!sandbox.has_target());
        assert!(sandbox.resolve_path("src/main.rs").is_err());
        // No target also means no shell, regardless of mode.
        let rt = tokio::runtime::Runtime::new().unwrap();
        assert!(rt.block_on(sandbox.bash("echo hi")).is_err());
    }

    #[tokio::test]
    async fn writes_stay_under_configured_target_not_crate_root() {
        let target = unique_root("dst");
        let crate_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let sandbox = Sandbox::with_root(target.clone());
        sandbox
            .write_file("hello_agent.txt", "hi")
            .await
            .unwrap();
        let written = target.join("hello_agent.txt");
        assert!(written.exists());
        assert!(written.starts_with(&target));
        assert!(!crate_root.join("hello_agent.txt").exists());
        let _ = std::fs::remove_dir_all(&target);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn bash_completes_within_timeout() {
        let dir = unique_root("bash-ok");
        let sandbox = sandbox_with_timeout(dir, 10);
        let out = sandbox.bash("echo ok").await.unwrap();
        assert!(out.contains("ok"));
    }

    #[test]
    fn shell_timeout_is_clamped_away_from_zero() {
        let mut cfg = SandboxConfig::default();
        cfg.shell_timeout = Duration::ZERO;
        assert_eq!(cfg.effective_shell_timeout(), Duration::from_secs(MIN_SHELL_TIMEOUT_SECS));
        cfg.shell_timeout = Duration::from_secs(45);
        assert_eq!(cfg.effective_shell_timeout(), Duration::from_secs(45));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn bash_timeout_is_enforced() {
        let dir = unique_root("timeout");
        let sandbox = sandbox_with_timeout(dir, 1);
        let started = Instant::now();
        let err = sandbox.bash("sleep 5").await.unwrap_err();
        let elapsed = started.elapsed();
        assert!(
            err.to_string().contains("timed out"),
            "unexpected error: {err}"
        );
        assert!(elapsed.as_secs() < 4, "timeout took too long: {elapsed:?}");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn bash_timeout_kills_background_processes() {
        // `sh` exits only after `wait`; the backgrounded sleep keeps the
        // output pipes open, so a naive implementation would hang forever
        // here. The timeout must fire AND the process group must die.
        let dir = unique_root("timeout-group");
        let sandbox = sandbox_with_timeout(dir, 1);
        let result = sandbox.bash("sleep 37 & wait").await;
        assert!(result.is_err());
        let mut gone = false;
        for _ in 0..40 {
            if !process_running("sleep 37") {
                gone = true;
                break;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        assert!(gone, "backgrounded `sleep 37` survived the timeout kill");
    }

    #[cfg(unix)]
    fn process_running(needle: &str) -> bool {
        let entries = match std::fs::read_dir("/proc") {
            Ok(entries) => entries,
            Err(_) => return false,
        };
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if !name.bytes().all(|b| b.is_ascii_digit()) {
                continue;
            }
            if let Ok(cmdline) = std::fs::read(entry.path().join("cmdline")) {
                let text = String::from_utf8_lossy(&cmdline).replace('\0', " ");
                if text.contains(needle) {
                    return true;
                }
            }
        }
        false
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn bash_does_not_inherit_sensitive_environment() {
        let dir = unique_root("env");
        let sandbox = Sandbox::with_root(dir);
        std::env::set_var("CHATTUI_SECRET_TEST", "top-secret-value");
        let secret = sandbox.bash("printenv CHATTUI_SECRET_TEST").await.unwrap();
        assert!(
            !secret.contains("top-secret-value"),
            "credentials leaked into the shell environment"
        );
        // The allowlist keeps the shell usable for builds.
        let path = sandbox.bash("printenv PATH").await.unwrap();
        assert!(!path.trim().is_empty(), "allowlisted PATH must survive");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn bash_denies_commands_naming_sensitive_files() {
        let dir = unique_root("shell-sensitive");
        let sandbox = Sandbox::with_root(dir);
        let err = sandbox.bash("cat .env").await.unwrap_err();
        assert!(err.to_string().contains("sensitive"), "{err}");
        assert!(sandbox.bash("cat config/id_rsa").await.is_err());
        assert!(sandbox.bash("less .git/config").await.is_err());
        // Documented trade-off: even harmless uses of a sensitive name are
        // refused (the scan is textual by design).
        assert!(sandbox.bash("echo .env").await.is_err());
        assert!(sandbox.bash("echo hello world").await.is_ok());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn shell_reads_outside_the_workspace_are_allowed_by_design() {
        // Pins the honest behavior in BOTH modes: even with kernel isolation
        // active, reads outside the workspace stay allowed (toolchains must
        // read system files; secret-read protection is application policy).
        // Writes outside are covered by the isolation-specific tests below.
        let dir = unique_root("no-sandbox");
        let sandbox = Sandbox::with_root(dir);
        let out = sandbox.bash("cat /etc/hosts").await.unwrap();
        assert!(!out.trim().is_empty());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn bash_kernel_isolation_denies_outside_writes_when_supported() {
        if !os_isolation::os_isolation_supported() {
            eprintln!("skipping: kernel lacks Landlock/seccomp support");
            return;
        }
        let Some(home) = std::env::var_os("HOME")
            .map(PathBuf::from)
            .filter(|p| !p.as_os_str().is_empty())
        else {
            eprintln!("skipping: HOME is not set");
            return;
        };
        let dir = unique_root("iso-write");
        let mut cfg = SandboxConfig::default();
        cfg.workspace_root = dir.canonicalize().unwrap();
        cfg.os_isolation = OsIsolation::Require;
        let sandbox = Sandbox::new(cfg);

        let escape = home.join("chattui_iso_escape_probe");
        let _ = std::fs::remove_file(&escape);

        // Write outside the workspace: the kernel must refuse it. The shell
        // still exits normally, so the denial shows up in stderr output.
        let out = sandbox
            .bash(&format!("touch '{}'", escape.display()))
            .await
            .expect("the shell itself must run");
        let lower = out.to_ascii_lowercase();
        assert!(
            lower.contains("permission denied") || lower.contains("operation not permitted"),
            "expected a filesystem denial, got: {out}"
        );
        assert!(
            !escape.exists(),
            "the sandboxed shell must not be able to write outside the workspace"
        );

        // The workspace itself stays writable, and normal output flows.
        let out = sandbox
            .bash("echo ok > inside.txt && cat inside.txt")
            .await
            .unwrap();
        assert!(out.contains("ok"), "unexpected output: {out}");
        assert!(dir.join("inside.txt").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn bash_network_access_is_denied_when_isolation_is_supported() {
        if !os_isolation::os_isolation_supported() {
            eprintln!("skipping: kernel lacks Landlock/seccomp support");
            return;
        }
        let dir = unique_root("iso-net");
        let sandbox = Sandbox::with_root(dir);
        // `ping` needs a socket; under isolation the kernel refuses it.
        let out = sandbox
            .bash("ping -c1 -W1 127.0.0.1")
            .await
            .expect("the shell itself must run");
        let lower = out.to_ascii_lowercase();
        if lower.contains("not found") {
            eprintln!("skipping: ping is not installed");
            return;
        }
        assert!(
            lower.contains("permitted") || lower.contains("denied"),
            "expected a socket permission failure, got: {out}"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn bash_auto_mode_fallback_is_explicit_not_silent() {
        let dir = unique_root("iso-auto");
        let sandbox = Sandbox::with_root(dir);
        let out = sandbox.bash("echo hi").await.unwrap();
        let supported = os_isolation::os_isolation_supported();
        if supported {
            assert!(
                !out.contains("os-level isolation unavailable"),
                "no warning may appear when isolation applied: {out}"
            );
        } else {
            assert!(
                out.contains("os-level isolation unavailable"),
                "the unrestricted fallback must say so explicitly: {out}"
            );
        }
        assert!(out.contains("hi"));
    }

    #[tokio::test]
    async fn bash_require_mode_fails_closed_when_isolation_unsupported() {
        // Only meaningful on machines without kernel support; elsewhere the
        // command simply runs (confined), so the test asserts the invariant
        // that holds in both worlds.
        let dir = unique_root("iso-require");
        let mut cfg = SandboxConfig::default();
        cfg.workspace_root = dir.canonicalize().unwrap();
        cfg.os_isolation = OsIsolation::Require;
        let sandbox = Sandbox::new(cfg);
        let result = sandbox.bash("echo hi").await;
        if os_isolation::os_isolation_supported() {
            assert!(result.unwrap().contains("hi"));
        } else {
            let error = result.unwrap_err().to_string();
            assert!(error.contains("required but unavailable"), "{error}");
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn advisory_denylist_still_blocks_obvious_footguns() {
        let dir = unique_root("advisory");
        let sandbox = Sandbox::with_root(dir);
        let err = sandbox.bash("rm -rf /").await.unwrap_err();
        assert!(err.to_string().contains("blocked dangerous command"));
    }

    #[tokio::test]
    async fn permission_modes_gate_tools() {
        let dir = unique_root("perm");

        // read-only: reads pass the policy (file-not-found, not permission),
        // writes/edits/shell are refused.
        let mut cfg = SandboxConfig::default();
        cfg.workspace_root = dir.canonicalize().unwrap();
        cfg.permission_mode = PermissionMode::ReadOnly;
        let sandbox = Sandbox::new(cfg);
        let err = sandbox.read_file("missing.txt").await.unwrap_err();
        assert!(err.to_string().contains("file not found"));
        let err = sandbox.write_file("x.txt", "y").await.unwrap_err();
        assert!(err.to_string().contains("permission denied"), "{err}");
        let err = sandbox.edit_file("x.txt", "a", "b").await.unwrap_err();
        assert!(err.to_string().contains("permission denied"));
        let err = sandbox.bash("echo hi").await.unwrap_err();
        assert!(err.to_string().contains("permission denied"));

        // workspace-write: files yes, shell no.
        let mut cfg = SandboxConfig::default();
        cfg.workspace_root = dir.canonicalize().unwrap();
        cfg.permission_mode = PermissionMode::WorkspaceWrite;
        let sandbox = Sandbox::new(cfg);
        sandbox.write_file("x.txt", "y").await.unwrap();
        assert_eq!(sandbox.read_file("x.txt").await.unwrap(), "y");
        let err = sandbox.bash("echo hi").await.unwrap_err();
        assert!(err.to_string().contains("permission denied"));

        // ask-before-write denies until interactive approval exists.
        let mut cfg = SandboxConfig::default();
        cfg.workspace_root = dir.canonicalize().unwrap();
        cfg.permission_mode = PermissionMode::AskBeforeWrite;
        let sandbox = Sandbox::new(cfg);
        let err = sandbox.write_file("y.txt", "y").await.unwrap_err();
        assert!(err.to_string().contains("permission denied"));
        assert!(err.to_string().contains("ask-before-write"));

        // The legacy master switch still wins over full-auto for shell.
        let mut cfg = SandboxConfig::default();
        cfg.workspace_root = dir.canonicalize().unwrap();
        cfg.permission_mode = PermissionMode::FullAuto;
        cfg.allow_shell = false;
        let sandbox = Sandbox::new(cfg);
        let err = sandbox.bash("echo hi").await.unwrap_err();
        assert!(err.to_string().contains("allow_shell"));
    }

    #[tokio::test]
    async fn execute_tool_refuses_unknown_tools_and_denied_kinds() {
        let dir = unique_root("dispatch");
        let mut cfg = SandboxConfig::default();
        cfg.workspace_root = dir.canonicalize().unwrap();
        cfg.permission_mode = PermissionMode::ReadOnly;
        let sandbox = Sandbox::new(cfg);

        let unknown = ToolCall::new("1", "rm_file", "{}");
        let err = sandbox.execute_tool(&unknown, None).await.unwrap_err();
        assert!(err.to_string().contains("unknown tool"));

        let bash = ToolCall::new("2", "bash", r#"{"command":"echo hi"}"#);
        let err = sandbox.execute_tool(&bash, None).await.unwrap_err();
        assert!(err.to_string().contains("permission denied"));
    }

    #[tokio::test]
    async fn execute_tool_runs_allowed_kinds_end_to_end() {
        let dir = unique_root("dispatch-ok");
        let sandbox = Sandbox::with_root(dir);
        let write = ToolCall::new("1", "write_file", r#"{"path":"a.txt","content":"hello"}"#);
        sandbox.execute_tool(&write, None).await.unwrap();
        let read = ToolCall::new("2", "read_file", r#"{"path":"a.txt"}"#);
        assert_eq!(sandbox.execute_tool(&read, None).await.unwrap(), "hello");
        let list = ToolCall::new("3", "list_files", "{}");
        assert!(sandbox.execute_tool(&list, None).await.unwrap().contains("a.txt"));
    }

    #[tokio::test]
    async fn execute_tool_reports_truncated_arguments_explicitly() {
        let dir = unique_root("trunc-args");
        let sandbox = Sandbox::with_root(dir);
        // Unterminated JSON — what a stream cut mid-arguments leaves behind.
        let call = ToolCall::new("1", "read_file", r#"{"path": "#);
        let err = sandbox.execute_tool(&call, None).await.unwrap_err();
        assert!(err.to_string().contains("truncated"), "unexpected: {err}");
    }

    #[tokio::test]
    async fn read_file_serves_numbered_windows() {
        let dir = unique_root("read-window");
        let body: String = (1..=600).map(|i| format!("line {i}\n")).collect();
        fs::write(dir.join("big.txt"), body).await.unwrap();
        let sandbox = Sandbox::with_root(dir);

        // First window: 250 numbered lines plus a pointer onward.
        let out = sandbox.read_file("big.txt", 1, 250).await.unwrap();
        assert!(out.contains("lines 1-250 of 600"), "header: {out}");
        assert!(out.contains("1│line 1"));
        assert!(out.contains("250│line 250"));
        assert!(!out.contains("line 251"));
        assert!(out.contains("offset 251"), "footer must point onward: {out}");

        // A later window; the tail carries no onward pointer.
        let out = sandbox.read_file("big.txt", 551, 250).await.unwrap();
        assert!(out.contains("lines 551-600 of 600"), "header: {out}");
        assert!(!out.contains("more line"));

        // Past the end: informative, not a crash.
        let out = sandbox.read_file("big.txt", 700, 250).await.unwrap();
        assert!(out.contains("past the end"), "{out}");

        // Empty file: plain and simple.
        let out = sandbox.read_file("empty-missing", 1, 250).await;
        assert!(out.is_err(), "a missing file is still an error");
    }

    #[tokio::test]
    async fn write_and_edit_report_line_stats() {
        let dir = unique_root("stats");
        let sandbox = Sandbox::with_root(dir);

        // New file: additions only.
        let out = sandbox.write_file("a.txt", "one\ntwo\n").await.unwrap();
        assert_eq!(out, "wrote a.txt (+2)");

        // Overwrite: additions and removals.
        let out = sandbox
            .write_file("a.txt", "one\ntwo\nthree\nfour\nfive\n")
            .await
            .unwrap();
        assert_eq!(out, "wrote a.txt (+5 -2)");

        // Edit: one line replaced by two.
        let out = sandbox
            .edit_file("a.txt", "three", "three-a\nthree-b")
            .await
            .unwrap();
        assert_eq!(out, "edited a.txt (+2 -1)");
    }

    #[tokio::test]
    async fn apply_patch_writes_several_files_at_once() {
        let dir = unique_root("patch-multi");
        fs::write(dir.join("a.txt"), "alpha\n").await.unwrap();
        let sandbox = Sandbox::with_root(dir.clone());

        let out = sandbox
            .apply_patch(
                "*** Begin Patch\n\
                 *** Add File: nested/b.txt\n+beta\n\
                 *** Update File: a.txt\n-alpha\n+alpha2\n\
                 *** End Patch",
            )
            .await
            .unwrap();
        assert!(out.contains("added nested/b.txt (+1 -0)"), "{out}");
        assert!(out.contains("updated a.txt (+1 -1)"), "{out}");
        assert_eq!(fs::read_to_string(dir.join("a.txt")).await.unwrap(), "alpha2\n");
        assert_eq!(
            fs::read_to_string(dir.join("nested/b.txt")).await.unwrap(),
            "beta\n"
        );
    }

    #[tokio::test]
    async fn apply_patch_refuses_to_escape_the_workspace() {
        let dir = unique_root("patch-escape");
        let sandbox = Sandbox::with_root(dir);
        let error = sandbox
            .apply_patch("*** Begin Patch\n*** Add File: ../outside.txt\n+x\n*** End Patch")
            .await
            .unwrap_err();
        assert!(error.to_string().contains("escapes workspace"), "{error}");
    }

    #[tokio::test]
    async fn apply_patch_in_read_only_mode_is_denied() {
        let dir = unique_root("patch-readonly");
        let mut cfg = SandboxConfig::default();
        cfg.workspace_root = dir.canonicalize().unwrap();
        cfg.permission_mode = PermissionMode::ReadOnly;
        let sandbox = Sandbox::new(cfg);
        let error = sandbox
            .apply_patch("*** Begin Patch\n*** Add File: a.txt\n+x\n*** End Patch")
            .await
            .unwrap_err();
        assert!(error.to_string().contains("permission denied"), "{error}");
    }

    #[test]
    fn output_truncation_never_panics_on_multibyte_boundaries() {
        let mut stdout = Vec::new();
        // Multi-byte characters straddling the truncation point.
        while stdout.len() < MAX_TOOL_OUTPUT_BYTES + 16 {
            stdout.extend_from_slice("é".as_bytes());
        }
        let truncated = format_shell_parts(&stdout, &[], "0");
        assert!(
            truncated.len() <= MAX_TOOL_OUTPUT_BYTES + 64,
            "grew past the budget: {}",
            truncated.len()
        );
        assert!(truncated.contains("output truncated"));
    }

    #[test]
    fn output_truncation_keeps_the_end_where_the_failure_is() {
        // A build log: the reason it failed is the last line, which is
        // exactly what head-only truncation deletes.
        let mut body = String::from("first line of the log\n");
        while body.len() < MAX_TOOL_OUTPUT_BYTES * 2 {
            body.push_str("compiling something\n");
        }
        body.push_str("error[E0308]: mismatched types — the actual failure\n");

        let truncated = format_shell_parts(body.as_bytes(), &[], "0");
        assert!(truncated.starts_with("first line of the log"), "head lost");
        assert!(
            truncated.contains("the actual failure"),
            "the failure at the end was truncated away"
        );
        assert!(truncated.contains("output truncated"), "marker missing");
        assert!(
            truncated.len() <= MAX_TOOL_OUTPUT_BYTES + 64,
            "grew past the budget: {}",
            truncated.len()
        );
    }

    #[test]
    fn short_output_is_never_marked_truncated() {
        assert_eq!(format_shell_parts(b"all good\n", &[], "0"), "all good\n");
    }
}
