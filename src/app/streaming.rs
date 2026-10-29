//! Running a chat request: spawning the background stream, consuming its
//! token events frame by frame, and reporting elapsed time while it runs.
//! Now with agent loop for tool calling.

use crate::api::types::{
    Message, Role, StreamEvent, ToolCall, ToolDefinition, MAX_RESPONSE_TEXT_BYTES,
    MAX_TOOL_ARGUMENT_BYTES, MAX_TOOL_CALLS_PER_TURN, MAX_REQUEST_BYTES,
};
use crate::app::{
    App, Cell, RetryView, ToolRound, AGENT_CONSECUTIVE_FAILURE_LIMIT, AGENT_STAGNATION_LIMIT,
};
use crate::sandbox::permissions::ToolKind;
use crate::session::manager::ToolCallRecord;
use crate::tools;
use anyhow::Result;
use std::collections::HashSet;
use std::sync::{atomic::AtomicBool, Arc};
use std::time::{Duration, Instant};
use tokio::sync::mpsc::{self, error::TryRecvError};

impl App {
    pub fn elapsed_secs(&self) -> u64 {
        self.stream_started
            .map(|started| started.elapsed().as_secs())
            .unwrap_or(0)
    }

    pub fn elapsed_ms(&self) -> u64 {
        self.stream_started
            .map(|started| started.elapsed().as_millis() as u64)
            .unwrap_or(0)
    }

    pub(crate) fn start_stream(&mut self) -> Result<()> {
        if self.config.api_key.is_none() {
            let env_key = self
                .config
                .find_provider(&self.config.provider)
                .map(|p| p.env_key)
                .unwrap_or_else(|| "API_KEY".to_string());
            return Err(anyhow::anyhow!(
                "{env_key} is not configured — set it in config.json or the environment"
            ));
        }
        let (tx, rx) = mpsc::channel(64);
        let mut messages: Vec<Message> = Vec::with_capacity(self.sessions.current().messages.len());
        let mut repaired_args = 0usize;
        for m in &self.sessions.current().messages {
            let role = Role::from(m.role.as_str());
            // Only assistant output may contain our reasoning markers. User
            // input and tool output are opaque data: stripping them corrupts
            // literal <think> text in a prompt or a file.
            let content = if role == Role::Assistant {
                crate::ui::thinking::strip(&m.content)
            } else {
                m.content.clone()
            };
            let mut msg = Message::new(role, content);
            if let Some(id) = &m.tool_call_id {
                msg = msg.with_tool_call_id(id.clone());
            }
            if let Some(tcs) = &m.tool_calls {
                // History integrity: tool-call arguments must be valid JSON
                // before they are replayed. A truncated call (a stream cut
                // mid-arguments) would otherwise poison this request and
                // every later one in the session.
                let tool_calls: Vec<crate::api::types::ToolCall> = tcs
                    .iter()
                    .map(|tc| {
                        let arguments =
                            if tc.arguments.len() <= MAX_TOOL_ARGUMENT_BYTES
                                && serde_json::from_str::<serde_json::Value>(&tc.arguments).is_ok() {
                                tc.arguments.clone()
                            } else {
                                repaired_args += 1;
                                "{}".to_string()
                            };
                        let mut call = crate::api::types::ToolCall::new(
                            tc.id.clone(),
                            tc.name.clone(),
                            arguments,
                        );
                        // Provider reasoning signatures (Gemini) must survive
                        // replay verbatim, or the turn is rejected with 400.
                        if let Some(signature) = tc.signature.clone() {
                            call = call.with_signature(signature);
                        }
                        call
                    })
                    .collect();
                msg = msg.with_tool_calls(tool_calls);
            }
            messages.push(msg);
        }

        // Pair results with their OWN assistant turn, not a global set of
        // ids: gateways may reuse the same id in later turns. Orphaned or
        // duplicate results are excluded from the API request, not from the
        // saved transcript. A missing result may mean the tool ran before a
        // crash, so never claim it definitely did not run.
        let (mut messages, backfilled, excluded, repaired_ids) = pair_history_tools(messages);
        if repaired_args > 0 {
            self.push_notice(format!(
                "repaired {repaired_args} truncated tool call(s) in history before replay"
            ));
        }
        if backfilled > 0 {
            self.push_notice(format!(
                "backfilled {backfilled} missing tool result(s) from an interrupted round"
            ));
        }
        if repaired_ids > 0 {
            self.push_notice(format!(
                "repaired {repaired_ids} duplicate or empty tool call id(s) for API replay"
            ));
        }
        if excluded > 0 {
            self.push_notice(format!(
                "excluded {excluded} orphan or duplicate tool result(s) from API replay; saved history is unchanged"
            ));
        }

        // Inject system prompt for agent mode if enabled
        let want_tools =
            self.agent_mode && self.config.sandbox.enabled && self.sandbox.config.enabled;
        let tools_enabled = want_tools && self.sandbox.has_target();
        if want_tools && !self.sandbox.has_target() {
            self.push_notice(
                "agent tools off — set a target with /sandbox <path> (tools will not use this app's directory)".into(),
            );
        }
        if tools_enabled {
            let has_system = messages.iter().any(|m| m.role == Role::System);
            if !has_system {
                let workspace = self.sandbox.config.workspace_root.display().to_string();
                let timeout_secs = self.sandbox.config.effective_shell_timeout().as_secs();
                let mode = self.sandbox.config.permission_mode.as_str();
                // The schema must describe what the program enforces — no
                // claims of sandboxing: file tools are workspace-restricted,
                // shell is not isolated (cwd only, full user privileges).
                let system_content = format!(
                    "You are chatTUI agent. Target directory (workspace): {workspace}\n\
                    File tools (read_file, write_file, edit_file, list_files) operate only inside this workspace; paths that escape it (including via symlinks) and sensitive files (.env*, key material, .git/config) are refused.\n\
                    bash runs `sh -c` with the workspace as the current directory. When OS-level isolation is active (Linux kernel 5.13+), the kernel blocks writes outside the workspace and its scratch roots (/tmp, CARGO_HOME, CARGO_TARGET_DIR), denies all network connections, and denies ptrace/kernel-module/namespace operations; on systems without it, commands run with full user privileges and the tool output says so. Secret files remain readable by shell commands by design (the name-based refusal is best-effort).\n\
                    Commands are killed after a {timeout_secs}s timeout. Permission mode: {mode}. Tools outside the mode return permission-denied errors.\n\
                    File handling policy: read_file returns a window of numbered lines (250 by default) — read large files in successive windows (offset 1, then 251, …) instead of pulling them whole; prefer small targeted edit_file calls over rewriting entire files.\n\
                    Use tools to help the user with file operations. Be concise, explain what you do. Never write files outside the target directory.",
                );
                messages.insert(0, Message::new(Role::System, system_content));
            }
        }
        let model = self.config.model.clone();
        let temperature = self.config.temperature;
        let client = self.config.api_client();

        // Prepare tools if agent mode enabled (reuse tools_enabled from above)
        let tool_defs: Vec<ToolDefinition> = if tools_enabled {
            tools::all_tools()
                .into_iter()
                .map(|t| ToolDefinition {
                    name: t.name,
                    description: t.description,
                    parameters: t.parameters,
                })
                .collect()
        } else {
            Vec::new()
        };

        let request_bytes: usize = messages.iter()
            .map(|message| message.content.len() + message.tool_call_id.as_ref().map_or(0, String::len)
                + message.tool_calls.as_ref().map_or(0, |calls| calls.iter()
                    .map(|call| call.arguments.len() + call.name.len() + call.id.len()
                        + call.signature.as_ref().map_or(0, String::len)).sum::<usize>()))
            .sum::<usize>()
            + tool_defs.iter().map(|tool| tool.name.len() + tool.description.len()
                + tool.parameters.to_string().len()).sum::<usize>();
        if request_bytes > MAX_REQUEST_BYTES {
            return Err(anyhow::anyhow!(
                "conversation exceeds the 8 MiB request limit — start a new session or shorten the history"
            ));
        }

        let task = tokio::spawn(async move {
            if let Err(error) = client
                .stream_chat(&messages, &model, temperature, tool_defs, tx.clone())
                .await
            {
                let _ = tx.send(StreamEvent::Error(error.to_string())).await;
            }
        });
        self.stream_task = Some(task);
        self.tokens = Some(rx);
        self.streaming = true;
        self.stream_started = Some(Instant::now());
        Ok(())
    }

    pub async fn receive_token(&mut self) {
        // A tool round parked on an approval prompt resumes as soon as the
        // user has answered; until then there is nothing to do.
        if self.tool_round.is_some() {
            // Poll one in-flight tool without awaiting unfinished work. Keys,
            // redraws and cancellation keep running on the terminal loop.
            self.run_tool_round().await;
            return;
        }
        let Some(mut rx) = self.tokens.take() else {
            return;
        };
        loop {
            match rx.try_recv() {
                Ok(StreamEvent::Delta(token)) => {
                    // Tokens are flowing again: the retry countdown is over.
                    self.retry_state = None;
                    if self.response.len().saturating_add(token.len()) > MAX_RESPONSE_TEXT_BYTES {
                        if let Some(task) = self.stream_task.take() {
                            task.abort();
                        }
                        self.tokens = None;
                        self.pending_tool_calls.clear();
                        self.truncated_tool_calls.clear();
                        self.finish_run();
                        self.push_error("response exceeded the size limit; stopped before using any tool calls".into());
                        return;
                    }
                    self.response.push_str(&token);
                }
                Ok(StreamEvent::ToolCall(tc)) => {
                    self.retry_state = None;
                    // Detect truncated tool calls at ingestion: arguments
                    // that do not parse were cut mid-stream. Record the id
                    // so execution returns an explicit error and the stored
                    // record keeps replayable JSON.
                    let existing_bytes: usize = self.pending_tool_calls.iter()
                        .map(|call| call.id.len() + call.name.len() + call.arguments.len()
                            + call.signature.as_ref().map_or(0, String::len)).sum();
                    if self.pending_tool_calls.len() >= MAX_TOOL_CALLS_PER_TURN
                        || existing_bytes.saturating_add(tc.id.len())
                            .saturating_add(tc.name.len()).saturating_add(tc.arguments.len())
                            .saturating_add(tc.signature.as_ref().map_or(0, String::len)) > MAX_REQUEST_BYTES
                        || tc.arguments.len() > MAX_TOOL_ARGUMENT_BYTES
                        || tc.id.is_empty() || tc.id.len() > 256 || tc.name.len() > 256
                        || tc.signature.as_ref().is_some_and(|signature| signature.len() > 64 * 1024)
                        || self.pending_tool_calls.iter().any(|previous| previous.id == tc.id)
                    {
                        if let Some(task) = self.stream_task.take() {
                            task.abort();
                        }
                        self.tokens = None;
                        self.pending_tool_calls.clear();
                        self.truncated_tool_calls.clear();
                        self.finish_run();
                        self.push_error("invalid, duplicate or oversized tool call — none were executed".into());
                        return;
                    }
                    if serde_json::from_str::<serde_json::Value>(&tc.arguments).is_err() {
                        self.truncated_tool_calls.insert(tc.id.clone());
                    }
                    self.pending_tool_calls.push(tc);
                }
                Ok(StreamEvent::Notice(message)) => {
                    self.push_notice(message);
                }
                Ok(StreamEvent::Retry {
                    attempt,
                    max,
                    wait_ms,
                    reason,
                }) => {
                    // Drive the animated status row: the countdown is
                    // computed from `started` on every redraw.
                    self.retry_state = Some(RetryView {
                        started: Instant::now(),
                        wait: Duration::from_millis(wait_ms),
                        attempt,
                        max,
                        reason,
                    });
                }
                Ok(StreamEvent::Error(error)) => {
                    self.retry_state = None;
                    self.finish_partial();
                    self.streaming = false;
                    self.stream_started = None;
                    self.pending_tool_calls.clear();
                    self.truncated_tool_calls.clear();
                    self.reset_agent_health();
                    self.push_error(error);
                    return;
                }
                Err(TryRecvError::Empty) => {
                    self.tokens = Some(rx);
                    return;
                }
                Err(TryRecvError::Disconnected) => break,
            }
        }

        // The stream ended cleanly; drop any leftover retry indicator.
        self.retry_state = None;

        // Stream finished - check for agent loop
        if self.streaming && !self.pending_tool_calls.is_empty() {
            // Save assistant message with tool calls
            let response_text = std::mem::take(&mut self.response);
            let pending = std::mem::take(&mut self.pending_tool_calls);
            let tool_records: Vec<ToolCallRecord> = pending
                .iter()
                .map(|tc| ToolCallRecord {
                    id: tc.id.clone(),
                    name: tc.name.clone(),
                    // Truncated arguments are stored as "{}" so the saved
                    // history never carries unparseable JSON.
                    arguments: if self.truncated_tool_calls.contains(&tc.id) {
                        "{}".to_string()
                    } else {
                        tc.arguments.clone()
                    },
                    signature: tc.signature.clone(),
                })
                .collect();

            if !response_text.is_empty() {
                self.cells.push(Cell::Assistant(response_text.clone()));
            }
            for tc in &pending {
                self.push_tool_call(tc);
            }
            let saved = self.sessions.add_assistant_with_tools(response_text, tool_records);
            if !self.report_save(saved) {
                // Never perform a side effect when we cannot first record
                // the model's intent for crash-safe history replay.
                for call in &pending {
                    let note = "not run: assistant tool calls could not be saved".to_string();
                    self.push_tool_result(call.id.clone(), note.clone(), true);
                    let saved = self.sessions.add_tool_result(call.id.clone(), note, true);
                    self.report_save(saved);
                }
                self.push_error("tools not run because the conversation could not be saved".into());
                self.finish_run();
                return;
            }

            // Run the round's tools. A call that needs the user's OK parks
            // the round (see `run_tool_round`); it resumes from the top of
            // this function once the user has answered.
            self.tool_round = Some(ToolRound {
                calls: pending,
                next: 0,
                failures: 0,
                answer: None,
                in_flight: None,
                cancelled: false,
                cancel: Arc::new(AtomicBool::new(false)),
            });
            self.run_tool_round().await;
            return;
        }

        self.finish_run();
    }

    /// Whether `call` needs the user's OK before it runs: manual approval
    /// covers every tool that changes something (write, edit, shell), and an
    /// `ask-*` permission mode always asks about the kinds it names. Reads
    /// never ask, and neither does a call the permission mode forbids
    /// outright — that fails with its permission error instead of asking
    /// about something that could not run anyway.
    fn needs_approval(&self, call: &ToolCall) -> bool {
        let mode = self.sandbox.config.permission_mode;
        match ToolKind::from_tool_name(&call.name) {
            Some(kind) if kind != ToolKind::Read => {
                mode.asks(kind) || (self.manual_approval && mode.allows(kind))
            }
            _ => false,
        }
    }

    /// The tool call the agent is paused on, while the approval prompt is up.
    pub fn approval_request(&self) -> Option<&ToolCall> {
        self.tool_round
            .as_ref()
            .filter(|round| round.in_flight.is_none() && !round.cancelled && round.answer.is_none())
            .and_then(|round| round.calls.get(round.next))
    }

    /// Whether an approval prompt is waiting for `y` / `n`.
    pub fn awaiting_approval(&self) -> bool {
        self.approval_request().is_some()
    }

    pub fn has_inflight_tool(&self) -> bool {
        self.tool_round.as_ref().is_some_and(|round| round.in_flight.is_some())
    }

    pub fn is_stopping_tool(&self) -> bool {
        self.tool_round.as_ref().is_some_and(|round| round.cancelled && round.in_flight.is_some())
    }

    /// `y` / `n` on the approval prompt: record the answer. The paused round
    /// acts on it at the next `receive_token`.
    pub fn answer_approval(&mut self, approved: bool) {
        if let Some(round) = self
            .tool_round
            .as_mut()
            .filter(|round| round.in_flight.is_none() && !round.cancelled && round.answer.is_none())
        {
            round.answer = Some(approved);
        }
    }

    /// Drive one tool round without waiting for a subprocess or filesystem
    /// operation on the UI loop. Only one call runs at a time, preserving the
    /// model's order. Esc signals the shell worker or waits for a file tool's
    /// actual result, then records errors for calls never started.
    async fn run_tool_round(&mut self) {
        let Some(mut round) = self.tool_round.take() else {
            return;
        };
        loop {
            if let Some(task) = round.in_flight.as_ref() {
                if !task.is_finished() {
                    self.tool_round = Some(round);
                    return;
                }
                let task = round.in_flight.take().expect("checked above");
                let outcome = match task.await {
                    Ok(result) => result,
                    Err(error) => Err(anyhow::anyhow!("tool task stopped unexpectedly: {error}")),
                };
                let id = round.calls[round.next].id.clone();
                if !self.record_tool_outcome(&id, outcome, &mut round.failures) {
                    round.cancelled = true;
                }
                round.next += 1;
                continue;
            }

            if round.cancelled {
                for call in round.calls.iter().skip(round.next) {
                    let note = "interrupted by the user before this tool could run".to_string();
                    self.push_tool_result(call.id.clone(), note.clone(), true);
                    let saved = self.sessions.add_tool_result(call.id.clone(), note, true);
                    self.report_save(saved);
                }
                self.truncated_tool_calls.clear();
                self.push_notice("agent stopped before running the remaining tool calls".into());
                self.finish_run();
                return;
            }
            if round.next == round.calls.len() {
                self.finish_tool_round(&round);
                return;
            }

            let call = round.calls[round.next].clone();
            let answer = round.answer.take();
            let outcome = if self.truncated_tool_calls.contains(&call.id) {
                Some(Err(anyhow::anyhow!(
                    "tool call arguments arrived truncated (the stream was cut mid-call) — retry with smaller edits"
                )))
            } else if self.needs_approval(&call) {
                match answer {
                    Some(false) => Some(Err(anyhow::anyhow!(
                        "the user declined to approve this {} call, so it was not run — do not repeat it as-is; ask the user how to proceed",
                        call.name
                    ))),
                    Some(true) => {
                        self.start_tool_task(round, call, true);
                        return;
                    }
                    None => {
                        self.tool_round = Some(round);
                        return;
                    }
                }
            } else {
                self.start_tool_task(round, call, false);
                return;
            };
            if let Some(outcome) = outcome {
                if !self.record_tool_outcome(&call.id, outcome, &mut round.failures) {
                    round.cancelled = true;
                }
                round.next += 1;
            }
        }
    }

    fn start_tool_task(&mut self, mut round: ToolRound, call: ToolCall, approved: bool) {
        let sandbox = self.sandbox.clone();
        let cancel = round.cancel.clone();
        round.in_flight = Some(tokio::spawn(async move {
            sandbox.execute_tool_cancellable(&call, approved, cancel).await
        }));
        self.tool_round = Some(round);
    }

    fn record_tool_outcome(&mut self, id: &str, outcome: anyhow::Result<String>, failures: &mut usize) -> bool {
        let (content, is_error) = match outcome {
            Ok(output) => (output, false),
            Err(error) => (format!("Error: {error}"), true),
        };
        *failures += usize::from(is_error);
        self.push_tool_result(id.to_string(), content.clone(), is_error);
        let saved = self.sessions.add_tool_result(id.to_string(), content, is_error);
        let persisted = self.report_save(saved);
        self.truncated_tool_calls.remove(id);
        persisted
    }

    /// Every call of the round has run: decide whether the agent loop may
    /// continue.
    fn finish_tool_round(&mut self, round: &ToolRound) {
        // -- Agent health guards: decide whether the loop may continue.
        // Progress, not a step counter, decides: a healthy run keeps
        // going; these guards stop the specific broken patterns.

        // Stagnation: fingerprint the round (sorted name+arguments) and
        // count identical failing rounds in a row.
        let mut keys: Vec<String> = round
            .calls
            .iter()
            .map(|tc| format!("{}\u{0}{}", tc.name, tc.arguments))
            .collect();
        keys.sort();
        let round_key = keys.join("\u{1f}");
        let round_failed = round.failures > 0;
        if round_failed && self.agent_last_round_key.as_deref() == Some(round_key.as_str()) {
            self.agent_repeat_failures += 1;
        } else {
            self.agent_repeat_failures = usize::from(round_failed);
        }
        self.agent_last_round_key = Some(round_key);

        // Consecutive rounds where every tool errored.
        if round.failures == round.calls.len() {
            self.agent_consecutive_failures += 1;
        } else {
            self.agent_consecutive_failures = 0;
        }

        let stop_reason: Option<String> = if self.agent_repeat_failures >= AGENT_STAGNATION_LIMIT {
            Some(format!(
                    "agent stopped: the same tool call failed {AGENT_STAGNATION_LIMIT} rounds in a row — it is not making progress"
                ))
        } else if self.agent_consecutive_failures >= AGENT_CONSECUTIVE_FAILURE_LIMIT {
            Some(format!(
                    "agent stopped: {AGENT_CONSECUTIVE_FAILURE_LIMIT} consecutive rounds failed — check the workspace and try again"
                ))
        } else if self.agent_iterations >= self.agent_max_rounds {
            // Safety net only — healthy runs are stopped by the
            // guards above, not by this ceiling.
            Some(format!(
                    "agent stopped after {} rounds — raise agent.max_rounds in config.json if this task needs more",
                    self.agent_max_rounds
                ))
        } else {
            None
        };

        if let Some(reason) = stop_reason {
            self.push_notice(reason);
            self.finish_run();
        } else {
            self.agent_iterations += 1;
            if let Err(e) = self.start_stream() {
                self.push_error(e.to_string());
                self.streaming = false;
                self.stream_started = None;
                self.reset_agent_health();
            }
        }
    }

    /// The agent run is over (no more tool calls, or a guard stopped it):
    /// commit any partial answer and reset the per-run state.
    fn finish_run(&mut self) {
        self.streaming = false;
        self.stream_started = None;
        self.finish_partial();
        self.reset_agent_health();
        self.pending_tool_calls.clear();
        self.truncated_tool_calls.clear();
    }
}


/// Normalize a saved transcript for API replay without modifying the on-disk
/// history. Results must immediately follow the assistant turn that requested
/// them, including when ids were reused in a later turn or a crash cut a
/// round short. Matching uses the original id before any duplicate-id repair.
fn pair_history_tools(messages: Vec<Message>) -> (Vec<Message>, usize, usize, usize) {
    let mut iter = messages.into_iter().peekable();
    let mut paired = Vec::new();
    let (mut backfilled, mut excluded, mut repaired_ids) = (0, 0, 0);
    while let Some(mut message) = iter.next() {
        if message.role == Role::Tool {
            excluded += 1;
            continue;
        }
        if message.role != Role::Assistant
            || message.tool_calls.as_ref().map_or(true, Vec::is_empty)
        {
            paired.push(message);
            continue;
        }
        let mut results = Vec::new();
        while iter.peek().is_some_and(|next| next.role == Role::Tool) {
            results.push(iter.next().expect("peeked a tool result"));
        }
        let mut seen = HashSet::new();
        let turn = paired.len();
        let mut ordered = Vec::new();
        for (index, call) in message.tool_calls.as_mut().expect("checked above").iter_mut().enumerate() {
            let original_id = call.id.clone();
            if call.id.is_empty() || !seen.insert(call.id.clone()) {
                call.id = format!("{}:replay_{turn}_{index}", call.name);
                // A malformed legacy history might already contain that id.
                while !seen.insert(call.id.clone()) {
                    call.id.push('_');
                }
                repaired_ids += 1;
            }
            if let Some(pos) = results.iter().position(|result: &Message| {
                result.tool_call_id.as_deref() == Some(original_id.as_str())
            }) {
                let mut result = results.remove(pos);
                result.tool_call_id = Some(call.id.clone());
                ordered.push(result);
            } else {
                ordered.push(Message::tool_result(
                    call.id.clone(),
                    "result unavailable after interrupted history; the tool may have run — verify before retrying",
                ));
                backfilled += 1;
            }
        }
        excluded += results.len();
        paired.push(message);
        paired.extend(ordered);
    }
    (paired, backfilled, excluded, repaired_ids)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::types::ToolCall;
    use crate::app::{App, Cell};
    use crate::config::Config;
    use crate::sandbox::permissions::PermissionMode;
    use crate::session::manager::SessionManager;

    fn test_app() -> App {
        App::new(Config::default(), SessionManager::for_tests())
    }

    /// An app with agent tools enabled against an empty temp workspace and
    /// no API key: a tool round that tries to continue the agent loop fails
    /// its restart immediately instead of touching the network.
    fn agent_app(name: &str) -> App {
        let config: Config = serde_json::from_str("{}").unwrap();
        let mut app = App::new(config, SessionManager::for_tests());
        let dir = std::env::temp_dir().join(format!("chatTUI_loop_{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        app.sandbox.set_target(dir.to_str().unwrap()).unwrap();
        app.agent_mode = true;
        app
    }

    fn tool_call_event(id: &str, name: &str, arguments: &str) -> StreamEvent {
        StreamEvent::ToolCall(ToolCall::new(id, name, arguments))
    }

    /// Feed one completed stream containing `events` into the app.
    fn feed(app: &mut App, events: Vec<StreamEvent>) {
        let (tx, rx) = tokio::sync::mpsc::channel(4);
        for event in events {
            tx.try_send(event).unwrap();
        }
        drop(tx);
        app.streaming = true;
        app.tokens = Some(rx);
    }

    #[test]
    fn stream_notices_become_transcript_rows() {
        let (tx, rx) = tokio::sync::mpsc::channel(4);
        tx.try_send(StreamEvent::Notice("switching to b/2".into()))
            .unwrap();
        tx.try_send(StreamEvent::Delta("hel".into())).unwrap();
        let mut app = test_app();
        app.streaming = true;
        app.tokens = Some(rx);
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        runtime.block_on(app.receive_token());
        assert_eq!(app.response, "hel");
        assert_eq!(app.cells.len(), 1);
        assert!(matches!(app.cells.last(), Some(Cell::Notice(_))));

        // Dropping the sender ends the stream and commits the partial answer.
        drop(tx);
        runtime.block_on(app.receive_token());
        assert!(!app.streaming);
        assert_eq!(app.cells.len(), 2);
        assert!(matches!(app.cells[0], Cell::Notice(_)));
        assert_eq!(app.cells[1], Cell::Assistant("hel".into()));
    }

    #[test]
    fn replay_pairs_reused_ids_with_their_own_turn_only() {
        let a = Message::new(Role::Assistant, "")
            .with_tool_calls(vec![ToolCall::new("same", "read_file", "{}")]);
        let (paired, backfilled, excluded, repaired) = pair_history_tools(vec![
            a.clone(), Message::tool_result("same", "first result"),
            a, Message::new(Role::User, "next question"),
            Message::tool_result("same", "orphaned late result"),
        ]);
        assert_eq!((backfilled, excluded, repaired), (1, 1, 0));
        assert_eq!(paired.len(), 5);
        assert_eq!(paired[1].content, "first result");
        assert_eq!(paired[2].role, Role::Assistant);
        assert!(paired[3].content.contains("may have run"));
        assert_eq!(paired[4].role, Role::User);
    }

    #[test]
    fn replay_repairs_duplicate_ids_and_reorders_results() {
        let assistant = Message::new(Role::Assistant, "").with_tool_calls(vec![
            ToolCall::new("same", "read_file", "{}"),
            ToolCall::new("same", "write_file", "{}"),
        ]);
        let (paired, backfilled, excluded, repaired) = pair_history_tools(vec![
            assistant,
            Message::tool_result("same", "first"),
            Message::tool_result("same", "second"),
        ]);
        assert_eq!((backfilled, excluded, repaired), (0, 0, 1));
        let calls = paired[0].tool_calls.as_ref().unwrap();
        assert_ne!(calls[0].id, calls[1].id);
        assert_eq!(paired[1].tool_call_id.as_deref(), Some(calls[0].id.as_str()));
        assert_eq!(paired[2].tool_call_id.as_deref(), Some(calls[1].id.as_str()));
    }

    /// Background tools need further UI ticks, but never block one tick.
    async fn settle(app: &mut App) {
        for _ in 0..200 {
            app.receive_token().await;
            if app.tool_round.is_none() || app.awaiting_approval() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        panic!("tool round did not settle");
    }

    #[tokio::test]
    async fn tool_call_survives_a_split_poll_and_runs_once() {
        let mut app = agent_app("split-poll");
        let (tx, rx) = tokio::sync::mpsc::channel(4);
        app.tokens = Some(rx);
        app.streaming = true;
        tx.send(tool_call_event("c1", "list_files", "{}"))
            .await.unwrap();
        app.receive_token().await;
        assert_eq!(app.pending_tool_calls.len(), 1);
        assert!(app.cells.iter().all(|cell| !matches!(cell, Cell::ToolResult { .. })));
        drop(tx);
        settle(&mut app).await;
        assert_eq!(saved_tool_result_ids(&app), vec!["c1"]);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn esc_during_a_shell_command_keeps_the_ui_responsive() {
        let mut app = agent_app("shell-interrupt");
        app.sandbox.config.permission_mode = PermissionMode::FullAuto;
        app.sandbox.config.shell_timeout = Duration::from_secs(3);
        feed(&mut app, vec![tool_call_event("c1", "bash", r#"{"command":"sleep 30"}"#)]);
        app.receive_token().await; // schedules work, does not await sleep
        assert!(app.streaming);
        app.interrupt();
        assert!(app.streaming); // wait for the actual outcome, not a fake denial
        tokio::time::timeout(Duration::from_secs(5), async {
            while app.streaming {
                app.receive_token().await;
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        }).await.expect("shell cancellation should complete promptly");
        assert_eq!(saved_tool_result_ids(&app), vec!["c1"]);
        assert!(app.cells.iter().any(|cell| matches!(cell, Cell::ToolResult { is_error: true, .. })));
    }

    #[tokio::test]
    async fn tool_rounds_execute_and_stop_predictably_when_restart_fails() {
        let mut app = agent_app("restart-fail");
        feed(&mut app, vec![tool_call_event("c1", "list_files", "{}")]);
        settle(&mut app).await;

        // The tool actually ran.
        assert!(app.cells.iter().any(|c| matches!(
            c,
            Cell::ToolResult {
                is_error: false,
                ..
            }
        )));
        // The loop tried to continue, the restart failed (no API key), and
        // the agent stopped with a reported error instead of looping.
        assert!(!app.streaming);
        assert_eq!(app.agent_iterations, 0);
        assert!(matches!(app.cells.last(), Some(Cell::Error(_))));
    }

    #[tokio::test]
    async fn agent_loop_stops_at_the_configured_ceiling() {
        let mut app = agent_app("ceiling");
        app.agent_iterations = app.agent_max_rounds;
        feed(&mut app, vec![tool_call_event("c1", "list_files", "{}")]);
        settle(&mut app).await;

        // The final round's tool still ran and its result was recorded…
        assert!(app
            .cells
            .iter()
            .any(|c| matches!(c, Cell::ToolResult { .. })));
        // …but no further LLM request was started and the agent stopped.
        assert!(!app.streaming);
        assert_eq!(app.agent_iterations, 0);
        assert!(app.cells.iter().any(|c| matches!(
            c,
            Cell::Notice(text) if text.contains("agent stopped") && text.contains("max_rounds")
        )));
    }

    #[tokio::test]
    async fn successful_rounds_run_until_the_ceiling() {
        // Restart streams succeed (the legacy `api_key` field satisfies
        // start_stream; the endpoint is a closed local port that fails
        // fast), so the agent loop really runs round after round — exactly
        // the setup that would spin forever without a backstop. Identical
        // SUCCEEDING rounds must not trip the stagnation guard.
        let config: Config = serde_json::from_str(
            r#"{"provider":"openai","base_url":"http://127.0.0.1:9/v1","api_key":"test-key","agent":{"max_rounds":12}}"#,
        )
        .unwrap();
        let mut app = App::new(config, SessionManager::for_tests());
        let dir = std::env::temp_dir().join(format!("chatTUI_loop_many_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        app.sandbox.set_target(dir.to_str().unwrap()).unwrap();
        app.agent_mode = true;

        let mut rounds = 0;
        loop {
            // The round counter may reach but never exceed the ceiling.
            assert!(
                app.agent_iterations <= app.agent_max_rounds,
                "round counter exceeded its ceiling"
            );
            feed(
                &mut app,
                vec![tool_call_event(&format!("c{rounds}"), "list_files", "{}")],
            );
            settle(&mut app).await;
            rounds += 1;
            assert!(rounds < 100, "agent loop failed to terminate");
            if !app.streaming {
                break;
            }
        }

        assert_eq!(rounds, app.agent_max_rounds + 1);
        assert!(app.cells.iter().any(|c| matches!(
            c,
            Cell::Notice(text) if text.contains("agent stopped")
        )));
        assert_eq!(app.agent_iterations, 0);
    }

    #[tokio::test]
    async fn tool_errors_cannot_bypass_the_iteration_limit() {
        // Every round fails (permission denied); the counter must still
        // advance and the loop must still terminate.
        let mut app = agent_app("errors-bounded");
        app.sandbox.config.permission_mode = PermissionMode::ReadOnly;
        let rounds = 3;
        for round in 0..rounds {
            assert!(app.agent_iterations < app.agent_max_rounds);
            feed(
                &mut app,
                vec![tool_call_event(
                    &format!("c{round}"),
                    "write_file",
                    r#"{"path":"x","content":"y"}"#,
                )],
            );
            settle(&mut app).await;
            assert!(!app.streaming);
            assert!(app.cells.iter().any(|c| matches!(
                c,
                Cell::ToolResult { is_error: true, content, .. }
                    if content.contains("permission denied")
            )));
        }
        assert!(!app.streaming);
    }

    #[tokio::test]
    async fn permission_denials_reach_the_model_as_tool_errors() {
        let mut app = agent_app("permission");
        app.sandbox.config.permission_mode = PermissionMode::ReadOnly;
        feed(
            &mut app,
            vec![tool_call_event("c1", "bash", r#"{"command":"echo hi"}"#)],
        );
        settle(&mut app).await;
        assert!(app.cells.iter().any(|c| matches!(
            c,
            Cell::ToolResult { is_error: true, content, .. }
                if content.contains("permission denied")
        )));
    }

    #[tokio::test]
    async fn truncated_tool_arguments_become_an_explicit_tool_error() {
        let mut app = agent_app("truncated");
        // Unterminated JSON — a tool call whose stream was cut mid-arguments.
        feed(
            &mut app,
            vec![StreamEvent::ToolCall(ToolCall::new(
                "c1",
                "edit_file",
                r#"{"new_string": "#,
            ))],
        );
        settle(&mut app).await;

        // The model sees an error that names the real problem…
        assert!(app.cells.iter().any(|c| matches!(
            c,
            Cell::ToolResult { is_error: true, content, .. } if content.contains("truncated")
        )));
        // …and the stored record carries clean JSON, never the broken text.
        assert!(app.sessions.current().messages.iter().any(|m| m
            .tool_calls
            .as_ref()
            .is_some_and(|calls| calls.iter().any(|tc| tc.arguments == "{}"))));
    }

    #[tokio::test]
    async fn replay_repairs_truncated_arguments_and_missing_tool_results() {
        let mut config = Config::default();
        config.api_key = Some("test-key".into());
        config.base_url = "http://127.0.0.1:9/v1".into();
        let mut app = App::new(config, SessionManager::for_tests());

        // Doubly broken history: unterminated arguments AND no tool result.
        app.sessions.add_assistant_with_tools(
            "",
            vec![ToolCallRecord {
                id: "c1".into(),
                name: "edit_file".into(),
                arguments: r#"{"new_string": "#.into(),
                signature: None,
            }],
        ).unwrap();

        app.start_stream().expect("stream starts");

        assert!(app.cells.iter().any(|c| matches!(
            c,
            Cell::Notice(text) if text.contains("repaired 1 truncated tool call")
        )));
        assert!(app.cells.iter().any(|c| matches!(
            c,
            Cell::Notice(text) if text.contains("backfilled 1 missing tool result")
        )));
        app.interrupt();
    }

    /// An app with a satisfied API key (pointed at a closed endpoint), a
    /// temp workspace and read-only permissions — for exercising the agent
    /// health guards round after round without real network traffic.
    fn guarded_agent_app(name: &str) -> App {
        let config: Config = serde_json::from_str(
            r#"{"provider":"openai","base_url":"http://127.0.0.1:9/v1","api_key":"test-key"}"#,
        )
        .unwrap();
        let mut app = App::new(config, SessionManager::for_tests());
        let dir = std::env::temp_dir().join(format!("chatTUI_guard_{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        app.sandbox.set_target(dir.to_str().unwrap()).unwrap();
        app.agent_mode = true;
        app.sandbox.config.permission_mode = PermissionMode::ReadOnly;
        app
    }

    #[tokio::test]
    async fn identical_failing_rounds_stop_the_agent() {
        let mut app = guarded_agent_app("stagnant");
        for round in 0..3 {
            feed(
                &mut app,
                vec![tool_call_event(
                    &format!("c{round}"),
                    "write_file",
                    r#"{"path":"x","content":"y"}"#,
                )],
            );
            settle(&mut app).await;
        }

        // Third identical failing round: the stagnation guard stops the run.
        assert!(!app.streaming);
        assert_eq!(app.agent_iterations, 0);
        assert!(app.cells.iter().any(|c| matches!(
            c,
            Cell::Notice(text) if text.contains("not making progress")
        )));
    }

    #[tokio::test]
    async fn consecutive_failing_rounds_stop_the_agent() {
        let mut app = guarded_agent_app("failing");
        // A different call each round keeps the stagnation guard quiet;
        // every round still fails (read-only permission).
        for round in 0..4 {
            feed(
                &mut app,
                vec![tool_call_event(
                    &format!("c{round}"),
                    "write_file",
                    &format!(r#"{{"path":"f{round}","content":"y"}}"#),
                )],
            );
            settle(&mut app).await;
        }

        assert!(!app.streaming);
        assert!(app.cells.iter().any(|c| matches!(
            c,
            Cell::Notice(text) if text.contains("consecutive rounds failed")
        )));
    }

    // ---- manual approval (`/approve`) and the `ask-*` permission modes ----

    #[test]
    fn default_shell_approval_gate_tracks_manual_and_explicit_ask_modes() {
        let config: Config = serde_json::from_str("{}").unwrap();
        let mut app = App::new(config, SessionManager::for_tests());
        let shell = ToolCall::new("c1", "bash", r#"{"command":"echo ok"}"#);
        assert_eq!(app.sandbox.config.permission_mode, PermissionMode::FullAuto);
        assert!(!app.manual_approval);
        assert!(!app.needs_approval(&shell));

        app.composer = "/approve manual".into();
        app.submit();
        assert!(app.needs_approval(&shell));
        app.composer = "/approve auto".into();
        app.submit();
        assert!(!app.needs_approval(&shell));

        let config: Config =
            serde_json::from_str(r#"{"sandbox":{"permission_mode":"ask-before-shell"}}"#).unwrap();
        let app = App::new(config, SessionManager::for_tests());
        assert_eq!(
            app.sandbox.config.permission_mode,
            PermissionMode::AskBeforeShell
        );
        assert!(app.needs_approval(&shell));
    }

    /// A manual-approval agent whose first round is parked on a `write_file`
    /// prompt (`a.txt` ← "hi").
    async fn parked_on_write(name: &str) -> App {
        let mut app = agent_app(name);
        app.manual_approval = true;
        feed(
            &mut app,
            vec![tool_call_event(
                "c1",
                "write_file",
                r#"{"path":"a.txt","content":"hi"}"#,
            )],
        );
        settle(&mut app).await;
        app
    }

    /// Ids of the tool results saved in the conversation, in order.
    fn saved_tool_result_ids(app: &App) -> Vec<String> {
        app.sessions
            .current()
            .messages
            .iter()
            .filter(|m| m.role == "tool")
            .filter_map(|m| m.tool_call_id.clone())
            .collect()
    }

    /// Everything `ui::render` draws for `app`, as one string.
    fn screen(app: &App) -> String {
        let mut terminal =
            ratatui::Terminal::new(ratatui::backend::TestBackend::new(100, 30)).unwrap();
        terminal
            .draw(|frame| crate::ui::render(frame, app))
            .unwrap();
        let buffer = terminal.backend().buffer();
        let mut out = String::new();
        for y in 0..buffer.area.height {
            for x in 0..buffer.area.width {
                out.push_str(buffer[(x, y)].symbol());
            }
            out.push('\n');
        }
        out
    }

    #[tokio::test]
    async fn manual_approval_pauses_the_round_until_the_user_answers() {
        let mut app = parked_on_write("manual-approve").await;
        let file = app.sandbox.config.workspace_root.join("a.txt");

        // Parked on the prompt: it names the pending call, nothing has run
        // yet, and the agent is still mid-run.
        assert!(app.awaiting_approval());
        assert_eq!(
            app.approval_request().map(|call| call.name.as_str()),
            Some("write_file")
        );
        assert!(app.streaming);
        assert!(!file.exists());
        assert!(!app
            .cells
            .iter()
            .any(|c| matches!(c, Cell::ToolResult { .. })));

        // The user sees the card and why the agent is idle.
        let shown = screen(&app);
        assert!(shown.contains("Approve this tool call?"), "{shown}");
        assert!(shown.contains("Waiting for your approval"), "{shown}");

        // More ticks change nothing while the user decides.
        settle(&mut app).await;
        settle(&mut app).await;
        assert!(app.awaiting_approval());
        assert!(!file.exists());

        // `y`: the prompt closes at once and the call runs on the next tick.
        // (The follow-up request then fails without an API key, ending the
        // run — the same way the other round tests end.)
        app.answer_approval(true);
        assert!(!app.awaiting_approval());
        assert!(!file.exists());
        settle(&mut app).await;
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "hi");
        assert!(app.cells.iter().any(|c| matches!(
            c,
            Cell::ToolResult {
                is_error: false,
                ..
            }
        )));
        assert!(!app.streaming);
        // Both prompts are gone from the screen.
        let shown = screen(&app);
        assert!(!shown.contains("Approve this tool call?"), "{shown}");
        assert!(!shown.contains("Waiting for your approval"), "{shown}");
    }

    #[tokio::test]
    async fn declining_a_call_tells_the_model_and_runs_nothing() {
        let mut app = parked_on_write("manual-deny").await;
        let file = app.sandbox.config.workspace_root.join("a.txt");

        app.answer_approval(false);
        settle(&mut app).await;

        assert!(!file.exists());
        assert!(app.cells.iter().any(|c| matches!(
            c,
            Cell::ToolResult { is_error: true, content, .. } if content.contains("declined")
        )));
        // The model gets the refusal as the call's own tool result.
        assert_eq!(saved_tool_result_ids(&app), vec!["c1"]);
        assert!(app
            .sessions
            .current()
            .messages
            .iter()
            .any(|m| m.role == "tool" && m.content.contains("declined")));
        assert!(!app.awaiting_approval());
        assert!(!app.streaming);
    }

    #[tokio::test]
    async fn manual_approval_only_asks_about_calls_that_change_something() {
        // Reads never ask.
        let mut app = agent_app("manual-read");
        app.manual_approval = true;
        feed(&mut app, vec![tool_call_event("c1", "list_files", "{}")]);
        settle(&mut app).await;
        assert!(!app.awaiting_approval());
        assert!(app.cells.iter().any(|c| matches!(
            c,
            Cell::ToolResult {
                is_error: false,
                ..
            }
        )));

        // Neither does a call the permission mode forbids outright: it fails
        // with its permission error instead of asking about the impossible.
        let mut app = agent_app("manual-forbidden");
        app.manual_approval = true;
        app.sandbox.config.permission_mode = PermissionMode::ReadOnly;
        feed(
            &mut app,
            vec![tool_call_event(
                "c1",
                "write_file",
                r#"{"path":"a.txt","content":"hi"}"#,
            )],
        );
        settle(&mut app).await;
        assert!(!app.awaiting_approval());
        assert!(app.cells.iter().any(|c| matches!(
            c,
            Cell::ToolResult { is_error: true, content, .. }
                if content.contains("permission denied")
        )));
    }

    #[tokio::test]
    async fn esc_on_a_prompt_stops_the_run_and_answers_every_unfinished_call() {
        use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
        let mut app = agent_app("manual-esc");
        app.manual_approval = true;
        let file = app.sandbox.config.workspace_root.join("a.txt");
        feed(
            &mut app,
            vec![
                tool_call_event("c1", "list_files", "{}"),
                tool_call_event("c2", "write_file", r#"{"path":"a.txt","content":"hi"}"#),
                tool_call_event("c3", "bash", r#"{"command":"echo hi"}"#),
            ],
        );
        settle(&mut app).await;
        // The read ran; the write waits, with the shell queued behind it.
        assert_eq!(
            app.approval_request().map(|call| call.id.as_str()),
            Some("c2")
        );
        assert_eq!(saved_tool_result_ids(&app), vec!["c1"]);

        assert!(crate::handle_key(
            &mut app,
            KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE)
        ));

        assert!(!app.streaming);
        assert!(!app.awaiting_approval());
        assert!(!file.exists());
        // Every call of the assistant message has a tool result, so the
        // saved conversation can be replayed to the API later.
        assert_eq!(saved_tool_result_ids(&app), vec!["c1", "c2", "c3"]);
        // And nothing resumes afterwards.
        settle(&mut app).await;
        assert!(!file.exists());
        assert_eq!(saved_tool_result_ids(&app).len(), 3);
    }

    #[tokio::test]
    async fn leaving_the_conversation_ends_a_paused_round_in_its_own_history() {
        let mut app = parked_on_write("manual-switch").await;
        let file = app.sandbox.config.workspace_root.join("a.txt");
        // A second, empty conversation exists (index 0); back to the first
        // one (now index 1), where the round is paused, then pick the new
        // one in the history list.
        app.sessions.new_session().unwrap();
        app.sessions.select(1).unwrap();
        app.overlay = Some(crate::app::Overlay::History { selected: 0 });
        app.load_selected_session();

        // The run ended: the paused call was answered in the conversation
        // it belongs to, and nothing leaked into the one switched to.
        assert!(!app.streaming);
        assert!(!app.awaiting_approval());
        assert_eq!(app.sessions.current_index(), 0);
        assert!(app.sessions.current().messages.is_empty());
        assert!(app.sessions.sessions()[1]
            .messages
            .iter()
            .any(|m| m.role == "tool" && m.tool_call_id.as_deref() == Some("c1")));
        settle(&mut app).await;
        assert!(!file.exists());
        assert!(app.sessions.current().messages.is_empty());
    }

    #[tokio::test]
    async fn ask_modes_prompt_even_in_auto_and_approval_never_widens_them() {
        let mut app = agent_app("ask-write");
        app.sandbox.config.permission_mode = PermissionMode::AskBeforeWrite;
        assert!(!app.manual_approval);
        let file = app.sandbox.config.workspace_root.join("a.txt");
        feed(
            &mut app,
            vec![
                tool_call_event("c1", "write_file", r#"{"path":"a.txt","content":"hi"}"#),
                tool_call_event("c2", "bash", r#"{"command":"echo hi"}"#),
            ],
        );
        settle(&mut app).await;
        // The mode asks about writes, so the write waits for a yes — even
        // though approval is on auto.
        assert_eq!(
            app.approval_request().map(|call| call.id.as_str()),
            Some("c1")
        );
        assert!(!file.exists());

        app.answer_approval(true);
        settle(&mut app).await;
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "hi");
        // The shell is not something this mode asks about, so approving the
        // write did not unlock it: refused, and without a prompt.
        assert!(!app.awaiting_approval());
        assert!(app.cells.iter().any(|c| matches!(
            c,
            Cell::ToolResult { is_error: true, content, .. }
                if content.contains("permission denied")
        )));
    }

    #[tokio::test]
    async fn only_a_deliberate_y_or_n_press_answers_the_prompt() {
        use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
        let press = |code, modifiers| KeyEvent::new(code, modifiers);
        let mut app = parked_on_write("manual-keys-y").await;
        let file = app.sandbox.config.workspace_root.join("a.txt");

        // Typing, other keys and chords neither answer nor reach the composer.
        for key in [
            press(KeyCode::Char('x'), KeyModifiers::NONE),
            press(KeyCode::Enter, KeyModifiers::NONE),
            press(KeyCode::Char('y'), KeyModifiers::CONTROL),
            press(KeyCode::Char('n'), KeyModifiers::ALT),
        ] {
            assert!(crate::handle_key(&mut app, key));
            assert!(app.awaiting_approval());
            assert!(app.composer.is_empty());
        }
        // A held key's auto-repeat is not an answer either.
        let mut repeat = press(KeyCode::Char('y'), KeyModifiers::NONE);
        repeat.kind = KeyEventKind::Repeat;
        assert!(crate::handle_key(&mut app, repeat));
        assert!(app.awaiting_approval());
        assert!(!file.exists());

        // A press of `Y` (terminals report shift) approves.
        assert!(crate::handle_key(
            &mut app,
            press(KeyCode::Char('Y'), KeyModifiers::SHIFT)
        ));
        assert!(!app.awaiting_approval());
        settle(&mut app).await;
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "hi");

        // `n` declines.
        let mut app = parked_on_write("manual-keys-n").await;
        let file = app.sandbox.config.workspace_root.join("a.txt");
        assert!(crate::handle_key(
            &mut app,
            press(KeyCode::Char('n'), KeyModifiers::NONE)
        ));
        settle(&mut app).await;
        assert!(!file.exists());
        assert!(app.cells.iter().any(|c| matches!(
            c,
            Cell::ToolResult { is_error: true, content, .. } if content.contains("declined")
        )));
    }
}
