//! Application state and the actions shared across all feature modules.
//!
//! `App` is the single source of truth for the TUI: the conversation
//! transcript (`cells`), the in-flight stream, the composer buffer, and the
//! open overlay. The feature modules add behaviour to `App` through their own
//! `impl App` blocks:
//!
//! - [`composer`]: text editing, cursor movement, paste and history recall
//! - [`commands`]: slash commands and the slash popup
//! - [`streaming`]: running a chat request and consuming its tokens
//! - [`models`]: the `/model` picker and availability-based default models
//! - [`providers`]: the `/provider` picker and switching
//! - [`overlay`]: the full-screen popups (shortcuts, history, code, models)

pub mod commands;
pub mod composer;
pub mod models;
pub mod overlay;
pub mod providers;
pub mod streaming;

pub use commands::SLASH_COMMANDS;
pub use models::ModelCatalog;
pub use overlay::Overlay;

use crate::api::types::{Question, ReasoningReplay, StreamEvent, ToolCall, Usage};
use crate::config::Config;
use crate::sandbox::os_isolation::OsIsolation;
use crate::sandbox::permissions::PermissionMode;
use crate::sandbox::{Sandbox, SandboxConfig};
use crate::session::manager::SessionManager;
use anyhow::Result;
use std::collections::HashSet;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use tokio::sync::mpsc::Receiver;
use tokio::task::JoinHandle;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const PLACEHOLDER: &str = "Ask chatTUI to do anything";
pub const MAX_COMPOSER_ROWS: usize = 8;
/// Number of rows visible inside overlay lists (history / models / code); also
/// the page size for pgup/pgdn navigation within an overlay.
pub const OVERLAY_ROWS: usize = 12;
const QUIT_PRIME_WINDOW: Duration = std::time::Duration::from_secs(2);

/// Identical failing tool rounds in a row before the agent is stopped —
/// the stagnation guard. The same call failing repeatedly means the model
/// is looping, not working; three repetitions prove it within a few rounds
/// instead of letting it run to a step counter.
pub const AGENT_STAGNATION_LIMIT: usize = 3;
/// Consecutive rounds in which every tool result is an error before the
/// agent is stopped — the "nothing is working anymore" guard.
pub const AGENT_CONSECUTIVE_FAILURE_LIMIT: usize = 4;
/// Backstop ceiling for agent tool rounds when config.json does not set
/// one. This is insurance, not policy: healthy runs keep going as long as
/// they make progress and are stopped by the guards above when they don't.
pub const DEFAULT_AGENT_MAX_ROUNDS: u64 = 50;

/// A rendered entry of the conversation transcript.
#[derive(Debug, Clone, PartialEq)]
pub enum Cell {
    User(String),
    Assistant(String),
    Error(String),
    Notice(String),
    ToolCall { name: String, args: String, id: String },
    ToolResult { id: String, content: String, is_error: bool },
}

/// What the status row shows while the API client waits out a retry backoff.
/// Set by `StreamEvent::Retry`, cleared as soon as tokens flow again, the
/// stream ends, or the user interrupts.
#[derive(Debug, Clone)]
pub struct RetryView {
    /// When the backoff started — the countdown is derived from this, so the
    /// animation needs no extra timer.
    pub started: Instant,
    /// How long the backoff lasts in total.
    pub wait: Duration,
    /// Which retry this is (1-based) and the maximum.
    pub attempt: u8,
    pub max: u8,
    /// Why the previous attempt failed, shown dimmed after the countdown.
    pub reason: String,
}

pub struct App {
    pub config: Config,
    pub sessions: SessionManager,
    pub cells: Vec<Cell>,
    pub response: String,
    pub streaming: bool,
    pub stream_started: Option<Instant>,
    pub tokens: Option<Receiver<StreamEvent>>,
    /// Handle for the in-flight LLM request task so `interrupt()` can abort
    /// it (dropping the request future closes the HTTP stream) instead of
    /// leaving it running after the user cancelled.
    stream_task: Option<JoinHandle<()>>,
    pub models: ModelCatalog,
    models_rx: Option<Receiver<Result<Vec<String>>>>,
    /// Whether the in-flight model-list fetch is a background
    /// availability-based default-model pick (`true`) rather than a fetch
    /// the `/model` picker explicitly requested (`false`).
    models_fetch_auto: bool,
    pub composer: String,
    /// Byte offset of the editing cursor inside `composer` (char boundary).
    pub cursor: usize,
    pub prompt_history: Vec<String>,
    pub history_nav: Option<(usize, String)>,
    pub scroll_from_bottom: u16,
    pub overlay: Option<Overlay>,
    pub slash_selected: usize,
    /// Keep finished `<think>` reasoning blocks expanded in the transcript.
    pub show_thinking: bool,
    pub quit_primed_at: Option<Instant>,
    pub should_quit: bool,
    // Agent / Sandbox state
    pub sandbox: Sandbox,
    pub pending_tool_calls: Vec<ToolCall>,
    /// Tool rounds completed in the current agent run (display + ceiling).
    pub agent_iterations: usize,
    pub agent_mode: bool,
    /// Configured backstop ceiling for tool rounds (`agent.max_rounds`).
    pub agent_max_rounds: usize,
    /// Fingerprint (sorted name+arguments) of the previous tool round —
    /// lets the stagnation guard recognize the same round coming back.
    pub agent_last_round_key: Option<String>,
    /// Identical failing rounds seen in a row so far.
    pub agent_repeat_failures: usize,
    /// Rounds in a row where every tool result was an error.
    pub agent_consecutive_failures: usize,
    /// Ids of tool calls whose arguments arrived unparseable (a stream cut
    /// mid-arguments). They get an explicit truncation error instead of a
    /// misleading "missing field", and are persisted with clean JSON.
    pub truncated_tool_calls: HashSet<String>,
    /// True when the current turn's stream stopped on the provider's output
    /// cap (`finish_reason: length` / `max_tokens`) rather than on a dropped
    /// connection. A tool call truncated by the cap needs different recovery
    /// advice from one cut off by a dead connection, so the verdict is kept
    /// here rather than guessed at when the error is written.
    pub output_limit_hit: bool,
    /// Live retry state for the animated status row; `None` outside a
    /// same-model retry backoff.
    pub retry_state: Option<RetryView>,
    /// Set when a tool result landed since the last frame. The main loop
    /// re-asserts the terminal state before drawing — a command the agent
    /// ran must never be able to leave the user's terminal in a state
    /// chatTUI did not choose.
    pub terminal_dirty: bool,
    /// Handle for the in-flight tool-execution task. Tools run off this
    /// thread so a long command cannot freeze the UI; the handle is what
    /// lets `interrupt()` stop the round instead of leaving it running.
    tool_task: Option<JoinHandle<()>>,
    /// The tool calls of the round currently executing. Needed to backfill
    /// an "interrupted" result for any call that never reported back: an
    /// assistant message with tool calls must be followed by exactly one
    /// result per call, or the next request is rejected.
    active_round_calls: Vec<ToolCall>,
    /// Ids in [`Self::active_round_calls`] that have already reported.
    answered_tool_ids: HashSet<String>,
    /// A tool call parked until the user says yes or no. While this is set
    /// the round is not running and the status row shows the prompt.
    pub awaiting_approval: Option<PendingApproval>,
    /// The model called `ask_user` and is waiting for answers.
    pub awaiting_questions: Option<PendingQuestions>,
    /// The signature the provider sent for the reasoning currently being
    /// streamed. Attached to the assistant message when it is stored.
    pending_reasoning_signature: Option<String>,
    /// Token counts from the most recent request, when the provider reported
    /// them. The only trustworthy basis for a compaction decision.
    pub last_usage: Option<Usage>,
}

/// A set of questions from `ask_user`, and the answers collected so far.
///
/// They are answered one at a time — five questions on one line is a wall of
/// text — so `current` walks forward and `answers` fills in behind it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PendingQuestions {
    pub call_id: String,
    pub questions: Vec<Question>,
    /// Index of the question on screen now.
    pub current: usize,
    /// One slot per question; `None` means unanswered so far.
    pub answers: Vec<Option<String>>,
}

/// Messages kept verbatim when the history is compacted. Everything older is
/// a candidate for elision; everything newer is what the model is actually
/// working from.
const KEEP_RECENT_MESSAGES: usize = 12;

/// A tool call the permission layer will not run without the user.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PendingApproval {
    /// Which call of the round this is about.
    pub call_id: String,
    /// What the call would do, in one line.
    pub summary: String,
    /// Why it needs permission.
    pub reason: String,
    /// True when the sandbox already refused it and the question is whether
    /// to run it unconfined instead.
    pub escalated: bool,
    /// For an escalated call: what the sandboxed attempt produced.
    pub output: Option<String>,
}

/// The user's answer to an approval prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Approval {
    /// Run this call only.
    Once,
    /// Run it, and remember the decision for the rest of the session.
    Always,
    /// Do not run it.
    Deny,
}

impl App {
    pub fn new(config: Config, sessions: SessionManager) -> Self {
        // Build sandbox from config — never default to process cwd.
        // An explicit `permission_mode` wins; when it is unset the legacy
        // auto_approve/allow_shell flags decide, as they always did. An
        // unknown mode fails closed to read-only until the config is fixed.
        let (permission_mode, mode_error) = match config.sandbox.permission_mode.as_deref() {
            Some(raw) if !raw.trim().is_empty() => match PermissionMode::parse(raw) {
                Some(mode) => (mode, None),
                None => (
                    PermissionMode::ReadOnly,
                    Some(format!(
                        "unknown sandbox.permission_mode '{raw}' — valid modes: read-only, \
                         workspace-write, ask-before-write, ask-before-shell, full-auto"
                    )),
                ),
            },
            _ => (
                PermissionMode::from_legacy_flags(
                    config.sandbox.auto_approve,
                    config.sandbox.allow_shell,
                ),
                None,
            ),
        };
        // Kernel isolation for shell commands: auto (default) / require /
        // off. An unknown value falls back to auto and reports the typo.
        let (os_isolation, isolation_error) = match config.sandbox.os_isolation.as_deref() {
            Some(raw) if !raw.trim().is_empty() => match OsIsolation::parse(raw) {
                Some(mode) => (mode, None),
                None => (
                    OsIsolation::Auto,
                    Some(format!(
                        "unknown sandbox.os_isolation '{raw}' — valid values: auto, require, off"
                    )),
                ),
            },
            _ => (OsIsolation::Auto, None),
        };
        let sandbox_config = SandboxConfig {
            enabled: config.sandbox.enabled,
            workspace_root: PathBuf::new(),
            auto_approve: config.sandbox.auto_approve,
            allow_shell: config.sandbox.allow_shell,
            max_file_size: 1024 * 1024,
            shell_timeout: Duration::from_secs(config.sandbox.shell_timeout_secs.max(1)),
            permission_mode,
            os_isolation,
            extra_sensitive_names: config.sandbox.extra_sensitive_names.clone(),
        };
        let mut sandbox = Sandbox::new(sandbox_config);
        let mut target_error = None;
        if !config.sandbox.workspace_root.trim().is_empty() {
            if let Err(error) = sandbox.set_target(&config.sandbox.workspace_root) {
                target_error = Some(error.to_string());
            }
        }

        // Bind before the struct literal: the `config` shorthand moves the
        // value, so later initializers may not read it.
        let agent_max_rounds = config.agent.max_rounds.max(1).min(10_000) as usize;
        let mut app = Self {
            config,
            sessions,
            cells: Vec::new(),
            response: String::new(),
            streaming: false,
            stream_started: None,
            tokens: None,
            stream_task: None,
            models: ModelCatalog::default(),
            models_rx: None,
            models_fetch_auto: false,
            composer: String::new(),
            cursor: 0,
            prompt_history: Vec::new(),
            history_nav: None,
            scroll_from_bottom: 0,
            overlay: None,
            slash_selected: 0,
            show_thinking: false,
            quit_primed_at: None,
            should_quit: false,
            sandbox,
            pending_tool_calls: Vec::new(),
            agent_iterations: 0,
            agent_mode: true, // Agent mode enabled by default when sandbox enabled
            agent_max_rounds,
            agent_last_round_key: None,
            agent_repeat_failures: 0,
            agent_consecutive_failures: 0,
            truncated_tool_calls: HashSet::new(),
            output_limit_hit: false,
            retry_state: None,
            terminal_dirty: false,
            tool_task: None,
            active_round_calls: Vec::new(),
            answered_tool_ids: HashSet::new(),
            awaiting_approval: None,
            awaiting_questions: None,
            pending_reasoning_signature: None,
            last_usage: None,
        };
        app.rebuild_cells();
        if let Some(error) = target_error {
            app.push_error(format!("sandbox target from config is invalid: {error}"));
        }
        if let Some(error) = mode_error {
            app.push_error(format!("sandbox {error}"));
        }
        if let Some(error) = isolation_error {
            app.push_error(format!("sandbox {error}"));
        }
        // For providers whose default model is availability-based (custom
        // providers without an explicit `model`),
        // resolve the default against the endpoint's live model list in the
        // background. Silently keeps the built-in default when the fetch
        // fails or no key is configured.
        app.request_available_default();
        app
    }

    pub(crate) fn rebuild_cells(&mut self) {
        self.cells = Vec::new();
        for message in self.sessions.current().messages.iter() {
            match message.role.as_str() {
                "assistant" => {
                    if let Some(tool_calls) = &message.tool_calls {
                        // Show tool calls as cells
                        for tc in tool_calls {
                            self.cells.push(Cell::ToolCall {
                                name: tc.name.clone(),
                                args: tc.arguments.clone(),
                                id: tc.id.clone(),
                            });
                        }
                        if !message.content.is_empty() {
                            self.cells.push(Cell::Assistant(message.content.clone()));
                        }
                    } else {
                        self.cells.push(Cell::Assistant(message.content.clone()));
                    }
                }
                "tool" => {
                    let id = message.tool_call_id.clone().unwrap_or_else(|| "unknown".to_string());
                    self.cells.push(Cell::ToolResult {
                        id,
                        content: message.content.clone(),
                        // Sessions saved before statuses were recorded have
                        // no verdict; treat those as successes rather than
                        // marking old history as broken.
                        is_error: message.is_error.unwrap_or(false),
                    });
                }
                "user" => {
                    self.cells.push(Cell::User(message.content.clone()));
                }
                _ => {
                    self.cells.push(Cell::User(message.content.clone()));
                }
            }
        }
    }

    // -- transient UI actions -------------------------------------------------

    /// `Esc`: close popups first, then interrupt a running stream, then clear the composer.
    pub fn escape(&mut self) {
        if self.overlay.take().is_some() {
            return;
        }
        // At an approval prompt esc means "no", not "abort everything": the
        // round survives and the model is told, so it can find another way
        // instead of losing the work it had already done.
        if self.awaiting_approval.is_some() {
            self.resolve_approval(Approval::Deny);
            return;
        }
        // Same reasoning for questions: dismissing them is an answer the
        // model can act on, losing the round is not.
        if self.awaiting_questions.is_some() {
            self.skip_questions();
            return;
        }
        if self.streaming {
            self.interrupt();
            return;
        }
        if !self.composer.is_empty() {
            self.clear_composer();
        }
    }

    pub fn interrupt(&mut self) {
        // Abort the in-flight LLM request: dropping its future closes the
        // HTTP stream. Tool execution runs inline on this loop and is
        // bounded by the shell timeout, so no separate tool process can
        // outlive the interrupt.
        if let Some(task) = self.stream_task.take() {
            task.abort();
        }
        // A command already running is not killed by this — it is bounded by
        // the shell timeout and its own process group. What stops here is the
        // round: no further tool starts, and no result restarts the loop.
        if let Some(task) = self.tool_task.take() {
            task.abort();
        }
        self.tokens = None;
        self.streaming = false;
        self.stream_started = None;
        self.pending_tool_calls.clear();
        self.backfill_interrupted_tools();
        self.reset_agent_health();
        self.truncated_tool_calls.clear();
        self.output_limit_hit = false;
        self.retry_state = None;
        self.finish_partial();
    }

    /// Reset the per-run agent health tracking (round count, stagnation
    /// fingerprint, failure streaks). Called whenever an agent run ends —
    /// finished normally, stopped by a guard, or interrupted.
    pub(crate) fn reset_agent_health(&mut self) {
        self.agent_iterations = 0;
        self.agent_last_round_key = None;
        self.agent_repeat_failures = 0;
        self.agent_consecutive_failures = 0;
        self.active_round_calls.clear();
        self.answered_tool_ids.clear();
        // A prompt nobody can answer any more is worse than no prompt.
        self.awaiting_approval = None;
        self.awaiting_questions = None;
    }

    /// Record an explicit "interrupted" result for every tool call of the
    /// in-flight round that never reported back.
    ///
    /// Without this the saved history holds an assistant message whose tool
    /// calls have no results, and the next request to the provider is
    /// rejected for the whole conversation.
    fn backfill_interrupted_tools(&mut self) {
        if self.active_round_calls.is_empty() {
            return;
        }
        let calls = std::mem::take(&mut self.active_round_calls);
        for call in calls {
            let id = call.id.clone();
            if self.answered_tool_ids.contains(&id) {
                continue;
            }
            let content = "interrupted by the user before this tool could finish".to_string();
            self.sessions
                .add_tool_result(id.clone(), content.clone(), true);
            self.push_tool_result(id, content, true);
        }
        self.answered_tool_ids.clear();
    }

    /// `ctrl+r`: expand / collapse completed reasoning blocks.
    pub fn toggle_thinking(&mut self) {
        self.show_thinking = !self.show_thinking;
    }

    /// True while the model is streaming tokens inside a `<think>` block.
    pub fn is_thinking(&self) -> bool {
        self.streaming && crate::ui::thinking::is_thinking(&self.response)
    }

    /// The reasoning line currently being written, for the status row.
    pub fn current_thought(&self) -> Option<String> {
        crate::ui::thinking::latest_thought(&self.response)
    }

    pub fn scroll(&mut self, delta: i32) {
        let next = self.scroll_from_bottom as i32 + delta;
        self.scroll_from_bottom = next.clamp(0, u16::MAX as i32) as u16;
    }

    // -- quit flow --------------------------------------------------------------

    pub fn prime_quit(&mut self) {
        if self.quit_primed() {
            self.should_quit = true;
        } else {
            self.quit_primed_at = Some(Instant::now());
        }
    }

    pub fn quit_primed(&self) -> bool {
        self.quit_primed_at
            .is_some_and(|at| at.elapsed() < QUIT_PRIME_WINDOW)
    }

    // -- transcript cells ---------------------------------------------------------

    /// Commit an in-flight assistant response (used on completion and interrupt).
    /// How full the context window is, in tokens.
    ///
    /// Prefers what the provider actually counted for the last request, and
    /// adds an estimate for whatever has been added since — a tool result can
    /// be 20 kB, and waiting for the next request to find that out is exactly
    /// the overflow this is meant to prevent.
    pub fn context_tokens(&self) -> u64 {
        let chars: usize = self
            .sessions
            .current()
            .messages
            .iter()
            .map(|message| message.content.len())
            .sum();
        let estimated = (chars / 4) as u64;
        match self.last_usage {
            // The provider counted the history as it was then; the estimate
            // covers the whole thing now, so the larger of the two is the
            // honest number.
            Some(usage) => usage.input_tokens.max(estimated),
            None => estimated,
        }
    }

    /// Compact the history before it overflows the window.
    ///
    /// Called before every request. Doing it here rather than after a failure
    /// is the point: an overflowed request is rejected outright, and by then
    /// the work in flight is already lost.
    fn maybe_compact(&mut self) {
        let window = self.config.context_window_tokens.max(1_000);
        let percent = self.config.compact_at_percent.clamp(10, 99) as u64;
        let target = window * percent / 100;
        if self.context_tokens() < target {
            return;
        }
        // Keep enough recent context to stay coherent: the last dozen messages
        // plus anything the current round still needs.
        let report = self
            .sessions
            .compact(KEEP_RECENT_MESSAGES, target as usize);
        // One real number is worth more than the estimate from here on.
        self.last_usage = None;
        if let Some(report) = report {
            self.push_notice(report);
        }
    }

    /// How this provider's reasoning is treated on replay.
    ///
    /// Anthropic signs its thinking blocks and validates the signature on the
    /// next turn, so its reasoning has to survive; OpenAI-compatible endpoints
    /// have no such contract and resent reasoning is mostly wasted tokens.
    pub(crate) fn reasoning_replay(&self) -> ReasoningReplay {
        ReasoningReplay::resolve(
            self.config.reasoning_replay.as_deref(),
            self.config.provider == "anthropic",
        )
    }

    /// Attach the signature that arrived with this turn's reasoning to the
    /// assistant message that was just stored.
    fn attach_reasoning_signature(&mut self) {
        if let Some(signature) = self.pending_reasoning_signature.take() {
            self.sessions
                .set_last_assistant_reasoning_signature(signature);
        }
    }

    pub(crate) fn finish_partial(&mut self) {
        if !self.response.is_empty() {
            let mut text = std::mem::take(&mut self.response);
            // An interrupted stream can leave a dangling `<think>`; close it so
            // the transcript shows a finished (collapsible) reasoning block.
            if crate::ui::thinking::is_thinking(&text) {
                text.push_str(crate::ui::thinking::CLOSE_TAG);
            }
            // If the model only streamed reasoning (reasoning-only reply), still
            // surface that text as the visible answer instead of an empty cell.
            if crate::ui::thinking::strip(&text).is_empty() {
                if let Some(inner) = crate::ui::thinking::reasoning_text(&text) {
                    if !inner.trim().is_empty() {
                        text = inner;
                    }
                }
            }
            self.cells.push(Cell::Assistant(text.clone()));
            self.sessions.add_message("assistant", text);
            self.attach_reasoning_signature();
        }
    }

    pub(crate) fn push_error(&mut self, message: String) {
        self.cells.push(Cell::Error(message));
    }

    pub(crate) fn push_notice(&mut self, message: String) {
        self.cells.push(Cell::Notice(message));
    }

    pub(crate) fn push_tool_call(&mut self, tc: &ToolCall) {
        self.cells.push(Cell::ToolCall {
            name: tc.name.clone(),
            args: tc.arguments.clone(),
            id: tc.id.clone(),
        });
    }

    pub(crate) fn push_tool_result(&mut self, id: String, content: String, is_error: bool) {
        self.cells.push(Cell::ToolResult { id, content, is_error });
        self.terminal_dirty = true;
    }

    /// The context summary shown on the right-hand side of the footer.
    pub fn context_summary(&self) -> String {
        let session = self.sessions.current();
        let messages = session.messages.len();
        let sandbox_status = if !self.config.sandbox.enabled {
            " · sandbox:off"
        } else if self.sandbox.has_target() {
            " · sandbox:on"
        } else {
            " · sandbox:no-target"
        };
        let tokens = self.context_tokens();
        let window = self.config.context_window_tokens.max(1_000);
        // A counted number and an estimate are not the same claim, so they do
        // not get the same formatting: "~" means chatTUI guessed.
        let counted = match self.last_usage {
            Some(usage) if usage.input_tokens > 0 => {
                crate::ui::theme::human_tokens(tokens as usize)
            }
            _ => format!("~{}", crate::ui::theme::human_tokens(tokens as usize)),
        };
        format!(
            "{messages} msgs · {counted}/{} tok ({}%){sandbox_status}",
            crate::ui::theme::human_tokens(window as usize),
            (tokens * 100 / window).min(999)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_app() -> App {
        App::new(Config::default(), SessionManager::for_tests())
    }

    #[test]
    fn escape_closes_overlay_before_stream() {
        let mut app = test_app();
        app.overlay = Some(Overlay::Shortcuts);
        app.escape();
        assert!(app.overlay.is_none());
        app.streaming = true;
        app.response = "partial".into();
        app.escape();
        assert!(!app.streaming);
        assert!(matches!(app.cells.last(), Some(Cell::Assistant(_))));
    }

    #[test]
    fn quit_needs_double_ctrl_c() {
        let mut app = test_app();
        app.prime_quit();
        assert!(!app.should_quit);
        app.prime_quit();
        assert!(app.should_quit);
    }
}
