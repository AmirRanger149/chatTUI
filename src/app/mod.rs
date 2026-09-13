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

use crate::api::types::StreamEvent;
use crate::config::Config;
use crate::session::manager::SessionManager;
use anyhow::Result;
use std::time::{Duration, Instant};
use tokio::sync::mpsc::Receiver;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const PLACEHOLDER: &str = "Ask chatTUI to do anything";
pub const MAX_COMPOSER_ROWS: usize = 8;
/// Number of rows visible inside overlay lists (history / models / code); also
/// the page size for pgup/pgdn navigation within an overlay.
pub const OVERLAY_ROWS: usize = 12;
const QUIT_PRIME_WINDOW: Duration = std::time::Duration::from_secs(2);

/// A rendered entry of the conversation transcript.
#[derive(Debug, Clone, PartialEq)]
pub enum Cell {
    User(String),
    Assistant(String),
    Error(String),
    Notice(String),
}

pub struct App {
    pub config: Config,
    pub sessions: SessionManager,
    pub cells: Vec<Cell>,
    pub response: String,
    pub streaming: bool,
    pub stream_started: Option<Instant>,
    pub tokens: Option<Receiver<StreamEvent>>,
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
}

impl App {
    pub fn new(config: Config, sessions: SessionManager) -> Self {
        let mut app = Self {
            config,
            sessions,
            cells: Vec::new(),
            response: String::new(),
            streaming: false,
            stream_started: None,
            tokens: None,
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
        };
        app.rebuild_cells();
        // For providers whose default model is availability-based (custom
        // providers without an explicit `model`),
        // resolve the default against the endpoint's live model list in the
        // background. Silently keeps the built-in default when the fetch
        // fails or no key is configured.
        app.request_available_default();
        app
    }

    pub(crate) fn rebuild_cells(&mut self) {
        self.cells = self
            .sessions
            .current()
            .messages
            .iter()
            .map(|message| match message.role.as_str() {
                "assistant" => Cell::Assistant(message.content.clone()),
                _ => Cell::User(message.content.clone()),
            })
            .collect();
    }

    // -- transient UI actions -------------------------------------------------

    /// `Esc`: close popups first, then interrupt a running stream, then clear the composer.
    pub fn escape(&mut self) {
        if self.overlay.take().is_some() {
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
        self.tokens = None;
        self.streaming = false;
        self.stream_started = None;
        self.finish_partial();
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
    pub(crate) fn finish_partial(&mut self) {
        if !self.response.is_empty() {
            let mut text = std::mem::take(&mut self.response);
            // An interrupted stream can leave a dangling `<think>`; close it so
            // the transcript shows a finished (collapsible) reasoning block.
            if crate::ui::thinking::is_thinking(&text) {
                text.push_str(crate::ui::thinking::CLOSE_TAG);
            }
            self.cells.push(Cell::Assistant(text.clone()));
            self.sessions.add_message("assistant", text);
        }
    }

    pub(crate) fn push_error(&mut self, message: String) {
        self.cells.push(Cell::Error(message));
    }

    pub(crate) fn push_notice(&mut self, message: String) {
        self.cells.push(Cell::Notice(message));
    }

    /// The context summary shown on the right-hand side of the footer.
    pub fn context_summary(&self) -> String {
        let session = self.sessions.current();
        let messages = session.messages.len();
        let chars: usize = session.messages.iter().map(|m| m.content.len()).sum();
        format!(
            "{messages} msgs · ~{} tok",
            crate::ui::theme::human_tokens(chars / 4)
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
