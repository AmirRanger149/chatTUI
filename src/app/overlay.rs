//! The full-screen popups: keyboard shortcuts, saved-conversation history,
//! and the code-block browser. Opening, navigating and acting within each
//! overlay lives here; the rendering side is in `crate::ui::overlay`.

use crate::app::{App, Cell};

/// Full-screen popups rendered on top of the interface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Overlay {
    Shortcuts,
    History { selected: usize },
    Code { selected: usize },
    Models { selected: usize },
    Providers { selected: usize },
    ToolDetail { selected: usize },
}

/// One inspectable tool activity for the `ctrl+t` overlay: what a write,
/// edit, list or bash call actually did. Reads are deliberately absent —
/// their windowed content is for the model, not the user's screen.
pub struct ToolDetailItem {
    pub title: String,
    pub lines: Vec<String>,
}

impl App {
    pub fn toggle_shortcuts(&mut self) {
        self.overlay = match self.overlay {
            Some(Overlay::Shortcuts) => None,
            _ => Some(Overlay::Shortcuts),
        };
    }

    pub fn toggle_history(&mut self) {
        if self.overlay.is_some_and(|o| matches!(o, Overlay::History { .. })) {
            self.overlay = None;
        } else {
            self.open_history();
        }
    }

    pub fn open_history(&mut self) {
        self.overlay = Some(Overlay::History {
            selected: self.sessions.current_index(),
        });
    }

    pub fn move_history_selection(&mut self, delta: i32) {
        let len = self.sessions.len();
        if len == 0 {
            return;
        }
        if let Some(Overlay::History { selected }) = &mut self.overlay {
            let next = (*selected as i32 + delta).rem_euclid(len as i32);
            *selected = next as usize;
        }
    }

    pub fn load_selected_session(&mut self) {
        let Some(Overlay::History { selected }) = self.overlay else {
            return;
        };
        if selected != self.sessions.current_index() {
            self.sessions.select(selected);
            self.rebuild_cells();
            self.scroll_from_bottom = 0;
        }
        self.overlay = None;
    }

    pub fn delete_selected_session(&mut self) {
        let Some(Overlay::History { selected }) = self.overlay else {
            return;
        };
        if self.sessions.len() <= 1 {
            return;
        }
        let was_current = selected == self.sessions.current_index();
        self.sessions.delete_at(selected);
        let len = self.sessions.len();
        if let Some(Overlay::History { selected }) = &mut self.overlay {
            *selected = (*selected).min(len - 1);
        }
        if was_current {
            self.rebuild_cells();
            self.scroll_from_bottom = 0;
        }
    }

    // -- code blocks -----------------------------------------------------------

    /// Every fenced code block in the conversation, in order. An in-flight
    /// streamed response is included too, so code can be copied while it is
    /// still being generated.
    pub fn code_blocks(&self) -> Vec<crate::code::CodeBlock> {
        let mut blocks = Vec::new();
        for cell in &self.cells {
            if let Cell::Assistant(text) = cell {
                blocks.extend(crate::code::extract(text));
            }
        }
        if !self.response.is_empty() {
            blocks.extend(crate::code::extract(&self.response));
        }
        blocks
    }

    pub fn open_code(&mut self) {
        if self.code_blocks().is_empty() {
            self.push_error("no code blocks in this conversation yet".into());
            return;
        }
        self.overlay = Some(Overlay::Code { selected: 0 });
    }

    pub fn toggle_code(&mut self) {
        if self.overlay.is_some_and(|o| matches!(o, Overlay::Code { .. })) {
            self.overlay = None;
        } else {
            self.open_code();
        }
    }

    pub fn move_code_selection(&mut self, delta: i32) {
        let len = self.code_blocks().len();
        if len == 0 {
            return;
        }
        if let Some(Overlay::Code { selected }) = &mut self.overlay {
            *selected = (*selected as i32 + delta).rem_euclid(len as i32) as usize;
        }
    }

    // -- tool detail ----------------------------------------------------------

    /// Every inspectable tool activity in the conversation, in order:
    /// writes show their added lines, edits their removed + added lines,
    /// bash its full command and output. Re-derived from the transcript
    /// cells on demand (like `code_blocks`), so nothing extra is stored.
    pub fn tool_details(&self) -> Vec<ToolDetailItem> {
        let mut items = Vec::new();
        for cell in &self.cells {
            let Cell::ToolCall { name, args, id } = cell else {
                continue;
            };
            let args_v = serde_json::from_str::<serde_json::Value>(args).ok();
            let result = self.cells.iter().rev().find_map(|c| match c {
                Cell::ToolResult {
                    id: rid, content, ..
                } if rid == id => Some(content.clone()),
                _ => None,
            });
            match name.as_str() {
                "write_file" => {
                    let path = args_v
                        .as_ref()
                        .and_then(|v| v["path"].as_str())
                        .unwrap_or("?");
                    let content = args_v
                        .as_ref()
                        .and_then(|v| v["content"].as_str())
                        .unwrap_or("");
                    let mut lines: Vec<String> =
                        content.lines().map(|l| format!("+ {l}")).collect();
                    if lines.is_empty() {
                        lines.push("(empty file)".to_string());
                    }
                    items.push(ToolDetailItem {
                        title: format!("Write {path}"),
                        lines,
                    });
                }
                "edit_file" => {
                    let path = args_v
                        .as_ref()
                        .and_then(|v| v["path"].as_str())
                        .unwrap_or("?");
                    let old = args_v
                        .as_ref()
                        .and_then(|v| v["old_string"].as_str())
                        .unwrap_or("");
                    let new = args_v
                        .as_ref()
                        .and_then(|v| v["new_string"].as_str())
                        .unwrap_or("");
                    let mut lines: Vec<String> =
                        old.lines().map(|l| format!("- {l}")).collect();
                    lines.extend(new.lines().map(|l| format!("+ {l}")));
                    items.push(ToolDetailItem {
                        title: format!("Edit {path}"),
                        lines,
                    });
                }
                "list_files" => {
                    let path = args_v
                        .as_ref()
                        .and_then(|v| v["path"].as_str())
                        .unwrap_or(".");
                    let lines = match result {
                        Some(content) => content.lines().map(str::to_string).collect(),
                        None => vec!["(no result yet)".to_string()],
                    };
                    items.push(ToolDetailItem {
                        title: format!("List {path}"),
                        lines,
                    });
                }
                "bash" => {
                    let cmd = args_v
                        .as_ref()
                        .and_then(|v| v["command"].as_str())
                        .unwrap_or("?");
                    let mut lines = vec![format!("$ {cmd}"), String::new()];
                    match result {
                        Some(content) => {
                            lines.extend(content.lines().map(str::to_string));
                        }
                        None => lines.push("(still running…)".to_string()),
                    }
                    items.push(ToolDetailItem {
                        title: format!("$ {}", crate::ui::theme::clamp_text(cmd, 80)),
                        lines,
                    });
                }
                _ => {}
            }
        }
        items
    }

    /// `ctrl+t`: open the tool-detail overlay on the most recent activity,
    /// or close it when it is already open — the same toggle feel as
    /// `ctrl+r` for reasoning.
    pub fn toggle_tool_detail(&mut self) {
        if self
            .overlay
            .is_some_and(|o| matches!(o, Overlay::ToolDetail { .. }))
        {
            self.overlay = None;
            return;
        }
        let count = self.tool_details().len();
        if count == 0 {
            self.push_notice("no tool activity to inspect yet".into());
            return;
        }
        self.overlay = Some(Overlay::ToolDetail {
            selected: count - 1,
        });
    }

    pub fn move_tool_detail_selection(&mut self, delta: i32) {
        let len = self.tool_details().len();
        if len == 0 {
            return;
        }
        if let Some(Overlay::ToolDetail { selected }) = &mut self.overlay {
            *selected = (*selected as i32 + delta).rem_euclid(len as i32) as usize;
        }
    }

    /// `Enter` in the code overlay: copy the selected block to the clipboard.
    pub fn copy_selected_code(&mut self) {
        let Some(Overlay::Code { selected }) = self.overlay else {
            return;
        };
        let blocks = self.code_blocks();
        let Some(block) = blocks.get(selected) else {
            self.overlay = None;
            return;
        };
        let lang = if block.lang.is_empty() {
            "code".to_string()
        } else {
            block.lang.clone()
        };
        let line_count = block.code.lines().count();
        match crate::clipboard::copy(&block.code) {
            Ok(outcome) if outcome.verified => self.push_notice(format!(
                "copied {lang} block ({line_count} lines) via {}",
                outcome.method
            )),
            Ok(outcome) => {
                // Best-effort OSC 52 path: the terminal may have ignored it,
                // so say "sent" instead of "copied" and point at the fix.
                let mut message = format!(
                    "sent {lang} block ({line_count} lines) via {} — paste to confirm",
                    outcome.method
                );
                if let Some(hint) = outcome.hint {
                    message.push_str(&format!(" ({hint})"));
                }
                self.push_notice(message);
            }
            Err(error) => self.push_error(format!("clipboard failed: {error}")),
        }
        self.overlay = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::App;
    use crate::config::Config;
    use crate::session::manager::SessionManager;

    fn test_app() -> App {
        App::new(Config::default(), SessionManager::for_tests())
    }

    #[test]
    fn tool_details_collect_write_edit_bash_but_not_reads() {
        let mut app = test_app();
        app.cells.push(Cell::ToolCall {
            name: "read_file".into(),
            args: r#"{"path":"a.txt"}"#.into(),
            id: "r1".into(),
        });
        app.cells.push(Cell::ToolResult {
            id: "r1".into(),
            content: "     1│hello".into(),
            is_error: false,
        });
        app.cells.push(Cell::ToolCall {
            name: "edit_file".into(),
            args: r#"{"path":"b.txt","old_string":"x","new_string":"y1\ny2"}"#.into(),
            id: "e1".into(),
        });
        app.cells.push(Cell::ToolResult {
            id: "e1".into(),
            content: "edited b.txt (+2 -1)".into(),
            is_error: false,
        });
        app.cells.push(Cell::ToolCall {
            name: "bash".into(),
            args: r#"{"command":"ls -la"}"#.into(),
            id: "b1".into(),
        });
        app.cells.push(Cell::ToolResult {
            id: "b1".into(),
            content: "total 8".into(),
            is_error: false,
        });

        let items = app.tool_details();
        assert_eq!(items.len(), 2, "reads stay out of the inspector");
        assert_eq!(items[0].title, "Edit b.txt");
        assert!(items[0].lines.contains(&"- x".to_string()));
        assert!(items[0].lines.contains(&"+ y1".to_string()));
        assert!(items[0].lines.contains(&"+ y2".to_string()));
        assert!(items[1].title.starts_with("$ ls -la"));
        assert!(items[1].lines.iter().any(|l| l == "total 8"));
    }

    #[test]
    fn toggle_tool_detail_opens_latest_and_closes() {
        let mut app = test_app();
        app.toggle_tool_detail();
        assert!(app.overlay.is_none(), "nothing to inspect yet");
        assert!(matches!(app.cells.last(), Some(Cell::Notice(_))));

        app.cells.push(Cell::ToolCall {
            name: "bash".into(),
            args: r#"{"command":"pwd"}"#.into(),
            id: "b1".into(),
        });
        app.cells.push(Cell::ToolResult {
            id: "b1".into(),
            content: "/tmp".into(),
            is_error: false,
        });
        app.toggle_tool_detail();
        assert!(matches!(
            app.overlay,
            Some(Overlay::ToolDetail { selected: 0 })
        ));
        app.toggle_tool_detail();
        assert!(app.overlay.is_none());
    }

    #[test]
    fn code_blocks_are_collected_from_assistant_messages() {
        let mut app = test_app();
        app.cells.push(Cell::Assistant(
            "intro\n```rust\nlet x = 1;\n```\nmiddle\n```js\nlet y = 2;\n```".into(),
        ));
        app.response = "```py\nprint(1)".into(); // streamed, still unclosed
        let blocks = app.code_blocks();
        assert_eq!(blocks.len(), 3);
        assert_eq!(blocks[0].lang, "rust");
        assert_eq!(blocks[0].code, "let x = 1;");
        assert_eq!(blocks[1].lang, "js");
        assert_eq!(blocks[2].lang, "py");
    }

    #[test]
    fn code_overlay_opens_only_with_blocks() {
        let mut app = test_app();
        app.open_code();
        assert!(app.overlay.is_none());
        assert!(matches!(app.cells.last(), Some(Cell::Error(_))));
        app.cells.push(Cell::Assistant("```\ncode here\n```".into()));
        app.open_code();
        assert!(matches!(app.overlay, Some(Overlay::Code { selected: 0 })));
        app.move_code_selection(3); // wraps around with a single block
        assert!(matches!(app.overlay, Some(Overlay::Code { selected: 0 })));
    }

    #[test]
    fn history_overlay_moves_and_clamps() {
        let mut app = test_app();
        app.sessions.new_session();
        app.sessions.new_session();
        app.open_history();
        app.move_history_selection(5);
        if let Some(Overlay::History { selected }) = app.overlay {
            assert!(selected < app.sessions.len());
        } else {
            panic!("history overlay should be open");
        }
    }
}
