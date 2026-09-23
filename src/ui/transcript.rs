//! The scrollback transcript: a session header card, `› `-prefixed user turns,
//! markdown assistant turns, notices and errors. It stays pinned to the bottom
//! until the user scrolls back.

use crate::app::{App, Cell};
use crate::ui::{markdown, theme, thinking};
use ratatui::prelude::*;
use ratatui::widgets::Paragraph;
use std::env;

pub fn render(frame: &mut Frame, area: Rect, app: &App) {
    let width = area.width.max(20) as usize;
    let mut lines: Vec<Line<'static>> = Vec::new();

    lines.extend(theme::with_border(header_lines(app, width.saturating_sub(4))));
    lines.push(Line::from(""));

    // Pair tool results with the call that produced them so results can be
    // rendered compactly (a successful read needs no row at all).
    let mut call_names: Vec<(String, String)> = Vec::new();
    for cell in &app.cells {
        if let Cell::ToolCall { name, id, .. } = cell {
            call_names.push((id.clone(), name.clone()));
        }
        let before = lines.len();
        match cell {
            Cell::ToolCall { name, args, id } => {
                lines.extend(tool_call_lines(name, args, id, width));
            }
            Cell::ToolResult { id, content, is_error } => {
                let call_name = call_names
                    .iter()
                    .rev()
                    .find(|(cid, _)| cid == id)
                    .map(|(_, name)| name.as_str());
                lines.extend(tool_result_lines(id, content, *is_error, width, call_name));
            }
            _ => lines.extend(cell_lines(cell, width, app)),
        }
        if lines.len() > before {
            lines.push(Line::from(""));
        }
    }
    if !app.response.is_empty() {
        lines.extend(assistant_lines(&app.response, width, app));
        lines.push(Line::from(""));
    }

    // Show pending tool calls during streaming
    if !app.pending_tool_calls.is_empty() {
        for tc in &app.pending_tool_calls {
            lines.extend(tool_call_lines(&tc.name, &tc.arguments, &tc.id, width));
            lines.push(Line::from(""));
        }
    }

    let total = lines.len();
    let height = area.height as usize;
    let skip = total
        .saturating_sub(height)
        .saturating_sub(app.scroll_from_bottom as usize);
    frame.render_widget(Paragraph::new(lines).scroll((skip as u16, 0)), area);
}

/// The `>_< chatTUI (vX)` card shown at the top of every session.
fn header_lines(app: &App, max_inner: usize) -> Vec<Line<'static>> {
    let inner = max_inner.min(56);
    let title = vec![
        Span::styled(">_< ", theme::dim()),
        Span::styled("chatTUI", Style::new().bold()),
        Span::styled(" ", theme::dim()),
        Span::styled(format!("(v{})", crate::app::VERSION), theme::dim()),
    ];
    let provider_name = app.active_provider_name();
    let mut provider_line = vec![
        Span::styled("provider: ", theme::dim()),
        Span::styled(provider_name.clone(), Style::new().bold()),
    ];
    let prov_hint_w = "   /provider to change".len();
    if inner > "provider: ".len() + provider_name.len() + prov_hint_w {
        provider_line.push(Span::styled("   ", theme::dim()));
        provider_line.push(Span::styled("/provider", Style::new().fg(theme::ACCENT)));
        provider_line.push(Span::styled(" to change", theme::dim()));
    }
    let mut model_line = vec![
        Span::styled("model: ", theme::dim()),
        Span::styled(app.config.model.clone(), Style::new().fg(theme::ACCENT)),
    ];
    let hint_w = "   /model to change".len();
    if inner > "model: ".len() + app.config.model.len() + hint_w {
        model_line.push(Span::styled("   ", theme::dim()));
        model_line.push(Span::styled("/model", Style::new().fg(theme::ACCENT)));
        model_line.push(Span::styled(" to change", theme::dim()));
    }
    let dir_label = "directory: ";
    let dir = theme::display_path(&current_dir(), inner.saturating_sub(dir_label.len()));
    let dir_line = vec![Span::styled(dir_label, theme::dim()), Span::raw(dir)];
    
    let sandbox_label = "sandbox: ";
    let (sandbox_status, sandbox_color) = if !app.config.sandbox.enabled {
        ("off".to_string(), Color::DarkGray)
    } else if app.sandbox.has_target() {
        (
            format!(
                "on ({})",
                theme::display_path(
                    app.sandbox.config.workspace_root.to_string_lossy().as_ref(),
                    30,
                )
            ),
            Color::Green,
        )
    } else {
        ("no target — /sandbox <dir>".to_string(), Color::Yellow)
    };
    let sandbox_line = vec![
        Span::styled(sandbox_label, theme::dim()),
        Span::styled(sandbox_status, Style::new().fg(sandbox_color)),
    ];
    
    vec![
        Line::from(title),
        Line::from(""),
        Line::from(provider_line),
        Line::from(model_line),
        Line::from(dir_line),
        Line::from(sandbox_line),
    ]
}

fn current_dir() -> String {
    env::current_dir()
        .map(|path| path.to_string_lossy().into_owned())
        .unwrap_or_else(|_| "?".into())
}

/// Assistant output, with any `<think> … </think>` reasoning peeled off into
/// its own animated / collapsible cell above the answer.
fn assistant_lines(text: &str, width: usize, app: &App) -> Vec<Line<'static>> {
    let segments = thinking::split(text);
    if !segments.iter().any(|segment| segment.thinking) {
        return markdown::render(text, width);
    }
    let mut lines = Vec::new();
    for segment in segments {
        if segment.thinking {
            lines.extend(thinking::render_segment(
                &segment.text,
                width,
                segment.open,
                app.show_thinking,
                app.elapsed_ms(),
            ));
            lines.push(Line::from(""));
        } else if !segment.text.trim().is_empty() {
            lines.extend(markdown::render(segment.text.trim_start_matches('\n'), width));
        }
    }
    lines
}

/// One compact activity row per tool call. The full arguments stay in the
/// model's history, but the transcript shows only what the user needs:
/// `Read src/main.rs · lines 1-250`, `Edit src/app.js · +2 -1`, …
fn tool_call_lines(name: &str, args: &str, id: &str, width: usize) -> Vec<Line<'static>> {
    let args_v = serde_json::from_str::<serde_json::Value>(args).ok();
    let path = args_v
        .as_ref()
        .and_then(|v| v["path"].as_str())
        .unwrap_or("?")
        .to_string();
    let summary = match name {
        "read_file" => {
            let offset = args_v
                .as_ref()
                .and_then(|v| v["offset"].as_u64())
                .unwrap_or(1)
                .max(1);
            let limit = args_v
                .as_ref()
                .and_then(|v| v["limit"].as_u64())
                .map(|v| v as usize)
                .unwrap_or(crate::sandbox::DEFAULT_READ_LIMIT)
                .clamp(1, crate::sandbox::MAX_READ_LIMIT);
            format!("Read {path} · lines {offset}-{}", offset + limit as u64 - 1)
        }
        "write_file" => {
            let count = args_v
                .as_ref()
                .and_then(|v| v["content"].as_str())
                .map(|c| c.lines().count())
                .unwrap_or(0);
            format!("Write {path} · {count} lines")
        }
        "edit_file" => {
            let added = args_v
                .as_ref()
                .and_then(|v| v["new_string"].as_str())
                .map(|s| s.lines().count())
                .unwrap_or(0);
            let removed = args_v
                .as_ref()
                .and_then(|v| v["old_string"].as_str())
                .map(|s| s.lines().count())
                .unwrap_or(0);
            format!("Edit {path} · +{added} -{removed}")
        }
        "list_files" => format!("List {}", if path == "?" { ".".into() } else { path }),
        "bash" => {
            let cmd = args_v
                .as_ref()
                .and_then(|v| v["command"].as_str())
                .unwrap_or("?");
            format!("$ {}", theme::clamp_text(&cmd.replace('\n', " "), 120))
        }
        _ => name.to_string(),
    };
    let header_style = Style::new().fg(Color::Yellow).bold();
    vec![Line::from(vec![
        Span::styled("🔧 ", header_style),
        Span::styled(
            theme::clamp_text(&summary, width.saturating_sub(18).max(20)),
            header_style,
        ),
        Span::styled(format!(" [{}]", short_id(id)), theme::dim()),
    ])]
}

/// First 12 characters of a tool-call id (char-safe, unlike a byte slice).
fn short_id(id: &str) -> String {
    id.chars().take(12).collect()
}

/// Compact result row. Successful reads render nothing at all — the call
/// row already said what was read, and the content is for the model's
/// history, not the user's screen. Everything else shows its first line
/// (write/edit results are one-line stats); errors keep a few lines so
/// they stay diagnosable.
fn tool_result_lines(
    id: &str,
    content: &str,
    is_error: bool,
    width: usize,
    call_name: Option<&str>,
) -> Vec<Line<'static>> {
    if !is_error && call_name == Some("read_file") {
        return Vec::new();
    }
    let (icon, style) = if is_error {
        ("❌ ", Style::new().fg(theme::ERROR_COLOR))
    } else {
        ("✓ ", Style::new().fg(Color::Green))
    };

    let total = content.lines().count();
    let first = content.lines().next().unwrap_or("(no output)");
    let mut lines = vec![Line::from(vec![
        Span::styled(icon, style),
        Span::styled(
            theme::clamp_text(first, width.saturating_sub(18).max(20)),
            style,
        ),
        Span::styled(format!(" [{}]", short_id(id)), theme::dim()),
    ])];

    // Errors stay a bit more verbose so they remain diagnosable.
    let keep: usize = if is_error { 4 } else { 1 };
    for line in content.lines().skip(1).take(keep.saturating_sub(1)) {
        lines.extend(theme::wrap_styled(
            vec![Span::styled(theme::clamp_text(line, 500), theme::dim())],
            width,
            Span::raw("  "),
            Span::raw("  "),
        ));
    }
    if total > keep {
        lines.push(Line::from(Span::styled(
            format!("  … ({} more lines)", total - keep),
            theme::dim(),
        )));
    }
    lines
}

fn cell_lines(cell: &Cell, width: usize, app: &App) -> Vec<Line<'static>> {
    match cell {
        Cell::User(text) => theme::wrap_styled(
            vec![Span::raw(theme::clamp_text(text, 2000))],
            width,
            theme::user_prefix(),
            Span::raw("  "),
        ),
        Cell::Assistant(text) => assistant_lines(text, width, app),
        Cell::Error(text) => {
            let style = Style::new().fg(theme::ERROR_COLOR);
            theme::wrap_styled(
                vec![Span::styled(
                    theme::clamp_text(text, 600),
                    style,
                )],
                width,
                Span::styled("⚠ ", style),
                Span::styled("  ", style),
            )
        }
        Cell::Notice(text) => theme::wrap_styled(
            vec![Span::raw(theme::clamp_text(text, 400))],
            width,
            Span::styled("• ", Style::new().fg(theme::ACCENT)),
            Span::raw("  "),
        ),
        Cell::ToolCall { name, args, id } => tool_call_lines(name, args, id, width),
        Cell::ToolResult { id, content, is_error } => {
            tool_result_lines(id, content, *is_error, width, None)
        }
    }
}
