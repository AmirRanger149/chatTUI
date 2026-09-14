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

    for cell in &app.cells {
        lines.extend(cell_lines(cell, width, app));
        lines.push(Line::from(""));
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

fn tool_call_lines(name: &str, args: &str, id: &str, width: usize) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    // Header: 🔧 tool_call
    let header_style = Style::new().fg(Color::Yellow).bold();
    lines.push(Line::from(vec![
        Span::styled("🔧 ", header_style),
        Span::styled(format!("{} ", name), header_style),
        Span::styled(format!("[{}]", &id[..id.len().min(12)]), theme::dim()),
    ]));
    
    // Args - try to pretty print JSON
    let pretty_args = if let Ok(v) = serde_json::from_str::<serde_json::Value>(args) {
        serde_json::to_string_pretty(&v).unwrap_or_else(|_| args.to_string())
    } else {
        args.to_string()
    };
    
    for line in pretty_args.lines().take(10) {
        lines.extend(theme::wrap_styled(
            vec![Span::styled(theme::clamp_text(line, 500), Style::new().fg(Color::DarkGray))],
            width,
            Span::raw("  "),
            Span::raw("  "),
        ));
    }
    if pretty_args.lines().count() > 10 {
        lines.push(Line::from(Span::styled("  ... (truncated)", theme::dim())));
    }
    
    lines
}

fn tool_result_lines(id: &str, content: &str, is_error: bool, width: usize) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    let (icon, style) = if is_error {
        ("❌ ", Style::new().fg(theme::ERROR_COLOR))
    } else {
        ("✓ ", Style::new().fg(Color::Green))
    };
    
    lines.push(Line::from(vec![
        Span::styled(icon, style),
        Span::styled("tool result ", Style::new().fg(Color::DarkGray)),
        Span::styled(format!("[{}]", &id[..id.len().min(12)]), theme::dim()),
    ]));
    
    // Content - clamp and wrap
    let clamped = theme::clamp_text(content, 2000);
    for line in clamped.lines().take(15) {
        lines.extend(theme::wrap_styled(
            vec![Span::raw(line.to_string())],
            width,
            Span::raw("  "),
            Span::raw("  "),
        ));
    }
    if clamped.lines().count() > 15 {
        lines.push(Line::from(Span::styled(
            format!("  ... ({} more lines)", clamped.lines().count() - 15),
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
        Cell::ToolResult { id, content, is_error } => tool_result_lines(id, content, *is_error, width),
    }
}
