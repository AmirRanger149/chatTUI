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
    vec![
        Line::from(title),
        Line::from(""),
        Line::from(provider_line),
        Line::from(model_line),
        Line::from(dir_line),
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
    }
}
