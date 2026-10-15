//! Full-screen overlays: the keyboard-shortcuts popup (`?` / `/help`) and the
//! conversation-history picker (`ctrl+h` / `/history`). Both are dim rounded
//! cards centered over the interface — as is the tool-approval prompt.

use crate::api::types::ToolCall;
use crate::app::{App, Overlay};
use crate::ui::theme;
use ratatui::prelude::*;
use ratatui::widgets::{Clear, Paragraph};
use unicode_segmentation::UnicodeSegmentation;

pub fn render(frame: &mut Frame, area: Rect, app: &App, overlay: Overlay) {
    match overlay {
        Overlay::Shortcuts => shortcuts(frame, area),
        Overlay::History { selected } => history(frame, area, app, selected),
        Overlay::Code { selected } => code(frame, area, app, selected),
        Overlay::Models { selected } => models(frame, area, app, selected),
        Overlay::Providers { selected } => providers(frame, area, app, selected),
        Overlay::ToolDetail { selected } => tool_detail(frame, area, app, selected),
    }
}

fn shortcuts(frame: &mut Frame, area: Rect) {
    let rows: [(&str, &str); 11] = [
        ("enter", "send message"),
        ("esc", "close popup · interrupt stream"),
        ("ctrl+h", "conversation history"),
        ("ctrl+t", "inspect tool activity"),
        ("ctrl+g", "copy code blocks"),
        ("ctrl+r", "show / hide model reasoning"),
        ("pgup / pgdn", "scroll transcript"),
        ("up / down", "prompt history"),
        ("← / →", "move cursor · ctrl jumps words"),
        ("shift+enter", "newline in composer"),
        ("ctrl+c ×2", "quit"),
    ];
    let key_w = rows.iter().map(|(key, _)| key.len()).max().unwrap_or(0);

    let mut lines = vec![Line::from(Span::styled(
        "Keyboard shortcuts",
        Style::new().bold(),
    ))];
    lines.push(Line::from(""));
    for (key, desc) in rows {
        lines.push(Line::from(vec![
            Span::raw(format!("{key:<key_w$}  ")),
            Span::styled(desc, theme::dim()),
        ]));
    }
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        "Slash commands",
        Style::new().bold(),
    )));
    lines.push(Line::from(""));
    for command in crate::app::SLASH_COMMANDS {
        lines.push(Line::from(vec![
            Span::styled(
                format!("{:<key_w$}  ", command.name),
                Style::new().fg(theme::ACCENT),
            ),
            Span::styled(command.desc, theme::dim()),
        ]));
    }
    render_card(frame, area, lines);
}

/// The most wrapped rows of a shell command the approval card shows.
const APPROVAL_COMMAND_ROWS: usize = 16;

/// The tool-approval prompt: what the agent wants to do, so the user can
/// answer `y` / `n`. Shown while the agent is paused in manual approval (see
/// `/approve`); `ctrl+t` opens the diff or command in the inspector.
pub fn approval(frame: &mut Frame, area: Rect, call: &ToolCall) {
    let mut lines = vec![
        Line::from(vec![
            Span::styled("Approve this tool call?", Style::new().bold()),
            Span::styled(
                "   y approve · n deny · ctrl+t details · esc interrupt",
                theme::dim(),
            ),
        ]),
        Line::from(""),
    ];
    // The border takes four columns. Vertically the card spends six rows on
    // everything but the command (border, title, heading, "more rows" note),
    // so a command that does not fit is cut with a note instead of being
    // silently clipped by the screen edge.
    let width = (area.width as usize).saturating_sub(4).clamp(20, 100);
    let rows = (area.height as usize)
        .saturating_sub(6)
        .clamp(1, APPROVAL_COMMAND_ROWS);
    lines.extend(approval_lines(call, width, rows));
    render_card(frame, area, lines);
}

/// The body of the approval card. A shell command is shown wrapped, up to
/// `max_rows` rows and with a note when more follow, because it is exactly
/// what the user is being asked to allow; file changes show the path and size
/// (the diff is behind `ctrl+t`).
fn approval_lines(call: &ToolCall, width: usize, max_rows: usize) -> Vec<Line<'static>> {
    let args = serde_json::from_str::<serde_json::Value>(&call.arguments).unwrap_or_default();
    let text = |key: &str| args[key].as_str().unwrap_or("?").to_string();
    let count = |key: &str| args[key].as_str().map(|s| s.lines().count()).unwrap_or(0);
    let header_style = Style::new().fg(Color::Yellow).bold();
    let header = |title: String| Line::from(Span::styled(format!("🔧 {title}"), header_style));
    match call.name.as_str() {
        "bash" => {
            // Wrap row by row: the wrapper itself would flatten line breaks.
            let mut rows: Vec<Line<'static>> = Vec::new();
            for row in text("command").lines() {
                rows.extend(theme::wrap_styled(
                    vec![Span::raw(row.to_string())],
                    width,
                    Span::styled("$ ", Style::new().fg(theme::ACCENT).bold()),
                    Span::raw("  "),
                ));
            }
            if rows.len() > max_rows {
                let hidden = rows.len() - max_rows;
                rows.truncate(max_rows);
                rows.push(Line::styled(
                    format!("… {hidden} more rows not shown"),
                    theme::dim(),
                ));
            }
            let mut lines = vec![header("Run shell command".to_string())];
            lines.extend(rows);
            lines
        }
        "write_file" => vec![header(format!(
            "Write {} · {} lines",
            text("path"),
            count("content")
        ))],
        "edit_file" => vec![header(format!(
            "Edit {} · +{} -{}",
            text("path"),
            count("new_string"),
            count("old_string")
        ))],
        other => vec![header(other.to_string())],
    }
}

/// The `ctrl+t` tool inspector: what a write/edit/list/bash call actually
/// did — added/removed lines for writes and edits, the full command and its
/// output for bash. Same toggle mechanic as the reasoning blocks: the key
/// opens it, the same key (or `esc`) closes it.
fn tool_detail(frame: &mut Frame, area: Rect, app: &App, selected: usize) {
    let items = app.tool_details();
    let mut lines = vec![
        Line::from(vec![
            Span::styled("Tool activity", Style::new().bold()),
            Span::styled("   ↑↓ switch · ctrl+t / esc close", theme::dim()),
        ]),
        Line::from(""),
    ];
    let index = selected.min(items.len().saturating_sub(1));
    let Some(item) = items.get(index) else {
        lines.push(Line::styled(
            "no tool activity to inspect yet",
            theme::dim(),
        ));
        render_card(frame, area, lines);
        return;
    };
    lines.push(Line::from(vec![
        Span::styled("🔧 ", Style::new().fg(Color::Yellow).bold()),
        Span::styled(theme::clamp_text(&item.title, 120), Style::new().bold()),
        Span::styled(format!("   ({}/{})", index + 1, items.len()), theme::dim()),
    ]));
    lines.push(Line::from(""));
    // Bounded view: enough lines to inspect, never enough to overflow.
    let max_detail = 28usize;
    for raw in item.lines.iter().take(max_detail) {
        let style = if raw.starts_with("+ ") {
            Style::new().fg(Color::Green)
        } else if raw.starts_with("- ") {
            Style::new().fg(theme::ERROR_COLOR)
        } else if raw.starts_with("$ ") {
            Style::new().fg(theme::ACCENT).bold()
        } else {
            Style::default()
        };
        lines.push(Line::from(Span::styled(theme::clamp_text(raw, 500), style)));
    }
    if item.lines.len() > max_detail {
        lines.push(Line::from(Span::styled(
            format!("… {} more lines", item.lines.len() - max_detail),
            theme::dim(),
        )));
    }
    render_card(frame, area, lines);
}

fn history(frame: &mut Frame, area: Rect, app: &App, selected: usize) {
    let sessions = app.sessions.sessions();
    let mut lines = vec![
        Line::from(vec![
            Span::styled("Conversation history", Style::new().bold()),
            Span::styled(
                "   ↑↓ select · enter open · d delete · esc close",
                theme::dim(),
            ),
        ]),
        Line::from(""),
    ];
    if sessions.is_empty() {
        lines.push(Line::styled("no saved conversations", theme::dim()));
        render_card(frame, area, lines);
        return;
    }
    let max_rows = crate::app::OVERLAY_ROWS;
    let start = selected.saturating_sub(max_rows - 1);
    let end = (start + max_rows).min(sessions.len());
    for (index, session) in sessions[start..end].iter().enumerate() {
        let index = start + index;
        let is_selected = index == selected;
        let marker = if is_selected { "> " } else { "  " };
        let label = format!("{marker}{}", session.title);
        let meta = format!("  · {} msgs", session.messages.len());
        let line = if is_selected {
            Line::from(vec![
                Span::styled(label, Style::new().bold().bg(theme::SELECT_BG)),
                Span::styled(meta, theme::dim().bg(theme::SELECT_BG)),
            ])
        } else {
            Line::from(vec![Span::raw(label), Span::styled(meta, theme::dim())])
        };
        lines.push(line);
    }
    render_card(frame, area, lines);
}

fn code(frame: &mut Frame, area: Rect, app: &App, selected: usize) {
    let blocks = app.code_blocks();
    let mut lines = vec![
        Line::from(vec![
            Span::styled("Code blocks", Style::new().bold()),
            Span::styled("   ↑↓ select · enter copy · esc close", theme::dim()),
        ]),
        Line::from(""),
    ];
    if blocks.is_empty() {
        lines.push(Line::styled(
            "no code blocks in this conversation",
            theme::dim(),
        ));
        render_card(frame, area, lines);
        return;
    }
    let max_rows = crate::app::OVERLAY_ROWS;
    let start = selected.saturating_sub(max_rows - 1);
    let end = (start + max_rows).min(blocks.len());
    for (index, block) in blocks[start..end].iter().enumerate() {
        let index = start + index;
        let is_selected = index == selected;
        let marker = if is_selected { "> " } else { "  " };
        let lang = if block.lang.is_empty() {
            "code"
        } else {
            block.lang.as_str()
        };
        let line_count = block.code.lines().count();
        let preview = preview_text(&block.code);
        let label = format!(
            "{marker}{:>2} · {lang} · {line_count} lines · {preview}",
            index + 1
        );
        let line = if is_selected {
            Line::from(Span::styled(
                label,
                Style::new().bold().bg(theme::SELECT_BG),
            ))
        } else {
            Line::from(label)
        };
        lines.push(line);
    }
    render_card(frame, area, lines);
}

fn models(frame: &mut Frame, area: Rect, app: &App, selected: usize) {
    let mut lines = vec![Line::from(vec![
        Span::styled("Models", Style::new().bold()),
        Span::styled(
            "   ↑↓ select · enter switch · r refresh · esc close",
            theme::dim(),
        ),
    ])];
    lines.push(Line::from(""));

    if app.models.ids.is_empty() {
        if app.models.loading {
            lines.push(Line::styled("loading models from the API…", theme::dim()));
        } else if let Some(error) = &app.models.error {
            lines.push(Line::from(Span::styled(
                format!("couldn't load models: {error}"),
                Style::new().fg(Color::Red),
            )));
            lines.push(Line::styled("press r to retry", theme::dim()));
        } else {
            lines.push(Line::styled(
                "no models available — press r to fetch",
                theme::dim(),
            ));
        }
        render_card(frame, area, lines);
        return;
    }

    let selected = selected.min(app.models.ids.len() - 1);
    let max_rows = crate::app::OVERLAY_ROWS;
    let start = selected.saturating_sub(max_rows - 1);
    let end = (start + max_rows).min(app.models.ids.len());
    for (offset, id) in app.models.ids[start..end].iter().enumerate() {
        let index = start + offset;
        let is_selected = index == selected;
        let marker = if is_selected { "> " } else { "  " };
        let label = format!("{marker}{id}");
        let meta = if id == &app.config.model {
            "  · active"
        } else {
            ""
        };
        let line = if is_selected {
            Line::from(vec![
                Span::styled(label, Style::new().bold().bg(theme::SELECT_BG)),
                Span::styled(meta, theme::dim().bg(theme::SELECT_BG)),
            ])
        } else {
            Line::from(vec![Span::raw(label), Span::styled(meta, theme::dim())])
        };
        lines.push(line);
    }
    if app.models.loading {
        lines.push(Line::styled("refreshing…", theme::dim()));
    } else if let Some(error) = &app.models.error {
        lines.push(Line::from(Span::styled(
            format!("refresh failed: {error}"),
            Style::new().fg(Color::Red),
        )));
    }
    render_card(frame, area, lines);
}

fn providers(frame: &mut Frame, area: Rect, app: &App, selected: usize) {
    let mut lines = vec![Line::from(vec![
        Span::styled("API Providers", Style::new().bold()),
        Span::styled("   ↑↓ select · enter switch · esc close", theme::dim()),
    ])];
    lines.push(Line::from(""));

    let registry = app.config.providers();
    let selected = selected.min(registry.len().saturating_sub(1));
    for (index, provider) in registry.iter().enumerate() {
        let is_selected = index == selected;
        let is_active = provider.id == app.config.provider;
        let marker = if is_selected { "> " } else { "  " };
        let has_key = app.config.api_key_for_provider(&provider.id).is_some();
        let key_status = if has_key { "key configured" } else { "no key" };

        let custom_tag = if provider.is_custom {
            "  · custom"
        } else {
            ""
        };
        let label = format!(
            "{marker}{:<13} · {}{}",
            provider.name, provider.base_url, custom_tag
        );
        let meta = if is_active {
            format!("  · active ({key_status})")
        } else {
            format!("  · ({key_status})")
        };
        let line = if is_selected {
            Line::from(vec![
                Span::styled(label, Style::new().bold().bg(theme::SELECT_BG)),
                Span::styled(meta, theme::dim().bg(theme::SELECT_BG)),
            ])
        } else {
            Line::from(vec![Span::raw(label), Span::styled(meta, theme::dim())])
        };
        lines.push(line);
    }
    lines.push(Line::from(""));
    lines.push(Line::styled(
        "add your own OpenAI-compatible endpoint under \"custom_providers\" in config.json",
        theme::dim(),
    ));
    render_card(frame, area, lines);
}

/// Maximum preview width in the code list, in terminal cells.
const PREVIEW_WIDTH: usize = 48;

/// First line of a code block, sanitized for single-line display in the code
/// list: tabs become spaces (a raw tab would jump to the terminal's next tab
/// stop and blow past the card edge), other control characters become `�`,
/// and the line is capped at [`PREVIEW_WIDTH`] cells without splitting
/// graphemes. Persian/Arabic text, ZWNJ included, passes through untouched;
/// only layout-breaking characters are replaced.
fn preview_text(code: &str) -> String {
    let line = code.lines().next().unwrap_or("");
    let mut out = String::new();
    let mut used = 0usize;
    for grapheme in line.graphemes(true) {
        let shown = match grapheme {
            "\t" => "  ",
            g if g.chars().all(|ch| !ch.is_control()) => g,
            _ => "�",
        };
        let width = theme::cell_width(shown);
        if used + width > PREVIEW_WIDTH {
            break;
        }
        out.push_str(shown);
        used += width;
    }
    out
}

/// Center a dim rounded card around `lines` and render it.
fn render_card(frame: &mut Frame, area: Rect, lines: Vec<Line<'static>>) {
    let bordered = theme::with_border(lines);
    let width = bordered
        .iter()
        .map(theme::line_width)
        .max()
        .unwrap_or(0)
        .min(area.width as usize);
    let height = bordered.len().min(area.height as usize);
    if width == 0 || height == 0 {
        return;
    }
    let x = area.x + (area.width.saturating_sub(width as u16)) / 2;
    let y = area.y + (area.height.saturating_sub(height as u16)) / 2;
    let rect = Rect {
        x,
        y,
        width: width as u16,
        height: height as u16,
    };
    // Erase the card's rows across the full screen width first. Without this,
    // content drawn behind the popup (shaded code blocks in the transcript are
    // the worst offender) bleeds into the card and the two layers fight each
    // other. Clearing whole rows instead of just the card rect also keeps
    // bidirectional text away from the card: if RTL transcript text stayed on
    // the same rows, a bidi-capable terminal would reorder the card along with
    // the background and the two would visibly interfere.
    let scrim = Rect {
        x: area.x,
        y: rect.y,
        width: area.width,
        height: rect.height,
    };
    frame.render_widget(Clear, scrim);
    frame.render_widget(Paragraph::new(bordered), rect);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::Cell;
    use crate::config::Config;
    use crate::session::manager::SessionManager;
    use ratatui::backend::TestBackend;

    #[test]
    fn models_card_renders_in_every_state() {
        let mut app = App::new(Config::default(), SessionManager::for_tests());
        app.config.model = "b/2".into();
        let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();

        // loading
        app.overlay = Some(Overlay::Models { selected: 0 });
        app.models.loading = true;
        terminal
            .draw(|frame| crate::ui::render(frame, &app))
            .unwrap();

        // loaded, with the active model marked
        app.models.loading = false;
        app.models.ids = vec!["a/1".into(), "b/2".into()];
        app.overlay = Some(Overlay::Models { selected: 1 });
        terminal
            .draw(|frame| crate::ui::render(frame, &app))
            .unwrap();

        // error state
        app.models.ids.clear();
        app.models.error = Some("HTTP 401".into());
        app.overlay = Some(Overlay::Models { selected: 0 });
        terminal
            .draw(|frame| crate::ui::render(frame, &app))
            .unwrap();

        // providers overlay
        app.overlay = Some(Overlay::Providers { selected: 0 });
        terminal
            .draw(|frame| crate::ui::render(frame, &app))
            .unwrap();
        app.overlay = Some(Overlay::Providers { selected: 1 });
        terminal
            .draw(|frame| crate::ui::render(frame, &app))
            .unwrap();
    }

    /// The text of each row, without styling.
    fn plain(lines: Vec<Line<'static>>) -> Vec<String> {
        lines
            .iter()
            .map(|line| {
                line.spans
                    .iter()
                    .map(|span| span.content.as_ref())
                    .collect::<String>()
            })
            .collect()
    }

    #[test]
    fn approval_card_shows_the_command_wrapped_with_its_line_breaks() {
        let call = ToolCall::new(
            "c1",
            "bash",
            r#"{"command":"echo alpha beta gamma delta\nrm -rf build && echo finished"}"#,
        );
        let rows = plain(approval_lines(&call, 24, APPROVAL_COMMAND_ROWS));
        let shown = rows.join("\n");
        // Every word of the command is on screen: nothing is cut off.
        for word in ["echo", "alpha", "delta", "rm", "-rf", "build", "finished"] {
            assert!(shown.contains(word), "{word} missing from:\n{shown}");
        }
        // A line break in the command stays a line break.
        assert!(rows.iter().any(|row| row.starts_with("$ rm")), "{shown}");
        assert!(
            rows.iter().all(|row| theme::cell_width(row) <= 24),
            "{shown}"
        );
    }

    #[test]
    fn approval_card_says_so_when_a_long_command_is_cut_short() {
        let command = (1..=30)
            .map(|n| format!("echo line{n}"))
            .collect::<Vec<_>>()
            .join("\n");
        let args = serde_json::json!({ "command": command }).to_string();
        let call = ToolCall::new("c1", "bash", args);

        let shown = plain(approval_lines(&call, 40, 5)).join("\n");
        assert!(shown.contains("echo line5"), "{shown}");
        assert!(!shown.contains("echo line6"), "{shown}");
        assert!(shown.contains("25 more rows not shown"), "{shown}");

        // Room for everything: no note.
        let shown = plain(approval_lines(&call, 40, 40)).join("\n");
        assert!(
            shown.contains("echo line30") && !shown.contains("more rows"),
            "{shown}"
        );
    }

    #[test]
    fn approval_card_summarizes_file_changes_and_renders_everywhere() {
        let write = ToolCall::new("c1", "write_file", r#"{"path":"a.txt","content":"x\ny"}"#);
        let text = plain(approval_lines(&write, 40, 16)).join("\n");
        assert!(
            text.contains("Write a.txt") && text.contains("2 lines"),
            "{text}"
        );
        let edit = ToolCall::new(
            "c1",
            "edit_file",
            r#"{"path":"a.txt","old_string":"a","new_string":"b\nc"}"#,
        );
        let text = plain(approval_lines(&edit, 40, 16)).join("\n");
        assert!(
            text.contains("Edit a.txt") && text.contains("+2 -1"),
            "{text}"
        );

        let draw = |terminal: &mut Terminal<TestBackend>, call: &ToolCall| {
            terminal
                .draw(|frame| {
                    let area = frame.area();
                    approval(frame, area, call);
                })
                .unwrap();
        };
        let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
        for (name, args) in [
            ("bash", r#"{"command":"cargo test"}"#),
            ("write_file", r#"{"path":"a.txt","content":"x\ny"}"#),
            (
                "edit_file",
                r#"{"path":"a.txt","old_string":"a","new_string":"b\nc"}"#,
            ),
            ("mystery", "{}"),
            ("bash", "not json"),
        ] {
            draw(&mut terminal, &ToolCall::new("c1", name, args));
        }
        // Tiny terminals must not panic either.
        let long = serde_json::json!({ "command": "echo hello world ".repeat(40) }).to_string();
        for (width, height) in [(10, 3), (30, 8), (200, 60)] {
            let mut tiny = Terminal::new(TestBackend::new(width, height)).unwrap();
            draw(&mut tiny, &ToolCall::new("c1", "bash", long.clone()));
        }
    }

    #[test]
    fn preview_sanitizes_layout_breaking_characters() {
        assert_eq!(preview_text("\tindented"), "  indented");
        assert_eq!(preview_text("ok\x07no"), "ok�no");
        // Persian passes through untouched, including ZWNJ.
        assert_eq!(preview_text("می‌شود سلام"), "می‌شود سلام");
        assert_eq!(preview_text(""), "");
        // Capped by cells, not chars.
        assert!(theme::cell_width(&preview_text(&"x".repeat(100))) <= PREVIEW_WIDTH);
        assert_eq!(preview_text(&"x".repeat(100)).len(), PREVIEW_WIDTH);
    }

    #[test]
    fn code_card_stays_aligned_with_persian_content() {
        let mut app = App::new(Config::default(), SessionManager::for_tests());
        app.cells.push(Cell::Assistant(
            concat!(
                "این یک پاسخ فارسی است\n",
                "```python\n",
                "print(\"سلام دنیا\")\n",
                "# توضیح فارسی\n",
                "```\n",
                "متن پایانی",
            )
            .into(),
        ));
        app.open_code();
        assert!(matches!(
            app.overlay,
            Some(crate::app::Overlay::Code { .. })
        ));

        let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
        terminal
            .draw(|frame| crate::ui::render(frame, &app))
            .unwrap();
        let buffer = terminal.backend().buffer();

        // The overlay card is the only centered (x > 0) bordered box: the
        // transcript's own boxes hug the left edge.
        let mut top = None;
        for y in 0..24u16 {
            for x in 0..80u16 {
                if buffer[(x, y)].symbol() == "╭" && x > 0 {
                    top = Some((x, y));
                }
            }
        }
        let (x0, y0) = top.expect("code card top border should be visible");
        let x1 = (x0..80)
            .find(|x| buffer[(*x, y0)].symbol() == "╮")
            .expect("code card top border should close");
        let y1 = (y0 + 1..24)
            .find(|y| buffer[(x0, *y)].symbol() == "╰")
            .expect("code card bottom border should be visible");
        assert_eq!(buffer[(x1, y1)].symbol(), "╯");

        // Every card row spans exactly the same columns with intact side
        // borders, and no background text leaks onto the card's rows (the
        // scrim keeps bidi-capable terminals from reordering the card with
        // the Persian transcript behind it).
        for y in y0 + 1..y1 {
            assert_eq!(buffer[(x0, y)].symbol(), "│", "left border at row {y}");
            assert_eq!(buffer[(x1, y)].symbol(), "│", "right border at row {y}");
            for x in 0..x0 {
                assert_eq!(buffer[(x, y)].symbol(), " ", "scrim gap at ({x}, {y})");
            }
            for x in x1 + 1..80 {
                assert_eq!(buffer[(x, y)].symbol(), " ", "scrim gap at ({x}, {y})");
            }
        }
    }
}
