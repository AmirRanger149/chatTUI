//! The bottom pane: a borderless `› ` composer that grows with the text in
//! it, the slash-command popup anchored above it, and the hint/context footer
//! row. The composer tracks a real editing cursor, so typing, backspace and
//! cursor/word navigation all happen where the cursor is.

use crate::app::{App, MAX_COMPOSER_ROWS};
use crate::ui::theme;
use ratatui::prelude::*;
use ratatui::widgets::Paragraph;
use unicode_width::UnicodeWidthChar;

/// Display width of the `› ` / continuation prefix in front of every row.
const PREFIX_W: usize = 2;

/// The composer's wrapped rows plus the cursor's position within them.
struct Layout {
    rows: Vec<String>,
    cursor_row: usize,
    /// Display column of the cursor, including the row prefix.
    cursor_col: usize,
}

/// Number of wrapped rows the composer needs at `width`.
pub fn row_count(app: &App, width: u16) -> u16 {
    let rows = layout(app, width.max(4) as usize).rows.len();
    rows.min(MAX_COMPOSER_ROWS).max(1) as u16
}

/// Wrap the composer text and locate the cursor inside the wrapped rows.
fn layout(app: &App, width: usize) -> Layout {
    let limit = width.saturating_sub(PREFIX_W).max(1);
    let cursor_char = app.composer[..app.cursor.min(app.composer.len())]
        .chars()
        .count();

    let mut rows: Vec<String> = Vec::new();
    let mut cursor_row = 0usize;
    let mut cursor_col = PREFIX_W;
    let mut consumed = 0usize; // chars consumed so far, incl. '\n' separators
    let mut found = false;

    for segment in app.composer.split('\n') {
        let seg_chars = segment.chars().count();
        let base = rows.len();
        let wrapped = wrap_rows(segment, limit);
        if !found && cursor_char >= consumed && cursor_char <= consumed + seg_chars {
            let local = cursor_char - consumed;
            let mut acc = 0usize;
            for (index, row) in wrapped.iter().enumerate() {
                let row_chars = row.chars().count();
                if local <= acc + row_chars {
                    let in_row = local - acc;
                    let byte = row
                        .char_indices()
                        .nth(in_row)
                        .map(|(i, _)| i)
                        .unwrap_or(row.len());
                    cursor_row = base + index;
                    cursor_col = PREFIX_W + theme::cell_width(&row[..byte]);
                    found = true;
                    break;
                }
                acc += row_chars;
            }
            if !found {
                // Cursor sits past the last wrapped row: pin to its end.
                cursor_row = base + wrapped.len().saturating_sub(1);
                let tail = wrapped.last().map(String::as_str).unwrap_or("");
                cursor_col = PREFIX_W + theme::cell_width(tail);
                found = true;
            }
        }
        rows.extend(wrapped);
        consumed += seg_chars + 1;
    }

    Layout {
        rows,
        cursor_row,
        cursor_col,
    }
}

/// Greedy word wrap whose rows concatenate back to the original text, so
/// cursor offsets map exactly onto (row, column) positions.
fn wrap_rows(text: &str, limit: usize) -> Vec<String> {
    let mut rows: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut used = 0usize;

    let mut chars = text.chars().peekable();
    loop {
        // One token: leading whitespace plus the word that follows it.
        let mut token = String::new();
        while matches!(chars.peek(), Some(ch) if ch.is_whitespace()) {
            token.push(chars.next().unwrap());
        }
        while matches!(chars.peek(), Some(ch) if !ch.is_whitespace()) {
            token.push(chars.next().unwrap());
        }
        if token.is_empty() {
            break;
        }
        let token_w: usize = token.chars().filter_map(|ch| ch.width()).sum();
        if used > 0 && used + token_w > limit {
            rows.push(std::mem::take(&mut current));
            used = 0;
        }
        // A token wider than a whole row is hard-split.
        for ch in token.chars() {
            let ch_w = ch.width().unwrap_or(0);
            if used > 0 && used + ch_w > limit {
                rows.push(std::mem::take(&mut current));
                used = 0;
            }
            current.push(ch);
            used += ch_w;
        }
    }
    rows.push(current);
    rows
}

pub fn render_prompt(frame: &mut Frame, area: Rect, app: &App) -> Option<Position> {
    if area.width == 0 || area.height == 0 {
        return None;
    }
    if app.composer.is_empty() {
        frame.render_widget(
            Paragraph::new(Line::from(vec![
                theme::user_prefix(),
                Span::styled(crate::app::PLACEHOLDER.to_string(), theme::dim()),
            ])),
            Rect {
                height: 1,
                ..area
            },
        );
        return Some(Position::new(area.x + PREFIX_W as u16, area.y));
    }

    let layout = layout(app, area.width as usize);
    let total = layout.rows.len();
    let window = total.min(MAX_COMPOSER_ROWS).min(area.height as usize);
    if window == 0 {
        return None;
    }
    // Pin to the bottom by default, but always keep the cursor row visible.
    let mut start = total.saturating_sub(window);
    if layout.cursor_row < start {
        start = layout.cursor_row;
    } else if layout.cursor_row >= start + window {
        start = layout.cursor_row + 1 - window;
    }

    for (offset, row) in layout.rows[start..start + window].iter().enumerate() {
        let prefix = if start + offset == 0 {
            theme::user_prefix()
        } else {
            Span::raw("  ")
        };
        frame.render_widget(
            Paragraph::new(Line::from(vec![prefix, Span::raw(row.clone())])),
            Rect {
                x: area.x,
                y: area.y + offset as u16,
                width: area.width,
                height: 1,
            },
        );
    }

    let row_in_window = layout.cursor_row.saturating_sub(start) as u16;
    let col = layout
        .cursor_col
        .min(area.width.saturating_sub(1) as usize) as u16;
    Some(Position::new(area.x + col, area.y + row_in_window))
}

pub fn render_footer(frame: &mut Frame, area: Rect, app: &App) {
    if area.width == 0 || area.height == 0 {
        return;
    }
    let left: Vec<Span<'static>> = if app.quit_primed() {
        vec![Span::styled(
            "press ctrl+c again to quit",
            Style::new().fg(Color::Yellow),
        )]
    } else if app.streaming {
        Vec::new()
    } else if app.composer.is_empty() && app.overlay.is_none() {
        vec![Span::styled("? for shortcuts", theme::dim())]
    } else {
        Vec::new()
    };
    let right = vec![Span::styled(app.context_summary(), theme::dim())];
    let right_w = theme::spans_width(&right);
    let left_w = theme::spans_width(&left);
    let gap = 2;
    let left_fits = left_w + gap + right_w <= area.width as usize;

    if left_fits && !left.is_empty() {
        frame.render_widget(
            Paragraph::new(Line::from(left)),
            Rect {
                width: area.width - right_w as u16 - gap as u16,
                ..area
            },
        );
    }
    if right_w < area.width as usize {
        frame.render_widget(
            Paragraph::new(Line::from(right)),
            Rect {
                x: area.right() - right_w as u16,
                width: right_w as u16,
                ..area
            },
        );
    }
}

/// Popup rows for the slash-command palette (already bordered), or empty.
pub fn slash_popup_lines(app: &App, width: usize) -> Vec<Line<'static>> {
    let items = app.slash_filtered();
    if items.is_empty() {
        return vec![];
    }
    let selected = app.slash_selected.min(items.len() - 1);
    let max_rows = 6usize;
    let start = selected.saturating_sub(max_rows - 1);
    let end = (start + max_rows).min(items.len());
    let shown = &items[start..end];

    let name_w = shown.iter().map(|c| theme::cell_width(c.name)).max().unwrap_or(0);
    let desc_limit = width.saturating_sub(name_w + 6).max(8);

    let mut rows: Vec<(String, bool)> = shown
        .iter()
        .enumerate()
        .map(|(index, item)| {
            let desc: String = item.desc.chars().take(desc_limit).collect();
            (
                format!("{:<name_w$}  {}", item.name, desc),
                start + index == selected,
            )
        })
        .collect();
    // Equalize widths so the highlight bar spans the full popup.
    let row_w = rows
        .iter()
        .map(|(text, _)| theme::cell_width(text))
        .max()
        .unwrap_or(0)
        .min(width.saturating_sub(6));
    for (text, _) in &mut rows {
        let pad = row_w.saturating_sub(theme::cell_width(text));
        text.push_str(&" ".repeat(pad));
    }
    let lines: Vec<Line<'static>> = rows
        .into_iter()
        .map(|(text, is_selected)| {
            Line::from(vec![
                Span::raw("  "),
                if is_selected {
                    Span::styled(text, Style::new().bg(theme::SELECT_BG))
                } else {
                    Span::raw(text)
                },
            ])
        })
        .collect();
    theme::with_border(lines)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::session::manager::SessionManager;

    #[test]
    fn cursor_column_counts_persian_like_the_terminal() {
        let mut app = App::new(Config::default(), SessionManager::for_tests());
        app.composer = "سلام".into();
        app.cursor = app.composer.len();
        // 4 cells for سلام, not 3: whole-string width() would collapse the
        // Lam-Alef ligature and park the cursor one cell too far left.
        assert_eq!(layout(&app, 80).cursor_col, PREFIX_W + 4);
    }
}
