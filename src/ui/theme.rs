//! Small visual helpers the rest of the UI leans on: dim rounded cards, word
//! wrap with a hanging indent, spinner glyphs, and a bit of formatting.

use ratatui::prelude::*;
use unicode_segmentation::UnicodeSegmentation;
use unicode_width::UnicodeWidthStr;

pub const ACCENT: Color = Color::Cyan;
pub const ERROR_COLOR: Color = Color::Red;
pub const DIM: Color = Color::DarkGray;
pub const SELECT_BG: Color = Color::DarkGray;

/// Animated glyph cycle for the "Working" indicator.
pub const SPINNER: [&str; 6] = ["·", "✢", "✳", "∗", "✻", "✽"];

pub fn dim() -> Style {
    Style::new().fg(DIM)
}

/// The `› ` prompt prefix used for user turns and the composer.
pub fn user_prefix() -> Span<'static> {
    Span::styled("› ", Style::new().bold().dim())
}

/// Visible width of `text` in terminal cells, counted per grapheme the way
/// ratatui fills its buffer.
///
/// Don't reach for `UnicodeWidthStr::width` on the whole string when doing
/// layout maths. unicode-width 0.2 applies multi-character ligature rules,
/// and the Arabic Lam-Alef `لا` (which turns up constantly in Persian text)
/// gets collapsed to a single cell. Ratatui places one grapheme at a time and
/// terminals draw the ligature across two cells, so whole-string widths come
/// out too small: padded rows overflow their area and ratatui clips the
/// trailing border. The code overlay is where that bit us first.
pub fn cell_width(text: &str) -> usize {
    text.graphemes(true).map(UnicodeWidthStr::width).sum()
}

pub fn line_width(line: &Line<'_>) -> usize {
    line.spans.iter().map(|span| cell_width(&span.content)).sum()
}

pub fn spans_width(spans: &[Span<'_>]) -> usize {
    spans.iter().map(|span| cell_width(&span.content)).sum()
}

/// Wrap `lines` in a dim rounded border sized to fit the widest one.
pub fn with_border(lines: Vec<Line<'static>>) -> Vec<Line<'static>> {
    let inner = lines.iter().map(line_width).max().unwrap_or(0);
    let rule = "─".repeat(inner + 2);
    let mut out = vec![Line::styled(format!("╭{rule}╮"), dim())];
    for line in lines {
        let used = line_width(&line);
        let mut spans = Vec::with_capacity(line.spans.len() + 3);
        spans.push(Span::styled("│ ", dim()));
        spans.extend(line.spans);
        if used < inner {
            spans.push(Span::styled(" ".repeat(inner - used), dim()));
        }
        spans.push(Span::styled(" │", dim()));
        out.push(Line::from(spans));
    }
    out.push(Line::styled(format!("╰{rule}╯"), dim()));
    out
}

/// Greedy word wrap for pre-styled spans with a hanging indent: `first_prefix`
/// is placed on the first row, `cont_prefix` on every wrapped row.
pub fn wrap_styled(
    spans: Vec<Span<'static>>,
    width: usize,
    first_prefix: Span<'static>,
    cont_prefix: Span<'static>,
) -> Vec<Line<'static>> {
    let width = width.max(8);
    let words = tokenize(spans);
    let first_w = cell_width(&first_prefix.content);
    let cont_w = cell_width(&cont_prefix.content);

    let mut rows: Vec<Vec<Span<'static>>> = Vec::new();
    let mut current: Vec<Span<'static>> = Vec::new();
    let mut used = 0usize;
    let mut first_row = true;

    for (mut word, style) in words {
        loop {
            let prefix_w = if first_row { first_w } else { cont_w };
            let limit = width.saturating_sub(prefix_w);
            let word_w = cell_width(&word);
            if used + word_w <= limit {
                if word_w > 0 {
                    current.push(Span::styled(word, style));
                    used += word_w;
                }
                break;
            }
            if current.is_empty() {
                // A single word longer than the line: hard-split it.
                let mut take: String = word.chars().take(limit).collect();
                let rest: String = word.chars().skip(limit).collect();
                let rest_width = cell_width(&rest);
                take.push_str(&" ".repeat(limit.saturating_sub(word_w.min(limit))));
                current.push(Span::styled(take, style));
                rows.push(std::mem::take(&mut current));
                used = 0;
                first_row = false;
                word = rest;
                if rest_width == 0 {
                    break;
                }
            } else {
                rows.push(std::mem::take(&mut current));
                used = 0;
                first_row = false;
            }
        }
    }
    rows.push(current);

    rows.into_iter()
        .enumerate()
        .map(|(index, row_spans)| {
            let prefix = if index == 0 {
                first_prefix.clone()
            } else {
                cont_prefix.clone()
            };
            let mut spans = vec![prefix];
            spans.extend(row_spans);
            Line::from(spans)
        })
        .collect()
}

/// Split styled spans into `(word_with_trailing_spaces, style)` tokens.
fn tokenize(spans: Vec<Span<'static>>) -> Vec<(String, Style)> {
    let mut words: Vec<(String, Style)> = Vec::new();
    for span in spans {
        let style = span.style;
        let mut word = String::new();
        let mut pending = String::new();
        for ch in span.content.chars() {
            let ch = if ch == '\n' { ' ' } else { ch };
            if ch.is_whitespace() {
                if !word.is_empty() {
                    words.push((std::mem::take(&mut word), style));
                }
                pending.push(ch);
            } else {
                if !pending.is_empty() {
                    word.push_str(&pending);
                    pending.clear();
                }
                word.push(ch);
            }
        }
        if !word.is_empty() {
            words.push((word, style));
        } else if !pending.is_empty() {
            match words.last_mut() {
                Some((last, _)) => last.push_str(&pending),
                None => words.push((std::mem::take(&mut pending), style)),
            }
        }
    }
    words
}

/// Compact elapsed time: `12s`, `1m 02s`, `1h 00m 00s`.
pub fn fmt_elapsed(secs: u64) -> String {
    if secs < 60 {
        return format!("{secs}s");
    }
    if secs < 3600 {
        return format!("{}m {:02}s", secs / 60, secs % 60);
    }
    format!("{}h {:02}m {:02}s", secs / 3600, (secs % 3600) / 60, secs % 60)
}

/// `980` → `980`, `12500` → `12.5k`.
pub fn human_tokens(value: usize) -> String {
    if value >= 1000 {
        format!("{:.1}k", value as f64 / 1000.0)
    } else {
        value.to_string()
    }
}

/// Abbreviate `$HOME` to `~` and center-truncate long paths.
pub fn display_path(path: &str, max_width: usize) -> String {
    let home = std::env::var("HOME").unwrap_or_default();
    let shortened = if !home.is_empty() && path.starts_with(&home) {
        format!("~{}", &path[home.len()..])
    } else {
        path.to_string()
    };
    if max_width == 0 {
        return String::new();
    }
    if cell_width(&shortened) <= max_width {
        return shortened;
    }
    let head = max_width / 2 - 1;
    let tail = max_width - head - 2;
    let chars: Vec<char> = shortened.chars().collect();
    format!(
        "{}…{}",
        chars[..head].iter().collect::<String>(),
        chars[chars.len() - tail..].iter().collect::<String>()
    )
}

/// Truncate prose (e.g. error bodies) to keep transcript cells manageable.
pub fn clamp_text(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    let clipped: String = text.chars().take(max_chars).collect();
    format!("{clipped}…")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elapsed_format_is_compact() {
        assert_eq!(fmt_elapsed(12), "12s");
        assert_eq!(fmt_elapsed(62), "1m 02s");
        assert_eq!(fmt_elapsed(3723), "1h 02m 03s");
    }

    #[test]
    fn wrap_places_hanging_indent() {
        let lines = wrap_styled(
            vec![Span::raw("hello brave new world of terminal interfaces")],
            12,
            user_prefix(),
            Span::raw("  "),
        );
        assert!(lines.len() >= 2);
        assert_eq!(lines[0].spans[0].content, "› ");
        assert_eq!(lines[1].spans[0].content, "  ");
        for line in &lines {
            assert!(line_width(line) <= 12 + 2); // prefix + content
        }
    }

    #[test]
    fn wrap_hard_splits_long_words() {
        let lines = wrap_styled(
            vec![Span::raw("aaaaaaaaaaaaaaaaaaaa")],
            10,
            Span::raw(""),
            Span::raw(""),
        );
        assert!(lines.len() >= 2);
    }

    #[test]
    fn display_path_abbreviates_home() {
        std::env::set_var("HOME", "/home/dev");
        let path = display_path("/home/dev/projects/thing", 100);
        assert_eq!(path, "~/projects/thing");
    }

    #[test]
    fn cell_width_counts_persian_like_the_terminal() {
        // سلام contains Lam-Alef (لا): whole-string width() collapses it to
        // one cell, but terminals and ratatui's per-grapheme layout use two.
        assert_eq!(cell_width("لا"), 2);
        assert_eq!(cell_width("سلام"), 4);
        assert_eq!(cell_width("سلام دنیا"), 9);
        // ZWNJ (نیم‌فاصله) is zero-width and must not disturb measurement.
        assert_eq!(cell_width("می‌شود"), 5);
    }
}
