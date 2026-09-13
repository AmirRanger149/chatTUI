//! A compact markdown renderer for assistant messages: bold headings, `•`
//! bullets with hanging indents, dim blockquotes and shaded code blocks.

use crate::ui::theme::{self, wrap_styled};
use ratatui::prelude::*;

pub fn render(src: &str, width: usize) -> Vec<Line<'static>> {
    let mut out: Vec<Line<'static>> = Vec::new();
    let mut in_code = false;

    for raw in src.lines() {
        let line = raw.trim_end();

        if let Some(info) = line.trim_start().strip_prefix("```") {
            if in_code {
                in_code = false;
                out.push(code_footer(width));
            } else {
                in_code = true;
                out.push(code_header(info, width));
            }
            continue;
        }
        if in_code {
            out.extend(code_body(line, width));
            continue;
        }
        if line.trim().is_empty() {
            out.push(Line::from(""));
            continue;
        }

        let trimmed = line.trim_start();
        let indent = line.len() - trimmed.len();

        // Horizontal rule
        if trimmed.len() >= 3 && trimmed.chars().all(|c| c == '-' || c == '_' || c == '*') {
            out.push(Line::styled(
                "─".repeat(24.min(width.max(1))),
                theme::dim(),
            ));
            continue;
        }

        // Headings (# .. ######)
        if let Some(rest) = trimmed.strip_prefix('#') {
            let level = rest.chars().take_while(|c| *c == '#').count();
            if level <= 6 && rest[level..].starts_with(' ') {
                let text = rest[level..].trim_start();
                let style = if level <= 2 {
                    Style::new().bold().underlined()
                } else {
                    Style::new().bold()
                };
                out.extend(wrap_styled(
                    inline(text, style),
                    width,
                    Span::raw(""),
                    Span::raw(""),
                ));
                continue;
            }
        }

        // Blockquote
        if let Some(rest) = trimmed.strip_prefix("> ") {
            let spans = vec![Span::styled(
                rest.to_string(),
                Style::new().dim().italic(),
            )];
            out.extend(wrap_styled(
                spans,
                width,
                Span::styled("▌ ", theme::dim()),
                Span::styled("  ", theme::dim()),
            ));
            continue;
        }

        // Unordered list
        if trimmed.starts_with("- ")
            || trimmed.starts_with("* ")
            || trimmed.starts_with("+ ")
        {
            let pad = " ".repeat(indent.min(8));
            let content = trimmed[2..].trim_start();
            let spans = inline(content, Style::new());
            out.extend(wrap_styled(
                spans,
                width,
                Span::raw(format!("{pad}• ")),
                Span::raw(format!("{}  ", pad)),
            ));
            continue;
        }

        // Ordered list: keep the original marker, indent the wrapped rows.
        if let Some(marker_end) = ordered_marker_len(trimmed) {
            let marker = trimmed[..marker_end].to_string();
            let content = trimmed[marker_end..].trim_start();
            let spans = inline(content, Style::new());
            let cont = " ".repeat(indent.min(8) + theme::cell_width(&marker) + 1);
            out.extend(wrap_styled(
                spans,
                width,
                Span::raw(format!("{}{} ", " ".repeat(indent.min(8)), marker)),
                Span::raw(cont),
            ));
            continue;
        }

        // Paragraph
        out.extend(wrap_styled(
            inline(line, Style::new()),
            width,
            Span::raw(""),
            Span::raw(""),
        ));
    }
    out
}

/// Top of a code box: `╭─ rust ─────────────── ctrl+g to copy ╮`.
fn code_header(info: &str, width: usize) -> Line<'static> {
    let inner = width.saturating_sub(2);
    let label = info.trim().split_whitespace().next().unwrap_or("");
    let label = if label.is_empty() { "code" } else { label };
    let left = format!("─ {label} ");
    let hint = " ctrl+g to copy ";
    let mut spans = vec![Span::styled("╭", theme::dim())];
    if left.len() + hint.len() + 1 <= inner {
        let rule = inner - left.len() - hint.len();
        spans.push(Span::styled(left, Style::new().bold()));
        spans.push(Span::styled("─".repeat(rule), theme::dim()));
        spans.push(Span::styled(hint, theme::dim()));
    } else if left.len() + 1 <= inner {
        let rule = inner - left.len();
        spans.push(Span::styled(left, Style::new().bold()));
        spans.push(Span::styled("─".repeat(rule), theme::dim()));
    } else {
        spans.push(Span::styled("─".repeat(inner), theme::dim()));
    }
    spans.push(Span::styled("╮", theme::dim()));
    Line::from(spans)
}

/// One code line inside the box: a shaded band padded to full width with dim
/// side borders. Long lines wrap inside the band.
fn code_body(line: &str, width: usize) -> Vec<Line<'static>> {
    let shade = Style::new().bg(Color::DarkGray);
    let inner = width.saturating_sub(2);
    let band = inner.saturating_sub(2).max(1);
    let rows = wrap_styled(
        vec![Span::styled(line.to_string(), shade)],
        band.saturating_sub(1).max(1),
        Span::raw(""),
        Span::raw(""),
    );
    rows.into_iter()
        .map(|row| {
            let used = theme::line_width(&row);
            let pad = band.saturating_sub(1 + used);
            let mut spans = vec![
                Span::styled("│", theme::dim()),
                Span::styled(" ", shade),
            ];
            spans.extend(row.spans);
            spans.push(Span::styled(" ".repeat(pad), shade));
            spans.push(Span::styled("│", theme::dim()));
            Line::from(spans)
        })
        .collect()
}

/// Bottom of a code box: `╰─────────────────────╯`.
fn code_footer(width: usize) -> Line<'static> {
    let inner = width.saturating_sub(2);
    Line::from(vec![
        Span::styled("╰", theme::dim()),
        Span::styled("─".repeat(inner), theme::dim()),
        Span::styled("╯", theme::dim()),
    ])
}

/// Recognize `12.` / `3)` markers, returning the byte length of the marker.
fn ordered_marker_len(text: &str) -> Option<usize> {
    let digits = text.chars().take_while(|c| c.is_ascii_digit()).count();
    if digits == 0 || digits > 9 {
        return None;
    }
    let rest = text[digits..].chars().next()?;
    if rest == '.' || rest == ')' {
        Some(digits + rest.len_utf8())
    } else {
        None
    }
}

/// Inline formatting: `**bold**`, `*italic*`, `` `code` `` and links.
fn flush_inline(
    out: &mut Vec<Span<'static>>,
    buf: &mut String,
    base: Style,
    bold: bool,
    italic: bool,
    code: bool,
) {
    if buf.is_empty() {
        return;
    }
    let mut style = base;
    if bold {
        style = style.bold();
    }
    if italic {
        style = style.italic();
    }
    if code {
        style = style.fg(theme::ACCENT);
    }
    out.push(Span::styled(std::mem::take(buf), style));
}

pub fn inline(text: &str, base: Style) -> Vec<Span<'static>> {
    let chars: Vec<char> = text.chars().collect();
    let mut out: Vec<Span<'static>> = Vec::new();
    let mut buf = String::new();
    let mut bold = false;
    let mut italic = false;
    let mut code = false;

    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c == '`' {
            flush_inline(&mut out, &mut buf, base, bold, italic, code);
            code = !code;
            i += 1;
        } else if c == '*' && i + 1 < chars.len() && chars[i + 1] == '*' {
            flush_inline(&mut out, &mut buf, base, bold, italic, code);
            bold = !bold;
            i += 2;
        } else if c == '*' {
            flush_inline(&mut out, &mut buf, base, bold, italic, code);
            italic = !italic;
            i += 1;
        } else if c == '[' {
            if let Some((label, end)) = parse_link(&chars, i) {
                flush_inline(&mut out, &mut buf, base, bold, italic, code);
                out.push(Span::styled(
                    label.iter().collect::<String>(),
                    Style::new().underlined(),
                ));
                i = end;
            } else {
                buf.push(c);
                i += 1;
            }
        } else {
            buf.push(c);
            i += 1;
        }
    }
    flush_inline(&mut out, &mut buf, base, bold, italic, code);
    out
}

/// Parse `[label](url)` starting at `[`; returns `(label, index_after_close)`.
fn parse_link(chars: &[char], start: usize) -> Option<(Vec<char>, usize)> {
    let mut i = start + 1;
    let mut label = Vec::new();
    while i < chars.len() && chars[i] != ']' {
        if chars[i] == '[' {
            return None; // nested brackets are not links
        }
        label.push(chars[i]);
        i += 1;
    }
    if i + 1 >= chars.len() || chars[i] != ']' || chars[i + 1] != '(' {
        return None;
    }
    i += 2;
    while i < chars.len() && chars[i] != ')' {
        i += 1;
    }
    if i >= chars.len() {
        return None;
    }
    Some((label, i + 1))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plain(lines: &[Line<'static>]) -> Vec<String> {
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
    fn bullets_use_the_dot_glyph_and_hanging_indent() {
        let lines = render("- first item\n- second item", 40);
        let text = plain(&lines);
        assert_eq!(text[0], "• first item");
        assert_eq!(text[1], "• second item");
    }

    #[test]
    fn headings_and_code_blocks_render() {
        let lines = render("## Title\n```rust\nlet x = 1;\n```\ndone", 40);
        let text = plain(&lines);
        assert_eq!(text[0], "Title");
        assert!(text.iter().any(|l| l.contains("let x = 1;")));
        assert_eq!(text.last().unwrap(), "done");
    }

    #[test]
    fn code_blocks_get_header_and_border() {
        let lines = render("```rust\nlet x = 1;\n```", 40);
        let text = plain(&lines);
        assert!(text[0].starts_with('╭'));
        assert!(text[0].contains("rust"));
        assert!(text.iter().any(|l| l.contains('│') && l.contains("let x = 1;")));
        assert!(text.last().unwrap().starts_with('╰'));
    }

    #[test]
    fn multiple_code_blocks_each_get_a_box() {
        let lines = render("```a\n1\n```\ntext\n```\n2\n```", 40);
        let text = plain(&lines);
        let headers = text.iter().filter(|l| l.starts_with('╭')).count();
        let footers = text.iter().filter(|l| l.starts_with('╰')).count();
        assert_eq!(headers, 2);
        assert_eq!(footers, 2);
    }

    #[test]
    fn inline_styles_are_applied() {
        let spans = inline("**bold** and `code`", Style::new());
        assert!(spans[0].style.add_modifier.contains(Modifier::BOLD));
        assert!(spans.iter().any(|s| s.content == "code"
            && s.style.fg == Some(theme::ACCENT)));
    }

    #[test]
    fn links_parse_to_label() {
        let spans = inline("see [docs](https://example.com) now", Style::new());
        let joined: String = spans.iter().map(|s| s.content.as_ref()).collect();
        assert_eq!(joined, "see docs now");
    }

    #[test]
    fn long_paragraphs_wrap_with_cont_indent() {
        let lines = render("word ".repeat(40).trim_end(), 20);
        assert!(lines.len() > 1);
    }
}
