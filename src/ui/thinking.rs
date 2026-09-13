//! Reasoning ("thinking") support.
//!
//! Some models (MiniMax-M2.7 is one) stream their private chain of thought
//! wrapped in `<think> … </think>` ahead of the real answer. This module splits
//! that out of the message body and renders it as its own animated cell: a
//! shimmering `✻ Thinking…` header while the tokens arrive, collapsing into a
//! quiet `✻ Thought for 12s` card once the answer starts.

use crate::ui::theme;
use ratatui::prelude::*;

pub const OPEN_TAG: &str = "<think>";
pub const CLOSE_TAG: &str = "</think>";

/// One chunk of an assistant message: either private reasoning or the answer.
#[derive(Debug, Clone, PartialEq)]
pub struct Segment {
    pub thinking: bool,
    /// True for a `<think>` block that has not been closed yet (still streaming).
    pub open: bool,
    pub text: String,
}

/// Split an assistant message into reasoning / answer segments.
///
/// Unclosed `<think>` blocks (mid-stream) are returned with `open: true`, and a
/// trailing partial tag such as `<thi` is dropped so it never flickers on screen.
pub fn split(src: &str) -> Vec<Segment> {
    let mut out: Vec<Segment> = Vec::new();
    let mut rest = src;

    while let Some(start) = rest.find(OPEN_TAG) {
        push(&mut out, false, false, &rest[..start]);
        let after = &rest[start + OPEN_TAG.len()..];
        match after.find(CLOSE_TAG) {
            Some(end) => {
                push(&mut out, true, false, &after[..end]);
                rest = &after[end + CLOSE_TAG.len()..];
            }
            None => {
                push(&mut out, true, true, after);
                return out;
            }
        }
    }
    push(&mut out, false, false, strip_partial_tag(rest));
    out
}

fn push(out: &mut Vec<Segment>, thinking: bool, open: bool, text: &str) {
    if text.trim().is_empty() && !open {
        return;
    }
    out.push(Segment {
        thinking,
        open,
        text: text.to_string(),
    });
}

/// Drop a half-received `<think` / `</think` tag at the very end of a stream.
fn strip_partial_tag(text: &str) -> &str {
    let Some(idx) = text.rfind('<') else {
        return text;
    };
    let tail = &text[idx..];
    let partial_open = tail.len() < OPEN_TAG.len() && OPEN_TAG.starts_with(tail);
    let partial_close = tail.len() < CLOSE_TAG.len() && CLOSE_TAG.starts_with(tail);
    if partial_open || partial_close {
        &text[..idx]
    } else {
        text
    }
}

/// The message with every `<think>` block removed. This is what we replay to the API.
pub fn strip(src: &str) -> String {
    let answer: String = split(src)
        .into_iter()
        .filter(|segment| !segment.thinking)
        .map(|segment| segment.text)
        .collect();
    answer.trim().to_string()
}

/// True while the message ends inside an unclosed `<think>` block.
pub fn is_thinking(src: &str) -> bool {
    split(src).last().is_some_and(|seg| seg.open)
}

/// The most recent non-empty line of live reasoning, for the status row.
pub fn latest_thought(src: &str) -> Option<String> {
    let last = split(src).into_iter().rev().find(|seg| seg.open)?;
    last.text
        .lines()
        .rev()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .map(|line| line.to_string())
}

/// Word count of a reasoning block, used in the collapsed summary.
fn word_count(text: &str) -> usize {
    text.split_whitespace().count()
}

/// Sweep a bright highlight along `text`, one character per tick.
pub fn shimmer(text: &str, tick: u64, base: Style) -> Vec<Span<'static>> {
    let chars: Vec<char> = text.chars().collect();
    let span = chars.len() as u64 + 8;
    let head = tick % span;
    chars
        .into_iter()
        .enumerate()
        .map(|(index, ch)| {
            let distance = (index as i64 - head as i64).unsigned_abs();
            let style = match distance {
                0 => base.fg(Color::White).bold(),
                1 => base.fg(theme::ACCENT).bold(),
                2 => base.fg(theme::ACCENT),
                _ => base.fg(theme::DIM),
            };
            Span::styled(ch.to_string(), style)
        })
        .collect()
}

/// The animated glyph used by the thinking header.
pub fn glyph(tick_ms: u64) -> &'static str {
    theme::SPINNER[(tick_ms / 120) as usize % theme::SPINNER.len()]
}

/// Render a reasoning segment for the transcript.
///
/// * `open`      – still streaming, so animate and always show the tail.
/// * `expanded`  – the user asked to keep finished reasoning visible (`ctrl+r`).
/// * `tick_ms`   – animation clock (elapsed stream time).
pub fn render_segment(
    text: &str,
    width: usize,
    open: bool,
    expanded: bool,
    tick_ms: u64,
) -> Vec<Line<'static>> {
    let mut lines: Vec<Line<'static>> = Vec::new();
    let bar = Span::styled("┃ ", Style::new().fg(theme::DIM));

    if open {
        let mut header = vec![Span::styled(
            format!("{} ", glyph(tick_ms)),
            Style::new().fg(theme::ACCENT).bold(),
        )];
        header.extend(shimmer("Thinking", tick_ms / 90, Style::new().italic()));
        header.push(Span::styled(
            format!("  {}", theme::fmt_elapsed(tick_ms / 1000)),
            theme::dim(),
        ));
        lines.push(Line::from(header));
    } else {
        let arrow = if expanded { "▾" } else { "▸" };
        lines.push(Line::from(vec![
            Span::styled("✻ ", Style::new().fg(theme::ACCENT)),
            Span::styled(
                format!("Thought {}", summary(text)),
                Style::new().fg(theme::DIM).italic(),
            ),
            Span::styled(format!("  {arrow} ctrl+r"), theme::dim()),
        ]));
    }

    if !open && !expanded {
        return lines;
    }

    // Body: dim italic prose behind a left rule. While streaming we only keep
    // the tail so the reasoning never pushes the answer off screen.
    let body: Vec<&str> = text.lines().map(str::trim_end).collect();
    let shown: Vec<&str> = if open && body.len() > 6 {
        body[body.len() - 6..].to_vec()
    } else {
        body
    };
    let style = Style::new().fg(theme::DIM).italic();
    for raw in shown {
        if raw.trim().is_empty() {
            continue;
        }
        lines.extend(theme::wrap_styled(
            vec![Span::styled(theme::clamp_text(raw, 2000), style)],
            width.saturating_sub(2).max(8),
            bar.clone(),
            bar.clone(),
        ));
    }
    lines
}

fn summary(text: &str) -> String {
    let words = word_count(text);
    if words == 0 {
        "for a moment".to_string()
    } else {
        format!("for {words} words")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_closed_think_block() {
        let segments = split("<think>reasoning here</think>Hello!");
        assert_eq!(segments.len(), 2);
        assert!(segments[0].thinking && !segments[0].open);
        assert_eq!(segments[0].text, "reasoning here");
        assert_eq!(segments[1].text, "Hello!");
        assert!(!segments[1].thinking);
    }

    #[test]
    fn marks_unclosed_block_as_open() {
        let segments = split("<think>still going");
        assert_eq!(segments.len(), 1);
        assert!(segments[0].open);
        assert!(is_thinking("<think>still going"));
        assert!(!is_thinking("<think>done</think>answer"));
    }

    #[test]
    fn hides_partial_tag_while_streaming() {
        let segments = split("answer <thi");
        assert_eq!(segments[0].text, "answer ");
    }

    #[test]
    fn latest_thought_is_last_non_empty_line() {
        let text = "<think>first line\n\nsecond line\n";
        assert_eq!(latest_thought(text).as_deref(), Some("second line"));
    }

    #[test]
    fn collapsed_segment_is_one_line() {
        let lines = render_segment("some long reasoning", 60, false, false, 0);
        assert_eq!(lines.len(), 1);
        assert!(render_segment("some long reasoning", 60, false, true, 0).len() > 1);
    }

    #[test]
    fn strip_removes_reasoning() {
        assert_eq!(strip("<think>hidden</think>Hi there"), "Hi there");
        assert_eq!(strip("plain"), "plain");
    }

    #[test]
    fn plain_text_has_single_segment() {
        let segments = split("no reasoning at all");
        assert_eq!(segments.len(), 1);
        assert!(!segments[0].thinking);
    }
}
