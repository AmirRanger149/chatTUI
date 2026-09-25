//! The `• Working (12s • esc to interrupt)` row drawn above the composer
//! while a response is streaming — plus the shimmering `Retrying` countdown
//! shown while the client waits out a same-model retry backoff.

use crate::app::{App, PendingApproval, PendingQuestions, RetryView};
use crate::ui::theme::{self, SPINNER};
use crate::ui::thinking;
use ratatui::prelude::*;
use ratatui::widgets::Paragraph;

pub fn render(frame: &mut Frame, area: Rect, app: &App) {
    if area.width == 0 || area.height == 0 {
        return;
    }
    // A question beats a spinner: nothing is running until it is answered.
    if let Some(pending) = &app.awaiting_approval {
        render_approval(frame, area, app, pending);
        return;
    }
    if let Some(pending) = &app.awaiting_questions {
        render_question(frame, area, app, pending);
        return;
    }
    if let Some(retry) = &app.retry_state {
        render_retry(frame, area, app, retry);
        return;
    }
    if app.is_thinking() {
        render_thinking(frame, area, app);
        return;
    }
    let glyph = SPINNER[(app.elapsed_ms() / 120) as usize % SPINNER.len()];
    
    let working_text = if app.agent_iterations > 0 {
        // Rounds completed so far — progress information, not a finish line:
        // healthy runs keep going until the health guards say otherwise.
        format!("Agent working (round {})", app.agent_iterations)
    } else if app.agent_mode && app.config.sandbox.enabled {
        "Working (agent)".to_string()
    } else {
        "Working".to_string()
    };
    
    let line = Line::from(vec![
        Span::styled(glyph, Style::new().fg(theme::ACCENT).bold()),
        Span::raw(" "),
        Span::styled(working_text, Style::new().bold()),
        Span::raw(" "),
        Span::styled(
            format!("({} • ", theme::fmt_elapsed(app.elapsed_secs())),
            theme::dim(),
        ),
        Span::raw("esc"),
        Span::styled(" to interrupt)", theme::dim()),
    ]);
    frame.render_widget(Paragraph::new(line), area);
}

/// The `✻ Thinking… (12s • esc to interrupt)` row, with a shimmering label and
/// a dim preview of the thought currently being written.
fn render_thinking(frame: &mut Frame, area: Rect, app: &App) {
    let tick = app.elapsed_ms();
    let mut spans = vec![
        Span::styled(
            format!("{} ", thinking::glyph(tick)),
            Style::new().fg(theme::ACCENT).bold(),
        ),
    ];
    spans.extend(thinking::shimmer("Thinking", tick / 90, Style::new().italic()));
    spans.push(Span::styled(
        format!(" ({} • ", theme::fmt_elapsed(app.elapsed_secs())),
        theme::dim(),
    ));
    spans.push(Span::raw("esc"));
    spans.push(Span::styled(" to interrupt)", theme::dim()));

    let used = theme::spans_width(&spans);
    if let Some(thought) = app.current_thought() {
        let room = (area.width as usize).saturating_sub(used + 4);
        if room > 8 {
            spans.push(Span::styled("  ", theme::dim()));
            spans.push(Span::styled(
                theme::clamp_text(&thought, room),
                Style::new().fg(theme::DIM).italic(),
            ));
        }
    }
    frame.render_widget(Paragraph::new(Line::from(spans)), area);
}

/// The `↻ Retrying in 1.8s (attempt 2/3 · reason • esc to interrupt)` row.
/// Same shimmer treatment as the thinking indicator; the countdown is
/// derived from the retry state on every redraw, so the ~20fps render loop
/// animates it for free. When the retry succeeds the state is cleared and
/// this row simply disappears.
fn render_retry(frame: &mut Frame, area: Rect, app: &App, retry: &RetryView) {
    let tick = app.elapsed_ms();
    let remaining = retry.wait.saturating_sub(retry.started.elapsed());
    let mut spans = vec![Span::styled(
        format!("{} ", thinking::glyph(tick)),
        Style::new().fg(Color::Yellow).bold(),
    )];
    spans.extend(thinking::shimmer("Retrying", tick / 90, Style::new().italic()));
    spans.push(Span::styled(
        format!(" in {:.1}s ", remaining.as_secs_f64()),
        Style::new().bold(),
    ));
    spans.push(Span::styled(
        format!("(attempt {}/{} · ", retry.attempt, retry.max),
        theme::dim(),
    ));
    let used = theme::spans_width(&spans);
    let tail = " • esc to interrupt)";
    let room = (area.width as usize).saturating_sub(used + theme::cell_width(tail) + 2);
    if room > 8 {
        spans.push(Span::styled(theme::clamp_text(&retry.reason, room), theme::dim()));
    }
    spans.push(Span::styled(" • ", theme::dim()));
    spans.push(Span::raw("esc"));
    spans.push(Span::styled(" to interrupt)", theme::dim()));
    frame.render_widget(Paragraph::new(Line::from(spans)), area);
}

/// The `? Approve? run: git push (y allow • a always • n deny)` row.
///
/// Drawn instead of the working spinner, because nothing is running. The
/// call itself is already in the transcript, so this row only has to fit
/// the question and the three keys on one line.
fn render_approval(frame: &mut Frame, area: Rect, app: &App, pending: &PendingApproval) {
    let tick = app.elapsed_ms();
    let mut spans = vec![Span::styled(
        format!("{} ", thinking::glyph(tick)),
        Style::new().fg(Color::Yellow).bold(),
    )];
    spans.extend(thinking::shimmer(
        if pending.escalated {
            "Unconfined?"
        } else {
            "Approve?"
        },
        tick / 90,
        Style::new().italic(),
    ));
    spans.push(Span::raw(" "));

    // Reserve room for the keys first: losing the tail would lose the only
    // place the answer keys are written down.
    let tail = if pending.escalated {
        " (y run it unconfined • n keep it blocked)"
    } else {
        " (y allow • a always • n deny)"
    };
    let used = theme::spans_width(&spans);
    let room = (area.width as usize).saturating_sub(used + theme::cell_width(tail) + 2);
    if room > 8 {
        spans.push(Span::styled(
            theme::clamp_text(&pending.summary, room),
            Style::new().bold(),
        ));
    }
    spans.push(Span::styled(" (", theme::dim()));
    spans.push(Span::raw("y"));
    if pending.escalated {
        spans.push(Span::styled(" run it unconfined • ", theme::dim()));
    } else {
        spans.push(Span::styled(" allow • ", theme::dim()));
        spans.push(Span::raw("a"));
        spans.push(Span::styled(" always • ", theme::dim()));
    }
    spans.push(Span::raw("n"));
    if pending.escalated {
        spans.push(Span::styled(" keep it blocked)", theme::dim()));
    } else {
        spans.push(Span::styled(" deny)", theme::dim()));
    }
    frame.render_widget(Paragraph::new(Line::from(spans)), area);
}

/// The `? Question 1/2 [Scope] Rewrite it? (1 Yes • 2 No • or type + enter)`
/// row. The options are listed here rather than in a popup so the composer
/// stays usable for the free-form answer.
fn render_question(frame: &mut Frame, area: Rect, app: &App, pending: &PendingQuestions) {
    let total = pending.questions.len();
    let index = pending.current.min(total.saturating_sub(1));
    let question = match pending.questions.get(index) {
        Some(question) => question,
        None => return,
    };

    let tick = app.elapsed_ms();
    let mut spans = vec![Span::styled(
        format!("{} ", thinking::glyph(tick)),
        Style::new().fg(Color::Cyan).bold(),
    )];
    spans.extend(thinking::shimmer(
        &format!("Question {}/{}", index + 1, total),
        tick / 90,
        Style::new().italic(),
    ));
    spans.push(Span::raw(" "));
    if !question.header.is_empty() {
        spans.push(Span::styled(
            format!("[{}] ", question.header),
            Style::new().fg(theme::ACCENT),
        ));
    }

    // Options first: they are the answer keys, and losing them would leave a
    // question with no visible way to answer it.
    let mut tail = String::from(" (");
    for (number, option) in question.options.iter().enumerate().take(9) {
        if number > 0 {
            tail.push_str(" • ");
        }
        tail.push_str(&format!("{} {}", number + 1, option.label));
    }
    if question.options.is_empty() {
        tail.push_str("type your answer + enter");
    } else {
        tail.push_str(" • or type + enter");
    }
    tail.push(')');

    let used = theme::spans_width(&spans);
    let room = (area.width as usize).saturating_sub(used + theme::cell_width(&tail) + 2);
    if room > 8 {
        spans.push(Span::styled(
            theme::clamp_text(&question.question, room),
            Style::new().bold(),
        ));
    }
    spans.push(Span::styled(tail, theme::dim()));
    frame.render_widget(Paragraph::new(Line::from(spans)), area);
}
