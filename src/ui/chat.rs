use crate::app::App;
use ratatui::{
    prelude::*,
    widgets::{Block, Borders, Paragraph, Wrap},
};

pub fn render(frame: &mut Frame, area: Rect, app: &App) {
    let lines: Vec<Line> = app
        .sessions
        .current()
        .messages
        .iter()
        .map(|message| {
            let (label, color) = if message.role == "user" {
                ("You", Color::Green)
            } else {
                ("Gemini", Color::Cyan)
            };
            vec![
                Line::from(Span::styled(
                    format!("{}  ", label),
                    Style::default().fg(color).add_modifier(Modifier::BOLD),
                )),
                Line::from(message.content.as_str()),
                Line::from(""),
            ]
        })
        .flatten()
        .collect();
    let mut text = lines;
    if !app.response.is_empty() {
        text.extend([
            Line::from(Span::styled(
                "Gemini  ",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            )),
            Line::from(app.response.as_str()),
        ]);
    }
    if text.is_empty() {
        text.push(Line::from(Span::styled(
            "Start with a question, an idea, or a piece of code.",
            Style::default().fg(Color::DarkGray),
        )));
    }
    frame.render_widget(
        Paragraph::new(text)
            .block(
                Block::default()
                    .title(" Conversation ")
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::Blue)),
            )
            .wrap(Wrap { trim: false })
            .scroll((app.scroll, 0)),
        area,
    );
}
