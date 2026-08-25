use crate::app::App;
use ratatui::{prelude::*, widgets::{Block, Borders, Paragraph, Wrap}};

pub fn render(frame: &mut Frame, area: Rect, input: &str, app: &App) {
    let title = match app.mode { crate::app::Mode::Insert => " Prompt [INSERT] ", crate::app::Mode::Normal => " Prompt [NORMAL] " };
    frame.render_widget(Paragraph::new(input).block(Block::default().title(title).borders(Borders::ALL).border_style(Style::default().fg(Color::White))).wrap(Wrap { trim: false }), area);
}
