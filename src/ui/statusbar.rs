use crate::app::{App, Mode};
use ratatui::{prelude::*, widgets::Paragraph};

pub fn render(frame: &mut Frame, area: Rect, app: &App) {
    let mode = match app.mode {
        Mode::Normal => "NORMAL",
        Mode::Insert => "INSERT",
    };
    let status = format!(
        " [{}]   {}   [?] Help  [h] History  [n] New Chat  [q] Quit ",
        mode, app.config.model
    );
    frame.render_widget(
        Paragraph::new(status).style(Style::default().fg(Color::Black).bg(Color::Cyan)),
        area,
    );
}
