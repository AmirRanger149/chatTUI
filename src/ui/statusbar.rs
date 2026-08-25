use crate::app::{App, Mode};
use ratatui::{prelude::*, widgets::Paragraph};

pub fn render(frame: &mut Frame, area: Rect, app: &App) {
    let mode = match app.mode {
        Mode::Normal => "NORMAL",
        Mode::Insert => "INSERT",
    };
    let api_status = if app
        .config
        .api_key
        .as_deref()
        .is_some_and(|key| !key.trim().is_empty())
    {
        "API: CONFIGURED"
    } else {
        "API: NOT CONFIGURED"
    };
    let status = format!(
        " [{}]   {}   {}   [?] Help  [h] History  [n] New Chat  [q] Quit ",
        mode, app.config.model, api_status
    );
    frame.render_widget(
        Paragraph::new(status).style(Style::default().fg(Color::Black).bg(Color::Cyan)),
        area,
    );
}
