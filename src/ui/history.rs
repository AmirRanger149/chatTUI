use crate::app::App;
use ratatui::{
    prelude::*,
    widgets::{Block, Borders, List, ListItem},
};

pub fn render(frame: &mut Frame, area: Rect, app: &App) {
    let current = app.sessions.current_index();
    let items = app
        .sessions
        .sessions()
        .iter()
        .enumerate()
        .map(|(index, session)| {
            let title = if index == current {
                format!("> {}", session.title)
            } else {
                format!("  {}", session.title)
            };
            ListItem::new(title)
        })
        .collect::<Vec<_>>();
    frame.render_widget(
        List::new(items)
            .block(
                Block::default()
                    .title(" History ")
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::Magenta)),
            )
            .highlight_style(Style::default().fg(Color::Yellow)),
        area,
    );
}
