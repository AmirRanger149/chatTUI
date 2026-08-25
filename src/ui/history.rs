use crate::app::App;
use ratatui::{prelude::*, widgets::{Block, Borders, List, ListItem}};

pub fn render(frame: &mut Frame, area: Rect, app: &App) {
    let items = app.sessions.sessions().iter().map(|session| ListItem::new(session.title.clone())).collect::<Vec<_>>();
    frame.render_widget(List::new(items).block(Block::default().title(" History ").borders(Borders::ALL).border_style(Style::default().fg(Color::Magenta))).highlight_style(Style::default().fg(Color::Yellow)), area);
}
