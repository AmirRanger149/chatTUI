pub mod chat;
pub mod history;
pub mod input;
pub mod statusbar;

use crate::app::App;
use ratatui::{prelude::*, widgets::{Block, Borders, Paragraph}};

pub fn render(frame: &mut Frame, app: &App, input: &str) {
    let root = frame.area();
    let vertical = Layout::vertical([Constraint::Length(1), Constraint::Min(5), Constraint::Length(5), Constraint::Length(1)]).split(root);
    frame.render_widget(Paragraph::new(" chatTUI  /  Gemini terminal workspace").style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)), vertical[0]);
    let main = Layout::horizontal([Constraint::Percentage(75), Constraint::Percentage(25)]).split(vertical[1]);
    chat::render(frame, main[0], app);
    if app.show_history { history::render(frame, main[1], app); } else { frame.render_widget(Block::default().title(" Tips ").borders(Borders::ALL).border_style(Style::default().fg(Color::DarkGray)), main[1]); }
    input::render(frame, vertical[2], input, app);
    statusbar::render(frame, vertical[3], app);
}
