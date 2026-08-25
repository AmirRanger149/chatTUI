use crate::app::App;
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::{prelude::*, widgets::{Block, Borders, Paragraph, Wrap}};

pub fn handle_key(app: &mut App, key: KeyEvent) -> bool {
    match app.mode {
        crate::app::Mode::Normal => match key.code {
            KeyCode::Char('i') => { app.dispatch(crate::app::Action::EnterInsert); true }
            KeyCode::Char('j') | KeyCode::Down => { app.dispatch(crate::app::Action::ScrollDown); true }
            KeyCode::Char('k') | KeyCode::Up => { app.dispatch(crate::app::Action::ScrollUp); true }
            KeyCode::Char('h') => { app.dispatch(crate::app::Action::ToggleHistory); true }
            KeyCode::Char('n') => { app.dispatch(crate::app::Action::NewChat); true }
            KeyCode::Char('?') => { app.dispatch(crate::app::Action::Help); true }
            KeyCode::Char('q') | KeyCode::Esc => app.dispatch(crate::app::Action::Quit),
            _ => true,
        },
        crate::app::Mode::Insert => match key.code {
            KeyCode::Esc => { app.dispatch(crate::app::Action::EnterNormal); true }
            KeyCode::Enter => true,
            _ => true,
        },
    }
}

pub fn render(frame: &mut Frame, area: Rect, input: &str, app: &App) {
    let title = match app.mode { crate::app::Mode::Insert => " Prompt [INSERT] ", crate::app::Mode::Normal => " Prompt [NORMAL] " };
    frame.render_widget(Paragraph::new(input).block(Block::default().title(title).borders(Borders::ALL).border_style(Style::default().fg(Color::White))).wrap(Wrap { trim: false }), area);
}
