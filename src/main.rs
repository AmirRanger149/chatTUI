mod api;
mod app;
mod config;
mod session;
mod ui;

use anyhow::Result;
use app::{Action, App, Mode};
use config::Config;
use crossterm::{
    event::{self, Event, KeyCode, KeyEvent},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{backend::CrosstermBackend, Terminal};
use std::io::{self, stdout};
use tokio::time::{self, Duration};

#[tokio::main]
async fn main() -> Result<()> {
    let config = Config::load()?;
    config.save()?;
    let sessions = session::manager::SessionManager::load()?;
    let mut app = App::new(config, sessions);
    let mut input = String::new();
    let mut terminal = setup_terminal()?;
    let result = run(&mut terminal, &mut app, &mut input).await;
    restore_terminal(&mut terminal)?;
    result
}

async fn run(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    app: &mut App,
    input: &mut String,
) -> Result<()> {
    let mut tick = time::interval(Duration::from_millis(50));
    loop {
        terminal.draw(|frame| ui::render(frame, app, input))?;
        tick.tick().await;
        app.receive_token().await;
        if event::poll(Duration::from_millis(1))? {
            if let Event::Key(key) = event::read()? {
                if !handle_key(app, input, key).await? {
                    break;
                }
            }
        }
    }
    Ok(())
}

async fn handle_key(app: &mut App, input: &mut String, key: KeyEvent) -> Result<bool> {
    match app.mode {
        Mode::Normal => match key.code {
            KeyCode::Char('i') => {
                app.dispatch(Action::EnterInsert);
                Ok(true)
            }
            KeyCode::Char('j') | KeyCode::Down => {
                app.dispatch(Action::ScrollDown);
                Ok(true)
            }
            KeyCode::Char('k') | KeyCode::Up => {
                app.dispatch(Action::ScrollUp);
                Ok(true)
            }
            KeyCode::PageDown => {
                app.dispatch(Action::ScrollDown);
                Ok(true)
            }
            KeyCode::PageUp => {
                app.dispatch(Action::ScrollUp);
                Ok(true)
            }
            KeyCode::Char('h') => {
                app.dispatch(Action::ToggleHistory);
                Ok(true)
            }
            KeyCode::Char('n') => {
                app.dispatch(Action::NewChat);
                Ok(true)
            }
            KeyCode::Char('?') => {
                app.dispatch(Action::Help);
                Ok(true)
            }
            KeyCode::Char('q') => Ok(app.dispatch(Action::Quit)),
            KeyCode::Char('d') => {
                app.dispatch(Action::DeleteChat);
                Ok(true)
            }
            _ => Ok(true),
        },
        Mode::Insert => match key.code {
            KeyCode::Esc => {
                app.dispatch(Action::EnterNormal);
                Ok(true)
            }
            KeyCode::Enter => {
                if !input.trim().is_empty() {
                    let text = std::mem::take(input);
                    app.dispatch(Action::Submit(text));
                    if let Err(error) = app.start_stream().await {
                        app.error = Some(error.to_string());
                    }
                }
                Ok(true)
            }
            KeyCode::Backspace => {
                input.pop();
                Ok(true)
            }
            KeyCode::Char(ch) if key.modifiers.is_empty() => {
                input.push(ch);
                Ok(true)
            }
            _ => Ok(true),
        },
    }
}

fn setup_terminal() -> Result<Terminal<CrosstermBackend<io::Stdout>>> {
    enable_raw_mode()?;
    execute!(stdout(), EnterAlternateScreen)?;
    Ok(Terminal::new(CrosstermBackend::new(stdout()))?)
}

fn restore_terminal(terminal: &mut Terminal<CrosstermBackend<io::Stdout>>) -> Result<()> {
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    Ok(())
}
