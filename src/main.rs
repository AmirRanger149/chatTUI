mod sandbox;
mod tools;
mod api;
mod clipboard;
mod code;
mod config;
mod session;
mod app;
mod ui;

use anyhow::{Context, Result};
use app::App;
use config::Config;
use crossterm::{
    event::{self, Event, KeyCode, KeyEvent, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{backend::CrosstermBackend, Terminal};
use session::manager::SessionManager;
use std::env;
use std::io::{self, stdout};
use tokio::time::Duration;

#[tokio::main]
async fn main() -> Result<()> {
    let mut config = Config::load()?;
    apply_cli_args(&mut config)?;
    let sessions = SessionManager::load()?;
    let mut app = App::new(config, sessions);
    let mut terminal = setup_terminal()?;
    let result = run(&mut terminal, &mut app).await;
    restore_terminal(&mut terminal)?;
    result
}

fn apply_cli_args(config: &mut Config) -> Result<()> {
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--sandbox" | "--target" | "--workspace" => {
                let path = args
                    .next()
                    .context("--sandbox / --target requires a directory path")?;
                config.sandbox.workspace_root = path;
            }
            "--help" | "-h" => {
                eprintln!(
                    "chatTUI\n  --sandbox <dir>   Agent target directory (required for file tools)\n  --target <dir>    Same as --sandbox\n"
                );
                std::process::exit(0);
            }
            other if other.starts_with('-') => {
                anyhow::bail!("unknown flag {other} — try --help");
            }
            _ => {}
        }
    }
    Ok(())
}

async fn run(terminal: &mut Terminal<CrosstermBackend<io::Stdout>>, app: &mut App) -> Result<()> {
    loop {
        app.receive_token().await;
        app.receive_models();
        terminal.draw(|frame| ui::render(frame, app))?;

        if event::poll(Duration::from_millis(50))? {
            loop {
                match event::read()? {
                    Event::Key(key) => {
                        if !handle_key(app, key) {
                            return Ok(());
                        }
                    }
                    Event::Paste(text) => app.paste(&text),
                    _ => {}
                }
                if !event::poll(Duration::ZERO)? {
                    break;
                }
            }
        }
        if app.should_quit {
            return Ok(());
        }
    }
}

/// Returns `false` when the app should exit.
fn handle_key(app: &mut App, key: KeyEvent) -> bool {
    // ctrl+c: press twice within 2s to quit, from any state.
    if key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL) {
        app.prime_quit();
        return !app.should_quit;
    }

    // Global keys.
    match key.code {
        KeyCode::Esc => {
            app.escape();
            return true;
        }
        KeyCode::Char('t') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            app.toggle_history();
            return true;
        }
        KeyCode::Char('g') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            app.toggle_code();
            return true;
        }
        KeyCode::Char('r') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            app.toggle_thinking();
            return true;
        }
        _ => {}
    }

    // Overlays consume every remaining key so an open popup and the transcript
    // (or the composer) can never fight over the same input.
    match app.overlay {
        Some(app::Overlay::Shortcuts) => {
            if matches!(key.code, KeyCode::Esc | KeyCode::Enter | KeyCode::Char('?')) {
                app.overlay = None;
            }
            return true;
        }
        Some(app::Overlay::History { .. }) => {
            match key.code {
                KeyCode::Up => app.move_history_selection(-1),
                KeyCode::Down => app.move_history_selection(1),
                KeyCode::PageUp => app.move_history_selection(-(app::OVERLAY_ROWS as i32)),
                KeyCode::PageDown => app.move_history_selection(app::OVERLAY_ROWS as i32),
                KeyCode::Enter => app.load_selected_session(),
                KeyCode::Char('d') => app.delete_selected_session(),
                _ => {}
            }
            return true;
        }
        Some(app::Overlay::Code { .. }) => {
            match key.code {
                KeyCode::Up => app.move_code_selection(-1),
                KeyCode::Down => app.move_code_selection(1),
                KeyCode::PageUp => app.move_code_selection(-(app::OVERLAY_ROWS as i32)),
                KeyCode::PageDown => app.move_code_selection(app::OVERLAY_ROWS as i32),
                KeyCode::Enter => app.copy_selected_code(),
                _ => {}
            }
            return true;
        }
        Some(app::Overlay::Models { .. }) => {
            match key.code {
                KeyCode::Up => app.move_model_selection(-1),
                KeyCode::Down => app.move_model_selection(1),
                KeyCode::PageUp => app.move_model_selection(-(app::OVERLAY_ROWS as i32)),
                KeyCode::PageDown => app.move_model_selection(app::OVERLAY_ROWS as i32),
                KeyCode::Enter => app.apply_selected_model(),
                KeyCode::Char('r') | KeyCode::Char('R') => app.refresh_models(),
                _ => {}
            }
            return true;
        }
        Some(app::Overlay::Providers { .. }) => {
            match key.code {
                KeyCode::Up => app.move_provider_selection(-1),
                KeyCode::Down => app.move_provider_selection(1),
                KeyCode::PageUp => app.move_provider_selection(-(app::OVERLAY_ROWS as i32)),
                KeyCode::PageDown => app.move_provider_selection(app::OVERLAY_ROWS as i32),
                KeyCode::Enter => app.apply_selected_provider(),
                _ => {}
            }
            return true;
        }
        None => {}
    }

    // Transcript scrolling (only when no overlay is open): pgup moves the
    // view up toward older messages, pgdn back down to the newest ones.
    match key.code {
        KeyCode::PageUp => {
            app.scroll(10);
            return true;
        }
        KeyCode::PageDown => {
            app.scroll(-10);
            return true;
        }
        _ => {}
    }

    // Nothing else claimed the key, so it belongs to the composer.
    match key.code {
        KeyCode::Enter if key.modifiers.contains(KeyModifiers::SHIFT) => {
            app.insert_newline();
            app.on_composer_changed();
        }
        KeyCode::Enter if key.modifiers.contains(KeyModifiers::ALT) => {
            app.insert_newline();
            app.on_composer_changed();
        }
        KeyCode::Enter => app.submit(),
        KeyCode::Tab => app.accept_slash(),
        KeyCode::Backspace => {
            app.backspace();
            app.on_composer_changed();
        }
        KeyCode::Up if app.slash_open() => app.slash_up(),
        KeyCode::Down if app.slash_open() => app.slash_down(),
        KeyCode::Up => app.recall_prev(),
        KeyCode::Down => app.recall_next(),
        // Cursor navigation: left/right arrows move letter by letter;
        // ctrl+arrow and the classic alt+b / alt+f jump whole words.
        KeyCode::Left if key.modifiers.contains(KeyModifiers::CONTROL) => app.move_word_left(),
        KeyCode::Right if key.modifiers.contains(KeyModifiers::CONTROL) => app.move_word_right(),
        KeyCode::Left => app.move_cursor_left(),
        KeyCode::Right => app.move_cursor_right(),
        KeyCode::Home => app.move_cursor_home(),
        KeyCode::End => app.move_cursor_end(),
        KeyCode::Char('b') if key.modifiers.contains(KeyModifiers::ALT) => app.move_word_left(),
        KeyCode::Char('f') if key.modifiers.contains(KeyModifiers::ALT) => app.move_word_right(),
        KeyCode::Char('u') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            app.clear_composer();
        }
        KeyCode::Char('?') if app.composer.is_empty() => app.toggle_shortcuts(),
        KeyCode::Char(ch)
            if !key.modifiers.contains(KeyModifiers::CONTROL)
                && !key.modifiers.contains(KeyModifiers::ALT) =>
        {
            app.insert_char(ch);
            app.on_composer_changed();
        }
        _ => {}
    }
    true
}

fn setup_terminal() -> Result<Terminal<CrosstermBackend<io::Stdout>>> {
    enable_raw_mode()?;
    execute!(stdout(), EnterAlternateScreen, event::EnableBracketedPaste)?;
    Ok(Terminal::new(CrosstermBackend::new(stdout()))?)
}

fn restore_terminal(terminal: &mut Terminal<CrosstermBackend<io::Stdout>>) -> Result<()> {
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen, event::DisableBracketedPaste)?;
    terminal.show_cursor()?;
    Ok(())
}
