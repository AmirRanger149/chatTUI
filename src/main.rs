mod sandbox;
mod tools;
mod api;
mod clipboard;
mod code;
mod config;
mod session;
mod app;
mod instructions;
mod ui;

use anyhow::{Context, Result};
use app::App;
use config::Config;
use crossterm::{
    cursor,
    event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers},
    execute,
    style::ResetColor,
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
    // A session is its own process session, so it would survive chatTUI and
    // leave the user with a dev server they did not ask to keep. Kill them
    // first, while the terminal is still ours and errors can be reported.
    app.sandbox.sessions.kill_all();
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
        // A tool ran since the last frame: make sure the terminal is still
        // ours before drawing into it. See `reassert_terminal`.
        if app.terminal_dirty {
            app.terminal_dirty = false;
            reassert_terminal()?;
        }
        terminal.draw(|frame| ui::render(frame, app))?;

        if event::poll(Duration::from_millis(50))? {
            loop {
                match event::read()? {
                    Event::Key(key) => {
                        // Windows sends a Press *and* a Release event per keystroke; only the
                        // press may reach handle_key() or every key would act twice.
                        if should_handle(&key) && !handle_key(app, key) {
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

/// Whether a key event may be forwarded to [`handle_key`].
///
/// crossterm's Windows console backend maps the console `key_down` flag straight onto
/// [`KeyEventKind`], so one physical keystroke arrives as **two** `Event::Key`s: a `Press` and a
/// `Release` (crossterm `src/event/sys/windows/parse.rs`). Handling both doubles every action on
/// Windows — typing `/` inserts `//`, backspace deletes two characters, arrows move twice,
/// ctrl+c quits on the first press. Unix backends only ever emit `Press` unless the kitty keyboard
/// protocol is turned on (`PushKeyboardEnhancementFlags` + `REPORT_EVENT_TYPES`), which this app
/// never does, so filtering here is a no-op on Linux/macOS.
///
/// `Repeat` is allowed through as well: no crossterm backend produces it today (Windows reports a
/// held key as a stream of `Press`es, plain Unix reports nothing), but it keeps auto-repeat working
/// if keyboard enhancement flags are ever enabled. `Release` is dropped outright — it carries no
/// character/code information that a press did not already deliver.
///
/// Do not "simplify" this check away: it looks redundant to anyone testing only on Linux.
fn should_handle(key: &KeyEvent) -> bool {
    matches!(key.kind, KeyEventKind::Press | KeyEventKind::Repeat)
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
            app.toggle_tool_detail();
            return true;
        }
        KeyCode::Char('h') if key.modifiers.contains(KeyModifiers::CONTROL) => {
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
            Some(app::Overlay::ToolDetail { .. }) => {
                match key.code {
                    KeyCode::Up => app.move_tool_detail_selection(-1),
                    KeyCode::Down => app.move_tool_detail_selection(1),
                    KeyCode::PageUp => app.move_tool_detail_selection(-(app::OVERLAY_ROWS as i32)),
                    KeyCode::PageDown => app.move_tool_detail_selection(app::OVERLAY_ROWS as i32),
                    KeyCode::Enter => app.overlay = None,
                    _ => {}
                }
                return true;
            }
            None => {}
    }

    // An approval prompt owns the keyboard until it is answered: y, a and n
    // are answers here, and letting them through would put stray letters in
    // the composer while the agent waits.
    if app.awaiting_approval.is_some() {
        match key.code {
            KeyCode::Char('y') | KeyCode::Char('Y') | KeyCode::Enter => {
                app.resolve_approval(app::Approval::Once)
            }
            KeyCode::Char('a') | KeyCode::Char('A') => {
                app.resolve_approval(app::Approval::Always)
            }
            KeyCode::Char('n') | KeyCode::Char('N') => {
                app.resolve_approval(app::Approval::Deny)
            }
            _ => {}
        }
        return true;
    }

    // A question is on screen: digits pick an option. Everything else still
    // belongs to the composer, because typing an answer is the other half of
    // the interaction.
    let picked = app
        .awaiting_questions
        .as_ref()
        .and_then(|pending| match key.code {
            KeyCode::Char(digit @ '1'..='9') => {
                let index = pending.current.min(pending.questions.len().saturating_sub(1));
                let option = digit as usize - '1' as usize;
                pending
                    .questions
                    .get(index)
                    .and_then(|question| question.options.get(option))
                    .map(|option| option.label.clone())
            }
            _ => None,
        });
    if let Some(label) = picked {
        app.answer_current_question(label);
        return true;
    }
    // Enter answers with whatever is in the composer; an empty composer is
    // not an answer, so it is swallowed rather than sent as a message.
    if app.awaiting_questions.is_some() && key.code == KeyCode::Enter {
        let text = app.composer.clone();
        if !text.trim().is_empty() {
            app.clear_composer();
            app.answer_current_question(text);
        }
        return true;
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

/// Put the terminal back into the exact state chatTUI needs.
///
/// A command the agent runs has no controlling terminal and its captured
/// output is stripped of control sequences, so nothing it does should reach
/// this terminal. This is the belt to those braces: the failure mode it
/// prevents is a session left in canonical mode with the cursor hidden,
/// which is unrecoverable without killing the process, and the cost is a few
/// terminal writes after a tool call.
///
/// Every part is idempotent, and the transcript is redrawn in full on the
/// next frame, so re-asserting state that is already correct is invisible.
fn reassert_terminal() -> Result<()> {
    enable_raw_mode()?;
    execute!(
        stdout(),
        EnterAlternateScreen,
        cursor::Show,
        event::EnableBracketedPaste,
        event::DisableMouseCapture,
        ResetColor
    )?;
    Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;

    fn key(code: KeyCode, kind: KeyEventKind) -> KeyEvent {
        KeyEvent::new_with_kind(code, KeyModifiers::NONE, kind)
    }

    #[test]
    fn press_is_forwarded_and_release_is_dropped() {
        for code in [
            KeyCode::Char('/'),
            KeyCode::Backspace,
            KeyCode::Left,
            KeyCode::Enter,
            KeyCode::Char('c'),
        ] {
            assert!(should_handle(&key(code, KeyEventKind::Press)), "{code:?} press");
            assert!(
                !should_handle(&key(code, KeyEventKind::Release)),
                "{code:?} release must never reach handle_key"
            );
        }
    }

    #[test]
    fn windows_press_release_pair_yields_a_single_forwarded_event() {
        // What the Windows console backend delivers for one physical "/" keystroke.
        let keystroke = [
            key(KeyCode::Char('/'), KeyEventKind::Press),
            key(KeyCode::Char('/'), KeyEventKind::Release),
        ];
        let forwarded: Vec<KeyEvent> = keystroke.into_iter().filter(should_handle).collect();
        assert_eq!(forwarded.len(), 1, "a keystroke must be handled exactly once");
        assert_eq!(forwarded[0].kind, KeyEventKind::Press);
    }

    #[test]
    fn unix_press_only_stream_is_untouched_by_the_filter() {
        // Unix backends emit Press for every keystroke, so nothing is dropped there.
        let typed = ['h', 'i'].map(|ch| key(KeyCode::Char(ch), KeyEventKind::Press));
        assert!(typed.iter().all(should_handle));
    }

    #[test]
    fn repeat_is_allowed_so_held_keys_still_auto_repeat() {
        assert!(should_handle(&key(KeyCode::Char('a'), KeyEventKind::Repeat)));
        assert!(should_handle(&key(KeyCode::Down, KeyEventKind::Repeat)));
    }
}
