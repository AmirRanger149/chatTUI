//! Root layout: a scrollback transcript filling the screen, with a borderless
//! bottom pane beneath it holding the slash-command popup, the optional
//! `Working` row, the `› ` composer, and a hint/context footer.

pub mod composer;
pub mod markdown;
pub mod overlay;
pub mod status;
pub mod theme;
pub mod thinking;
pub mod transcript;

use crate::app::App;
use ratatui::prelude::*;
use ratatui::widgets::Paragraph;

pub fn render(frame: &mut Frame, app: &App) {
    let area = frame.area();
    if area.width == 0 || area.height == 0 {
        return;
    }
    let width = area.width;

    let popup_h = if app.slash_open() {
        composer::slash_popup_lines(app, width as usize).len() as u16
    } else {
        0
    };
    let working_h = u16::from(app.streaming);
    let prompt_h = composer::row_count(app, width).max(1);
    // [popup][working][gap][prompt][gap][footer]
    let bottom_h = (popup_h + working_h + 1 + prompt_h + 1 + 1).min(area.height.saturating_sub(1));

    let [transcript_area, bottom] =
        Layout::vertical([Constraint::Min(1), Constraint::Length(bottom_h)]).areas(area);

    transcript::render(frame, transcript_area, app);

    let mut y = bottom.y;
    let row = |y: u16, height: u16| -> Rect {
        Rect {
            x: bottom.x,
            y,
            width: bottom.width,
            height,
        }
    };

    if popup_h > 0 {
        let lines = composer::slash_popup_lines(app, width as usize);
        frame.render_widget(Paragraph::new(lines), row(y, popup_h.min(bottom.bottom() - y)));
        y += popup_h;
    }
    if working_h > 0 && y < bottom.bottom() {
        status::render(frame, row(y, 1), app);
        y += 1;
    }
    y = (y + 1).min(bottom.bottom()); // gap
    if y < bottom.bottom() {
        let prompt_area = row(y, prompt_h.min(bottom.bottom() - y));
        if let Some(cursor) = composer::render_prompt(frame, prompt_area, app) {
            frame.set_cursor_position(cursor);
        }
        y += prompt_area.height;
    }
    y = (y + 1).min(bottom.bottom()); // gap
    if y < bottom.bottom() {
        composer::render_footer(frame, row(y, 1), app);
    }

    if let Some(overlay) = app.overlay {
        overlay::render(frame, area, app, overlay);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::session::manager::SessionManager;
    use ratatui::backend::TestBackend;

    #[test]
    fn renders_full_ui_without_panicking() {
        let app = App::new(Config::default(), SessionManager::for_tests());
        let mut terminal = Terminal::new(TestBackend::new(100, 30)).unwrap();
        terminal
            .draw(|frame| render(frame, &app))
            .expect("root render should succeed");
    }

    #[test]
    fn renders_ui_in_every_state() {
        let mut app = App::new(Config::default(), SessionManager::for_tests());
        let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();

        // slash popup open
        app.composer = "/he".into();
        terminal.draw(|frame| render(frame, &app)).unwrap();

        // streaming state
        app.composer.clear();
        app.streaming = true;
        app.response = "streaming **response**".into();
        terminal.draw(|frame| render(frame, &app)).unwrap();

        // streaming reasoning (<think> …) and a finished, collapsed block
        app.response = "<think>weighing the options".into();
        terminal.draw(|frame| render(frame, &app)).unwrap();
        app.streaming = false;
        app.response.clear();
        app.cells.push(crate::app::Cell::Assistant(
            "<think>because of X</think>Here is the answer.".into(),
        ));
        terminal.draw(|frame| render(frame, &app)).unwrap();
        app.show_thinking = true;
        terminal.draw(|frame| render(frame, &app)).unwrap();
        app.show_thinking = false;

        // overlays
        app.overlay = Some(crate::app::Overlay::Shortcuts);
        terminal.draw(|frame| render(frame, &app)).unwrap();
        app.open_history();
        terminal.draw(|frame| render(frame, &app)).unwrap();

        // code overlay
        app.cells.push(crate::app::Cell::Assistant(
            "```rust\nlet x = 1;\n```".into(),
        ));
        app.open_code();
        terminal.draw(|frame| render(frame, &app)).unwrap();

        // models overlay (loading + loaded)
        app.overlay = Some(crate::app::Overlay::Models { selected: 0 });
        app.models.loading = true;
        terminal.draw(|frame| render(frame, &app)).unwrap();
        app.models.loading = false;
        app.models.ids = vec!["MiniMaxAI/MiniMax-M2.7".into()];
        terminal.draw(|frame| render(frame, &app)).unwrap();
        app.models.ids.clear();

        // providers overlay
        app.overlay = Some(crate::app::Overlay::Providers { selected: 0 });
        terminal.draw(|frame| render(frame, &app)).unwrap();

        // tiny terminal
        app.overlay = None;
        let mut tiny = Terminal::new(TestBackend::new(20, 6)).unwrap();
        tiny.draw(|frame| render(frame, &app)).unwrap();
    }
}
