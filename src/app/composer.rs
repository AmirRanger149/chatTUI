//! Composer text editing: character insertion, backspace, letter/word cursor
//! navigation, home/end, bracketed paste, and prompt-history recall.

use crate::app::App;

impl App {
    pub fn on_composer_changed(&mut self) {
        self.slash_selected = 0;
    }

    /// Clamp the cursor to a valid char boundary inside `composer`.
    fn clamp_cursor(&mut self) {
        if self.cursor > self.composer.len() || !self.composer.is_char_boundary(self.cursor) {
            self.cursor = self.composer.len();
        }
    }

    /// Replace the composer text and park the cursor at its end.
    pub(crate) fn set_composer_text(&mut self, text: String) {
        self.cursor = text.len();
        self.composer = text;
    }

    pub fn clear_composer(&mut self) {
        self.composer.clear();
        self.cursor = 0;
        self.on_composer_changed();
    }

    /// Insert a character at the cursor position.
    pub fn insert_char(&mut self, ch: char) {
        self.clamp_cursor();
        self.composer.insert(self.cursor, ch);
        self.cursor += ch.len_utf8();
    }

    /// `shift+enter` / `alt+enter`: newline at the cursor position.
    pub fn insert_newline(&mut self) {
        self.insert_char('\n');
    }

    /// Backspace: delete the character before the cursor.
    pub fn backspace(&mut self) {
        self.clamp_cursor();
        if self.cursor == 0 {
            return;
        }
        let prev = self.composer[..self.cursor]
            .char_indices()
            .next_back()
            .map(|(index, _)| index)
            .unwrap_or(0);
        self.composer.replace_range(prev..self.cursor, "");
        self.cursor = prev;
    }

    /// Move the cursor one letter to the left.
    pub fn move_cursor_left(&mut self) {
        self.clamp_cursor();
        if self.cursor == 0 {
            return;
        }
        self.cursor = self.composer[..self.cursor]
            .char_indices()
            .next_back()
            .map(|(index, _)| index)
            .unwrap_or(0);
    }

    /// Move the cursor one letter to the right.
    pub fn move_cursor_right(&mut self) {
        self.clamp_cursor();
        if let Some(ch) = self.composer[self.cursor..].chars().next() {
            self.cursor += ch.len_utf8();
        }
    }

    /// Word navigation: jump to the start of the previous word.
    pub fn move_word_left(&mut self) {
        self.clamp_cursor();
        let mut i = self.cursor;
        // Skip whitespace, then the word itself.
        while i > 0 {
            let Some((index, ch)) = self.composer[..i].char_indices().next_back() else {
                break;
            };
            if !ch.is_whitespace() {
                break;
            }
            i = index;
        }
        while i > 0 {
            let Some((index, ch)) = self.composer[..i].char_indices().next_back() else {
                break;
            };
            if ch.is_whitespace() {
                break;
            }
            i = index;
        }
        self.cursor = i;
    }

    /// Word navigation: jump to the start of the next word.
    pub fn move_word_right(&mut self) {
        self.clamp_cursor();
        let len = self.composer.len();
        let mut i = self.cursor;
        // Skip the current word, then any whitespace after it.
        while i < len {
            let Some(ch) = self.composer[i..].chars().next() else {
                break;
            };
            if ch.is_whitespace() {
                break;
            }
            i += ch.len_utf8();
        }
        while i < len {
            let Some(ch) = self.composer[i..].chars().next() else {
                break;
            };
            if !ch.is_whitespace() {
                break;
            }
            i += ch.len_utf8();
        }
        self.cursor = i;
    }

    /// Home: jump to the start of the composer.
    pub fn move_cursor_home(&mut self) {
        self.cursor = 0;
    }

    /// End: jump to the end of the composer.
    pub fn move_cursor_end(&mut self) {
        self.cursor = self.composer.len();
    }

    pub fn paste(&mut self, text: &str) {
        self.clamp_cursor();
        let cleaned = text.replace("\r\n", "\n");
        self.composer.insert_str(self.cursor, &cleaned);
        self.cursor += cleaned.len();
        self.on_composer_changed();
    }

    pub fn recall_prev(&mut self) {
        if self.prompt_history.is_empty() {
            return;
        }
        match self.history_nav {
            None => {
                let index = self.prompt_history.len() - 1;
                self.history_nav = Some((index, self.composer.clone()));
                let text = self.prompt_history[index].clone();
                self.set_composer_text(text);
            }
            Some((index, ref draft)) => {
                if index > 0 {
                    self.history_nav = Some((index - 1, draft.clone()));
                    let text = self.prompt_history[index - 1].clone();
                    self.set_composer_text(text);
                }
            }
        }
    }

    pub fn recall_next(&mut self) {
        let Some((index, draft)) = self.history_nav.take() else {
            return;
        };
        if index + 1 < self.prompt_history.len() {
            self.history_nav = Some((index + 1, draft));
            let text = self.prompt_history[index + 1].clone();
            self.set_composer_text(text);
        } else {
            self.set_composer_text(draft);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::app::App;
    use crate::config::Config;
    use crate::session::manager::SessionManager;

    fn test_app() -> App {
        App::new(Config::default(), SessionManager::for_tests())
    }

    #[test]
    fn prompt_history_recalls_and_restores_draft() {
        let mut app = test_app();
        app.composer = "first".into();
        app.prompt_history.push("first".into());
        app.composer = "draft".into();
        app.recall_prev();
        assert_eq!(app.composer, "first");
        app.recall_next();
        assert_eq!(app.composer, "draft");
    }

    #[test]
    fn letter_navigation_moves_one_char_at_a_time() {
        let mut app = test_app();
        app.set_composer_text("aé b".into()); // 'é' is 2 bytes, len = 5
        assert_eq!(app.cursor, 5);
        app.move_cursor_left();
        assert_eq!(app.cursor, 4); // before 'b'
        app.move_cursor_left();
        assert_eq!(app.cursor, 3); // before ' '
        app.move_cursor_left();
        assert_eq!(app.cursor, 1); // before 'é', on its byte boundary
        app.move_cursor_left();
        assert_eq!(app.cursor, 0);
        app.move_cursor_left();
        assert_eq!(app.cursor, 0); // stays at the start
        app.move_cursor_right();
        assert_eq!(app.cursor, 1);
        app.move_cursor_right();
        assert_eq!(app.cursor, 3); // 'é' is skipped as one letter
        app.move_cursor_right();
        app.move_cursor_right();
        app.move_cursor_right();
        assert_eq!(app.cursor, 5); // stays at the end
    }

    #[test]
    fn word_navigation_jumps_between_words() {
        let mut app = test_app();
        app.set_composer_text("hello brave new world".into());
        app.move_word_left();
        assert_eq!(app.cursor, 16); // start of "world"
        app.move_word_left();
        assert_eq!(app.cursor, 12); // start of "new"
        app.move_word_right();
        assert_eq!(app.cursor, 16); // end of "new"
        app.move_word_right();
        assert_eq!(app.cursor, 21); // end of "world"
        app.move_word_right();
        assert_eq!(app.cursor, 21); // stays at the end
        app.move_cursor_home();
        assert_eq!(app.cursor, 0);
        app.move_cursor_end();
        assert_eq!(app.cursor, 21);
    }

    #[test]
    fn editing_happens_at_cursor() {
        let mut app = test_app();
        app.set_composer_text("hello world".into());
        app.move_word_left(); // cursor at 6, before "world"
        app.insert_char('X');
        assert_eq!(app.composer, "hello Xworld");
        assert_eq!(app.cursor, 7);
        app.backspace();
        assert_eq!(app.composer, "hello world");
        assert_eq!(app.cursor, 6);
        app.paste("pasted\n");
        assert_eq!(app.composer, "hello pasted\nworld");
    }
}
