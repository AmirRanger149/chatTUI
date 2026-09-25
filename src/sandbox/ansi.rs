//! Stripping terminal control sequences out of captured command output.
//!
//! A command's stdout/stderr is captured through pipes, but the bytes in it
//! were written by a program that may believe it owns a terminal: a TUI, a
//! coloured build log, a progress bar. Those bytes include escape sequences
//! that switch screens, hide the cursor, enable mouse reporting and remap
//! keys. Displaying them verbatim in the transcript hands the user's
//! terminal to whatever the agent last ran, and chatTUI's own rendering
//! cannot survive that.
//!
//! So captured output is reduced to plain text before it becomes a tool
//! result: escape sequences are dropped entirely, newlines and tabs are
//! kept, and every other control character is removed.
//!
//! The cost is that colours are lost from build output. That is the right
//! trade: chatTUI styles the transcript itself, and most tools disable
//! colour anyway once they see their output is not a terminal.

/// The 8-bit (single-byte) form of CSI, `ESC [`. Rare, but cheap to handle.
const CSI_8BIT: char = '\u{9b}';
/// BEL terminates an OSC sequence, and is the more common of its two forms
/// (the other being `ESC \`).
const BEL: char = '\u{7}';

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    /// Ordinary text.
    Text,
    /// Saw `ESC`; the next byte says which kind of sequence follows.
    Escape,
    /// Inside a Control Sequence Introducer: parameters, then a final byte.
    Csi,
    /// Inside an Operating System Command (window title, hyperlinks, …).
    Osc,
    /// Inside a string-terminated sequence (DCS, SOS, PM, APC).
    StringSequence,
    /// Saw `ESC` inside a string-terminated sequence; expecting `\`.
    EscapeInString,
}

/// Remove every terminal control sequence and control character from
/// `input`, keeping newlines and tabs.
///
/// Operates on `char`s rather than bytes on purpose: `0x9b` — the 8-bit CSI
/// — is also a valid UTF-8 continuation byte, so a byte-oriented scan would
/// corrupt multi-byte text. Iterating characters keeps that unambiguous.
///
/// A sequence truncated by the end of the input is simply dropped, which is
/// what a truncated capture should do anyway.
pub(crate) fn strip(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut state = State::Text;

    for ch in input.chars() {
        match state {
            State::Text => match ch {
                '\u{1b}' => state = State::Escape,
                CSI_8BIT => state = State::Csi,
                '\n' | '\t' => out.push(ch),
                // Other C0 controls, DEL, and the C1 range: dropped. They
                // carry no meaning in a transcript and can move the cursor.
                _ if ch.is_control() => {}
                _ => out.push(ch),
            },
            State::Escape => {
                state = match ch {
                    '[' => State::Csi,
                    ']' => State::Osc,
                    // DCS, SOS, PM, APC: all run until a String Terminator.
                    'P' | 'X' | '^' | '_' => State::StringSequence,
                    _ => State::Text,
                };
            }
            State::Csi => {
                // Parameter bytes (0x30-0x3f) and intermediates (0x20-0x2f)
                // continue the sequence; a final byte (0x40-0x7e) ends it.
                // Anything else is malformed, so bail out to text rather
                // than swallowing the rest of the output.
                let code = ch as u32;
                if !(0x20..=0x3f).contains(&code) {
                    state = State::Text;
                }
            }
            State::Osc => match ch {
                BEL => state = State::Text,
                '\u{1b}' => state = State::EscapeInString,
                _ => {}
            },
            State::StringSequence => {
                if ch == '\u{1b}' {
                    state = State::EscapeInString;
                }
            }
            State::EscapeInString => {
                // `ESC \` ends the sequence. Anything else is malformed;
                // either way the sequence is over.
                state = State::Text;
            }
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_text_survives_untouched() {
        assert_eq!(strip("cargo test\nall good\tok"), "cargo test\nall good\tok");
    }

    #[test]
    fn colour_codes_are_removed() {
        assert_eq!(strip("\u{1b}[31merror\u{1b}[0m: failed"), "error: failed");
        assert_eq!(strip("\u{1b}[1;33mwarning\u{1b}[m"), "warning");
    }

    #[test]
    fn cursor_and_screen_control_is_removed() {
        // Alternate screen, cursor hide/show, clear, cursor positioning and
        // mouse reporting — the sequences that wreck a host terminal.
        assert_eq!(strip("\u{1b}[?1049h\u{1b}[2J\u{1b}[?25lhi\u{1b}[?25h\u{1b}[?1049l"), "hi");
        assert_eq!(strip("\u{1b}[10;20H\u{1b}[?1000hx"), "x");
    }

    #[test]
    fn osc_sequences_are_removed_whether_bel_or_st_terminated() {
        assert_eq!(strip("\u{1b}]0;window title\u{7}text"), "text");
        assert_eq!(strip("\u{1b}]8;;https://example.com\u{1b}\\link"), "link");
    }

    #[test]
    fn two_character_escapes_are_removed() {
        assert_eq!(strip("\u{1b}Mscroll up\u{1b}7ok"), "scroll upok");
    }

    #[test]
    fn other_control_characters_are_removed_but_layout_is_kept() {
        assert_eq!(strip("a\u{0}b\u{8}c\r\nd"), "abc\nd");
        assert_eq!(strip("line one\nline two\n"), "line one\nline two\n");
    }

    #[test]
    fn a_sequence_truncated_by_end_of_input_is_dropped() {
        assert_eq!(strip("done\u{1b}[31"), "done");
        assert_eq!(strip("done\u{1b}"), "done");
        assert_eq!(strip("done\u{1b}]0;title"), "done");
    }

    #[test]
    fn multibyte_text_is_preserved() {
        // U+065B encodes to bytes D9 9B: a byte-oriented scanner would read
        // the 0x9B as an 8-bit CSI and eat the rest of the line.
        let arabic = "نص\u{65b} عربي";
        assert_eq!(strip(arabic), arabic);
        assert_eq!(strip("日本語\u{1b}[31mテスト\u{1b}[0m"), "日本語テスト");
        assert_eq!(strip("emoji 🦀 ok"), "emoji 🦀 ok");
    }

    #[test]
    fn malformed_sequences_do_not_swallow_the_rest() {
        // A control byte where a parameter was expected ends the sequence
        // instead of consuming everything after it.
        assert_eq!(strip("\u{1b}[\u{1}kept"), "kept");
    }

    #[test]
    fn empty_and_whitespace_only_input() {
        assert_eq!(strip(""), "");
        assert_eq!(strip("\n\n"), "\n\n");
    }
}
