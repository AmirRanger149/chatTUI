//! Clipboard support for copying code blocks.
//!
//! Strategy: copy through a real system-clipboard utility whenever one works
//! (its exit status proves the bytes landed), and only then fall back to the
//! terminal's OSC 52 escape sequence. OSC 52 is fire and forget: the terminal
//! may quietly ignore it (missing tmux/screen passthrough, permission prompts,
//! length limits), so that path is always reported as best-effort, with a hint
//! on how to get reliable copies.

use anyhow::Result;
use std::borrow::Cow;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// How long to wait for a single clipboard utility before assuming it hung
/// (stale `$DISPLAY` / `$WAYLAND_DISPLAY` pointing at a dead socket) and
/// moving on to the next candidate. Utilities normally return in milliseconds.
const COMMAND_TIMEOUT: Duration = Duration::from_secs(2);

/// Outcome of a copy attempt.
pub struct CopyOutcome {
    /// Short description of the method used, e.g. `"system clipboard (wl-copy)"`.
    pub method: String,
    /// True only when bytes verifiably reached a clipboard the user can paste
    /// from. The OSC 52 fallback is one-way and the terminal never confirms
    /// anything, so it is always unverified.
    pub verified: bool,
    /// Advice shown when the copy is best-effort (OSC 52 may have been ignored).
    pub hint: Option<String>,
}

/// Copy `text` to the clipboard.
pub fn copy(text: &str) -> Result<CopyOutcome> {
    let env = Env::gather();
    let system = platform_commands(&env).iter().find_map(|argv| try_command(argv, text));

    if env.ssh {
        // Over SSH a remote clipboard utility copies to the *remote* machine,
        // which the user cannot paste from, but the terminal forwards OSC 52
        // to the machine in front of them. So always emit OSC 52 there, even
        // when a remote tool also succeeded.
        write_osc52(text, &env)?;
        return match system {
            Some(method) => Ok(CopyOutcome {
                method: format!("{method} + terminal clipboard (OSC 52)"),
                verified: false,
                hint: install_hint(&env),
            }),
            None => Ok(osc52_outcome(&env)),
        };
    }

    match system {
        Some(method) => Ok(CopyOutcome { method, verified: true, hint: None }),
        None => {
            write_osc52(text, &env)?;
            Ok(osc52_outcome(&env))
        }
    }
}

/// Best-effort outcome for the OSC 52 fallback path.
fn osc52_outcome(env: &Env) -> CopyOutcome {
    CopyOutcome {
        method: "terminal clipboard (OSC 52)".to_string(),
        verified: false,
        hint: install_hint(env),
    }
}

fn install_hint(env: &Env) -> Option<String> {
    if env.ssh {
        return Some("SSH session: local paste needs terminal OSC 52 support".to_string());
    }
    if cfg!(target_os = "macos") || cfg!(target_os = "windows") {
        None
    } else if env.wayland {
        Some("install wl-copy for reliable copies".to_string())
    } else if env.x11 {
        Some("install xclip or xsel for reliable copies".to_string())
    } else {
        Some("install wl-copy (Wayland) or xclip (X11) for reliable copies".to_string())
    }
}

/// The slice of process environment that influences clipboard selection.
/// Gathered once per copy so the selection logic stays pure and testable.
#[derive(Default)]
struct Env {
    /// `$WAYLAND_DISPLAY` is set.
    wayland: bool,
    /// `$DISPLAY` is set.
    x11: bool,
    /// Running inside tmux (`$TMUX`, or a tmux `$TERM`).
    tmux: bool,
    /// Running inside GNU screen (a `screen*` `$TERM` without tmux).
    screen: bool,
    /// Running over SSH.
    ssh: bool,
    /// Running inside WSL (`/proc/version` mentions Microsoft).
    wsl: bool,
}

impl Env {
    fn gather() -> Self {
        let is_set = |key: &str| std::env::var_os(key).is_some_and(|v| !v.is_empty());
        let term = std::env::var("TERM").unwrap_or_default();
        let tmux = is_set("TMUX") || term.starts_with("tmux");
        Self {
            wayland: is_set("WAYLAND_DISPLAY"),
            x11: is_set("DISPLAY"),
            tmux,
            screen: !tmux && term.starts_with("screen"),
            ssh: is_set("SSH_TTY") || is_set("SSH_CONNECTION") || is_set("SSH_CLIENT"),
            wsl: is_wsl(),
        }
    }
}

#[cfg(target_os = "linux")]
fn is_wsl() -> bool {
    std::fs::read_to_string("/proc/version")
        .map(|version| version.to_ascii_lowercase().contains("microsoft"))
        .unwrap_or(false)
}

#[cfg(not(target_os = "linux"))]
fn is_wsl() -> bool {
    false
}

/// Candidate clipboard utilities in preference order: the native session
/// type first, then platform extras (WSL / Termux), then blind retries for
/// when the session env is missing or stale but a tool might still work
/// (e.g. tmux not refreshing the environment on attach).
fn platform_commands(env: &Env) -> Vec<Vec<String>> {
    if cfg!(target_os = "macos") {
        return vec![vec!["pbcopy".to_string()]];
    }
    if cfg!(target_os = "windows") {
        return vec![vec!["clip".to_string()]];
    }
    let mut candidates = Vec::new();
    if env.wayland {
        candidates.push(vec!["wl-copy".to_string()]);
    }
    if env.x11 {
        candidates.push(vec![
            "xclip".to_string(),
            "-selection".to_string(),
            "clipboard".to_string(),
        ]);
        candidates.push(vec![
            "xsel".to_string(),
            "--clipboard".to_string(),
            "--input".to_string(),
        ]);
    }
    if env.wsl {
        candidates.push(vec!["clip.exe".to_string()]);
    }
    candidates.push(vec!["termux-clipboard-set".to_string()]);
    if !env.wayland {
        candidates.push(vec!["wl-copy".to_string()]);
    }
    if !env.x11 {
        candidates.push(vec![
            "xclip".to_string(),
            "-selection".to_string(),
            "clipboard".to_string(),
        ]);
        candidates.push(vec![
            "xsel".to_string(),
            "--clipboard".to_string(),
            "--input".to_string(),
        ]);
    }
    candidates
}

/// Windows' `clip` expects CRLF line endings; every other tool takes the text as-is.
fn clipboard_payload<'a>(program: &str, text: &'a str) -> Cow<'a, str> {
    if program == "clip" || program == "clip.exe" {
        Cow::Owned(text.replace("\r\n", "\n").replace('\n', "\r\n"))
    } else {
        Cow::Borrowed(text)
    }
}

/// Pipe `text` into a clipboard utility; report the method on success.
/// Returns `None` when the tool is missing, fails, or hangs past
/// [`COMMAND_TIMEOUT`]; the caller then tries the next candidate.
fn try_command(argv: &[String], text: &str) -> Option<String> {
    let (program, args) = argv.split_first()?;
    let mut child = Command::new(program)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    if let Some(mut stdin) = child.stdin.take() {
        // Ignore pipe errors: the tool may have exited early, successfully or
        // not. Its exit status below is the verdict, not the pipe.
        let _ = stdin.write_all(clipboard_payload(program, text).as_bytes());
    }
    let started = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                return status.success().then(|| format!("system clipboard ({program})"));
            }
            Ok(None) => {
                if started.elapsed() >= COMMAND_TIMEOUT {
                    let _ = child.kill();
                    let _ = child.wait();
                    return None;
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            Err(_) => return None,
        }
    }
}

/// OSC 52: ask the terminal itself to store the text in the system clipboard.
/// Best-effort: the terminal may ignore it, so callers have to report this
/// path as unverified.
fn write_osc52(text: &str, env: &Env) -> Result<()> {
    let sequence = osc52_sequence(&base64(text.as_bytes()), env);
    let mut stdout = std::io::stdout().lock();
    stdout.write_all(&sequence)?;
    stdout.flush()?;
    Ok(())
}

/// Build the OSC 52 sequence, wrapped in DCS passthrough when running inside
/// tmux or GNU screen so the multiplexer forwards it to the real terminal
/// instead of swallowing it.
fn osc52_sequence(payload: &str, env: &Env) -> Vec<u8> {
    let inner = format!("\x1b]52;c;{payload}\x07");
    if env.tmux {
        format!("\x1bPtmux;{}\x1b\\", inner.replace('\x1b', "\x1b\x1b")).into_bytes()
    } else if env.screen {
        format!("\x1bP{inner}\x1b\\").into_bytes()
    } else {
        inner.into_bytes()
    }
}

/// Minimal standard base64 encoder, so we do not need another dependency.
fn base64(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] =
        b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity((bytes.len() + 2) / 3 * 4);
    for chunk in bytes.chunks(3) {
        let n = ((chunk[0] as u32) << 16)
            | ((*chunk.get(1).unwrap_or(&0) as u32) << 8)
            | (*chunk.get(2).unwrap_or(&0) as u32);
        out.push(ALPHABET[(n >> 18) as usize & 63] as char);
        out.push(ALPHABET[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 {
            ALPHABET[(n >> 6) as usize & 63] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            ALPHABET[n as usize & 63] as char
        } else {
            '='
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_matches_rfc_vectors() {
        assert_eq!(base64(b""), "");
        assert_eq!(base64(b"f"), "Zg==");
        assert_eq!(base64(b"fo"), "Zm8=");
        assert_eq!(base64(b"foo"), "Zm9v");
        assert_eq!(base64(b"foob"), "Zm9vYg==");
        assert_eq!(base64(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn osc52_goes_directly_outside_multiplexers() {
        let sequence = osc52_sequence("Zm9v", &Env::default());
        assert_eq!(sequence, b"\x1b]52;c;Zm9v\x07");
    }

    #[test]
    fn osc52_wraps_for_tmux_passthrough() {
        let env = Env { tmux: true, ..Env::default() };
        let sequence = osc52_sequence("Zm9v", &env);
        assert_eq!(sequence, b"\x1bPtmux;\x1b\x1b]52;c;Zm9v\x07\x1b\\");
    }

    #[test]
    fn osc52_wraps_for_screen_passthrough() {
        let env = Env { screen: true, ..Env::default() };
        let sequence = osc52_sequence("Zm9v", &env);
        assert_eq!(sequence, b"\x1bP\x1b]52;c;Zm9v\x07\x1b\\");
    }

    #[test]
    fn clip_gets_crlf_line_endings() {
        assert_eq!(clipboard_payload("clip", "a\nb\r\nc"), "a\r\nb\r\nc");
        assert_eq!(clipboard_payload("clip.exe", "a\nb"), "a\r\nb");
        assert_eq!(clipboard_payload("wl-copy", "a\nb"), "a\nb");
    }

    #[cfg(target_os = "linux")]
    mod linux_selection {
        use super::*;

        fn programs(env: &Env) -> Vec<String> {
            platform_commands(env).iter().map(|argv| argv[0].clone()).collect()
        }

        #[test]
        fn wayland_prefers_wl_copy() {
            let env = Env { wayland: true, ..Env::default() };
            let order = programs(&env);
            assert_eq!(order[0], "wl-copy");
            // X11 tools are still retried blind in case the session is mixed.
            assert!(order.contains(&"xclip".to_string()));
        }

        #[test]
        fn x11_prefers_xclip_then_xsel() {
            let env = Env { x11: true, ..Env::default() };
            let order = programs(&env);
            assert_eq!(order[0], "xclip");
            assert_eq!(order[1], "xsel");
        }

        #[test]
        fn wsl_falls_back_to_clip_exe() {
            let env = Env { wsl: true, ..Env::default() };
            assert!(programs(&env).contains(&"clip.exe".to_string()));
        }

        #[test]
        fn empty_env_still_retries_everything_blind() {
            let order = programs(&Env::default());
            for tool in ["wl-copy", "xclip", "xsel", "termux-clipboard-set"] {
                assert!(order.contains(&tool.to_string()), "missing {tool}");
            }
        }
    }

    #[cfg(unix)]
    mod command_results {
        use super::*;

        fn argv(program: &str, args: &[&str]) -> Vec<String> {
            std::iter::once(program).chain(args.iter().copied()).map(str::to_string).collect()
        }

        #[test]
        fn exit_zero_counts_as_success() {
            assert_eq!(
                try_command(&argv("cat", &[]), "hello").as_deref(),
                Some("system clipboard (cat)")
            );
        }

        #[test]
        fn nonzero_exit_falls_through() {
            assert_eq!(try_command(&argv("false", &[]), "hello"), None);
        }

        #[test]
        fn missing_tool_falls_through() {
            assert_eq!(try_command(&argv("chat-tui-no-such-tool", &[]), "hello"), None);
        }

        #[test]
        fn hung_tool_times_out_instead_of_blocking_forever() {
            let started = Instant::now();
            assert_eq!(try_command(&argv("sleep", &["30"]), "hello"), None);
            assert!(started.elapsed() < Duration::from_secs(10));
        }
    }
}
