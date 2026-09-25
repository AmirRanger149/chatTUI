//! Slash commands: the command list, the popup shown while a `/` prefix is
//! being typed, and `Enter` dispatch (command vs. chat message).

use crate::app::{App, Cell, Overlay};

pub struct SlashCmd {
    pub name: &'static str,
    pub desc: &'static str,
}

pub const SLASH_COMMANDS: &[SlashCmd] = &[
    SlashCmd { name: "/help", desc: "Show keyboard shortcuts" },
    SlashCmd { name: "/new", desc: "Start a new conversation" },
    SlashCmd { name: "/history", desc: "Browse saved conversations" },
    SlashCmd { name: "/code", desc: "Browse & copy code blocks" },
    SlashCmd { name: "/model", desc: "Pick a model from the API's list" },
    SlashCmd { name: "/provider", desc: "Select API provider" },
    SlashCmd { name: "/agent", desc: "Toggle agent mode (sandbox tools)" },
    SlashCmd { name: "/sandbox", desc: "Show or set agent target dir" },
    SlashCmd { name: "/approve", desc: "Auto-approve tools, or ask first" },
    SlashCmd { name: "/quit", desc: "Exit chatTUI" },
];

impl App {
    /// Tab: complete the highlighted slash command into the composer.
    pub fn accept_slash(&mut self) {
        let Some(name) = self
            .slash_filtered()
            .get(self.slash_selected)
            .map(|command| command.name)
        else {
            return;
        };
        self.set_composer_text(format!("{} ", name));
        self.on_composer_changed();
    }

    pub fn slash_up(&mut self) {
        let len = self.slash_filtered().len();
        if len > 0 {
            self.slash_selected = (self.slash_selected + len - 1) % len;
        }
    }

    pub fn slash_down(&mut self) {
        let len = self.slash_filtered().len();
        if len > 0 {
            self.slash_selected = (self.slash_selected + 1) % len;
        }
    }

    /// The slash popup is open while the composer starts with `/` and an
    /// unambiguous command prefix is being typed.
    pub fn slash_filtered(&self) -> Vec<&'static SlashCmd> {
        if self.overlay.is_some() {
            return Vec::new();
        }
        let text = self.composer.trim_start();
        if !text.starts_with('/') {
            return Vec::new();
        }
        let rest = &text[1..];
        if rest.contains(char::is_whitespace) {
            return Vec::new();
        }
        SLASH_COMMANDS
            .iter()
            .filter(|command| command.name[1..].starts_with(rest))
            .collect()
    }

    pub fn slash_open(&self) -> bool {
        !self.slash_filtered().is_empty()
    }

    /// `Enter`: dispatch the highlighted slash command, run a typed command,
    /// or send the message to the model.
    pub fn submit(&mut self) {
        if self.overlay.is_some() || self.streaming {
            return;
        }
        if self.slash_open() {
            let name = match self.slash_filtered().get(self.slash_selected) {
                Some(command) => command.name.to_string(),
                None => return,
            };
            self.composer.clear();
            self.cursor = 0;
            self.history_nav = None;
            self.run_command(&name);
            return;
        }
        let text = self.composer.trim().to_string();
        if text.is_empty() {
            return;
        }
        self.composer.clear();
        self.cursor = 0;
        self.history_nav = None;
        if text.starts_with('/') {
            self.run_command(&text);
            return;
        }
        self.prompt_history.push(text.clone());
        self.cells.push(Cell::User(text.clone()));
        self.sessions.add_message("user", text);
        if let Err(error) = self.start_stream() {
            self.push_error(error.to_string());
        }
    }

    fn run_command(&mut self, text: &str) {
        let mut parts = text.splitn(2, char::is_whitespace);
        let command = parts
            .next()
            .unwrap_or("")
            .trim()
            .trim_end_matches(':')
            .to_ascii_lowercase();
        let argument = parts.next().unwrap_or("").trim().to_string();
        match command.as_str() {
            "/help" => self.overlay = Some(Overlay::Shortcuts),
            "/new" => self.new_chat(),
            "/history" => self.open_history(),
            "/code" => self.open_code(),
            "/agent" => {
                self.agent_mode = !self.agent_mode;
                let status = if self.agent_mode { "enabled" } else { "disabled" };
                self.push_notice(format!("agent mode {}", status));
                if self.agent_mode && !self.config.sandbox.enabled {
                    self.push_notice("sandbox is disabled in config.json - tools will not work".into());
                } else if self.agent_mode && !self.sandbox.has_target() {
                    self.push_notice(
                        "no target directory — set one with /sandbox <path> before agent writes".into(),
                    );
                }
            }
            "/sandbox" => {
                if argument.is_empty() {
                    self.push_sandbox_status();
                } else if let Err(error) = self.set_sandbox_target(&argument) {
                    self.push_error(error);
                } else {
                    self.push_notice(format!(
                        "sandbox target set to {}",
                        self.sandbox.config.workspace_root.display()
                    ));
                }
            }
            "/approve" => {
                use crate::sandbox::permissions::PermissionMode as Mode;
                let target = match argument.as_str() {
                    // No argument toggles between the two ends, which is what
                    // most people want from it.
                    "" => {
                        if self.sandbox.config.permission_mode == Mode::FullAuto {
                            Mode::AskBeforeWrite
                        } else {
                            Mode::FullAuto
                        }
                    }
                    "auto" | "on" | "yes" => Mode::FullAuto,
                    "manual" | "ask" | "off" | "no" => Mode::AskBeforeWrite,
                    "reset" | "forget" => {
                        self.sandbox.clear_allowances();
                        self.push_notice(
                            "forgot every approval remembered this session".to_string(),
                        );
                        return;
                    }
                    other => {
                        self.push_error(format!(
                            "unknown /approve argument '{other}' — use auto, manual or reset"
                        ));
                        return;
                    }
                };
                self.sandbox.config.permission_mode = target;
                // Approvals were remembered against the old mode; carrying
                // them over would let a "manual" setting silently keep
                // running whatever "auto" had waved through.
                self.sandbox.clear_allowances();
                self.config.sandbox.permission_mode = Some(target.as_str().to_string());
                let blurb = match target {
                    Mode::FullAuto => "auto — every tool runs without asking",
                    Mode::AskBeforeWrite => "manual — reads run free, writes and shell ask first",
                    _ => target.as_str(),
                };
                self.push_notice(format!("approvals: {blurb}"));
            }
            "/model" => {
                if argument.is_empty() {
                    self.open_models();
                } else {
                    // Resolve against the cached model list when one is
                    // available (exact, unique prefix or unique suffix);
                    // anything else is set exactly as typed.
                    let model = self.lookup_model(&argument);
                    self.config.model = model.clone();
                    if !self.models.ids.is_empty()
                        && !self.models.ids.iter().any(|id| *id == model)
                    {
                        self.push_notice(format!(
                            "model set to {model} — not in the API's model list"
                        ));
                    } else {
                        self.push_notice(format!("model set to {model}"));
                    }
                }
            }
            "/provider" => {
                if argument.is_empty() {
                    self.open_providers();
                } else {
                    if let Some(p) = self.config.find_provider(&argument) {
                        self.set_active_provider(&p.id);
                    } else {
                        let available = self
                            .config
                            .providers()
                            .iter()
                            .map(|p| p.id.clone())
                            .collect::<Vec<_>>()
                            .join(", ");
                        self.push_error(format!(
                            "unknown provider: '{argument}' — available: {available}"
                        ));
                    }
                }
            }
            "/quit" => self.should_quit = true,
            other => self.push_error(format!("unknown command: {other} — try /help")),
        }
    }

    pub fn set_sandbox_target(&mut self, raw: &str) -> Result<(), String> {
        match self.sandbox.set_target(raw) {
            Ok(path) => {
                self.config.sandbox.workspace_root = path.display().to_string();
                Ok(())
            }
            Err(error) => Err(error.to_string()),
        }
    }

    fn push_sandbox_status(&mut self) {
        let root = if self.sandbox.has_target() {
            self.sandbox.config.workspace_root.display().to_string()
        } else {
            "(unset — /sandbox <path>)".to_string()
        };
        let status = if self.sandbox.config.enabled { "enabled" } else { "disabled" };
        let agent = if self.agent_mode { "on" } else { "off" };
        self.push_notice(format!(
            "sandbox: {status} | agent: {agent} | root: {root} | mode: {} | shell: {} | shell-timeout: {}s | os-isolation: {} | auto-approve: {}",
            self.sandbox.config.permission_mode.as_str(),
            self.sandbox.config.allow_shell,
            self.sandbox.config.effective_shell_timeout().as_secs(),
            self.sandbox.config.os_isolation.as_str(),
            self.sandbox.config.auto_approve
        ));
        if !self.sandbox.allowances.is_empty() {
            self.push_notice(format!(
                "approved for this session: {} — /approve reset forgets them",
                self.sandbox.allowances.join(", ")
            ));
        }
        if !self.sandbox.has_target() {
            self.push_notice(
                "agent tools will not write until a target is set with /sandbox <dir>".into(),
            );
        }
    }

    pub fn new_chat(&mut self) {
        if self.sessions.current().messages.is_empty() {
            self.push_notice("already in a new conversation".into());
            return;
        }
        self.sessions.new_session();
        self.rebuild_cells();
        self.scroll_from_bottom = 0;
        self.agent_iterations = 0;
        self.pending_tool_calls.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::App;
    use crate::config::Config;
    use crate::session::manager::SessionManager;

    fn test_app() -> App {
        App::new(Config::default(), SessionManager::for_tests())
    }

    #[test]
    fn submit_routes_slash_commands() {
        let mut app = test_app();
        app.composer = "/model test-model".into();
        app.submit();
        assert_eq!(app.config.model, "test-model");
        assert!(matches!(app.cells.last(), Some(Cell::Notice(_))));
        assert!(app.composer.is_empty());
    }

    #[test]
    fn slash_popup_filters_by_prefix() {
        let mut app = test_app();
        app.composer = "/m".into();
        let names: Vec<&str> = app.slash_filtered().iter().map(|c| c.name).collect();
        assert_eq!(names, vec!["/model"]);
        app.composer = "/model something".into();
        assert!(!app.slash_open());
    }

    #[test]
    fn unknown_command_is_reported_not_sent() {
        let mut app = test_app();
        app.composer = "/nope".into();
        app.submit();
        assert!(matches!(app.cells.last(), Some(Cell::Error(_))));
        assert!(app.sessions.current().messages.is_empty());
    }

    #[test]
    fn slash_model_argument_sets_directly_and_resolves() {
        let mut app = test_app();
        app.models.ids = vec!["AcmeAI/acme-model-1".into(), "OtherOrg/other-model".into()];
        // Exact match, case-insensitive.
        app.composer = "/model acmeai/acme-model-1".into();
        app.submit();
        assert_eq!(app.config.model, "AcmeAI/acme-model-1");
        // Unique suffix shorthand.
        app.composer = "/model acme-model-1".into();
        app.submit();
        assert_eq!(app.config.model, "AcmeAI/acme-model-1");
        // Unknown → set exactly as typed.
        app.composer = "/model totally/unknown".into();
        app.submit();
        assert_eq!(app.config.model, "totally/unknown");
        assert!(matches!(app.cells.last(), Some(Cell::Notice(_))));
    }

    #[test]
    fn submit_routes_provider_slash_command() {
        let mut app = test_app();
        // Gateways are declared as custom providers.
        app.config.custom_providers.push(crate::config::CustomProvider {
            id: "acme".into(),
            name: Some("Acme".into()),
            base_url: "https://api.acme.example/v1".into(),
            api_key: None,
            model: Some("acme-model-9".into()),
        });
        app.composer = "/provider acme".into();
        app.submit();
        assert_eq!(app.config.provider, "acme");
        assert_eq!(app.config.base_url, "https://api.acme.example/v1");
        assert_eq!(app.config.model, "acme-model-9");
        assert!(matches!(app.cells.last(), Some(Cell::Notice(_))));
        assert!(app.composer.is_empty());
    }

    #[test]
    fn unknown_provider_reports_error() {
        let mut app = test_app();
        app.composer = "/provider unknown_prov".into();
        app.submit();
        assert!(matches!(app.cells.last(), Some(Cell::Error(_))));
    }

    #[test]
    fn agent_toggle_works() {
        let mut app = test_app();
        let initial = app.agent_mode;
        app.composer = "/agent".into();
        app.submit();
        assert_ne!(app.agent_mode, initial);
        assert!(matches!(app.cells.last(), Some(Cell::Notice(_))));
    }

    #[test]
    fn sandbox_status_does_not_default_to_crate_root() {
        let app = test_app();
        assert!(!app.sandbox.has_target());
        let crate_root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        assert_ne!(app.sandbox.config.workspace_root, crate_root);
    }

    #[test]
    fn sandbox_command_sets_explicit_target() {
        let mut app = test_app();
        let dir = std::env::temp_dir().join(format!("chatTUI_cmd_dst_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        app.composer = format!("/sandbox {}", dir.display());
        app.submit();
        assert!(app.sandbox.has_target());
        let root = app.sandbox.config.workspace_root.canonicalize().unwrap();
        let expected = dir.canonicalize().unwrap();
        assert_eq!(root, expected);
        let crate_root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .canonicalize()
            .unwrap();
        assert_ne!(root, crate_root);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn sandbox_colon_syntax_sets_target() {
        let mut app = test_app();
        let dir = std::env::temp_dir().join(format!("chatTUI_cmd_colon_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        app.composer = format!("/sandbox: {}", dir.display());
        app.submit();
        assert!(app.sandbox.has_target());
        assert_eq!(
            app.sandbox.config.workspace_root.canonicalize().unwrap(),
            dir.canonicalize().unwrap()
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
