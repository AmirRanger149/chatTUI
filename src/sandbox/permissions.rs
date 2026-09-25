//! Minimal permission layer for agent tool execution.
//!
//! This is an *application-level policy*, not an OS security boundary: it
//! decides which classes of tools may run at all, and which ones have to go
//! in front of the user first. The kernel sandbox is the actual boundary;
//! this layer is what keeps the agent inside the workspace by default.

use anyhow::Result;

/// The class of capability a tool needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolKind {
    /// `read_file`, `list_files`.
    Read,
    /// `write_file`.
    Write,
    /// `edit_file`.
    Edit,
    /// `bash`.
    Shell,
    /// `ask_user` — it talks to the human, so it needs no capability and is
    /// never gated.
    Ask,
}

impl ToolKind {
    pub fn as_str(self) -> &'static str {
        match self {
            ToolKind::Read => "read",
            ToolKind::Write => "write",
            ToolKind::Edit => "edit",
            ToolKind::Shell => "shell",
            ToolKind::Ask => "ask",
        }
    }

    /// Map a tool name from [`crate::tools`] to its capability class.
    /// Unknown tools return `None` (and are refused by the executor).
    pub fn from_tool_name(name: &str) -> Option<ToolKind> {
        match name {
            "read_file" | "list_files" => Some(ToolKind::Read),
            "write_file" => Some(ToolKind::Write),
            "edit_file" | "apply_patch" => Some(ToolKind::Edit),
            "bash" | "write_stdin" | "kill_session" => Some(ToolKind::Shell),
            "ask_user" => Some(ToolKind::Ask),
            _ => None,
        }
    }
}

/// Application-level permission mode.
///
/// Enabling agent mode never widens permissions by itself: the mode comes
/// from configuration only. `Ask*` modes model "ask before doing X"; the
/// interactive approval prompt is not implemented yet, so those requests
/// are **denied** with an explanatory error rather than silently allowed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PermissionMode {
    /// Only read-class tools.
    ReadOnly,
    /// Read + write/edit inside the workspace; no shell.
    WorkspaceWrite,
    /// Read; writes would ask first (currently denied).
    AskBeforeWrite,
    /// Read + write; shell would ask first (currently denied).
    AskBeforeShell,
    /// Everything the master switches (`allow_shell`, target set) permit.
    FullAuto,
}

impl PermissionMode {
    /// Parse the `sandbox.permission_mode` config value.
    pub fn parse(raw: &str) -> Option<PermissionMode> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "read-only" | "readonly" => Some(PermissionMode::ReadOnly),
            "workspace-write" => Some(PermissionMode::WorkspaceWrite),
            "ask-before-write" => Some(PermissionMode::AskBeforeWrite),
            "ask-before-shell" => Some(PermissionMode::AskBeforeShell),
            "full-auto" | "fullauto" => Some(PermissionMode::FullAuto),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            PermissionMode::ReadOnly => "read-only",
            PermissionMode::WorkspaceWrite => "workspace-write",
            PermissionMode::AskBeforeWrite => "ask-before-write",
            PermissionMode::AskBeforeShell => "ask-before-shell",
            PermissionMode::FullAuto => "full-auto",
        }
    }

    /// Backward-compatible default for configs that predate
    /// `permission_mode`: the legacy flags keep their original meaning.
    pub fn from_legacy_flags(auto_approve: bool, allow_shell: bool) -> PermissionMode {
        match (auto_approve, allow_shell) {
            (false, _) => PermissionMode::AskBeforeWrite,
            (true, true) => PermissionMode::FullAuto,
            (true, false) => PermissionMode::WorkspaceWrite,
        }
    }

    /// Whether `kind` runs without asking in this mode.
    pub fn allows(self, kind: ToolKind) -> bool {
        matches!(gate(self, kind, "", None, &[]), Gate::Allow)
    }

    /// True when this mode puts `kind` in front of the user.
    pub fn asks(self, kind: ToolKind) -> bool {
        matches!(gate(self, kind, "", None, &[]), Gate::Ask { .. })
    }

    /// How to change the mode so `kind` becomes allowed.
    pub fn remedy(self) -> &'static str {
        match self {
            PermissionMode::ReadOnly => {
                "use 'workspace-write' to allow file tools or 'full-auto' to also allow shell"
            }
            PermissionMode::WorkspaceWrite => {
                "shell additionally needs 'full-auto' (and sandbox.allow_shell = true)"
            }
            PermissionMode::AskBeforeWrite | PermissionMode::AskBeforeShell => {
                "this mode asks first — approve the call, or set 'workspace-write' (file tools) or 'full-auto' (file tools + shell) in config.json"
            }
            PermissionMode::FullAuto => "nothing to change: this mode allows everything",
        }
    }
}

/// What the permission layer decides about one call, before it runs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Gate {
    /// Run it.
    Allow,
    /// Never run it in this mode.
    Deny { reason: String },
    /// The user decides.
    Ask { reason: String },
}

/// The key an approval is remembered under: the tool name for file tools,
/// the first two words of the command for shell — so approving one
/// `git push` covers the next one without approving every single command.
pub fn allowance_key(tool_name: &str, command: Option<&str>) -> String {
    match (tool_name, command) {
        ("bash", Some(command)) => {
            let words: Vec<&str> = command.split_whitespace().take(2).collect();
            if words.is_empty() {
                "bash".to_string()
            } else {
                words.join(" ")
            }
        }
        _ => tool_name.to_string(),
    }
}

/// Decide what one call may do.
///
/// The mode decides what is *possible*; `allowed` holds what the user has
/// already approved this session. Both halves are needed: without the second
/// one an ask-mode has only two real settings — deny everything or allow
/// everything — which is why those modes went unused before.
pub fn gate(
    mode: PermissionMode,
    kind: ToolKind,
    tool_name: &str,
    command: Option<&str>,
    allowed: &[String],
) -> Gate {
    // Talking to the user needs no capability of its own.
    if kind == ToolKind::Ask {
        return Gate::Allow;
    }

    match mode {
        PermissionMode::ReadOnly => {
            if kind != ToolKind::Read {
                return Gate::Deny {
                    reason: format!(
                        "permission mode is read-only and {} needs {}",
                        tool_name,
                        kind.as_str()
                    ),
                };
            }
            return Gate::Allow;
        }
        PermissionMode::WorkspaceWrite => {
            if kind == ToolKind::Shell {
                return Gate::Deny {
                    reason: format!(
                        "permission mode is workspace-write, which does not allow {tool_name}"
                    ),
                };
            }
            return Gate::Allow;
        }
        PermissionMode::FullAuto => return Gate::Allow,
        PermissionMode::AskBeforeWrite | PermissionMode::AskBeforeShell => {}
    }

    // Ask modes: reads are always free, and what else asks depends on which
    // ask-mode we are in.
    let needs_approval = match mode {
        PermissionMode::AskBeforeShell => kind == ToolKind::Shell,
        // AskBeforeWrite is the strict one: anything that changes state or
        // runs a command goes in front of the user.
        _ => kind != ToolKind::Read,
    };
    if !needs_approval {
        return Gate::Allow;
    }

    let key = allowance_key(tool_name, command);
    if allowed.iter().any(|entry| *entry == key) {
        return Gate::Allow;
    }
    Gate::Ask {
        reason: format!("{} needs approval in {} mode", tool_name, mode.as_str()),
    }
}

/// One line for the approval prompt: what the call is about to do.
pub fn describe_call(tool_name: &str, command: Option<&str>, target: Option<&str>) -> String {
    match tool_name {
        "bash" => format!("run: {}", command.unwrap_or("(empty)")),
        "write_stdin" => format!("send input to session {}", target.unwrap_or("?")),
        "kill_session" => format!("kill session {}", target.unwrap_or("?")),
        "apply_patch" => "apply a patch".to_string(),
        other => format!("{other}: {}", target.unwrap_or("(no path)")),
    }
}

/// Refuse `kind` when the mode rules it out entirely.
///
/// Whether a call still needs the user's say-so is *not* decided here: that
/// question is answered once, by [`gate`], before the call is dispatched. By
/// the time an individual tool runs it has already been approved, so an ask
/// mode must not deny it a second time.
pub(crate) fn authorize(mode: PermissionMode, kind: ToolKind) -> Result<()> {
    match gate(mode, kind, "", None, &[]) {
        Gate::Deny { reason } => Err(anyhow::anyhow!(
            "permission denied: {reason}. {}",
            mode.remedy()
        )),
        Gate::Allow | Gate::Ask { .. } => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mode_parses_all_documented_names() {
        for name in [
            "read-only",
            "workspace-write",
            "ask-before-write",
            "ask-before-shell",
            "full-auto",
        ] {
            assert_eq!(PermissionMode::parse(name).unwrap().as_str(), name);
        }
        assert_eq!(PermissionMode::parse("FULL-AUTO"), Some(PermissionMode::FullAuto));
        assert_eq!(PermissionMode::parse("yolo"), None);
        assert_eq!(PermissionMode::parse(""), None);
    }

    #[test]
    fn modes_gate_capability_classes() {
        assert!(PermissionMode::ReadOnly.allows(ToolKind::Read));
        assert!(!PermissionMode::ReadOnly.allows(ToolKind::Write));
        assert!(!PermissionMode::ReadOnly.allows(ToolKind::Shell));

        assert!(PermissionMode::WorkspaceWrite.allows(ToolKind::Write));
        assert!(PermissionMode::WorkspaceWrite.allows(ToolKind::Edit));
        assert!(!PermissionMode::WorkspaceWrite.allows(ToolKind::Shell));

        assert!(PermissionMode::AskBeforeShell.allows(ToolKind::Write));
        assert!(!PermissionMode::AskBeforeShell.allows(ToolKind::Shell));
        assert!(PermissionMode::AskBeforeShell.asks(ToolKind::Shell));
        assert!(!PermissionMode::WorkspaceWrite.asks(ToolKind::Shell));

        assert!(PermissionMode::FullAuto.allows(ToolKind::Shell));
    }

    #[test]
    fn legacy_flags_map_to_modes_without_changing_old_behavior() {
        // auto_approve + allow_shell (the historical defaults) stay full-auto.
        assert_eq!(
            PermissionMode::from_legacy_flags(true, true),
            PermissionMode::FullAuto
        );
        assert_eq!(
            PermissionMode::from_legacy_flags(true, false),
            PermissionMode::WorkspaceWrite
        );
        assert_eq!(
            PermissionMode::from_legacy_flags(false, true),
            PermissionMode::AskBeforeWrite
        );
        assert_eq!(
            PermissionMode::from_legacy_flags(false, false),
            PermissionMode::AskBeforeWrite
        );
    }

    #[test]
    fn every_agent_tool_maps_to_a_capability_class() {
        for tool in crate::tools::all_tools() {
            assert!(
                ToolKind::from_tool_name(&tool.name).is_some(),
                "tool '{}' has no permission class",
                tool.name
            );
        }
        assert_eq!(ToolKind::from_tool_name("read_file"), Some(ToolKind::Read));
        assert_eq!(ToolKind::from_tool_name("list_files"), Some(ToolKind::Read));
        assert_eq!(ToolKind::from_tool_name("write_file"), Some(ToolKind::Write));
        assert_eq!(ToolKind::from_tool_name("edit_file"), Some(ToolKind::Edit));
        assert_eq!(ToolKind::from_tool_name("bash"), Some(ToolKind::Shell));
        assert_eq!(ToolKind::from_tool_name("ask_user"), Some(ToolKind::Ask));
        assert_eq!(ToolKind::from_tool_name("apply_patch"), Some(ToolKind::Edit));
        assert_eq!(ToolKind::from_tool_name("nope"), None);
    }

    #[test]
    fn ask_modes_ask_instead_of_denying() {
        // The strict mode puts writes and shell in front of the user…
        assert!(matches!(
            gate(
                PermissionMode::AskBeforeWrite,
                ToolKind::Write,
                "write_file",
                None,
                &[]
            ),
            Gate::Ask { .. }
        ));
        // …and the shell-only mode leaves writes alone.
        assert!(matches!(
            gate(
                PermissionMode::AskBeforeShell,
                ToolKind::Write,
                "write_file",
                None,
                &[]
            ),
            Gate::Allow
        ));
        // Reads are never gated, in any mode that allows them.
        assert!(matches!(
            gate(
                PermissionMode::AskBeforeWrite,
                ToolKind::Read,
                "read_file",
                None,
                &[]
            ),
            Gate::Allow
        ));
    }

    #[test]
    fn an_approval_is_remembered_per_command_prefix() {
        let allowed = vec!["git push".to_string(), "write_file".to_string()];
        // Same first two words → already approved.
        assert!(matches!(
            gate(
                PermissionMode::AskBeforeShell,
                ToolKind::Shell,
                "bash",
                Some("git push --force origin main"),
                &allowed
            ),
            Gate::Allow
        ));
        // Different command → asked again. Approving one thing must not
        // quietly approve everything else.
        assert!(matches!(
            gate(
                PermissionMode::AskBeforeShell,
                ToolKind::Shell,
                "bash",
                Some("rm -rf target"),
                &allowed
            ),
            Gate::Ask { .. }
        ));
        // A file tool is remembered by name.
        assert!(matches!(
            gate(
                PermissionMode::AskBeforeWrite,
                ToolKind::Write,
                "write_file",
                None,
                &allowed
            ),
            Gate::Allow
        ));
    }

    #[test]
    fn the_allowance_key_is_the_first_two_words() {
        assert_eq!(allowance_key("bash", Some("cargo test --release")), "cargo test");
        assert_eq!(allowance_key("bash", Some("ls")), "ls");
        assert_eq!(allowance_key("bash", Some("   ")), "bash");
        assert_eq!(allowance_key("write_file", None), "write_file");
    }

    #[test]
    fn asking_the_user_needs_no_capability() {
        for mode in [
            PermissionMode::ReadOnly,
            PermissionMode::WorkspaceWrite,
            PermissionMode::AskBeforeWrite,
            PermissionMode::AskBeforeShell,
            PermissionMode::FullAuto,
        ] {
            assert!(
                matches!(gate(mode, ToolKind::Ask, "ask_user", None, &[]), Gate::Allow),
                "{mode:?} should let the model ask the user"
            );
        }
    }

    #[test]
    fn full_auto_allows_and_read_only_denies_without_asking() {
        assert!(matches!(
            gate(PermissionMode::FullAuto, ToolKind::Shell, "bash", Some("ls"), &[]),
            Gate::Allow
        ));
        assert!(matches!(
            gate(PermissionMode::ReadOnly, ToolKind::Write, "write_file", None, &[]),
            Gate::Deny { .. }
        ));
        assert!(matches!(
            gate(PermissionMode::WorkspaceWrite, ToolKind::Shell, "bash", Some("ls"), &[]),
            Gate::Deny { .. }
        ));
    }

    #[test]
    fn denial_mentions_the_config_knob() {
        let err = authorize(PermissionMode::ReadOnly, ToolKind::Write).unwrap_err();
        assert!(err.to_string().contains("permission denied"));
        assert!(err.to_string().contains("read-only"));
        assert!(err.to_string().contains("workspace-write"));
    }
}
