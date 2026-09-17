//! Minimal permission layer for agent tool execution.
//!
//! This is an *application-level policy*, not an OS security boundary: it
//! decides which classes of tools may run at all. It is deliberately small
//! so a future phase can add interactive approval without changing its
//! shape: the `Ask*` modes mark requests that should one day prompt the
//! user instead of being allowed silently.

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
}

impl ToolKind {
    pub fn as_str(self) -> &'static str {
        match self {
            ToolKind::Read => "read",
            ToolKind::Write => "write",
            ToolKind::Edit => "edit",
            ToolKind::Shell => "shell",
        }
    }

    /// Map a tool name from [`crate::tools`] to its capability class.
    /// Unknown tools return `None` (and are refused by the executor).
    pub fn from_tool_name(name: &str) -> Option<ToolKind> {
        match name {
            "read_file" | "list_files" => Some(ToolKind::Read),
            "write_file" => Some(ToolKind::Write),
            "edit_file" => Some(ToolKind::Edit),
            "bash" => Some(ToolKind::Shell),
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

    /// Whether `kind` is allowed outright by this mode.
    pub fn allows(self, kind: ToolKind) -> bool {
        match self {
            PermissionMode::ReadOnly => kind == ToolKind::Read,
            PermissionMode::WorkspaceWrite => kind != ToolKind::Shell,
            PermissionMode::AskBeforeWrite => kind == ToolKind::Read,
            PermissionMode::AskBeforeShell => kind != ToolKind::Shell,
            PermissionMode::FullAuto => true,
        }
    }

    /// True when this mode would want to ask the user before running
    /// `kind` (interactive approval is not implemented yet, so `authorize`
    /// turns these into denials).
    pub fn asks(self, kind: ToolKind) -> bool {
        !self.allows(kind)
            && match (self, kind) {
                (PermissionMode::AskBeforeWrite, ToolKind::Write)
                | (PermissionMode::AskBeforeWrite, ToolKind::Edit)
                | (PermissionMode::AskBeforeShell, ToolKind::Shell) => true,
                _ => false,
            }
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
                "interactive approval is not implemented yet — set 'workspace-write' (file tools) or 'full-auto' (file tools + shell) explicitly in config.json"
            }
            PermissionMode::FullAuto => "check sandbox.allow_shell in config.json",
        }
    }
}

/// Refuse `kind` unless the configured mode allows it. The error text tells
/// the model (and the user) exactly which knob to change.
pub(crate) fn authorize(mode: PermissionMode, kind: ToolKind) -> Result<()> {
    if mode.allows(kind) {
        return Ok(());
    }
    if mode.asks(kind) {
        return Err(anyhow::anyhow!(
            "permission denied: '{}' requires interactive approval (mode '{}'), which is not implemented yet — the request was denied. {}",
            kind.as_str(),
            mode.as_str(),
            mode.remedy()
        ));
    }
    Err(anyhow::anyhow!(
        "permission denied: '{}' is not allowed in permission mode '{}'. {}",
        kind.as_str(),
        mode.as_str(),
        mode.remedy()
    ))
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
        assert_eq!(ToolKind::from_tool_name("nope"), None);
    }

    #[test]
    fn denial_mentions_the_config_knob() {
        let err = authorize(PermissionMode::ReadOnly, ToolKind::Write).unwrap_err();
        assert!(err.to_string().contains("permission denied"));
        assert!(err.to_string().contains("read-only"));
        assert!(err.to_string().contains("workspace-write"));
    }
}
