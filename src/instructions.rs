//! What the agent is told before it starts working.
//!
//! Workspace, permission mode, timeout, isolation mode and output budget are
//! all read from the same [`SandboxConfig`] the enforcement code uses, so the
//! model can never be told a rule the program does not apply. That matters
//! most for isolation, where a hand-written promise keeps being repeated long
//! after the setting that backed it changed.

use crate::sandbox::os_isolation::OsIsolation;
use crate::sandbox::permissions::PermissionMode;
use crate::sandbox::{SandboxConfig, MAX_TOOL_OUTPUT_BYTES};

/// Describe the isolation the shell actually runs under.
///
/// This is the claim most worth getting right: it is the difference between
/// the model believing it is confined and knowing it is not.
fn describe_isolation(mode: OsIsolation) -> &'static str {
    match mode {
        OsIsolation::Off => "Kernel isolation is OFF. Commands run with this user's full permissions: they can read and write anything the user can and reach the network. Ask before running anything destructive.",
        OsIsolation::Require => "Kernel isolation is REQUIRED. Every command is confined, and if the kernel cannot confine it the command is refused rather than run unconfined.",
        OsIsolation::Auto => "Kernel isolation is applied when the kernel supports it (Linux 5.13+): writes are limited to the workspace and its scratch roots (/tmp, $TMPDIR, /dev/shm, $CARGO_HOME, $CARGO_TARGET_DIR), and network sockets, ptrace, kernel-module loading and namespace operations are denied. Where the kernel cannot do this, the command runs unconfined and the tool output says so explicitly.",
    }
}

/// The system prompt for an agent turn, derived from the enforced policy.
pub(crate) fn agent_system_prompt(config: &SandboxConfig) -> String {
    let workspace = config.workspace_root.display();
    let timeout = config.effective_shell_timeout().as_secs();
    let mode = config.permission_mode.as_str();
    let isolation = describe_isolation(config.os_isolation);
    let shell_allowed = describe_shell_access(config.permission_mode, config.allow_shell);

    let mut prompt = format!(
        "You are the chatTUI agent, working in one directory — the workspace: {workspace}\n\
         \n\
         ## Tools\n\
         - read_file: a window of numbered lines (250 by default, 1000 max) from a file. \
         Read large files in successive windows (offset 1, then 251, …) rather than whole.\n\
         - apply_patch: edits one or many files in a single atomic operation, anchored on \
         surrounding context. Prefer it for any change that touches more than a line or two, \
         and for multi-file changes.\n\
         - edit_file: replaces an exact old_string with new_string. It must match byte for \
         byte and appear exactly once.\n\
         - write_file: replaces an entire file. Prefer apply_patch or edit_file instead.\n\
         - list_files: names and types in a directory.\n\
         - bash: runs `sh -c` with the workspace as the working directory. Pass timeout_secs \
         for a command you know is slow; pass session=true for one that does not exit on its \
         own (a dev server, a watch loop, a REPL) and you get a session_id back.\n\
         - write_stdin: sends input to a running session and returns its new output.\n\
         - kill_session: kills a session and its process tree. Kill dev servers when you are \
         done with them.\n\
         - ask_user: asks the user up to 5 questions and waits for the answers before you \
         continue.\n\
         \n\
         ## What is enforced\n\
         - File tools resolve every path inside the workspace, following symlinks; anything \
         that escapes is refused. Sensitive files are refused for reading and writing: dotenv \
         files (.env, *.env), key material (*.pem, *.key, *.p12, *.pfx, *.jks, SSH private \
         keys) and .git/config. Writes under .git/ are refused.\n\
         - {isolation}\n\
         - bash has NO terminal. stdin is /dev/null and there is no controlling tty, so \
         interactive or full-screen programs cannot work — they fail or hang until the \
         timeout. Use non-interactive equivalents: builds, tests, --help, headless flags.\n\
         - Commands are killed after {timeout}s unless the call asked for a longer \
         timeout_secs, and nothing interactive can run at all — use a session for a server.\n\
         - Permission mode: {mode}. {shell_allowed}\n\
         - In an ask-mode, a gated call pauses until the user approves it. If one is denied, \
         the denial comes back as a tool result: take a different approach or ask what they \
         want — never retry the identical call.\n\
         - Tool output over {MAX_TOOL_OUTPUT_BYTES} bytes is truncated, keeping the beginning \
         and the end.\n\
         \n\
         ## How to work\n\
         - Read a file before editing it, and re-read after a failed edit — the file may not \
         be what you assume.\n\
         - Prefer small, targeted edits over rewriting whole files, and re-read a file \
         before patching it: a hunk that does not match means the file is not what you think.\n\
         - Verify your work by running the project's build or tests when there are any.\n\
         - Ask with ask_user instead of guessing when a choice changes what you build — scope, \
         naming, which of two valid designs, whether to touch files you were not asked about. \
         Ask them together in one call, not one at a time, and never to report progress.\n\
         - Never write outside the workspace.\n\
         - Be concise: say what you changed and why, not what you are about to consider.",
    );

    prompt
}

/// Whether the shell is reachable at all in this mode — worth stating
/// plainly, because a model that tries `bash` in a read-only mode wastes a
/// round discovering it.
fn describe_shell_access(mode: PermissionMode, allow_shell: bool) -> &'static str {
    match mode {
        PermissionMode::ReadOnly => "Only reading is allowed: writes and shell commands return permission-denied.",
        PermissionMode::WorkspaceWrite => "Writes inside the workspace are allowed; shell commands return permission-denied.",
        PermissionMode::AskBeforeWrite => "Writes and shell commands are not approved in this mode and return permission-denied.",
        PermissionMode::AskBeforeShell => "Writes inside the workspace are allowed; shell commands are not approved in this mode and return permission-denied.",
        PermissionMode::FullAuto => {
            if allow_shell {
                "Reads, writes inside the workspace and shell commands are all allowed."
            } else {
                "Reads and writes inside the workspace are allowed; the shell is disabled in config, so bash returns an error."
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sandbox::SandboxConfig;

    #[test]
    fn prompt_reports_the_configured_policy() {
        let mut config = SandboxConfig::default();
        config.workspace_root = PathBuf::from("/srv/project");
        config.permission_mode = PermissionMode::ReadOnly;
        config.os_isolation = OsIsolation::Off;
        config.shell_timeout = std::time::Duration::from_secs(7);

        let prompt = agent_system_prompt(&config);
        assert!(prompt.contains("/srv/project"), "{prompt}");
        assert!(prompt.contains("7s"), "timeout missing: {prompt}");
        assert!(prompt.contains("read-only"), "mode missing: {prompt}");
        // The isolation claim must follow the setting, not a fixed promise.
        assert!(
            prompt.contains("Kernel isolation is OFF"),
            "isolation text ignored the setting: {prompt}"
        );
        assert!(
            !prompt.contains("writes are limited to the workspace"),
            "off-mode prompt still claims confinement: {prompt}"
        );
    }

    #[test]
    fn prompt_claims_confinement_only_when_it_is_on() {
        let mut config = SandboxConfig::default();
        config.workspace_root = PathBuf::from("/srv/project");
        config.os_isolation = OsIsolation::Auto;
        let prompt = agent_system_prompt(&config);
        assert!(
            prompt.contains("Linux 5.13+"),
            "auto mode must state the requirement: {prompt}"
        );

        config.os_isolation = OsIsolation::Require;
        let prompt = agent_system_prompt(&config);
        assert!(
            prompt.contains("refused rather than run unconfined"),
            "require mode must describe fail-closed: {prompt}"
        );
    }

}
