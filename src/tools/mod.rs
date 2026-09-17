//! Tool definitions for agent mode - shared across all providers.
//! These are the functions the LLM can call to interact with the sandbox.

use serde_json::{json, Value};

/// A tool the model can call.
#[derive(Debug, Clone)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

impl ToolDefinition {
    pub fn new(name: &str, description: &str, parameters: Value) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            parameters,
        }
    }
}

/// All tools available in agent mode.
///
/// Every security-related claim here must match what the program actually
/// enforces (see `src/sandbox/mod.rs` for the exact security model): file
/// tools are workspace-restricted with a sensitive-file policy; `bash` has
/// a filtered environment, a hard timeout (default 30s — always enforced),
/// and kernel-level isolation (filesystem + network + process restrictions)
/// when the Linux kernel supports it — with an explicit, honest fallback
/// where it does not.
pub fn all_tools() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition::new(
            "read_file",
            "Read the contents of a file within the agent workspace. The path must resolve inside the workspace root (including through symlinks). Sensitive files are refused: dotenv files (.env, *.env), key material (*.pem, *.key, *.p12, *.pfx, *.jks, SSH private keys) and .git/config. Files above the size limit are refused. Returns file content or a structured error.",
            json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path from workspace root, e.g. 'src/main.rs' or 'README.md'"
                    }
                },
                "required": ["path"]
            }),
        ),
        ToolDefinition::new(
            "write_file",
            "Create or overwrite a file inside the agent workspace. The path must resolve inside the workspace root (including through symlinks); writing into .git/ or to sensitive files (.env*, key material) is refused. Parent directories are created automatically. Requires a permission mode that allows writes ('workspace-write' or 'full-auto'); otherwise a permission-denied error is returned.",
            json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path from workspace root"
                    },
                    "content": {
                        "type": "string",
                        "description": "Full file content to write"
                    }
                },
                "required": ["path", "content"]
            }),
        ),
        ToolDefinition::new(
            "edit_file",
            "Edit a file inside the agent workspace by replacing exact old_string with new_string. old_string must match exactly (including whitespace) and appear exactly once. The same workspace-path and sensitive-file restrictions as write_file apply, and write permission is required ('workspace-write' or 'full-auto').",
            json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path from workspace root"
                    },
                    "old_string": {
                        "type": "string",
                        "description": "Exact string to replace (must appear once)"
                    },
                    "new_string": {
                        "type": "string",
                        "description": "Replacement string"
                    }
                },
                "required": ["path", "old_string", "new_string"]
            }),
        ),
        ToolDefinition::new(
            "list_files",
            "List files and directories in a workspace path. Returns names and types. Sensitive file names (dotenv files, key material) are hidden from the listing. Use to explore project structure.",
            json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path to directory, defaults to '.' (workspace root)"
                    }
                },
                "required": []
            }),
        ),
        ToolDefinition::new(
            "bash",
            "Execute a shell command via `sh -c` with the workspace root as the current directory. When OS-level isolation is active (Linux kernel 5.13+, sandbox.os_isolation in config.json, default auto), the kernel enforces the boundary: writes outside the workspace and its scratch roots (/tmp, $TMPDIR, /dev/shm, CARGO_HOME, CARGO_TARGET_DIR) are denied, all network sockets are denied (Unix sockets too), and ptrace/kernel-module/namespace operations are denied. Where the kernel lacks that support, the command runs with full user privileges and the tool output states this explicitly — reads outside the workspace are always possible by design (system files, secret files). The refusal of sensitive-file names in commands is a best-effort scan. The environment is reduced to a small allowlist (API keys and credentials are not passed through). A hard timeout is enforced (default 30 seconds, configurable via sandbox.shell_timeout_secs in config.json): on timeout the command's process group is killed and a structured timeout error is returned. Requires permission mode 'full-auto' (and sandbox.allow_shell enabled). Use for building, testing, git, etc. Returns stdout/stderr.",
            json!({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to execute, e.g. 'cargo test' or 'ls -la'"
                    }
                },
                "required": ["command"]
            }),
        ),
    ]
}

/// Convert our internal ToolDefinition to OpenAI-compatible function tool format.
pub fn to_openai_tools(tools: &[ToolDefinition]) -> Value {
    let arr: Vec<Value> = tools
        .iter()
        .map(|t| {
            json!({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters
                }
            })
        })
        .collect();
    Value::Array(arr)
}

/// Convert to Anthropic tools format.
pub fn to_anthropic_tools(tools: &[ToolDefinition]) -> Value {
    let arr: Vec<Value> = tools
        .iter()
        .map(|t| {
            json!({
                "name": t.name,
                "description": t.description,
                "input_schema": t.parameters
            })
        })
        .collect();
    Value::Array(arr)
}

/// Convert to Gemini functionDeclarations format.
pub fn to_gemini_tools(tools: &[ToolDefinition]) -> Value {
    let decls: Vec<Value> = tools
        .iter()
        .map(|t| {
            json!({
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters
            })
        })
        .collect();
    json!([{
        "functionDeclarations": decls
    }])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_tools_have_valid_schema() {
        let tools = all_tools();
        assert_eq!(tools.len(), 5);
        for t in tools {
            assert!(!t.name.is_empty());
            assert!(t.parameters["type"] == "object");
        }
    }

    /// The model's tool schema must accurately describe what the program
    /// enforces: timeouts are real, workspace claims are scoped to the file
    /// tools, and shell is explicitly documented as not sandboxed.
    #[test]
    fn tool_descriptions_match_enforced_behavior() {
        let tools = all_tools();
        let find = |name: &str| {
            tools
                .iter()
                .find(|t| t.name == name)
                .unwrap_or_else(|| panic!("tool {name} missing"))
        };

        let bash = find("bash");
        let desc = bash.description.to_ascii_lowercase();
        assert!(desc.contains("timeout"), "bash must document its timeout");
        assert!(desc.contains("killed"), "bash must document that the command is killed on timeout");
        assert!(desc.contains("allowlist"), "bash must document the environment filtering");
        assert!(desc.contains("full-auto"), "bash must document its permission requirement");
        // Isolation claims must be conditional and honest: enforced where
        // supported, explicitly unrestricted (with a warning) elsewhere.
        assert!(desc.contains("os-level isolation"), "bash must document the kernel isolation layer");
        assert!(desc.contains("kernel 5.13"), "bash must state the isolation requirement");
        assert!(
            desc.contains("full user privileges") && desc.contains("states this explicitly"),
            "bash must describe the honest fallback when isolation is unavailable"
        );
        assert!(desc.contains("best-effort"), "bash must label the name scan as best-effort");

        for name in ["read_file", "write_file", "edit_file"] {
            let desc = find(name).description.to_ascii_lowercase();
            assert!(
                desc.contains("workspace"),
                "{name} must document the workspace restriction"
            );
            assert!(
                desc.contains("sensitive"),
                "{name} must document the sensitive-file policy"
            );
        }
    }

    #[test]
    fn openai_conversion() {
        let tools = all_tools();
        let v = to_openai_tools(&tools);
        assert!(v.is_array());
        assert_eq!(v.as_array().unwrap()[0]["type"], "function");
    }
}
