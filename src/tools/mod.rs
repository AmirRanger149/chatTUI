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

/// Most questions one `ask_user` call may carry. The user answers them one at
/// a time, so a long list is a wall of prompts rather than a decision.
pub const MAX_QUESTIONS: usize = 5;

/// Most options offered for a single question.
pub const MAX_OPTIONS: usize = 6;

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
            "Read a WINDOW of a file within the agent workspace: up to 'limit' numbered lines (default 250, max 1000) starting at the 1-based line 'offset' (default 1). The output header says which range of the file was served, and a footer points to the next offset when more lines exist. Read large files in successive windows (e.g. offset 1, then 251, then 501…) until you have the region you need — never try to pull a huge file in one call. The line numbers in the output are display-only; use the raw text (without numbers) for edit_file old_string. The path must resolve inside the workspace root (including through symlinks). Sensitive files are refused: dotenv files (.env, *.env), key material (*.pem, *.key, *.p12, *.pfx, *.jks, SSH private keys) and .git/config. Files above the size limit are refused.",
            json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path from workspace root, e.g. 'src/main.rs' or 'README.md'"
                    },
                    "offset": {
                        "type": "integer",
                        "description": "1-based first line to read (default 1)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max lines to read in this window (default 250, max 1000)"
                    }
                },
                "required": ["path"]
            }),
        ),
        ToolDefinition::new(
            "write_file",
            "Create or overwrite a file inside the agent workspace. Prefer edit_file for targeted changes; write_file replaces the ENTIRE file, so for big files keep it to when a full rewrite is really what you want. The whole file travels as ONE string in this call's arguments, so a large file can exceed the response's output limit and arrive cut off mid-way: keep write_file to files you can emit comfortably in a single response, and build anything larger in pieces (write_file the first part, then apply_patch to append the rest). The path must resolve inside the workspace root (including through symlinks); writing into .git/ or to sensitive files (.env*, key material) is refused. Parent directories are created automatically. The result reports the change as +added -removed lines. Requires a permission mode that allows writes ('workspace-write' or 'full-auto'); otherwise a permission-denied error is returned.",
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
            "Edit a file inside the agent workspace by replacing exact old_string with new_string. old_string must match exactly (including whitespace) and appear exactly once. Keep edits small and targeted — one focused change per call. The result reports the change as +added -removed lines. The same workspace-path and sensitive-file restrictions as write_file apply, and write permission is required ('workspace-write' or 'full-auto').",
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
            "apply_patch",
            "Edit several files in ONE atomic operation, anchored on surrounding context instead of on one exact string. PREFER THIS over edit_file and write_file: the whole patch is validated against the current file contents before anything is written, so if any hunk fails, nothing changes. Format (send it as the single 'patch' string, exactly, with the leading space on context lines):\n*** Begin Patch\n*** Add File: path/new.txt\n+added line\n*** Update File: path/existing.txt\n@@ optional unique text near the hunk\n context line (unchanged, note the leading space)\n-removed line\n+added line\n*** Delete File: path/old.txt\n*** End Patch\nRules: every body line starts with ' ', '+' or '-'; a blank line inside a hunk is a blank line in the file; use @@ when the context repeats elsewhere in the file; Add must not already exist, Update must already exist. If a hunk does not match, re-read the file first — it has changed since you last saw it. Paths are relative to the workspace and the same restrictions apply as to write_file (no escaping the workspace, no sensitive files), and write permission is required.",
            json!({
                "type": "object",
                "properties": {
                    "patch": {
                        "type": "string",
                        "description": "The whole patch document, from '*** Begin Patch' to '*** End Patch'"
                    }
                },
                "required": ["patch"]
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
            "Execute a shell command via `sh -c` with the workspace root as the current directory. When OS-level isolation is active (Linux kernel 5.13+, sandbox.os_isolation in config.json, default auto), the kernel enforces the boundary: writes outside the workspace and its scratch roots (/tmp, $TMPDIR, /dev/shm, CARGO_HOME, CARGO_TARGET_DIR) are denied, all network sockets are denied (Unix sockets too), and ptrace/kernel-module/namespace operations are denied. Where the kernel lacks that support, the command runs with full user privileges and the tool output states this explicitly — reads outside the workspace are always possible by design (system files, secret files). The refusal of sensitive-file names in commands is a best-effort scan. The environment is reduced to a small allowlist (API keys and credentials are not passed through). There is NO terminal: stdin is /dev/null and the command has no controlling tty, so interactive programs (editors, pagers, TUI apps, ssh, sudo, dev servers) cannot work and will only ever reach the timeout — use a build, a test run, `--help`, or the program's headless flags instead. Captured output has terminal control sequences stripped, so do not rely on colour. A hard timeout is enforced (default 30 seconds, configurable via sandbox.shell_timeout_secs in config.json): on timeout the command's process group is killed and a structured timeout error is returned. Requires permission mode 'full-auto' (and sandbox.allow_shell enabled). Use for building, testing, git, etc. Returns stdout/stderr.",
            json!({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to execute, e.g. 'cargo test' or 'ls -la'"
                    },
                    "timeout_secs": {
                        "type": "integer",
                        "description": "How long this command may run before it is killed, 1-1800. Set it when you know the command is slow (a release build, a full test suite) rather than letting it die at the default."
                    },
                    "session": {
                        "type": "boolean",
                        "description": "Start the command as a session that keeps running after this call returns, and get back whatever it printed so far plus a session_id. Use it for anything that does not exit on its own (a dev server, a watch loop, a REPL) or that would outlast the timeout."
                    },
                    "yield_time_ms": {
                        "type": "integer",
                        "description": "With session=true: how long to wait for output before returning, 100-120000 (default 10000)."
                    }
                },
                "required": ["command"]
            }),
        ),
        ToolDefinition::new(
            "write_stdin",
            "Send input to a running session started with bash(session=true), and return whatever it printed next. Use it to answer a prompt, feed a REPL, or just collect more output from a long-running command by sending an empty string. A newline is added if you leave one off. The session keeps running afterwards.",
            json!({
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "The session id returned by bash(session=true)"
                    },
                    "input": {
                        "type": "string",
                        "description": "Text to send. Empty to just read more output."
                    },
                    "yield_time_ms": {
                        "type": "integer",
                        "description": "How long to wait for output before returning, 100-120000 (default 5000)"
                    }
                },
                "required": ["session_id"]
            }),
        ),
        ToolDefinition::new(
            "kill_session",
            "Kill a running session and its whole process tree. Do this when you are finished with a dev server or a watch loop — a session lives on after the round that started it, up to 30 minutes.",
            json!({
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "The session id returned by bash(session=true)"
                    }
                },
                "required": ["session_id"]
            }),
        ),
        ToolDefinition::new(
            "ask_user",
            "Ask the user something you cannot decide for yourself, then wait for the answer. Use it for decisions that change what you build — scope, naming, which of two valid designs, whether to touch files outside what was asked — not for things you can settle by reading the repo. The call blocks until every question is answered, so ask them together in ONE call rather than one at a time; prefer a single question and never send more than 5. Give 2-4 concrete options when you can, most recommended first, and leave options empty when the answer is genuinely open — the user can always type free text instead of picking. Do not use it to report progress, and do not use it after the work is done.",
            json!({
                "type": "object",
                "properties": {
                    "questions": {
                        "type": "array",
                        "maxItems": MAX_QUESTIONS,
                        "description": "The questions to ask, in the order they should be answered",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {
                                    "type": "string",
                                    "description": "Short stable identifier for this question, e.g. 'scope'"
                                },
                                "header": {
                                    "type": "string",
                                    "description": "Very short topic label shown next to the question, e.g. 'Scope'"
                                },
                                "question": {
                                    "type": "string",
                                    "description": "The question itself, specific enough to answer in one line"
                                },
                                "options": {
                                    "type": "array",
                                    "maxItems": MAX_OPTIONS,
                                    "description": "Concrete answers to choose from, most recommended first. Omit for an open question.",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "label": {
                                                "type": "string",
                                                "description": "A few words, e.g. 'Rewrite the parser'"
                                            },
                                            "description": {
                                                "type": "string",
                                                "description": "One sentence on what this choice means"
                                            }
                                        },
                                        "required": ["label"]
                                    }
                                }
                            },
                            "required": ["question"]
                        }
                    }
                },
                "required": ["questions"]
            }),
        )
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
        assert_eq!(tools.len(), 9);
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

        for name in ["read_file", "write_file", "edit_file", "apply_patch"] {
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
