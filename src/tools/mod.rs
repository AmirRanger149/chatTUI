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
pub fn all_tools() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition::new(
            "read_file",
            "Read the contents of a file within the workspace. Returns file content or error. Use for understanding code, config, docs.",
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
            "Create or overwrite a file in the workspace. Use for creating new files or fully rewriting existing ones. Will create parent directories.",
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
            "Edit a file by replacing exact old_string with new_string. old_string must match exactly including whitespace. Use for surgical edits.",
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
            "List files and directories in a workspace path. Returns names and types. Use to explore project structure.",
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
            "Execute a shell command in the workspace. Use for building, testing, git, etc. Returns stdout/stderr. Command runs with workspace root as cwd. Timeout 30s.",
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

    #[test]
    fn openai_conversion() {
        let tools = all_tools();
        let v = to_openai_tools(&tools);
        assert!(v.is_array());
        assert_eq!(v.as_array().unwrap()[0]["type"], "function");
    }
}
