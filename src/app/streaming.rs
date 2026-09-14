//! Running a chat request: spawning the background stream, consuming its
//! token events frame by frame, and reporting elapsed time while it runs.
//! Now with agent loop for tool calling.

use crate::api::types::{Message, Role, StreamEvent, ToolDefinition};
use crate::app::{App, Cell, MAX_AGENT_ITERATIONS};
use crate::session::manager::ToolCallRecord;
use crate::tools;
use anyhow::Result;
use std::time::Instant;
use tokio::sync::mpsc::{self, error::TryRecvError};

impl App {
    pub fn elapsed_secs(&self) -> u64 {
        self.stream_started
            .map(|started| started.elapsed().as_secs())
            .unwrap_or(0)
    }

    pub fn elapsed_ms(&self) -> u64 {
        self.stream_started
            .map(|started| started.elapsed().as_millis() as u64)
            .unwrap_or(0)
    }

    pub(crate) fn start_stream(&mut self) -> Result<()> {
        if self.config.api_key.is_none() {
            let env_key = self
                .config
                .find_provider(&self.config.provider)
                .map(|p| p.env_key)
                .unwrap_or_else(|| "API_KEY".to_string());
            return Err(anyhow::anyhow!(
                "{env_key} is not configured — set it in config.json or the environment"
            ));
        }
        let (tx, rx) = mpsc::channel(64);
        let mut messages: Vec<Message> = self
            .sessions
            .current()
            .messages
            .iter()
            .map(|m| {
                let role = Role::from(m.role.as_str());
                let mut msg = Message::new(
                    role,
                    crate::ui::thinking::strip(&m.content),
                );
                if let Some(id) = &m.tool_call_id {
                    msg = msg.with_tool_call_id(id.clone());
                }
                if let Some(tcs) = &m.tool_calls {
                    let tool_calls: Vec<crate::api::types::ToolCall> = tcs
                        .iter()
                        .map(|tc| crate::api::types::ToolCall::new(tc.id.clone(), tc.name.clone(), tc.arguments.clone()))
                        .collect();
                    msg = msg.with_tool_calls(tool_calls);
                }
                msg
            })
            .collect();

        // Inject system prompt for agent mode if enabled
        let want_tools = self.agent_mode && self.config.sandbox.enabled && self.sandbox.config.enabled;
        let tools_enabled = want_tools && self.sandbox.has_target();
        if want_tools && !self.sandbox.has_target() {
            self.push_notice(
                "agent tools off — set a target with /sandbox <path> (tools will not use this app's directory)".into(),
            );
        }
        if tools_enabled {
            let has_system = messages.iter().any(|m| m.role == Role::System);
            if !has_system {
                let workspace = self.sandbox.config.workspace_root.display().to_string();
                let system_content = format!(
                    "You are chatTUI agent with sandbox access. Target directory (dst): {workspace}\n\
                    All file and shell tools operate only inside this target directory.\n\
                    You can read, write, edit, list files and run shell commands via tools.\n\
                    - read_file(path): read file content (relative to the target)\n\
                    - write_file(path, content): create/overwrite file in the target\n\
                    - edit_file(path, old_string, new_string): surgical edit (old_string must be unique)\n\
                    - list_files(path): list directory\n\
                    - bash(command): run shell command (cwd is the target directory)\n\
                    Always use tools to help the user with file operations. Be concise, explain what you do.\n\
                    Never write files outside the target directory.",
                );
                messages.insert(0, Message::new(Role::System, system_content));
            }
        }
        let model = self.config.model.clone();
        let temperature = self.config.temperature;
        let client = self.config.api_client();

        // Prepare tools if agent mode enabled (reuse tools_enabled from above)
        let tool_defs: Vec<ToolDefinition> = if tools_enabled {
            tools::all_tools()
                .into_iter()
                .map(|t| ToolDefinition {
                    name: t.name,
                    description: t.description,
                    parameters: t.parameters,
                })
                .collect()
        } else {
            Vec::new()
        };

        tokio::spawn(async move {
            if let Err(error) = client
                .stream_chat(&messages, &model, temperature, tool_defs, tx.clone())
                .await
            {
                let _ = tx.send(StreamEvent::Error(error.to_string())).await;
            }
        });
        self.tokens = Some(rx);
        self.streaming = true;
        self.stream_started = Some(Instant::now());
        Ok(())
    }

    pub async fn receive_token(&mut self) {
        let Some(mut rx) = self.tokens.take() else {
            return;
        };
        let mut has_tool_calls = false;
        loop {
            match rx.try_recv() {
                Ok(StreamEvent::Delta(token)) => self.response.push_str(&token),
                Ok(StreamEvent::ToolCall(tc)) => {
                    self.pending_tool_calls.push(tc);
                    has_tool_calls = true;
                }
                Ok(StreamEvent::Notice(message)) => {
                    self.push_notice(message);
                }
                Ok(StreamEvent::Error(error)) => {
                    self.finish_partial();
                    self.streaming = false;
                    self.stream_started = None;
                    self.pending_tool_calls.clear();
                    self.agent_iterations = 0;
                    self.push_error(error);
                    return;
                }
                Err(TryRecvError::Empty) => {
                    self.tokens = Some(rx);
                    return;
                }
                Err(TryRecvError::Disconnected) => break,
            }
        }

        // Stream finished - check for agent loop
        if has_tool_calls && !self.pending_tool_calls.is_empty() {
            // Save assistant message with tool calls
            let response_text = std::mem::take(&mut self.response);
            let pending = std::mem::take(&mut self.pending_tool_calls);
            let tool_records: Vec<ToolCallRecord> = pending
                .iter()
                .map(|tc| ToolCallRecord {
                    id: tc.id.clone(),
                    name: tc.name.clone(),
                    arguments: tc.arguments.clone(),
                })
                .collect();

            if !response_text.is_empty() {
                self.cells.push(Cell::Assistant(response_text.clone()));
            }
            for tc in &pending {
                self.push_tool_call(tc);
            }
            self.sessions.add_assistant_with_tools(response_text, tool_records);

            // Execute tools
            let sandbox = self.sandbox.clone();
            let mut tool_results = Vec::new();
            for tc in &pending {
                match sandbox.execute_tool(tc).await {
                    Ok(output) => {
                        tool_results.push((tc.id.clone(), output, false));
                    }
                    Err(e) => {
                        tool_results.push((tc.id.clone(), format!("Error: {}", e), true));
                    }
                }
            }

            // Save tool results to session and cells
            for (id, content, is_error) in &tool_results {
                self.push_tool_result(id.clone(), content.clone(), *is_error);
                self.sessions.add_tool_result(id.clone(), content.clone());
            }

            // Continue agent loop if iterations left
            if self.agent_iterations < MAX_AGENT_ITERATIONS {
                self.agent_iterations += 1;
                if let Err(e) = self.start_stream() {
                    self.push_error(e.to_string());
                    self.streaming = false;
                    self.stream_started = None;
                    self.agent_iterations = 0;
                }
                return;
            } else {
                self.push_notice(format!(
                    "agent stopped after {} iterations",
                    MAX_AGENT_ITERATIONS
                ));
                self.agent_iterations = 0;
            }
        }

        self.streaming = false;
        self.stream_started = None;
        self.finish_partial();
        self.agent_iterations = 0;
        self.pending_tool_calls.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::{App, Cell};
    use crate::config::Config;
    use crate::session::manager::SessionManager;

    fn test_app() -> App {
        App::new(Config::default(), SessionManager::for_tests())
    }

    #[test]
    fn stream_notices_become_transcript_rows() {
        let (tx, rx) = tokio::sync::mpsc::channel(4);
        tx.try_send(StreamEvent::Notice("switching to b/2".into()))
            .unwrap();
        tx.try_send(StreamEvent::Delta("hel".into())).unwrap();
        let mut app = test_app();
        app.streaming = true;
        app.tokens = Some(rx);
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        runtime.block_on(app.receive_token());
        assert_eq!(app.response, "hel");
        assert_eq!(app.cells.len(), 1);
        assert!(matches!(app.cells.last(), Some(Cell::Notice(_))));

        // Dropping the sender ends the stream and commits the partial answer.
        drop(tx);
        runtime.block_on(app.receive_token());
        assert!(!app.streaming);
        assert_eq!(app.cells.len(), 2);
        assert!(matches!(app.cells[0], Cell::Notice(_)));
        assert_eq!(app.cells[1], Cell::Assistant("hel".into()));
    }
}
