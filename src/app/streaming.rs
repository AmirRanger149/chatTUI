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
                let timeout_secs = self.sandbox.config.effective_shell_timeout().as_secs();
                let mode = self.sandbox.config.permission_mode.as_str();
                // The schema must describe what the program enforces — no
                // claims of sandboxing: file tools are workspace-restricted,
                // shell is not isolated (cwd only, full user privileges).
                let system_content = format!(
                    "You are chatTUI agent. Target directory (workspace): {workspace}\n\
                    File tools (read_file, write_file, edit_file, list_files) operate only inside this workspace; paths that escape it (including via symlinks) and sensitive files (.env*, key material, .git/config) are refused.\n\
                    bash runs `sh -c` with the workspace as the current directory. When OS-level isolation is active (Linux kernel 5.13+), the kernel blocks writes outside the workspace and its scratch roots (/tmp, CARGO_HOME, CARGO_TARGET_DIR), denies all network connections, and denies ptrace/kernel-module/namespace operations; on systems without it, commands run with full user privileges and the tool output says so. Secret files remain readable by shell commands by design (the name-based refusal is best-effort).\n\
                    Commands are killed after a {timeout_secs}s timeout. Permission mode: {mode}. Tools outside the mode return permission-denied errors.\n\
                    Use tools to help the user with file operations. Be concise, explain what you do. Never write files outside the target directory.",
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

        let task = tokio::spawn(async move {
            if let Err(error) = client
                .stream_chat(&messages, &model, temperature, tool_defs, tx.clone())
                .await
            {
                let _ = tx.send(StreamEvent::Error(error.to_string())).await;
            }
        });
        self.stream_task = Some(task);
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
        if self.streaming && has_tool_calls && !self.pending_tool_calls.is_empty() {
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

            // Execute tools (permission-gated inside the sandbox). Each
            // error — including timeouts, path escapes and permission
            // denials — is returned to the model as a structured tool error
            // result instead of aborting the loop.
            let sandbox = self.sandbox.clone();
            let mut tool_results = Vec::new();
            for tc in &pending {
                if !self.streaming {
                    break; // cancelled while earlier tools were running
                }
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
    use crate::api::types::ToolCall;
    use crate::app::{App, Cell, MAX_AGENT_ITERATIONS};
    use crate::config::Config;
    use crate::sandbox::permissions::PermissionMode;
    use crate::session::manager::SessionManager;

    fn test_app() -> App {
        App::new(Config::default(), SessionManager::for_tests())
    }

    /// An app with agent tools enabled against an empty temp workspace and
    /// no API key: a tool round that tries to continue the agent loop fails
    /// its restart immediately instead of touching the network.
    fn agent_app(name: &str) -> App {
        let config: Config = serde_json::from_str("{}").unwrap();
        let mut app = App::new(config, SessionManager::for_tests());
        let dir = std::env::temp_dir().join(format!("chatTUI_loop_{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        app.sandbox.set_target(dir.to_str().unwrap()).unwrap();
        app.agent_mode = true;
        app
    }

    fn tool_call_event(id: &str, name: &str, arguments: &str) -> StreamEvent {
        StreamEvent::ToolCall(ToolCall::new(id, name, arguments))
    }

    /// Feed one completed stream containing `events` into the app.
    fn feed(app: &mut App, events: Vec<StreamEvent>) {
        let (tx, rx) = tokio::sync::mpsc::channel(4);
        for event in events {
            tx.try_send(event).unwrap();
        }
        drop(tx);
        app.streaming = true;
        app.tokens = Some(rx);
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

    #[tokio::test]
    async fn tool_rounds_execute_and_stop_predictably_when_restart_fails() {
        let mut app = agent_app("restart-fail");
        feed(&mut app, vec![tool_call_event("c1", "list_files", "{}")]);
        app.receive_token().await;

        // The tool actually ran.
        assert!(app.cells.iter().any(|c| matches!(
            c,
            Cell::ToolResult { is_error: false, .. }
        )));
        // The loop tried to continue, the restart failed (no API key), and
        // the agent stopped with a reported error instead of looping.
        assert!(!app.streaming);
        assert_eq!(app.agent_iterations, 0);
        assert!(matches!(app.cells.last(), Some(Cell::Error(_))));
    }

    #[tokio::test]
    async fn agent_loop_stops_at_max_iterations() {
        let mut app = agent_app("max-iterations");
        app.agent_iterations = MAX_AGENT_ITERATIONS;
        feed(&mut app, vec![tool_call_event("c1", "list_files", "{}")]);
        app.receive_token().await;

        // The final round's tool still ran and its result was recorded…
        assert!(app
            .cells
            .iter()
            .any(|c| matches!(c, Cell::ToolResult { .. })));
        // …but no further LLM request was started and the agent stopped.
        assert!(!app.streaming);
        assert_eq!(app.agent_iterations, 0);
        assert!(app.cells.iter().any(|c| matches!(
            c,
            Cell::Notice(text) if text.contains("agent stopped")
        )));
    }

    #[tokio::test]
    async fn repeated_tool_rounds_cannot_exceed_max_iterations() {
        // Restart streams succeed (the legacy `api_key` field satisfies
        // start_stream; the endpoint is a closed local port that fails
        // fast), so the agent loop really runs round after round — exactly
        // the setup that would spin forever if the cap were bypassable.
        let config: Config = serde_json::from_str(
            r#"{"provider":"openai","base_url":"http://127.0.0.1:9/v1","api_key":"test-key"}"#,
        )
        .unwrap();
        let mut app = App::new(config, SessionManager::for_tests());
        let dir =
            std::env::temp_dir().join(format!("chatTUI_loop_many_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        app.sandbox.set_target(dir.to_str().unwrap()).unwrap();
        app.agent_mode = true;

        let mut rounds = 0;
        loop {
            // The counter may reach but never exceed the maximum.
            assert!(
                app.agent_iterations <= MAX_AGENT_ITERATIONS,
                "iteration counter exceeded its maximum"
            );
            feed(
                &mut app,
                vec![tool_call_event(&format!("c{rounds}"), "list_files", "{}")],
            );
            app.receive_token().await;
            rounds += 1;
            assert!(rounds < 100, "agent loop failed to terminate");
            if !app.streaming {
                break;
            }
        }

        assert_eq!(rounds, MAX_AGENT_ITERATIONS + 1);
        assert!(app.cells.iter().any(|c| matches!(
            c,
            Cell::Notice(text) if text.contains("agent stopped")
        )));
        assert_eq!(app.agent_iterations, 0);
    }

    #[tokio::test]
    async fn tool_errors_cannot_bypass_the_iteration_limit() {
        // Every round fails (permission denied); the counter must still
        // advance and the loop must still terminate.
        let mut app = agent_app("errors-bounded");
        app.sandbox.config.permission_mode = PermissionMode::ReadOnly;
        let rounds = 3;
        for round in 0..rounds {
            assert!(app.agent_iterations < MAX_AGENT_ITERATIONS);
            feed(
                &mut app,
                vec![tool_call_event(&format!("c{round}"), "write_file", r#"{"path":"x","content":"y"}"#)],
            );
            app.receive_token().await;
            assert!(!app.streaming);
            assert!(app.cells.iter().any(|c| matches!(
                c,
                Cell::ToolResult { is_error: true, content, .. }
                    if content.contains("permission denied")
            )));
        }
        assert!(!app.streaming);
    }

    #[tokio::test]
    async fn permission_denials_reach_the_model_as_tool_errors() {
        let mut app = agent_app("permission");
        app.sandbox.config.permission_mode = PermissionMode::ReadOnly;
        feed(
            &mut app,
            vec![tool_call_event("c1", "bash", r#"{"command":"echo hi"}"#)],
        );
        app.receive_token().await;
        assert!(app.cells.iter().any(|c| matches!(
            c,
            Cell::ToolResult { is_error: true, content, .. }
                if content.contains("permission denied")
        )));
    }
}
