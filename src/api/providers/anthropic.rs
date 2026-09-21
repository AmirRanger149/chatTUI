//! The Anthropic Messages API backend (`POST /v1/messages`).
//!
//! Note the differences from the OpenAI dialect that this adapter papers over:
//! - the system prompt is a top-level `system` field, not a message;
//! - message content is a list of content blocks (`{"type":"text", …}`);
//! - `max_tokens` is mandatory (a conservative bound here, trimmed by the
//!   model list when available);
//! - streaming deltas arrive as `content_block_delta` events with
//!   `delta.text`, and the model list is `{"data":[{"id": …}]}`.
//! - Tool calling uses `tools` + `tool_use` blocks and `tool_result` in user messages.

use super::{ChatBackend, HttpTimeouts};
use crate::api::error::{classify_failure, error_detail, MAX_CONSECUTIVE_SSE_PARSE_FAILURES};
use crate::api::sse::SseReader;
use crate::api::types::{CompletionRequest, Failure, Role, StreamEvent, ToolCall};
use crate::tools;
use anyhow::{anyhow, Context, Result};
use futures_util::StreamExt;
use reqwest::Client;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::time::Duration;
use tokio::sync::mpsc::Sender;

/// Fallback token cap when the model list cannot be consulted.
const DEFAULT_MAX_TOKENS: u64 = 8192;

pub struct AnthropicBackend {
    http: Client,
    api_key: String,
    base_url: String,
    /// How long the stream may stay quiet between chunks before the
    /// connection counts as dead. There is deliberately no whole-request
    /// timeout: it would cut long generations off mid-answer.
    idle_timeout: Duration,
}

impl AnthropicBackend {
    pub fn new(api_key: String, base_url: String, timeouts: HttpTimeouts) -> Self {
        Self {
            http: Client::builder()
                .connect_timeout(timeouts.connect)
                .build()
                .expect("HTTP client"),
            api_key,
            base_url: base_url.trim_end_matches('/').into(),
            idle_timeout: timeouts.idle,
        }
    }

    /// Pick a sensible `max_tokens` for the model. Consult the model list's
    /// `context_window` when available, otherwise use a conservative bound.
    async fn max_tokens_for(&self, model: &str) -> u64 {
        match self.resolve_context_window(model).await {
            Some(window) => window.saturating_sub(1024).max(512),
            None => DEFAULT_MAX_TOKENS,
        }
    }

    /// Best-effort lookup of a model's `context_window` from `GET /models`.
    /// Returns `None` on any failure; callers fall back to a constant.
    async fn resolve_context_window(&self, model: &str) -> Option<u64> {
        let response = tokio::time::timeout(
            self.idle_timeout,
            self.http.get(format!("{}/models", self.base_url)).send(),
        )
        .await
        .ok()?
        .ok()?;
        let json: Value = response.json().await.ok()?;
        let entry = json["data"]
            .as_array()?
            .iter()
            .find(|m| m["id"].as_str() == Some(model))?;
        entry["context_window"]
            .as_u64()
            .or_else(|| entry["context_window"].as_str()?.parse().ok())
    }

    async fn list_models_inner(&self) -> Result<Vec<String>> {
        let url = format!("{}/models", self.base_url);
        let response = tokio::time::timeout(
            self.idle_timeout,
            self.http
                .get(url)
                .header("x-api-key", &self.api_key)
                .header("anthropic-version", "2023-06-01")
                .send(),
        )
        .await
        .map_err(|_| {
            anyhow!(
                "model list request timed out after {}s",
                self.idle_timeout.as_secs()
            )
        })?
        .context("sending GET /models request")?;
        let status = response.status();
        if !status.is_success() {
            return Err(anyhow!(
                "model list request failed (HTTP {}): {}",
                status,
                error_detail(&response.text().await.unwrap_or_default())
            ));
        }
        let value: Value = response
            .json()
            .await
            .context("parsing the model list response")?;
        let mut ids = value["data"]
            .as_array()
            .map(Vec::as_slice)
            .unwrap_or(&[])
            .iter()
            .filter_map(|item| item["id"].as_str().map(str::to_string))
            .collect::<Vec<_>>();
        ids.sort_by_key(|id| id.to_ascii_lowercase());
        if ids.is_empty() {
            return Err(anyhow!("the endpoint returned no models"));
        }
        Ok(ids)
    }

    async fn stream_inner(
        &self,
        request: CompletionRequest,
        tx: Sender<StreamEvent>,
    ) -> Result<(), Failure> {
        // The system prompt is a top-level field, not a turn.
        let mut system: Option<String> = None;
        let mut messages = Vec::with_capacity(request.messages.len());
        for message in &request.messages {
            match message.role {
                Role::System => {
                    let mut prompt = system.take().unwrap_or_default();
                    if !prompt.is_empty() {
                        prompt.push_str("\n\n");
                    }
                    prompt.push_str(&message.content);
                    system = Some(prompt);
                }
                Role::Assistant => {
                    // Check if assistant has tool_calls
                    if let Some(tool_calls) = &message.tool_calls {
                        let mut content_blocks = Vec::new();
                        if !message.content.is_empty() {
                            content_blocks.push(json!({ "type": "text", "text": message.content }));
                        }
                        for tc in tool_calls {
                            let input: Value = serde_json::from_str(&tc.arguments).unwrap_or(json!({}));
                            content_blocks.push(json!({
                                "type": "tool_use",
                                "id": tc.id,
                                "name": tc.name,
                                "input": input
                            }));
                        }
                        messages.push(json!({
                            "role": "assistant",
                            "content": content_blocks
                        }));
                    } else {
                        messages.push(json!({
                            "role": "assistant",
                            "content": [{ "type": "text", "text": message.content }],
                        }));
                    }
                }
                Role::User => {
                    messages.push(json!({
                        "role": "user",
                        "content": [{ "type": "text", "text": message.content }],
                    }));
                }
                Role::Tool => {
                    // Tool result -> user message with tool_result block
                    let tool_use_id = message.tool_call_id.clone().unwrap_or_else(|| "unknown".to_string());
                    messages.push(json!({
                        "role": "user",
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": message.content
                        }]
                    }));
                }
            }
        }
        let mut body = json!({
            "model": request.model,
            "messages": messages,
            "max_tokens": self.max_tokens_for(&request.model).await,
            "temperature": request.temperature,
            "stream": true,
        });
        if let Some(text) = system {
            body["system"] = Value::String(text);
        }

        // Add tools if present
        if !request.tools.is_empty() {
            let tool_defs: Vec<crate::tools::ToolDefinition> = request
                .tools
                .iter()
                .map(|t| crate::tools::ToolDefinition::new(&t.name, &t.description, t.parameters.clone()))
                .collect();
            body["tools"] = tools::to_anthropic_tools(&tool_defs);
        }

        let url = format!("{}/messages", self.base_url);
        let response = self
            .http
            .post(url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .json(&body)
            .send()
            .await
            .map_err(|error| Failure::Transient(format!("request failed: {error}")))?;
        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(classify_failure(status.as_u16(), &body));
        }
        let mut stream = response.bytes_stream();
        let mut reader = SseReader::new();
        let mut emitted = false;
        let mut bad_lines = 0usize;
        let mut tool_builders: HashMap<usize, AnthropicToolBuilder> = HashMap::new();

        let mut stream_ended = false;
        loop {
            // Idle-timeout guard: a stream may run as long as it keeps
            // producing chunks, but a connection that stays quiet past
            // `idle_timeout` is dead and must not hang the UI forever.
            // When the stream closes cleanly, whatever trailing frame the
            // reader still holds (a provider may end without a final
            // newline) is processed too — losing it can truncate tool-call
            // arguments.
            let payloads = if stream_ended {
                break;
            } else {
                match tokio::time::timeout(self.idle_timeout, stream.next()).await {
                    Ok(Some(Ok(chunk))) => reader.feed(&chunk),
                    Ok(Some(Err(error))) => {
                        return Err(if emitted || !tool_builders.is_empty() {
                            Failure::Fatal(format!(
                                "stream interrupted after output started: {error}"
                            ))
                        } else {
                            Failure::Transient(format!(
                                "stream interrupted before any output: {error}"
                            ))
                        });
                    }
                    Ok(None) => {
                        stream_ended = true;
                        reader.flush()
                    }
                    Err(_elapsed) => {
                        let secs = self.idle_timeout.as_secs();
                        return Err(if emitted || !tool_builders.is_empty() {
                            Failure::Fatal(format!(
                                "the connection went quiet for {secs}s after output started"
                            ))
                        } else {
                            Failure::Transient(format!(
                                "the connection went quiet for {secs}s before any output"
                            ))
                        });
                    }
                }
            };
            for data in payloads {
                let value: Value = match serde_json::from_str(&data) {
                    Ok(value) => {
                        bad_lines = 0;
                        value
                    }
                    Err(_) => {
                        // Gateways occasionally emit junk or keep-alive
                        // lines; a single one must not kill the stream. Only
                        // a run of consecutive garbage is treated as broken.
                        bad_lines += 1;
                        if bad_lines >= MAX_CONSECUTIVE_SSE_PARSE_FAILURES {
                            return Err(Failure::Fatal(format!(
                                "the stream sent {bad_lines} unparseable lines in a row (last: {data})"
                            )));
                        }
                        continue;
                    }
                };
                match value["type"].as_str() {
                    Some("content_block_start") => {
                        let index = value["index"].as_u64().unwrap_or(0) as usize;
                        if let Some(block_type) = value["content_block"]["type"].as_str() {
                            if block_type == "tool_use" {
                                let id = value["content_block"]["id"].as_str().unwrap_or("").to_string();
                                let name = value["content_block"]["name"].as_str().unwrap_or("").to_string();
                                tool_builders.insert(
                                    index,
                                    AnthropicToolBuilder {
                                        id,
                                        name,
                                        input_json: String::new(),
                                    },
                                );
                            }
                        }
                    }
                    Some("content_block_delta") => {
                        let index = value["index"].as_u64().unwrap_or(0) as usize;
                        if let Some(text) = value["delta"]["text"].as_str() {
                            tx.send(StreamEvent::Delta(text.into()))
                                .await
                                .map_err(|_| Failure::Fatal("stream receiver closed".into()))?;
                            emitted = true;
                        }
                        if let Some(partial) = value["delta"]["partial_json"].as_str() {
                            if let Some(builder) = tool_builders.get_mut(&index) {
                                builder.input_json.push_str(partial);
                            }
                        }
                    }
                    Some("content_block_stop") => {
                        let index = value["index"].as_u64().unwrap_or(0) as usize;
                        if let Some(builder) = tool_builders.remove(&index) {
                            if !builder.name.is_empty() {
                                let tc = ToolCall::new(builder.id, builder.name, builder.input_json);
                                let _ = tx.send(StreamEvent::ToolCall(tc)).await;
                            }
                        }
                    }
                    Some("error") => {
                        if let Some(message) = value["error"]["message"].as_str() {
                            if emitted {
                                return Err(Failure::Fatal(format!("API error mid-stream: {message}")));
                            }
                            return Err(classify_failure(0, message));
                        }
                    }
                    _ => {}
                }
            }
        }

        // Emit any remaining builders
        for (_, builder) in tool_builders.drain() {
            if !builder.name.is_empty() {
                let tc = ToolCall::new(builder.id, builder.name, builder.input_json);
                let _ = tx.send(StreamEvent::ToolCall(tc)).await;
            }
        }

        Ok(())
    }
}

struct AnthropicToolBuilder {
    id: String,
    name: String,
    input_json: String,
}

impl ChatBackend for AnthropicBackend {
    fn list_models(&self) -> Pin<Box<dyn Future<Output = Result<Vec<String>>> + Send + '_>> {
        Box::pin(self.list_models_inner())
    }

    fn stream_completion(
        &self,
        request: &CompletionRequest,
        tx: Sender<StreamEvent>,
    ) -> Pin<Box<dyn Future<Output = Result<(), Failure>> + Send + '_>> {
        Box::pin(self.stream_inner(request.clone(), tx))
    }
}
