//! The Anthropic Messages API backend (`POST /v1/messages`).
//!
//! Note the differences from the OpenAI dialect that this adapter papers over:
//! - the system prompt is a top-level `system` field, not a message;
//! - message content is a list of content blocks (`{"type":"text", …}`);
//! - `max_tokens` is mandatory (a conservative output bound, unrelated to
//!   the model's input context window);
//! - streaming deltas arrive as `content_block_delta` events with
//!   `delta.text`, and the model list is `{"data":[{"id": …}]}`.
//! - Tool calling uses `tools` + `tool_use` blocks and `tool_result` in user messages.

use super::{bounded_body, check_request_size, ChatBackend, HttpTimeouts};
use crate::api::error::{classify_failure, error_detail};
use crate::api::sse::{SseReader, MAX_CONSECUTIVE_SSE_PARSE_FAILURES};
use crate::api::types::{
    CompletionRequest, Failure, Role, StreamEvent, ToolCall, MAX_TOOL_ARGUMENT_BYTES,
    MAX_TOOL_CALLS_PER_TURN,
};
use crate::tools;
use anyhow::{anyhow, Context, Result};
use futures_util::StreamExt;
use reqwest::Client;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;
use std::time::Duration;
use tokio::sync::mpsc::Sender;

/// Conservative output cap. The model's input context window does *not*
/// describe the maximum number of tokens it may generate.
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
        let bytes = bounded_body(response, self.idle_timeout).await
            .context("reading the model list response")?;
        if !status.is_success() {
            return Err(anyhow!(
                "model list request failed (HTTP {}): {}",
                status,
                error_detail(&String::from_utf8_lossy(&bytes))
            ));
        }
        let value: Value = serde_json::from_slice(&bytes)
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
                            let input: Value =
                                serde_json::from_str(&tc.arguments).unwrap_or(json!({}));
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
                    let tool_use_id = message
                        .tool_call_id
                        .clone()
                        .unwrap_or_else(|| "unknown".to_string());
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
            "max_tokens": DEFAULT_MAX_TOKENS,
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
                .map(|t| {
                    crate::tools::ToolDefinition::new(&t.name, &t.description, t.parameters.clone())
                })
                .collect();
            body["tools"] = tools::to_anthropic_tools(&tool_defs);
        }

        check_request_size(&body)?;

        let url = format!("{}/messages", self.base_url);
        let response = self
            .http
            .post(url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .json(&body)
            .send();
        let response = tokio::time::timeout(self.idle_timeout, response)
            .await
            .map_err(|_| Failure::Transient(format!(
                "request headers timed out after {}s", self.idle_timeout.as_secs()
            )))?
            .map_err(|error| Failure::Transient(format!("request failed: {error}")))?;
        let status = response.status();
        if !status.is_success() {
            let body = bounded_body(response, self.idle_timeout).await
                .map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
                .unwrap_or_else(|error| format!("reading error response failed: {error}"));
            return Err(classify_failure(status.as_u16(), &body));
        }
        let mut stream = response.bytes_stream();
        let mut reader = SseReader::new();
        // Keep-alive pings do not count as progress: require actual answer
        // text or tool-call data within the configured progress window.
        let mut progress_deadline = std::time::Instant::now() + self.idle_timeout;
        let mut emitted = false;
        let mut bad_lines = 0usize;
        let mut stop_reason = String::new();
        let mut saw_message_stop = false;
        let mut tool_builders: BTreeMap<usize, AnthropicToolBuilder> = BTreeMap::new();
        let mut completed_calls: BTreeMap<usize, ToolCall> = BTreeMap::new();

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
                let remaining = progress_deadline.saturating_duration_since(std::time::Instant::now());
                match tokio::time::timeout(remaining, stream.next()).await {
                    Ok(Some(Ok(chunk))) => reader.feed(&chunk)
                        .map_err(|error| Failure::Fatal(error.into()))?,
                    Ok(Some(Err(error))) => {
                        return Err(if emitted {
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
                        return Err(if emitted {
                            Failure::Fatal(format!(
                                "no model progress for {secs}s after output started"
                            ))
                        } else {
                            Failure::Transient(format!(
                                "no model progress for {secs}s before any output"
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
                                let id = value["content_block"]["id"]
                                    .as_str()
                                    .unwrap_or("")
                                    .to_string();
                                let name = value["content_block"]["name"]
                                    .as_str()
                                    .unwrap_or("")
                                    .to_string();
                                if id.len() > 256 || name.len() > 256
                                    || tool_builders.len() + completed_calls.len() >= MAX_TOOL_CALLS_PER_TURN
                                {
                                    return Err(Failure::Fatal("tool call exceeded the response limit".into()));
                                }
                                let fallback_input = value["content_block"]["input"].is_object()
                                    .then(|| serde_json::to_string(&value["content_block"]["input"]).unwrap_or_else(|_| "{}".into()));
                                if fallback_input.as_ref().is_some_and(|input| input.len() > MAX_TOOL_ARGUMENT_BYTES) {
                                    return Err(Failure::Fatal("tool arguments exceeded the size limit".into()));
                                }
                                if tool_builders.contains_key(&index) || completed_calls.contains_key(&index) {
                                    return Err(Failure::Fatal("duplicate tool block index in response".into()));
                                }
                                progress_deadline = std::time::Instant::now() + self.idle_timeout;
                                tool_builders.insert(index, AnthropicToolBuilder {
                                    id, name, input_json: String::new(), fallback_input,
                                });
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
                            if !text.is_empty() {
                                progress_deadline = std::time::Instant::now() + self.idle_timeout;
                            }
                        }
                        if let Some(partial) = value["delta"]["partial_json"].as_str() {
                            if let Some(builder) = tool_builders.get_mut(&index) {
                                if builder.input_json.len().saturating_add(partial.len()) > MAX_TOOL_ARGUMENT_BYTES {
                                    return Err(Failure::Fatal("tool arguments exceeded the size limit".into()));
                                }
                                builder.input_json.push_str(partial);
                                if !partial.is_empty() {
                                    progress_deadline = std::time::Instant::now() + self.idle_timeout;
                                }
                            }
                        }
                    }
                    Some("content_block_stop") => {
                        let index = value["index"].as_u64().unwrap_or(0) as usize;
                        if let Some(builder) = tool_builders.remove(&index) {
                            if builder.name.is_empty() {
                                return Err(Failure::Fatal("tool use omitted its function name".into()));
                            }
                            completed_calls.insert(index, builder.build());
                        }
                    }
                    Some("message_stop") => {
                        saw_message_stop = true;
                    }
                    Some("message_delta") => {
                        if let Some(reason) = value["delta"]["stop_reason"].as_str() {
                            stop_reason = reason.to_string();
                        }
                    }
                    Some("error") => {
                        if let Some(message) = value["error"]["message"].as_str() {
                            if emitted {
                                return Err(Failure::Fatal(format!(
                                    "API error mid-stream: {message}"
                                )));
                            }
                            return Err(classify_failure(0, message));
                        }
                    }
                    _ => {}
                }
            }
        }

        if !matches!(stop_reason.as_str(), "" | "end_turn" | "tool_use" | "stop_sequence") {
            return Err(Failure::Fatal(format!(
                "model stopped before completing its response ({stop_reason}); tool calls were not run"
            )));
        }
        if !saw_message_stop && (!tool_builders.is_empty() || !completed_calls.is_empty()) {
            return Err(Failure::Fatal("tool-call stream closed without message_stop; no calls were run".into()));
        }
        // Publish executable calls only after this attempt finished cleanly.
        // A retry/fallback cannot then leave calls from an abandoned attempt.
        for (index, builder) in tool_builders {
            if builder.name.is_empty() {
                return Err(Failure::Fatal("tool use omitted its function name".into()));
            }
            completed_calls.insert(index, builder.build());
        }
        for call in completed_calls.into_values() {
            tx.send(StreamEvent::ToolCall(call)).await
                .map_err(|_| Failure::Fatal("stream receiver closed".into()))?;
        }
        Ok(())
    }
}

struct AnthropicToolBuilder {
    id: String,
    name: String,
    input_json: String,
    fallback_input: Option<String>,
}

impl AnthropicToolBuilder {
    fn build(self) -> ToolCall {
        let args = if self.input_json.is_empty() {
            self.fallback_input.unwrap_or_else(|| "{}".into())
        } else {
            self.input_json
        };
        ToolCall::new(self.id, self.name, args)
    }
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
