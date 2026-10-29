//! An OpenAI-compatible chat backend. Covers OpenAI itself and every gateway
//! that speaks the `POST /chat/completions` dialect (custom providers such
//! as Ollama, Groq, Mistral, Together, OpenRouter, Azure OpenAI, …). This
//! is the workhorse backend: every `custom_providers` entry in `config.json`
//! uses it.
//!
//! Now with tool calling support for agent/sandbox mode.

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

pub struct OpenAICompatibleBackend {
    http: Client,
    api_key: String,
    base_url: String,
    /// How long the stream may stay quiet between chunks before the
    /// connection counts as dead. There is deliberately no whole-request
    /// timeout: it would cut long generations off mid-answer.
    idle_timeout: Duration,
}

impl OpenAICompatibleBackend {
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
            self.http.get(url).bearer_auth(&self.api_key).send(),
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
        let mut ids = parse_models(&value);
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
        // Build OpenAI messages with tool support
        let mut openai_messages = Vec::with_capacity(request.messages.len());
        for message in &request.messages {
            match message.role {
                Role::System => {
                    openai_messages.push(json!({ "role": "system", "content": message.content }));
                }
                Role::Assistant => {
                    if let Some(tool_calls) = &message.tool_calls {
                        // Assistant message with tool_calls
                        let mut tc_json = Vec::new();
                        for tc in tool_calls {
                            tc_json.push(json!({
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.name,
                                    "arguments": tc.arguments
                                }
                            }));
                        }
                        openai_messages.push(json!({
                            "role": "assistant",
                            "content": message.content,
                            "tool_calls": tc_json
                        }));
                    } else {
                        openai_messages
                            .push(json!({ "role": "assistant", "content": message.content }));
                    }
                }
                Role::User => {
                    openai_messages.push(json!({ "role": "user", "content": message.content }));
                }
                Role::Tool => {
                    // Tool result
                    let tool_call_id = message
                        .tool_call_id
                        .clone()
                        .unwrap_or_else(|| "unknown".to_string());
                    openai_messages.push(json!({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": message.content
                    }));
                }
            }
        }

        let mut body = json!({
            "model": request.model,
            "messages": openai_messages,
            "temperature": request.temperature,
            "stream": true,
        });

        // Add tools if present
        if !request.tools.is_empty() {
            // Convert our internal ToolDefinition to OpenAI format via tools module
            let tool_defs: Vec<crate::tools::ToolDefinition> = request
                .tools
                .iter()
                .map(|t| {
                    crate::tools::ToolDefinition::new(&t.name, &t.description, t.parameters.clone())
                })
                .collect();
            body["tools"] = tools::to_openai_tools(&tool_defs);
            body["tool_choice"] = json!(request.tool_choice.unwrap_or_else(|| "auto".to_string()));
        }

        check_request_size(&body)?;

        let url = format!("{}/chat/completions", self.base_url);
        let response = self
            .http
            .post(url)
            .bearer_auth(&self.api_key)
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
            let body_text = bounded_body(response, self.idle_timeout).await
                .map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
                .unwrap_or_else(|error| format!("reading error response failed: {error}"));
            // If tools not supported, try to detect and give better error
            if body_text.to_ascii_lowercase().contains("tool")
                || body_text.to_ascii_lowercase().contains("function")
            {
                // Check if it's a tools-not-supported error - we treat as retryable without tools? For now, classify normally
                // The client will fallback to other models; but we also want to inform user
            }
            return Err(classify_failure(status.as_u16(), &body_text));
        }
        let mut stream = response.bytes_stream();
        let mut reader = SseReader::new();
        // Keep-alive pings do not count as progress: require actual answer
        // text or tool-call data within the configured progress window.
        let mut progress_deadline = std::time::Instant::now() + self.idle_timeout;
        let mut emitted = false;
        let mut in_think = false;
        let mut got_visible_content = false;
        let mut bad_lines = 0usize;
        let mut finish_reason = String::new();
        // Accumulate tool calls by index
        let mut tool_builders: BTreeMap<usize, ToolCallBuilder> = BTreeMap::new();
        let mut message_calls: BTreeMap<usize, ToolCall> = BTreeMap::new();

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
                if data == "[DONE]" {
                    if in_think {
                        let _ = tx.send(StreamEvent::Delta("</think>".into())).await;
                    }
                    // Commit calls only once the attempt succeeds. A failed
                    // request can then be retried without leaving executable
                    // calls from the previous attempt in the UI's channel.
                    check_finish_reason(&finish_reason)?;
                    return emit_tool_calls(&tx, tool_builders, message_calls).await;
                }
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

                // Check for error in stream
                if let Some(message) = value["error"]["message"].as_str() {
                    if emitted {
                        return Err(Failure::Fatal(format!("API error mid-stream: {message}")));
                    }
                    return Err(classify_failure(0, message));
                }

                if let Some(reason) = value["choices"][0]["finish_reason"].as_str() {
                    finish_reason = reason.to_string();
                }

                // Content / reasoning deltas (content + reasoning_content variants)
                let (reasoning, content) = extract_delta_text(&value);
                if !reasoning.is_empty() {
                    if !in_think {
                        tx.send(StreamEvent::Delta("<think>".into()))
                            .await
                            .map_err(|_| Failure::Fatal("stream receiver closed".into()))?;
                        in_think = true;
                    }
                    tx.send(StreamEvent::Delta(reasoning))
                        .await
                        .map_err(|_| Failure::Fatal("stream receiver closed".into()))?;
                    emitted = true;
                    progress_deadline = std::time::Instant::now() + self.idle_timeout;
                }
                if !content.is_empty() {
                    if in_think {
                        tx.send(StreamEvent::Delta("</think>".into()))
                            .await
                            .map_err(|_| Failure::Fatal("stream receiver closed".into()))?;
                        in_think = false;
                    }
                    tx.send(StreamEvent::Delta(content))
                        .await
                        .map_err(|_| Failure::Fatal("stream receiver closed".into()))?;
                    emitted = true;
                    got_visible_content = true;
                    progress_deadline = std::time::Instant::now() + self.idle_timeout;
                } else if !got_visible_content {
                    if let Some(text) = value["choices"][0]["message"]["content"].as_str() {
                        if !text.is_empty() {
                            if in_think {
                                tx.send(StreamEvent::Delta("</think>".into()))
                                    .await
                                    .map_err(|_| Failure::Fatal("stream receiver closed".into()))?;
                                in_think = false;
                            }
                            tx.send(StreamEvent::Delta(text.into()))
                                .await
                                .map_err(|_| Failure::Fatal("stream receiver closed".into()))?;
                            emitted = true;
                            got_visible_content = true;
                            progress_deadline = std::time::Instant::now() + self.idle_timeout;
                        }
                    }
                }

                // Tool calls delta - OpenAI streams tool_calls as array
                if let Some(tool_calls) = value["choices"][0]["delta"]["tool_calls"].as_array() {
                    for tc in tool_calls {
                        let index = tc["index"].as_u64().unwrap_or(0) as usize;
                        if !tool_builders.contains_key(&index)
                            && tool_builders.len() >= MAX_TOOL_CALLS_PER_TURN
                        {
                            return Err(Failure::Fatal("too many tool calls in one response".into()));
                        }
                        let builder = tool_builders
                            .entry(index)
                            .or_insert_with(|| ToolCallBuilder::new(index));
                        let mut advanced = false;
                        if let Some(id) = tc["id"].as_str() {
                            if id.len() > 256 {
                                return Err(Failure::Fatal("tool call id too long".into()));
                            }
                            if !id.is_empty() && builder.id != id {
                                builder.id = id.to_string();
                                advanced = true;
                            }
                        }
                        if let Some(name) = tc["function"]["name"].as_str() {
                            if name.len() > 256 {
                                return Err(Failure::Fatal("tool name too long".into()));
                            }
                            if !name.is_empty() && builder.name != name {
                                builder.name = name.to_string();
                                advanced = true;
                            }
                        }
                        if let Some(args) = tc["function"]["arguments"].as_str() {
                            if builder.arguments.len().saturating_add(args.len()) > MAX_TOOL_ARGUMENT_BYTES {
                                return Err(Failure::Fatal("tool arguments exceeded the size limit".into()));
                            }
                            if !args.is_empty() {
                                builder.arguments.push_str(args);
                                advanced = true;
                            }
                        }
                        if advanced {
                            progress_deadline = std::time::Instant::now() + self.idle_timeout;
                        }
                    }
                }

                // Some gateways send complete message calls instead of deltas.
                // Index-keyed replacement tolerates cumulative repeated frames;
                // delta builders take precedence if both forms are present.
                if let Some(tool_calls) = value["choices"][0]["message"]["tool_calls"].as_array() {
                    if tool_calls.len() > MAX_TOOL_CALLS_PER_TURN {
                        return Err(Failure::Fatal("too many tool calls in one response".into()));
                    }
                    for (index, tc) in tool_calls.iter().enumerate() {
                        let id = tc["id"].as_str().unwrap_or("");
                        let name = tc["function"]["name"].as_str().unwrap_or("");
                        let args = tc["function"]["arguments"].as_str().unwrap_or("{}");
                        if id.len() > 256 || name.len() > 256 || args.len() > MAX_TOOL_ARGUMENT_BYTES {
                            return Err(Failure::Fatal("tool call exceeded the size limit".into()));
                        }
                        if !name.is_empty() {
                            let call = ToolCall::new(id, name, args);
                            if message_calls.get(&index) != Some(&call) {
                                progress_deadline = std::time::Instant::now() + self.idle_timeout;
                                message_calls.insert(index, call);
                            }
                        }
                    }
                }
                // `finish_reason: tool_calls` is not a commit: the stream may
                // still fail before [DONE] / a clean close. Keep the builders.
            }
        }

        if in_think {
            let _ = tx.send(StreamEvent::Delta("</think>".into())).await;
        }
        // A clean connection close without [DONE] is accepted by some
        // gateways; no calls have been published during the attempt.
        check_finish_reason(&finish_reason)?;
        if finish_reason.is_empty() && (!tool_builders.is_empty() || !message_calls.is_empty()) {
            return Err(Failure::Fatal("tool-call stream closed without a completion marker; no calls were run".into()));
        }
        emit_tool_calls(&tx, tool_builders, message_calls).await
    }
}

fn check_finish_reason(reason: &str) -> Result<(), Failure> {
    match reason {
        "" | "stop" | "tool_calls" | "function_call" => Ok(()),
        other => Err(Failure::Fatal(format!("model stopped before completing its response ({other}); tool calls were not run"))),
    }
}

async fn emit_tool_calls(
    tx: &Sender<StreamEvent>,
    builders: BTreeMap<usize, ToolCallBuilder>,
    message_calls: BTreeMap<usize, ToolCall>,
) -> Result<(), Failure> {
    let calls: Vec<ToolCall> = if builders.is_empty() {
        message_calls.into_values().collect()
    } else {
        builders.into_values().map(|builder| {
            builder.build().ok_or_else(|| Failure::Fatal(
                "tool call omitted its function name; no calls were run".into()
            ))
        }).collect::<Result<Vec<_>, _>>()?
    };
    for call in calls {
        tx.send(StreamEvent::ToolCall(call)).await
            .map_err(|_| Failure::Fatal("stream receiver closed".into()))?;
    }
    Ok(())
}

#[derive(Debug)]
struct ToolCallBuilder {
    index: usize,
    id: String,
    name: String,
    arguments: String,
}

impl ToolCallBuilder {
    fn new(index: usize) -> Self {
        Self {
            index,
            id: String::new(),
            name: String::new(),
            arguments: String::new(),
        }
    }

    fn build(self) -> Option<ToolCall> {
        if self.name.is_empty() {
            return None;
        }
        let id = if self.id.is_empty() {
            format!("call_{}_{}", self.index, self.name)
        } else {
            self.id
        };
        Some(ToolCall::new(id, self.name, self.arguments))
    }
}

impl ChatBackend for OpenAICompatibleBackend {
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

/// Pull visible + reasoning text out of one OpenAI-compatible SSE object.
/// Some models (and gateways) stream thinking in `reasoning_content` /
/// `reasoning` while the user-visible answer is `delta.content`.
fn extract_delta_text(value: &Value) -> (String, String) {
    let delta = &value["choices"][0]["delta"];
    if delta.is_null() {
        return (String::new(), String::new());
    }
    let reasoning = ["reasoning_content", "reasoning", "reasoning_text"]
        .iter()
        .find_map(|key| {
            delta[key]
                .as_str()
                .filter(|text| !text.is_empty())
                .map(str::to_string)
        })
        .unwrap_or_default();
    let content = delta["content"]
        .as_str()
        .filter(|text| !text.is_empty())
        .unwrap_or("")
        .to_string();
    (reasoning, content)
}

/// Pull model ids out of an OpenAI-compatible `GET /models` response: either
/// `{"data": [{"id": …}, …]}` or a bare `[{"id": …}, …]` array.
fn parse_models(value: &Value) -> Vec<String> {
    let entries: &[Value] = match value {
        Value::Array(items) => items,
        value => value["data"].as_array().map(Vec::as_slice).unwrap_or(&[]),
    };
    entries
        .iter()
        .filter_map(|item| item["id"].as_str().map(str::to_string))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_openai_and_bare_model_lists() {
        let openai = serde_json::json!({
            "object": "list",
            "data": [{"id": "b"}, {"id": "a"}, {"object": "model"}]
        });
        assert_eq!(parse_models(&openai), vec!["b", "a"]);
        let bare = serde_json::json!([{"id": "y"}, {"id": "x"}]);
        assert_eq!(parse_models(&bare), vec!["y", "x"]);
        assert!(parse_models(&serde_json::json!({})).is_empty());
    }

    #[test]
    fn tool_builder_accumulates() {
        let mut b = ToolCallBuilder::new(0);
        b.id = "call_123".into();
        b.name = "read_file".into();
        b.arguments.push_str("{\"path\":");
        b.arguments.push_str(" \"src/main.rs\"}");
        let tc = b.build().unwrap();
        assert_eq!(tc.name, "read_file");
        assert!(tc.arguments.contains("src/main.rs"));
    }

    #[test]
    fn reasoning_content_is_not_dropped() {
        let reasoning_only = serde_json::json!({
            "choices": [{
                "delta": {
                    "role": "assistant",
                    "content": null,
                    "reasoning_content": "Let me think."
                }
            }]
        });
        let (reasoning, content) = extract_delta_text(&reasoning_only);
        assert_eq!(reasoning, "Let me think.");
        assert!(content.is_empty());

        let answer = serde_json::json!({
            "choices": [{
                "delta": {
                    "content": "Hello there.",
                    "reasoning_content": ""
                }
            }]
        });
        let (reasoning, content) = extract_delta_text(&answer);
        assert!(reasoning.is_empty());
        assert_eq!(content, "Hello there.");
    }
}
