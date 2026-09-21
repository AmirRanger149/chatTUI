//! An OpenAI-compatible chat backend. Covers OpenAI itself and every gateway
//! that speaks the `POST /chat/completions` dialect (custom providers such
//! as Dahl, APInex, Ollama, Groq, Mistral, Together, Azure OpenAI, …). This
//! is the workhorse backend: every `custom_providers` entry in `config.json`
//! uses it.
//! 
//! Now with tool calling support for agent/sandbox mode.

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
                        openai_messages.push(json!({ "role": "assistant", "content": message.content }));
                    }
                }
                Role::User => {
                    openai_messages.push(json!({ "role": "user", "content": message.content }));
                }
                Role::Tool => {
                    // Tool result
                    let tool_call_id = message.tool_call_id.clone().unwrap_or_else(|| "unknown".to_string());
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
                .map(|t| crate::tools::ToolDefinition::new(&t.name, &t.description, t.parameters.clone()))
                .collect();
            body["tools"] = tools::to_openai_tools(&tool_defs);
            body["tool_choice"] = json!(request.tool_choice.unwrap_or_else(|| "auto".to_string()));
        }

        let url = format!("{}/chat/completions", self.base_url);
        let response = self
            .http
            .post(url)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await
            .map_err(|error| Failure::Transient(format!("request failed: {error}")))?;
        let status = response.status();
        if !status.is_success() {
            let body_text = response.text().await.unwrap_or_default();
            // If tools not supported, try to detect and give better error
            if body_text.to_ascii_lowercase().contains("tool") || body_text.to_ascii_lowercase().contains("function") {
                // Check if it's a tools-not-supported error - we treat as retryable without tools? For now, classify normally
                // The client will fallback to other models; but we also want to inform user
            }
            return Err(classify_failure(status.as_u16(), &body_text));
        }
        let mut stream = response.bytes_stream();
        let mut reader = SseReader::new();
        let mut emitted = false;
        let mut in_think = false;
        let mut got_visible_content = false;
        let mut bad_lines = 0usize;
        // Accumulate tool calls by index
        let mut tool_builders: HashMap<usize, ToolCallBuilder> = HashMap::new();

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
                if data == "[DONE]" {
                    if in_think {
                        let _ = tx.send(StreamEvent::Delta("</think>".into())).await;
                    }
                    // Emit any accumulated tool calls
                    for (_, builder) in tool_builders.drain() {
                        if let Some(tc) = builder.build() {
                            let _ = tx.send(StreamEvent::ToolCall(tc)).await;
                        }
                    }
                    return Ok(());
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
                    if emitted || !tool_builders.is_empty() {
                        return Err(Failure::Fatal(format!("API error mid-stream: {message}")));
                    }
                    return Err(classify_failure(0, message));
                }

                // Content / reasoning deltas (OpenAI content + MiniMax reasoning_content)
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
                        }
                    }
                }

                // Tool calls delta - OpenAI streams tool_calls as array
                if let Some(tool_calls) = value["choices"][0]["delta"]["tool_calls"].as_array() {
                    for tc in tool_calls {
                        let index = tc["index"].as_u64().unwrap_or(0) as usize;
                        let builder = tool_builders.entry(index).or_insert_with(|| ToolCallBuilder::new(index));
                        if let Some(id) = tc["id"].as_str() {
                            if !id.is_empty() {
                                builder.id = id.to_string();
                            }
                        }
                        if let Some(name) = tc["function"]["name"].as_str() {
                            if !name.is_empty() {
                                builder.name = name.to_string();
                            }
                        }
                        if let Some(args) = tc["function"]["arguments"].as_str() {
                            builder.arguments.push_str(args);
                        }
                    }
                }

                // Some providers send tool_calls in message (non-delta) at end
                if let Some(tool_calls) = value["choices"][0]["message"]["tool_calls"].as_array() {
                    for tc in tool_calls {
                        let id = tc["id"].as_str().unwrap_or("").to_string();
                        let name = tc["function"]["name"].as_str().unwrap_or("").to_string();
                        let args = tc["function"]["arguments"].as_str().unwrap_or("{}").to_string();
                        if !name.is_empty() {
                            let _ = tx
                                .send(StreamEvent::ToolCall(ToolCall::new(id, name, args)))
                                .await;
                        }
                    }
                }

                // Handle finish_reason to emit tool calls
                if let Some(finish) = value["choices"][0]["finish_reason"].as_str() {
                    if finish == "tool_calls" {
                        for (_, builder) in tool_builders.drain() {
                            if let Some(tc) = builder.build() {
                                let _ = tx.send(StreamEvent::ToolCall(tc)).await;
                            }
                        }
                    }
                }
            }
        }

        if in_think {
            let _ = tx.send(StreamEvent::Delta("</think>".into())).await;
        }
        // Stream ended without [DONE] - emit any remaining tool calls
        for (_, builder) in tool_builders.drain() {
            if let Some(tc) = builder.build() {
                let _ = tx.send(StreamEvent::ToolCall(tc)).await;
            }
        }

        Ok(())
    }
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
/// MiniMax (and some gateways) stream thinking in `reasoning_content` /
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
    fn minimax_reasoning_content_is_not_dropped() {
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
