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
use crate::api::sse::{abnormal_stream_end, first_token_timeout_failure, SseReader};
use crate::api::types::{CompletionRequest, Failure, Role, StreamEvent, ToolCall};
use crate::tools;
use anyhow::{anyhow, Context, Result};
use futures_util::StreamExt;
use reqwest::Client;
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
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
    /// How long the stream may stay quiet between chunks *after output has
    /// started* before the connection counts as dead. There is deliberately
    /// no whole-request timeout: it would cut long generations off
    /// mid-answer.
    idle_timeout: Duration,
    /// How long to wait for the model's first output. Extended thinking can
    /// run for many minutes before any visible text arrives, so this is much
    /// longer than [`Self::idle_timeout`].
    first_token_timeout: Duration,
    /// Extended-thinking budget. `None` means do not ask for thinking, in
    /// which case there is nothing to round-trip either.
    thinking_budget: Option<u32>,
    /// User-configured cap on tokens per response, overriding the value
    /// derived from the model list. `0` means use the derived value.
    max_output_tokens: u64,
}

impl AnthropicBackend {
    pub fn new(
        api_key: String,
        base_url: String,
        timeouts: HttpTimeouts,
        thinking_budget: Option<u32>,
        max_output_tokens: u64,
    ) -> Self {
        Self {
            http: Client::builder()
                .connect_timeout(timeouts.connect)
                .build()
                .expect("HTTP client"),
            api_key,
            base_url: base_url.trim_end_matches('/').into(),
            idle_timeout: timeouts.idle,
            first_token_timeout: timeouts.first_token,
            thinking_budget,
            max_output_tokens,
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
                        // A signed thinking block must come back verbatim and
                        // must precede the text and tool_use blocks it was
                        // issued with, or the request is rejected.
                        if let Some(block) = thinking_block(&message.reasoning) {
                            content_blocks.push(block);
                        }
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
                        let mut content_blocks = Vec::new();
                        if let Some(block) = thinking_block(&message.reasoning) {
                            content_blocks.push(block);
                        }
                        content_blocks.push(json!({ "type": "text", "text": message.content }));
                        messages.push(json!({
                            "role": "assistant",
                            "content": content_blocks,
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
        // A user-configured cap wins over the value derived from the model
        // list; `0` means "not configured", so the derived value stands.
        let max_tokens = if self.max_output_tokens > 0 {
            self.max_output_tokens
        } else {
            self.max_tokens_for(&request.model).await
        };
        let mut body = json!({
            "model": request.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": request.temperature,
            "stream": true,
        });
        if let Some(text) = system {
            body["system"] = Value::String(text);
        }
        // Extended thinking is opt-in, and the API rejects a temperature
        // other than the default while it is on — so asking for thinking
        // means giving up the temperature knob for that request.
        if let Some(budget) = self.thinking_budget {
            body["thinking"] = json!({ "type": "enabled", "budget_tokens": budget });
            body.as_object_mut().map(|body| body.remove("temperature"));
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
        // Block indices that are extended-thinking rather than text. Their
        // deltas are streamed as `<think>` so the transcript and the status
        // row behave exactly as they do for models that emit the tag inline,
        // and the signature that arrives with them is kept for replay.
        let mut thinking_blocks: HashSet<usize> = HashSet::new();
        let mut reasoning_signature: Option<String> = None;
        // Whether the provider ended the stream the way it intended: its
        // `message_stop` event, or a `stop_reason` on a `message_delta`.
        let mut saw_stop = false;
        let mut stop_reason: Option<String> = None;
        // Whether any tool call was emitted. `tool_builders` is drained
        // before the ending is classified, so this has to be remembered.
        let mut emitted_tool_call = false;

        let mut stream_ended = false;
        loop {
            // Two idle windows: a generous one until the model produces
            // anything (extended thinking can run for many minutes), and a
            // tight one between chunks once output has started. Neither caps
            // the total response time — a stream that keeps producing tokens
            // is never cut off mid-answer.
            //
            // When the stream closes cleanly, whatever trailing frame the
            // reader still holds (a provider may end without a final
            // newline) is processed too — losing it can truncate tool-call
            // arguments.
            let output_started = emitted || emitted_tool_call || !tool_builders.is_empty();
            let quiet_limit = if output_started {
                self.idle_timeout
            } else {
                self.first_token_timeout
            };
            let payloads = if stream_ended || saw_stop {
                break;
            } else {
                match tokio::time::timeout(quiet_limit, stream.next()).await {
                    Ok(Some(Ok(chunk))) => reader.feed(&chunk),
                    Ok(Some(Err(error))) => {
                        return Err(if output_started {
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
                    Err(_elapsed) if output_started => {
                        return Err(Failure::Fatal(format!(
                            "the connection went quiet for {}s after output started",
                            self.idle_timeout.as_secs()
                        )));
                    }
                    Err(_elapsed) => {
                        // No output yet, so the long first-token window
                        // applied. A long wait means the model was very
                        // likely still thinking, and a retry would throw
                        // that away and restart the same wait — so past a
                        // threshold this is reported instead of retried.
                        return Err(first_token_timeout_failure(self.first_token_timeout));
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
                            if block_type == "thinking" {
                                thinking_blocks.insert(index);
                                tx.send(StreamEvent::Delta(crate::ui::thinking::OPEN_TAG.into()))
                                    .await
                                    .map_err(|_| Failure::Fatal("stream receiver closed".into()))?;
                                emitted = true;
                            }
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
                        if let Some(thought) = value["delta"]["thinking"].as_str() {
                            tx.send(StreamEvent::Delta(thought.into()))
                                .await
                                .map_err(|_| Failure::Fatal("stream receiver closed".into()))?;
                            emitted = true;
                        }
                        // The signature validates this block on replay. It
                        // arrives with the block it belongs to.
                        if let Some(signature) = value["delta"]["signature"].as_str() {
                            if !signature.is_empty() {
                                reasoning_signature = Some(signature.to_string());
                            }
                        }
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
                        if thinking_blocks.remove(&index) {
                            tx.send(StreamEvent::Delta(crate::ui::thinking::CLOSE_TAG.into()))
                                .await
                                .map_err(|_| Failure::Fatal("stream receiver closed".into()))?;
                        }
                        if let Some(builder) = tool_builders.remove(&index) {
                            if !builder.name.is_empty() {
                                let tc = ToolCall::new(builder.id, builder.name, builder.input_json);
                                emitted_tool_call = true;
                                let _ = tx.send(StreamEvent::ToolCall(tc)).await;
                            }
                        }
                    }
                    // The provider's own end-of-stream signals. Without
                    // these, a connection that drops mid-answer looks
                    // exactly like a finished one.
                    Some("message_stop") => {
                        saw_stop = true;
                    }
                    // message_start carries the input count, message_delta the
                    // output count; neither alone is the whole picture.
                    Some("message_start") => {
                        if let Some(usage) = crate::api::types::usage_from(&value["message"]) {
                            let _ = tx.send(StreamEvent::Usage(usage)).await;
                        }
                    }
                    Some("message_delta") => {
                        if let Some(usage) = crate::api::types::usage_from(&value) {
                            let _ = tx.send(StreamEvent::Usage(usage)).await;
                        }
                        if let Some(reason) = value["delta"]["stop_reason"].as_str() {
                            stop_reason = Some(reason.to_string());
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
                emitted_tool_call = true;
                let _ = tx.send(StreamEvent::ToolCall(tc)).await;
            }
        }

        // Classify the ending: a response cut short by a dropped connection
        // or by `max_tokens` must not be passed off as a finished answer.
        if !saw_stop && stop_reason.is_none() && !emitted && !emitted_tool_call {
            // Nothing arrived at all, so a retry costs nothing and may work.
            return Err(Failure::Transient(
                "the provider closed the stream before sending anything".into(),
            ));
        }
        // Hand the signature over last: it belongs to the reasoning this turn
        // produced, and the app attaches it to the assistant message it is
        // about to store.
        if let Some(signature) = reasoning_signature {
            let _ = tx.send(StreamEvent::ReasoningSignature(signature)).await;
        }
        if let Some(notice) = abnormal_stream_end(saw_stop, stop_reason.as_deref()) {
            let output_limit = crate::api::sse::ended_by_output_limit(stop_reason.as_deref());
            let _ = tx.send(StreamEvent::Notice(notice)).await;
            let _ = tx.send(StreamEvent::EndedEarly { output_limit }).await;
        }

        Ok(())
    }
}

/// Rebuild the thinking block a signed reasoning payload came from.
///
/// Returns `None` when there is no signature: an unsigned thinking block is
/// not something this API accepts, and sending one turns a working request
/// into a 400.
fn thinking_block(reasoning: &Option<crate::api::types::Reasoning>) -> Option<Value> {
    let reasoning = reasoning.as_ref()?;
    let signature = reasoning.signature.as_ref()?;
    if reasoning.text.trim().is_empty() {
        return None;
    }
    Some(json!({
        "type": "thinking",
        "thinking": reasoning.text,
        "signature": signature,
    }))
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
