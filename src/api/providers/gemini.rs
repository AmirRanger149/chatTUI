//! The Google Gemini backend (`POST /v1beta/models/{model}:streamGenerateContent`).
//!
//! Gemini's dialect differs from OpenAI's in several ways the adapter papers
//! over:
//! - streaming is enabled with a `?alt=sse` query parameter (without it the
//!   endpoint returns one JSON array instead of a stream);
//! - the system prompt goes in `systemInstruction` (a single content block);
//! - turns are `contents` with a `parts: [{text}]` list, alternating
//!   `user`/`model` (an assistant reply is sent as `role: "model"`);
//! - the model id is part of the URL (the `{model}` from the endpoint may be
//!   a `models/`-prefixed display name, so the prefix is normalised away);
//! - the key travels as an `x-goog-api-key` header, not a bearer token;
//! - the streamed `parts[].text` is *cumulative* (each chunk repeats the whole
//!   answer so far), so only the new suffix is emitted, and thinking models
//!   put their reasoning in `"thought": true` parts which are skipped;
//! - the model list lives under `models[]` with a `displayName` (usually the
//!   full `models/{name}` path).
//! - Tool calling uses `functionDeclarations` and `functionCall`/`functionResponse`.

use super::{ChatBackend, HttpTimeouts};
use crate::api::error::{classify_failure, error_detail, MAX_CONSECUTIVE_SSE_PARSE_FAILURES};
use crate::api::sse::SseReader;
use crate::api::types::{CompletionRequest, Failure, Role, StreamEvent, ToolCall};
use crate::tools;
use anyhow::{anyhow, Context, Result};
use futures_util::StreamExt;
use reqwest::Client;
use serde_json::{json, Value};
use std::future::Future;
use std::pin::Pin;
use std::time::Duration;
use tokio::sync::mpsc::Sender;

pub struct GeminiBackend {
    http: Client,
    api_key: String,
    base_url: String,
    /// How long the stream may stay quiet between chunks before the
    /// connection counts as dead. There is deliberately no whole-request
    /// timeout: it would cut long generations off mid-answer.
    idle_timeout: Duration,
}

impl GeminiBackend {
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

    /// Turn a model id into the `{model}` URL segment. Accepts `gemini-2.0-x`,
    /// `models/gemini-2.0-x`, or a full `publishers/…/models/…` name.
    fn path_model(model: &str) -> String {
        let path = model.trim();
        if let Some(pos) = path.rfind("models/") {
            return path[pos + "models/".len()..].to_string();
        }
        path.to_string()
    }

    async fn list_models_inner(&self) -> Result<Vec<String>> {
        let url = format!("{}/models", self.base_url);
        let response = tokio::time::timeout(
            self.idle_timeout,
            self.http
                .get(url)
                .header("x-goog-api-key", &self.api_key)
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
        let mut ids = value["models"]
            .as_array()
            .map(Vec::as_slice)
            .unwrap_or(&[])
            .iter()
            .filter_map(|item| {
                item["name"]
                    .as_str()
                    .map(str::to_string)
                    .or_else(|| item["displayName"].as_str().map(str::to_string))
            })
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
        let mut system: Option<String> = None;
        let mut contents = Vec::with_capacity(request.messages.len());
        for message in &request.messages {
            match message.role {
                Role::System => {
                    system = Some(message.content.clone());
                }
                Role::Assistant => {
                    if let Some(tool_calls) = &message.tool_calls {
                        let mut parts = Vec::new();
                        if !message.content.is_empty() {
                            parts.push(json!({ "text": message.content }));
                        }
                        for tc in tool_calls {
                            let args: Value = serde_json::from_str(&tc.arguments).unwrap_or(json!({}));
                            parts.push(json!({
                                "functionCall": {
                                    "name": tc.name,
                                    "args": args
                                }
                            }));
                        }
                        contents.push(json!({
                            "role": "model",
                            "parts": parts
                        }));
                    } else {
                        contents.push(json!({
                            "role": "model",
                            "parts": [{ "text": message.content }],
                        }));
                    }
                }
                Role::User => contents.push(json!({
                    "role": "user",
                    "parts": [{ "text": message.content }],
                })),
                Role::Tool => {
                    // Tool result -> functionResponse
                    let tool_name = message.tool_call_id.clone().unwrap_or_else(|| "unknown".to_string());
                    // tool_call_id actually holds the function name for Gemini? We need to parse.
                    // For Gemini, we store id as name for simplicity, but we need to handle.
                    // We'll use the content as response, and need to know function name.
                    // We'll store tool name in tool_call_id as "name:id" or just name.
                    let (func_name, _) = if let Some((name, _id)) = tool_name.split_once(':') {
                        (name, _id)
                    } else {
                        (tool_name.as_str(), "")
                    };
                    let func_name = if func_name.is_empty() { "unknown" } else { func_name };
                    contents.push(json!({
                        "role": "user",
                        "parts": [{
                            "functionResponse": {
                                "name": func_name,
                                "response": { "result": message.content }
                            }
                        }]
                    }));
                }
            }
        }
        let mut body = json!({
            "contents": contents,
            "generationConfig": { "temperature": request.temperature },
        });
        if let Some(text) = system {
            body["systemInstruction"] = json!({ "parts": [{ "text": text }] });
        }

        // Add tools if present
        if !request.tools.is_empty() {
            let tool_defs: Vec<crate::tools::ToolDefinition> = request
                .tools
                .iter()
                .map(|t| crate::tools::ToolDefinition::new(&t.name, &t.description, t.parameters.clone()))
                .collect();
            body["tools"] = tools::to_gemini_tools(&tool_defs);
        }

        let model = Self::path_model(&request.model);
        let url = format!(
            "{}/models/{model}:streamGenerateContent?alt=sse",
            self.base_url
        );
        let response = self
            .http
            .post(url)
            .header("x-goog-api-key", &self.api_key)
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
        let mut emitted = String::new();
        let mut any_text = false;
        let mut finish_reason = String::new();
        let mut pending_tool_calls: Vec<ToolCall> = Vec::new();
        let mut bad_lines = 0usize;

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
                        return Err(if any_text || !pending_tool_calls.is_empty() {
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
                        return Err(if any_text || !pending_tool_calls.is_empty() {
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
                    for tc in pending_tool_calls.drain(..) {
                        let _ = tx.send(StreamEvent::ToolCall(tc)).await;
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
                if let Some(reason) = value["promptFeedback"]["blockReason"].as_str() {
                    return Err(Failure::Fatal(format!(
                        "Gemini blocked the prompt: {reason}"
                    )));
                }
                if let Some(message) = value["error"]["message"].as_str() {
                    if any_text {
                        return Err(Failure::Fatal(format!("API error mid-stream: {message}")));
                    }
                    return Err(classify_failure(0, message));
                }
                if let Some(reason) = value["candidates"][0]["finishReason"].as_str() {
                    finish_reason = reason.to_string();
                }

                // Check for functionCall
                if let Some(parts) = value["candidates"][0]["content"]["parts"].as_array() {
                    for part in parts {
                        if let Some(func_call) = part.get("functionCall") {
                            if let Some(name) = func_call["name"].as_str() {
                                let args = func_call["args"].clone();
                                let args_str = if args.is_object() {
                                    serde_json::to_string(&args).unwrap_or_else(|_| "{}".to_string())
                                } else {
                                    "{}".to_string()
                                };
                                let id = format!("{}:{}", name, pending_tool_calls.len());
                                let tc = ToolCall::new(id, name, args_str);
                                pending_tool_calls.push(tc);
                            }
                        }
                    }
                }

                // Text handling - cumulative
                let mut full = String::new();
                if let Some(parts) = value["candidates"][0]["content"]["parts"].as_array() {
                    for part in parts {
                        if part["thought"].as_bool() == Some(true) {
                            continue;
                        }
                        if part.get("functionCall").is_some() {
                            continue; // Skip function calls for text
                        }
                        if let Some(text) = part["text"].as_str() {
                            full.push_str(text);
                        }
                    }
                }
                if let Some(delta) = full.strip_prefix(emitted.as_str()) {
                    if !delta.is_empty() {
                        tx.send(StreamEvent::Delta(delta.to_string()))
                            .await
                            .map_err(|_| Failure::Fatal("stream receiver closed".into()))?;
                        emitted.push_str(delta);
                        any_text = true;
                    }
                } else if !full.is_empty() && full != emitted {
                    // Handle non-cumulative case
                    if full.len() > emitted.len() {
                        if let Some(delta) = full.strip_prefix(&emitted) {
                            tx.send(StreamEvent::Delta(delta.to_string()))
                                .await
                                .map_err(|_| Failure::Fatal("stream receiver closed".into()))?;
                            emitted = full;
                            any_text = true;
                        }
                    }
                }
            }
        }

        // Emit pending tool calls at end
        for tc in pending_tool_calls.drain(..) {
            let _ = tx.send(StreamEvent::ToolCall(tc)).await;
            any_text = true;
        }

        if !any_text {
            if !finish_reason.is_empty() && finish_reason != "STOP" && finish_reason != "MAX_TOKENS" {
                return Err(Failure::Fatal(format!(
                    "Gemini stopped without an answer ({finish_reason})"
                )));
            }
            return Err(Failure::Fatal("Gemini returned an empty response".into()));
        }
        Ok(())
    }
}

impl ChatBackend for GeminiBackend {
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
