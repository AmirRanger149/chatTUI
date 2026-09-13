//! The Anthropic Messages API backend (`POST /v1/messages`).
//!
//! Note the differences from the OpenAI dialect that this adapter papers over:
//! - the system prompt is a top-level `system` field, not a message;
//! - message content is a list of content blocks (`{"type":"text", …}`);
//! - `max_tokens` is mandatory (a conservative bound here, trimmed by the
//!   model list when available);
//! - streaming deltas arrive as `content_block_delta` events with
//!   `delta.text`, and the model list is `{"data":[{"id": …}]}`.

use super::ChatBackend;
use crate::api::error::{classify_failure, error_detail};
use crate::api::sse::SseReader;
use crate::api::types::{CompletionRequest, Failure, Role, StreamEvent};
use anyhow::{anyhow, Context, Result};
use futures_util::StreamExt;
use reqwest::Client;
use serde_json::{json, Value};
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
}

impl AnthropicBackend {
    pub fn new(api_key: String, base_url: String) -> Self {
        Self {
            http: Client::builder()
                .timeout(Duration::from_secs(90))
                .build()
                .expect("HTTP client"),
            api_key,
            base_url: base_url.trim_end_matches('/').into(),
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
        let response = self
            .http
            .get(format!("{}/models", self.base_url))
            .send()
            .await
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
        let response = self
            .http
            .get(url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .send()
            .await
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
            if message.role == Role::System {
                let mut prompt = system.take().unwrap_or_default();
                if !prompt.is_empty() {
                    prompt.push_str("\n\n");
                }
                prompt.push_str(&message.content);
                system = Some(prompt);
                continue;
            }
            let role = if message.role == Role::Assistant {
                "assistant"
            } else {
                "user"
            };
            messages.push(json!({
                "role": role,
                "content": [{ "type": "text", "text": message.content }],
            }));
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
        let url = format!("{}/messages", self.base_url);
        let response = self
            .http
            .post(url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .json(&body)
            .send()
            .await
            .map_err(|error| Failure::Retryable(format!("request failed: {error}")))?;
        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(classify_failure(status.as_u16(), &body));
        }
        let mut stream = response.bytes_stream();
        let mut reader = SseReader::new();
        let mut emitted = false;
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|error| {
                if emitted {
                    Failure::Fatal(format!("stream interrupted after output started: {error}"))
                } else {
                    Failure::Retryable(format!("stream interrupted before any output: {error}"))
                }
            })?;
            for data in reader.feed(&chunk) {
                let value: Value = serde_json::from_str(&data)
                    .map_err(|error| Failure::Fatal(format!("parsing streaming response failed: {error}")))?;
                match value["type"].as_str() {
                    // `content_block_delta` carries the token text.
                    Some("content_block_delta") => {
                        if let Some(text) = value["delta"]["text"].as_str() {
                            tx.send(StreamEvent::Delta(text.into()))
                                .await
                                .map_err(|_| Failure::Fatal("stream receiver closed".into()))?;
                            emitted = true;
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
                    // message_start, content_block_start, message_delta,
                    // message_stop, ping: nothing to emit for these.
                    _ => {}
                }
            }
        }
        Ok(())
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
