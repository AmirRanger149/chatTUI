//! Chat backends: one trait over the wire protocols, plus a dispatch enum so
//! callers build a backend by provider kind.
//!
//! Each backend is a pure protocol adapter: it translates a
//! [`CompletionRequest`] into its vendor's request schema and parses the
//! stream back into [`StreamEvent::Delta`]s. Everything protocol-agnostic
//! (model fallback, retries, family-first ordering) lives in
//! [`crate::api::client`] on top of this trait.

mod anthropic;
mod gemini;
mod openai_compatible;

pub use anthropic::AnthropicBackend;
pub use gemini::GeminiBackend;
pub use openai_compatible::OpenAICompatibleBackend;

use crate::api::types::{CompletionRequest, Failure, StreamEvent, MAX_REQUEST_BYTES};
use anyhow::{anyhow, Result};
use futures_util::StreamExt;
use std::future::Future;
use std::pin::Pin;
use std::time::Duration;
use tokio::sync::mpsc::Sender;

/// HTTP timeouts shared by every backend.
///
/// Deliberately no blanket whole-request timeout: it would include reading
/// the streamed body and cut long generations off mid-answer. Streams are
/// instead guarded by a meaningful-progress gap (see the backends' read
/// loops), so keep-alive heartbeats alone cannot hold a dead response open.
/// Request headers and small response bodies have their own bounds.
#[derive(Debug, Clone, Copy)]
pub struct HttpTimeouts {
    /// How long establishing the connection may take before giving up.
    pub connect: Duration,
    /// Maximum gap without answer text or tool-call data. SSE keep-alive
    /// frames do not reset this timer.
    pub idle: Duration,
}

impl Default for HttpTimeouts {
    fn default() -> Self {
        Self {
            connect: Duration::from_secs(15),
            idle: Duration::from_secs(90),
        }
    }
}

/// HTTP error/model-list responses are not SSE. Never collect them with
/// `response.text()` / `response.json()` without a size or progress bound.
const MAX_HTTP_BODY_BYTES: usize = 1024 * 1024;

pub(crate) async fn bounded_body(response: reqwest::Response, idle: Duration) -> Result<Vec<u8>> {
    let mut stream = response.bytes_stream();
    let mut body = Vec::new();
    let deadline = std::time::Instant::now() + idle.min(Duration::from_secs(30));
    loop {
        let remaining = deadline.saturating_duration_since(std::time::Instant::now());
        let chunk = tokio::time::timeout(remaining, stream.next())
            .await
            .map_err(|_| anyhow!("HTTP response body stalled or exceeded its 30s deadline"))?;
        match chunk {
            Some(Ok(chunk)) => {
                if body.len().saturating_add(chunk.len()) > MAX_HTTP_BODY_BYTES {
                    anyhow::bail!("HTTP response body exceeded 1 MiB");
                }
                body.extend_from_slice(&chunk);
            }
            Some(Err(error)) => return Err(error.into()),
            None => return Ok(body),
        }
    }
}

/// A raw-message count misses JSON escaping and provider-specific wrappers.
/// Check the actual serialized body before handing it to reqwest so a large
/// conversation cannot quietly multiply its size on the wire.
pub(crate) fn check_request_size(body: &serde_json::Value) -> Result<(), Failure> {
    let len = serde_json::to_vec(body)
        .map_err(|error| Failure::Fatal(format!("serializing the request failed: {error}")))?
        .len();
    if len > MAX_REQUEST_BYTES {
        return Err(Failure::Fatal("serialized request exceeds the 8 MiB limit — start a new session or shorten the history".into()));
    }
    Ok(())
}

/// The wire protocol a provider speaks. OpenAI and every OpenAI-compatible
/// gateway (the custom providers in `config.json`: Ollama, Groq, Mistral,
/// Together, OpenRouter, …) share one backend; Anthropic and Gemini each
/// have their own.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderKind {
    OpenAICompatible,
    Anthropic,
    Gemini,
}

impl ProviderKind {
    /// Build the backend for this protocol, bound to a key, endpoint and
    /// timeout policy.
    pub fn build(
        self,
        api_key: String,
        base_url: String,
        timeouts: HttpTimeouts,
    ) -> Box<dyn ChatBackend> {
        match self {
            Self::OpenAICompatible => {
                Box::new(OpenAICompatibleBackend::new(api_key, base_url, timeouts))
            }
            Self::Anthropic => Box::new(AnthropicBackend::new(api_key, base_url, timeouts)),
            Self::Gemini => Box::new(GeminiBackend::new(api_key, base_url, timeouts)),
        }
    }
}

/// A chat provider backend.
///
/// # Fallback contract
///
/// The orchestrator in [`crate::api::client`] reacts to failures in three
/// ways: [`Failure::Transient`] retries the *same* model with backoff
/// (network, timeouts, throttling), [`Failure::Retryable`] switches to
/// another model, and [`Failure::Fatal`] ends the request. A backend must
/// therefore return [`Failure::Fatal`] whenever it has emitted answer text:
/// a retry must not splice a second answer into the first. Tool calls are
/// executable effects: buffer them *inside the attempt* and send them only
/// after the attempt has completed successfully. Otherwise a retry/fallback
/// could execute a stale call from the failed attempt.
///
/// [`stream_completion`]: ChatBackend::stream_completion
pub trait ChatBackend: Send + Sync {
    /// The model ids the endpoint currently offers, sorted case-insensitively.
    fn list_models(&self) -> Pin<Box<dyn Future<Output = Result<Vec<String>>> + Send + '_>>;

    /// Stream one completion, sending deltas to `tx`. `Ok(())` means a clean
    /// finish; `Err(Failure)` tells the orchestrator how to classify the
    /// failure (and thus whether another model may be tried).
    fn stream_completion(
        &self,
        request: &CompletionRequest,
        tx: Sender<StreamEvent>,
    ) -> Pin<Box<dyn Future<Output = Result<(), Failure>> + Send + '_>>;
}
