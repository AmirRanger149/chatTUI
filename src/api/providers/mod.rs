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

use crate::api::types::{CompletionRequest, Failure, StreamEvent};
use anyhow::Result;
use std::future::Future;
use std::pin::Pin;
use std::time::Duration;
use tokio::sync::mpsc::Sender;

/// HTTP timeouts shared by every backend.
///
/// Deliberately no blanket whole-request timeout: it would include reading
/// the streamed body and cut long generations off mid-answer. Streams are
/// instead guarded by an *idle* timeout on each chunk (see the backends'
/// read loops), so a response that keeps producing tokens may run as long
/// as it needs while a dead connection is still detected.
#[derive(Debug, Clone, Copy)]
pub struct HttpTimeouts {
    /// How long establishing the connection may take before giving up.
    pub connect: Duration,
    /// How long the stream may stay quiet between chunks before the
    /// connection is treated as dead.
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
/// therefore return [`Failure::Fatal`] whenever it has already emitted some
/// text — neither a retry nor a second model may be spliced into a
/// half-written answer, or you get one Frankenstein reply. (This is how the
/// old single-backend code behaved; it is written down here so new backends
/// keep doing it.)
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
