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
/// instead guarded by *two* idle timeouts (see the backends' read loops):
/// a generous one until the model produces its first output, and a tight one
/// between chunks after that.
#[derive(Debug, Clone, Copy)]
pub struct HttpTimeouts {
    /// How long establishing the connection may take before giving up.
    pub connect: Duration,
    /// How long the stream may stay quiet between chunks *after output has
    /// started* before the connection is treated as dead.
    pub idle: Duration,
    /// How long to wait for the model's **first** output before treating the
    /// connection as dead. Much longer than [`Self::idle`] on purpose:
    /// reasoning models can think for many minutes, and some gateways buffer
    /// the entire chain of thought before sending a single byte. Keeping this
    /// separate means a long think is not mistaken for a dead connection,
    /// while a stream that dies mid-answer is still caught quickly.
    pub first_token: Duration,
}

impl Default for HttpTimeouts {
    fn default() -> Self {
        Self {
            connect: Duration::from_secs(15),
            idle: Duration::from_secs(90),
            first_token: Duration::from_secs(1800),
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
        // Extended-thinking budget, for the providers that have one. Only
        // Anthropic reads it today; the others ignore it.
        thinking_budget: Option<u32>,
        // Cap on tokens per response. `0` means "send no cap", which leaves
        // the provider's own default in force — often only a few thousand
        // tokens, and the usual cause of a tool call whose arguments arrive
        // cut off mid-JSON.
        max_output_tokens: u64,
    ) -> Box<dyn ChatBackend> {
        match self {
            Self::OpenAICompatible => Box::new(OpenAICompatibleBackend::new(
                api_key,
                base_url,
                timeouts,
                max_output_tokens,
            )),
            Self::Anthropic => Box::new(AnthropicBackend::new(
                api_key,
                base_url,
                timeouts,
                thinking_budget,
                max_output_tokens,
            )),
            Self::Gemini => Box::new(GeminiBackend::new(
                api_key,
                base_url,
                timeouts,
                max_output_tokens,
            )),
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
