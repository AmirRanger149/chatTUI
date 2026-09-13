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
use tokio::sync::mpsc::Sender;

/// The wire protocol a provider speaks. OpenAI and every OpenAI-compatible
/// gateway (the custom providers in `config.json`: Dahl, APInex, Ollama,
/// Groq, Mistral, Together, …) share one backend; Anthropic and Gemini each
/// have their own.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderKind {
    OpenAICompatible,
    Anthropic,
    Gemini,
}

impl ProviderKind {
    /// Build the backend for this protocol, bound to a key and endpoint.
    pub fn build(self, api_key: String, base_url: String) -> Box<dyn ChatBackend> {
        match self {
            Self::OpenAICompatible => {
                Box::new(OpenAICompatibleBackend::new(api_key, base_url))
            }
            Self::Anthropic => Box::new(AnthropicBackend::new(api_key, base_url)),
            Self::Gemini => Box::new(GeminiBackend::new(api_key, base_url)),
        }
    }
}

/// A chat provider backend.
///
/// # Fallback contract
///
/// The orchestrator in [`crate::api::client`] retries a failed attempt with a
/// different model only when [`stream_completion`] returns
/// [`Failure::Retryable`]. A backend must therefore return [`Failure::Fatal`]
/// whenever it has already emitted some text. Splicing a second model into a
/// half-written answer gives you one Frankenstein reply. (This is how the old
/// single-backend code behaved; it is written down here so new backends keep
/// doing it.)
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
