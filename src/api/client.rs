//! Protocol-agnostic chat orchestration, layered on top of [`ChatBackend`].
//!
//! This module owns everything that does not depend on a vendor's wire
//! format: the model-fallback loop, family-first ordering of fallbacks and
//! the stream events a chat request produces. Each [`ChatBackend`] only
//! translates requests into its protocol and parses deltas back out.

use crate::api::providers::ChatBackend;
use crate::api::types::{CompletionRequest, Failure, Message, StreamEvent, ToolDefinition, MAX_MODELS_PER_SEND};
use anyhow::{anyhow, Result};
use std::collections::HashSet;
use std::sync::Arc;
use tokio::sync::mpsc::Sender;

/// A chat request bound to a backend, ready to be streamed.
///
/// `Arc<dyn ChatBackend>` keeps the client cheap to clone and share across
/// spawned tasks (e.g. the model-list fetch and the chat stream).
#[derive(Clone)]
pub struct ApiClient {
    backend: Arc<dyn ChatBackend>,
}

impl ApiClient {
    /// `backend` must be built with [`crate::api::providers::ProviderKind`].
    pub fn new(backend: Box<dyn ChatBackend>) -> Self {
        Self {
            backend: Arc::from(backend),
        }
    }

    /// The model ids the endpoint currently offers (`GET /models`),
    /// sorted case-insensitively.
    pub async fn list_models(&self) -> Result<Vec<String>> {
        self.backend.list_models().await
    }

    /// Stream a chat completion, falling back to other models the endpoint
    /// offers when the requested one rejects the request or is at high
    /// demand. Every switch is announced as a [`StreamEvent::Notice`] so the
    /// transcript can tell the user which model actually answered.
    pub async fn stream_chat(
        &self,
        messages: &[Message],
        model: &str,
        temperature: f32,
        tools: Vec<ToolDefinition>,
        tx: Sender<StreamEvent>,
    ) -> Result<()> {
        let mut tried: Vec<String> = vec![model.to_string()];
        let mut fallbacks: Option<Vec<String>> = None;

        loop {
            let current = tried.last().cloned().unwrap_or_default();
            let mut request = CompletionRequest::new(messages.to_vec(), current.clone(), temperature);
            if !tools.is_empty() {
                request = request.with_tools(tools.clone());
            }
            match self
                .backend
                .stream_completion(&request, tx.clone())
                .await
            {
                Ok(()) => return Ok(()),
                Err(Failure::Fatal(reason)) => return Err(anyhow!(reason)),
                Err(Failure::Retryable(reason)) => {
                    // The backend already returned Fatal if any text was
                    // emitted (see the `ChatBackend` fallback contract), so a
                    // Retryable failure here means no answer was spliced.
                    if fallbacks.is_none() {
                        let _ = tx
                            .send(StreamEvent::Notice(format!(
                                "{current} is unavailable — {reason}"
                            )))
                            .await;
                        fallbacks = Some(match self.backend.list_models().await {
                            Ok(models) => order_fallbacks(models, &current),
                            Err(list_error) => {
                                return Err(anyhow!(
                                    "{current} is unavailable — {reason}; fetching the fallback model list failed too ({list_error})"
                                ));
                            }
                        });
                    }
                    let list = fallbacks.as_ref().expect("fallback list loaded above");
                    let Some(next) = list.iter().find(|id| !tried.contains(*id)) else {
                        return Err(anyhow!(
                            "all {} available model(s) failed — last error: {reason}",
                            tried.len()
                        ));
                    };
                    if tried.len() >= MAX_MODELS_PER_SEND {
                        return Err(anyhow!(
                            "tried {} models without success — last error: {reason}",
                            tried.len()
                        ));
                    }
                    let next = next.clone();
                    let _ = tx
                        .send(StreamEvent::Notice(format!("switching to {next}")))
                        .await;
                    tried.push(next);
                }
            }
        }
    }
}

/// Rank the models left after a failure: models from the same family (the
/// namespace before the `/`, e.g. `MiniMaxAI/…`) first, since a sibling
/// model is the likeliest drop-in replacement; everything else keeps the
/// API's own order. The failed model and duplicates are dropped.
fn order_fallbacks(models: Vec<String>, failed: &str) -> Vec<String> {
    let mut seen = HashSet::new();
    let family = failed.split('/').next().unwrap_or("").to_ascii_lowercase();
    let mut kin = Vec::new();
    let mut rest = Vec::new();
    for id in models {
        if id.eq_ignore_ascii_case(failed) || !seen.insert(id.clone()) {
            continue;
        }
        let same_family = !family.is_empty()
            && id.split('/').next().unwrap_or("").eq_ignore_ascii_case(&family);
        if same_family {
            kin.push(id);
        } else {
            rest.push(id);
        }
    }
    kin.extend(rest);
    kin
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fallback_order_prefers_the_same_family() {
        let ordered = order_fallbacks(
            vec![
                "Other/Model".into(),
                "Minimaxai/MiniMax-M1".into(),
                "MiniMaxAI/MiniMax-M2.7".into(),
                "Other/Model".into(),
                "MiniMaxAI/MiniMax-M2.7".into(),
            ],
            "MiniMaxAI/MiniMax-M3",
        );
        assert_eq!(
            ordered,
            vec![
                "Minimaxai/MiniMax-M1",
                "MiniMaxAI/MiniMax-M2.7",
                "Other/Model"
            ]
        );
    }
}
