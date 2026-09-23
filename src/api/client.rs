//! Protocol-agnostic chat orchestration, layered on top of [`ChatBackend`].
//!
//! This module owns everything that does not depend on a vendor's wire
//! format: the model-fallback loop, family-first ordering of fallbacks and
//! the stream events a chat request produces. Each [`ChatBackend`] only
//! translates requests into its protocol and parses deltas back out.

use crate::api::providers::ChatBackend;
use crate::api::types::{
    CompletionRequest, Failure, Message, StreamEvent, ToolDefinition, MAX_MODELS_PER_SEND,
    MAX_RETRIES_PER_MODEL,
};
use anyhow::{anyhow, Result};
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;
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

    /// Stream a chat completion with two layers of resilience:
    ///
    /// 1. **Same-model retries** — [`Failure::Transient`] errors (network,
    ///    idle timeouts, throttling) retry the *same* model up to
    ///    [`MAX_RETRIES_PER_MODEL`] times with exponential backoff. Each
    ///    retry is announced as [`StreamEvent::Retry`] (for the animated
    ///    status row) plus a permanent [`StreamEvent::Notice`].
    /// 2. **Model fallback** — [`Failure::Retryable`] errors (and transient
    ///    errors that exhausted their retries) fall back to other models the
    ///    endpoint offers, as before.
    ///
    /// Every fallback switch is announced as a [`StreamEvent::Notice`] so the
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

            // -- Layer 1: retry the same model on transient failures. ----
            // The backend already returned Fatal if any text was emitted
            // (see the `ChatBackend` fallback contract), so every failure
            // reaching this loop means no answer was spliced.
            let mut reason = String::new();
            for attempt in 1..=MAX_RETRIES_PER_MODEL {
                let mut request =
                    CompletionRequest::new(messages.to_vec(), current.clone(), temperature);
                if !tools.is_empty() {
                    request = request.with_tools(tools.clone());
                }
                match self.backend.stream_completion(&request, tx.clone()).await {
                    Ok(()) => return Ok(()),
                    Err(Failure::Fatal(fatal)) => return Err(anyhow!(fatal)),
                    Err(Failure::Retryable(why)) => {
                        // This model specifically is the problem: go straight
                        // to the fallback layer.
                        reason = why;
                        break;
                    }
                    Err(Failure::Transient(why)) => {
                        if attempt == MAX_RETRIES_PER_MODEL {
                            // Retries exhausted: give another model a chance.
                            reason = why;
                            break;
                        }
                        let wait = backoff(attempt);
                        let _ = tx
                            .send(StreamEvent::Retry {
                                attempt: attempt as u8,
                                max: MAX_RETRIES_PER_MODEL as u8,
                                wait_ms: wait.as_millis() as u64,
                                reason: why.clone(),
                            })
                            .await;
                        let _ = tx
                            .send(StreamEvent::Notice(format!(
                                "⚠ attempt {attempt}/{MAX_RETRIES_PER_MODEL} for {current} failed — {why} · retrying in {}s",
                                wait.as_secs()
                            )))
                            .await;
                        // `why` stays owned until both events have used it.
                        reason = why;
                        tokio::time::sleep(wait).await;
                    }
                }
            }

            // -- Layer 2: fall back to another model. ----------------------
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

/// Exponential backoff with a pinch of clock-derived jitter: ~1s, ~2s, ~4s,
/// capped at 8s. The jitter keeps several clients from retrying a shared
/// gateway in lockstep without pulling in an RNG crate.
fn backoff(attempt: usize) -> Duration {
    let base_secs = 1u64 << attempt.saturating_sub(1).min(3);
    let jitter_ms = u64::from(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|elapsed| elapsed.subsec_millis())
            .unwrap_or(0)
            % 250,
    );
    Duration::from_millis(base_secs * 1000 + jitter_ms)
}

/// Rank the models left after a failure: models from the same family (the
/// namespace before the `/`, e.g. `AcmeAI/…`) first, since a sibling
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
    fn backoff_grows_exponentially_and_stays_bounded() {
        for attempt in 1..=MAX_RETRIES_PER_MODEL {
            let wait = backoff(attempt);
            let base = 1u64 << (attempt - 1);
            assert!(wait >= Duration::from_secs(base), "attempt {attempt}");
            assert!(
                wait < Duration::from_secs(base) + Duration::from_millis(250),
                "attempt {attempt}"
            );
        }
        // Huge attempt counts must not overflow the cap.
        assert!(backoff(usize::MAX) <= Duration::from_secs(8) + Duration::from_millis(250));
    }

    #[test]
    fn fallback_order_prefers_the_same_family() {
        let ordered = order_fallbacks(
            vec![
                "OtherOrg/other-model".into(),
                "Acmeai/acme-model-1".into(),
                "AcmeAI/acme-model-2".into(),
                "OtherOrg/other-model".into(),
                "AcmeAI/acme-model-2".into(),
            ],
            "AcmeAI/acme-model-3",
        );
        assert_eq!(
            ordered,
            vec![
                "Acmeai/acme-model-1",
                "AcmeAI/acme-model-2",
                "OtherOrg/other-model"
            ]
        );
    }
}
