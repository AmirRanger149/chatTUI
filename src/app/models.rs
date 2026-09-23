//! The `/model` picker: the model catalog (background-fetched list from the
//! API), argument resolution for `/model <arg>`, and availability-based
//! default-model resolution for custom providers (their config entry may
//! omit `model`).

use crate::app::{App, Overlay};
use std::time::{Duration, Instant};
use tokio::sync::mpsc::{self, error::TryRecvError};

/// How long a fetched model list stays fresh before `/model` refetches it.
const MODELS_CACHE_TTL: Duration = Duration::from_secs(5 * 60);

/// The API's model list, fetched in the background for the `/model` picker.
#[derive(Debug, Default)]
pub struct ModelCatalog {
    pub ids: Vec<String>,
    pub loading: bool,
    pub error: Option<String>,
    pub fetched_at: Option<Instant>,
}

impl App {
    /// Index of the active model in the catalog, or `0` when unknown.
    fn current_model_index(&self) -> usize {
        self.models
            .ids
            .iter()
            .position(|id| *id == self.config.model)
            .unwrap_or(0)
    }

    /// Resolve a `/model <arg>` argument against the cached model list:
    /// exact (case-insensitive) first, then a unique prefix, then a unique
    /// suffix, so `acmeai/acme-model-1` and even `acme-model-1` both find
    /// `AcmeAI/acme-model-1`. Anything ambiguous or unknown comes back
    /// exactly as typed.
    pub(crate) fn lookup_model(&self, argument: &str) -> String {
        if let Some(id) = self
            .models
            .ids
            .iter()
            .find(|id| id.eq_ignore_ascii_case(argument))
        {
            return id.clone();
        }
        let lower = argument.to_ascii_lowercase();
        let starts: Vec<&String> = self
            .models
            .ids
            .iter()
            .filter(|id| id.to_ascii_lowercase().starts_with(&lower))
            .collect();
        if starts.len() == 1 {
            return starts[0].clone();
        }
        let ends: Vec<&String> = self
            .models
            .ids
            .iter()
            .filter(|id| id.to_ascii_lowercase().ends_with(&lower))
            .collect();
        if ends.len() == 1 {
            return ends[0].clone();
        }
        argument.to_string()
    }

    /// `/model` with no argument: open the picker backed by the API's model
    /// list and fetch it in the background if there is no fresh copy.
    pub fn open_models(&mut self) {
        if self
            .config
            .api_key
            .as_deref()
            .unwrap_or("")
            .trim()
            .is_empty()
        {
            let env_key = self
                .config
                .find_provider(&self.config.provider)
                .map(|p| p.env_key)
                .unwrap_or_else(|| "API_KEY".to_string());
            self.push_error(format!(
                "set {env_key} in config.json or environment to list models"
            ));
            return;
        }
        let selected = self.current_model_index();
        self.overlay = Some(Overlay::Models { selected });
        self.ensure_models();
    }

    /// Fetch the model list unless a fetch is running or the cache is fresh.
    fn ensure_models(&mut self) {
        if self.models_rx.is_some() || self.models.loading {
            return;
        }
        if self
            .models
            .fetched_at
            .is_some_and(|at| at.elapsed() < MODELS_CACHE_TTL)
        {
            return;
        }
        self.request_models(false);
    }

    /// Force a re-fetch (`r` inside the picker).
    pub fn refresh_models(&mut self) {
        if self.models_rx.is_some() {
            return;
        }
        self.request_models(false);
    }

    fn request_models(&mut self, auto: bool) {
        if self.config.api_key.is_none() {
            self.models.loading = false;
            self.models.error = Some("no API key configured".into());
            return;
        }
        let client = self.config.api_client();
        let (tx, rx) = mpsc::channel(1);
        tokio::spawn(async move {
            let _ = tx.send(client.list_models().await).await;
        });
        self.models_rx = Some(rx);
        self.models_fetch_auto = auto;
        self.models.loading = true;
        self.models.error = None;
    }

    /// Background fetch that resolves an availability-based default model:
    /// a custom provider whose config entry omits `model` gets its active
    /// model resolved against the endpoint's live list (a free model first,
    /// e.g. `free/…` on gateways that publish a free namespace). When it
    /// finishes, the active model is set to the best available match, unless
    /// the user has picked one themselves. Failures stay quiet: the send-time
    /// model fallback still covers an empty choice.
    pub(crate) fn request_available_default(&mut self) {
        // Custom providers are the availability-based ones; the built-ins
        // (OpenAI, Anthropic, Gemini) keep their static default models.
        let Some(provider) = self.config.find_provider(&self.config.provider) else {
            return;
        };
        if !provider.availability_based_model() {
            return;
        }
        if self.config.api_key.as_deref().unwrap_or("").trim().is_empty() {
            return;
        }
        // Don't override an explicit user choice: a model set through
        // `{ID}_MODEL`, the config entry's `model`, or `/model` is
        // non-empty here.
        if !self.config.model.is_empty() {
            return;
        }
        if self.models_rx.is_some() || self.models.loading {
            return;
        }
        self.request_models(true);
    }

    /// Drain the background model-list fetch, if one finished.
    pub fn receive_models(&mut self) {
        if self.models_rx.is_none() {
            return;
        }
        let mut rx = self.models_rx.take().expect("checked above");
        loop {
            match rx.try_recv() {
                Ok(Ok(ids)) => {
                    let auto = self.models_fetch_auto;
                    self.models.ids = ids;
                    self.models.loading = false;
                    self.models.error = None;
                    self.models.fetched_at = Some(Instant::now());
                    self.models_fetch_auto = false;
                    if auto {
                        // Background availability check: point the active
                        // model at the best model the endpoint actually
                        // offers. The fetched list doubles as the `/model`
                        // picker's cache.
                        self.apply_available_default();
                    }
                    // The fetch sends exactly one message and then closes the
                    // channel: drop the receiver with it, so a later poll
                    // can't mistake that close for a failed fetch.
                    self.retarget_models_overlay();
                    return;
                }
                Ok(Err(error)) => {
                    let auto = self.models_fetch_auto;
                    self.models_fetch_auto = false;
                    self.models.loading = false;
                    if auto {
                        // A background default-model check failed quietly:
                        // keep the built-in default (the send-time fallback
                        // still covers it) instead of surfacing an error.
                        self.models.error = None;
                    } else {
                        self.models.error = Some(error.to_string());
                    }
                    return;
                }
                Err(TryRecvError::Empty) => {
                    // Still in flight; poll again next frame.
                    self.models_rx = Some(rx);
                    return;
                }
                Err(TryRecvError::Disconnected) => {
                    // Only reachable when the task ended without sending a
                    // result at all (it always sends one), so this is a real
                    // failure and not the normal end of a completed fetch.
                    let auto = self.models_fetch_auto;
                    self.models_fetch_auto = false;
                    self.models.loading = false;
                    if !auto {
                        self.models.error = Some("model list fetch ended unexpectedly".into());
                    }
                    return;
                }
            }
        }
    }

    /// Point the picker's selection at the active model when fresh data lands.
    fn retarget_models_overlay(&mut self) {
        if !matches!(self.overlay, Some(Overlay::Models { .. })) {
            return;
        }
        let index = self.current_model_index();
        if let Some(Overlay::Models { selected }) = &mut self.overlay {
            *selected = index;
        }
    }

    /// After a background fetch, set the active model to the best model the
    /// endpoint currently offers: a free model first (`free/…`), then the
    /// provider's configured default if one is set and still listed, and
    /// failing that the first model in the list. Does nothing if the user has
    /// picked a model themselves in the meantime.
    fn apply_available_default(&mut self) {
        let Some(provider) = self.config.find_provider(&self.config.provider) else {
            return;
        };
        if !provider.availability_based_model() {
            return;
        }
        // Respect an explicit choice made while the fetch was in flight.
        if !self.config.model.is_empty() {
            return;
        }
        let Some(picked) = pick_available_default(&self.models.ids, &provider.default_model) else {
            return;
        };
        self.config.model = picked.clone();
        let was_free = picked.to_ascii_lowercase().starts_with("free/");
        if was_free {
            self.push_notice(format!(
                "{} default set to available free model: {picked}",
                provider.name
            ));
        } else {
            self.push_notice(format!(
                "{} default set to available model: {picked}",
                provider.name
            ));
        }
    }

    pub fn move_model_selection(&mut self, delta: i32) {
        let len = self.models.ids.len();
        if len == 0 {
            return;
        }
        if let Some(Overlay::Models { selected }) = &mut self.overlay {
            *selected = (*selected as i32 + delta).rem_euclid(len as i32) as usize;
        }
    }

    /// `Enter` in the picker: make the highlighted model the active one.
    pub fn apply_selected_model(&mut self) {
        let Some(Overlay::Models { selected }) = self.overlay else {
            return;
        };
        let Some(id) = self.models.ids.get(selected).cloned() else {
            self.overlay = None;
            return;
        };
        if id != self.config.model {
            self.config.model = id.clone();
            self.push_notice(format!("model set to {id}"));
        }
        self.overlay = None;
    }
}

/// Choose the default model for an availability-based provider from its live
/// model list: a free model (the `free/` namespace, e.g.
/// a `free/` model) wins; otherwise the built-in default if it
/// is still offered; otherwise the first model in the list. The list arrives
/// sorted case-insensitively (see `crate::api::client::ApiClient::list_models`), so both the
/// free pick and the fallback are deterministic.
fn pick_available_default(ids: &[String], builtin_default: &str) -> Option<String> {
    if let Some(free) = ids
        .iter()
        .find(|id| id.trim().to_ascii_lowercase().starts_with("free/"))
    {
        return Some(free.clone());
    }
    if let Some(current) = ids
        .iter()
        .find(|id| id.eq_ignore_ascii_case(builtin_default))
    {
        return Some(current.clone());
    }
    ids.first().cloned()
}

#[cfg(test)]
mod tests {
    use crate::app::{App, Cell, Overlay};
    use crate::config::Config;
    use crate::session::manager::SessionManager;

    fn test_app() -> App {
        App::new(Config::default(), SessionManager::for_tests())
    }

    #[test]
    fn model_picker_needs_an_api_key() {
        let mut app = test_app();
        app.config.api_key = None;
        app.open_models();
        assert!(app.overlay.is_none());
        assert!(matches!(app.cells.last(), Some(Cell::Error(_))));
    }

    #[tokio::test]
    async fn model_picker_opens_and_retargets_when_models_arrive() {
        let mut app = test_app();
        app.config.model = "b/2".into();
        app.config.api_key = Some("key".into());
        // Keep the background fetch away from the network.
        app.config.base_url = "http://127.0.0.1:9".into();
        app.open_models();
        assert!(matches!(app.overlay, Some(Overlay::Models { selected: 0 })));
        assert!(app.models.loading);

        // Simulate the background fetch completing (list_models already
        // sorted the ids).
        let (tx, rx) = tokio::sync::mpsc::channel(1);
        tx.send(Ok(vec!["a/1".into(), "b/2".into(), "c/3".into()]))
            .await
            .unwrap();
        drop(tx);
        app.models_rx = Some(rx);
        app.receive_models();
        assert_eq!(app.models.ids, vec!["a/1", "b/2", "c/3"]);
        assert!(!app.models.loading);
        assert!(app.models.error.is_none());
        // The selection jumped to the active model, `b/2`.
        assert!(matches!(app.overlay, Some(Overlay::Models { selected: 1 })));

        // A later poll sees the fetch's closed channel. It must NOT be read
        // as a failure and wipe the freshly loaded list.
        app.receive_models();
        assert!(app.models.error.is_none());
        assert_eq!(app.models.ids, vec!["a/1", "b/2", "c/3"]);
        assert!(!app.models.loading);
    }

    #[tokio::test]
    async fn model_picker_reports_a_failed_fetch() {
        let mut app = test_app();
        app.config.api_key = Some("key".into());
        app.config.base_url = "http://127.0.0.1:9".into();
        app.open_models();
        let (tx, rx) = tokio::sync::mpsc::channel(1);
        tx.send(Err(anyhow::anyhow!("HTTP 401"))).await.unwrap();
        drop(tx);
        app.models_rx = Some(rx);
        app.receive_models();
        assert!(!app.models.loading);
        assert_eq!(app.models.error.as_deref(), Some("HTTP 401"));
        // A later poll over the closed channel keeps the original error.
        app.receive_models();
        assert_eq!(app.models.error.as_deref(), Some("HTTP 401"));
    }

    #[tokio::test]
    async fn model_picker_selection_wraps_and_applies() {
        let mut app = test_app();
        app.config.api_key = Some("key".into());
        app.config.base_url = "http://127.0.0.1:9".into();
        app.models.ids = vec!["a/1".into(), "b/2".into(), "c/3".into()];
        app.overlay = Some(Overlay::Models { selected: 2 });
        app.move_model_selection(1); // wraps to the top
        assert!(matches!(app.overlay, Some(Overlay::Models { selected: 0 })));
        app.apply_selected_model();
        assert_eq!(app.config.model, "a/1");
        assert!(app.overlay.is_none());
        assert!(matches!(app.cells.last(), Some(Cell::Notice(_))));

        // Re-applying the already-active model stays quiet.
        let notices = app.cells.len();
        app.open_models();
        assert!(matches!(app.overlay, Some(Overlay::Models { selected: 0 })));
        app.apply_selected_model();
        assert_eq!(app.cells.len(), notices);
        assert_eq!(app.config.model, "a/1");
    }

    #[test]
    fn model_lookup_resolves_exact_prefix_and_suffix() {
        let mut app = test_app();
        app.models.ids = vec![
            "AcmeAI/acme-model-1".into(),
            "AcmeAI/acme-model-2".into(),
            "OtherOrg/other-model".into(),
        ];
        assert_eq!(
            app.lookup_model("otherorg/other-model"),
            "OtherOrg/other-model"
        );
        assert_eq!(
            app.lookup_model("acmeai/acme-model-1"),
            "AcmeAI/acme-model-1"
        );
        // Unique suffix shorthand.
        assert_eq!(app.lookup_model("acme-model-2"), "AcmeAI/acme-model-2");
        // Ambiguous prefixes/suffixes stay untouched.
        assert_eq!(app.lookup_model("acme"), "acme");
        assert_eq!(app.lookup_model("nonexistent"), "nonexistent");
        // No catalog → as typed.
        app.models.ids.clear();
        assert_eq!(app.lookup_model("anything"), "anything");
    }
}
