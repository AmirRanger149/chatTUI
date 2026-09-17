use crate::api::client::ApiClient;
use crate::api::providers::ProviderKind;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::{env, fs, path::PathBuf};

/// A chat provider: an id/name, the wire protocol it speaks, its endpoint,
/// default model, and the environment variable that may hold its API key.
///
/// There are two kinds: the three built-ins (`openai`, `anthropic`, `gemini`)
/// and user-defined custom gateways declared under `custom_providers` in
/// `config.json` (Dahl, APInex, Ollama, Groq, and anything else that speaks
/// the OpenAI-compatible protocol).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Provider {
    pub id: String,
    pub name: String,
    pub kind: ProviderKind,
    pub base_url: String,
    pub default_model: String,
    pub env_key: String,
    /// Custom providers come from `config.json`; their default model is
    /// availability-based (resolved against the endpoint's live model list).
    pub is_custom: bool,
}

impl Provider {
    fn builtin(
        id: &str,
        name: &str,
        kind: ProviderKind,
        base_url: &str,
        default_model: &str,
        env_key: &str,
    ) -> Self {
        Self {
            id: id.to_string(),
            name: name.to_string(),
            kind,
            base_url: base_url.to_string(),
            default_model: default_model.to_string(),
            env_key: env_key.to_string(),
            is_custom: false,
        }
    }

    /// The default model is resolved from the endpoint's live model list
    /// rather than a pinned id. True for custom providers whose config entry
    /// omits `model`.
    pub fn availability_based_model(&self) -> bool {
        self.is_custom
    }
}

/// The built-in providers, in default-selection order.
pub fn builtin_providers() -> Vec<Provider> {
    vec![
        Provider::builtin(
            "openai",
            "OpenAI",
            ProviderKind::OpenAICompatible,
            "https://api.openai.com/v1",
            "gpt-4o-mini",
            "OPENAI_API_KEY",
        ),
        Provider::builtin(
            "anthropic",
            "Anthropic",
            ProviderKind::Anthropic,
            "https://api.anthropic.com/v1",
            "claude-sonnet-4-5",
            "ANTHROPIC_API_KEY",
        ),
        Provider::builtin(
            "gemini",
            "Google Gemini",
            ProviderKind::Gemini,
            "https://generativelanguage.googleapis.com/v1beta",
            "gemini-2.5-flash",
            "GEMINI_API_KEY",
        ),
    ]
}

/// A user-defined OpenAI-compatible provider, declared under
/// `custom_providers` in `config.json`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CustomProvider {
    /// Unique id used with `/provider <id>` and the `provider` field. Ids
    /// that shadow a built-in (`openai`, `anthropic`, `gemini`) are ignored.
    pub id: String,
    /// Display name; defaults to the id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// OpenAI-compatible base URL, e.g. `https://inference.dahl.global/v1`.
    pub base_url: String,
    /// API key; `{ID}_API_KEY` in the environment is the fallback.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    /// Default model. When omitted, chatTUI picks one from the endpoint's
    /// live model list (a `free/` model first).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

fn default_temperature() -> f32 {
    0.7
}

fn is_default_temperature(val: &f32) -> bool {
    (*val - 0.7).abs() < f32::EPSILON
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SandboxConfigFile {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default, alias = "target_dir")]
    pub workspace_root: String,
    #[serde(default = "default_true")]
    pub auto_approve: bool,
    #[serde(default = "default_true")]
    pub allow_shell: bool,
    /// Hard timeout in seconds for agent shell commands. The command's
    /// process group is killed when it expires; clamped to at least 1s so
    /// the agent can never be blocked indefinitely.
    #[serde(default = "default_shell_timeout_secs")]
    pub shell_timeout_secs: u64,
    /// Explicit permission mode: "read-only", "workspace-write",
    /// "ask-before-write", "ask-before-shell" or "full-auto". When unset,
    /// the mode is derived from the legacy `auto_approve` / `allow_shell`
    /// flags exactly as in earlier versions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub permission_mode: Option<String>,
    /// Extra sensitive file names appended to the built-in policy. An entry
    /// starting with "." matches any file name ending with it; anything
    /// else must equal the file name (case-insensitively).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub extra_sensitive_names: Vec<String>,
}

fn default_true() -> bool {
    true
}

fn default_shell_timeout_secs() -> u64 {
    30
}

impl Default for SandboxConfigFile {
    fn default() -> Self {
        Self {
            enabled: true,
            workspace_root: String::new(),
            auto_approve: true,
            allow_shell: true,
            shell_timeout_secs: default_shell_timeout_secs(),
            permission_mode: None,
            extra_sensitive_names: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub openai_api_key: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub anthropic_api_key: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gemini_api_key: Option<String>,
    /// User-defined OpenAI-compatible gateways (Dahl, APInex, Ollama, Groq, …).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub custom_providers: Vec<CustomProvider>,
    /// Deprecated fields of the removed Dahl/APInex built-ins; migrated into
    /// `custom_providers` on load so old config files keep working.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dahl_api_key: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub apinex_api_key: Option<String>,
    /// Legacy single-key field: routed to its provider on load.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub base_url: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub model: String,
    #[serde(default = "default_temperature", skip_serializing_if = "is_default_temperature")]
    pub temperature: f32,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub provider: String,
    #[serde(default, skip_serializing_if = "is_default_sandbox")]
    pub sandbox: SandboxConfigFile,
}

fn is_default_sandbox(cfg: &SandboxConfigFile) -> bool {
    cfg.enabled
        && cfg.workspace_root.is_empty()
        && cfg.auto_approve
        && cfg.allow_shell
        && cfg.shell_timeout_secs == default_shell_timeout_secs()
        && cfg.permission_mode.is_none()
        && cfg.extra_sensitive_names.is_empty()
}

impl Default for Config {
    fn default() -> Self {
        let mut config = Self {
            openai_api_key: None,
            anthropic_api_key: None,
            gemini_api_key: None,
            custom_providers: Vec::new(),
            dahl_api_key: None,
            apinex_api_key: None,
            api_key: None,
            base_url: String::new(),
            model: String::new(),
            temperature: 0.7,
            provider: String::new(),
            sandbox: SandboxConfigFile::default(),
        };
        // Pick up every provider's key from its environment variable.
        for provider in builtin_providers() {
            if let Ok(key) = env::var(&provider.env_key) {
                if !key.trim().is_empty() {
                    config.set_provider_key(&provider.id, key);
                }
            }
        }
        // First provider with a key wins; OpenAI is the fallback when
        // nothing is set.
        let builtins = builtin_providers();
        let provider = builtins
            .iter()
            .find(|p| config.api_key_for_provider(&p.id).is_some())
            .cloned()
            .unwrap_or_else(|| builtins[0].clone());

        // A provider preset only pins defaults; explicit `{ID}_BASE_URL` /
        // `{ID}_MODEL` environment variables override them per provider.
        config.provider = provider.id.clone();
        config.base_url = env::var(format!("{}_BASE_URL", provider.id.to_uppercase()))
            .unwrap_or_else(|_| provider.base_url.clone());
        config.model = env::var(format!("{}_MODEL", provider.id.to_uppercase()))
            .unwrap_or_else(|_| provider.default_model.clone());
        config.api_key = config.api_key_for_provider(&provider.id);
        config
    }
}

impl Config {
    /// Every provider available to the app: the three built-ins first, then
    /// the custom gateways from `config.json`. Custom entries with empty ids
    /// or ids that shadow a built-in are skipped; duplicate custom ids keep
    /// their first occurrence.
    pub fn providers(&self) -> Vec<Provider> {
        let mut providers = builtin_providers();
        let mut seen: Vec<String> = providers.iter().map(|p| p.id.clone()).collect();
        for custom in &self.custom_providers {
            let id = custom.id.trim().to_ascii_lowercase();
            if id.is_empty() || seen.iter().any(|known| *known == id) {
                continue;
            }
            seen.push(id.clone());
            providers.push(Provider {
                name: custom
                    .name
                    .clone()
                    .filter(|name| !name.trim().is_empty())
                    .unwrap_or_else(|| id.clone()),
                kind: ProviderKind::OpenAICompatible,
                base_url: custom.base_url.trim().to_string(),
                default_model: custom.model.clone().unwrap_or_default(),
                env_key: format!("{}_API_KEY", id.to_uppercase()),
                is_custom: true,
                id,
            });
        }
        providers
    }

    /// Look a provider up by id or display name: exact match
    /// (case-insensitive) first, then a unique prefix.
    pub fn find_provider(&self, id_or_name: &str) -> Option<Provider> {
        let lower = id_or_name.trim().to_ascii_lowercase();
        if lower.is_empty() {
            return None;
        }
        let providers = self.providers();
        if let Some(p) = providers
            .iter()
            .find(|p| p.id == lower || p.name.to_ascii_lowercase() == lower)
        {
            return Some(p.clone());
        }
        let mut starts: Vec<&Provider> = Vec::new();
        for p in &providers {
            let is_match = p.id.starts_with(&lower)
                || p.name.to_ascii_lowercase().starts_with(&lower);
            if is_match && !starts.iter().any(|m| m.id == p.id) {
                starts.push(p);
            }
        }
        if starts.len() == 1 {
            return Some(starts[0].clone());
        }
        None
    }

    fn custom_has(&self, id: &str) -> bool {
        self.custom_providers
            .iter()
            .any(|c| c.id.eq_ignore_ascii_case(id))
    }

    /// Legacy `config.json` support: the Dahl/APInex gateways used to be
    /// built-in providers with dedicated `dahl_api_key` / `apinex_api_key`
    /// fields (and, before that, a single `api_key`). They are now regular
    /// custom providers, so old fields are quietly migrated into
    /// `custom_providers`, keeping each gateway's endpoint and default model.
    /// Dahl keeps its pinned MiniMax default; APInex keeps its
    /// availability-based default (no `model`, resolved from the live list).
    fn migrate_legacy_fields(&mut self) {
        const DAHL_URL: &str = "https://inference.dahl.global/v1";
        const DAHL_MODEL: &str = "MiniMaxAI/MiniMax-M2.7";
        const APINEX_URL: &str = "https://api.apinex.bond/v1";

        let legacy = [
            (
                "dahl",
                "Dahl",
                DAHL_URL,
                Some(DAHL_MODEL),
                self.dahl_api_key.take(),
            ),
            ("apinex", "APInex", APINEX_URL, None, self.apinex_api_key.take()),
        ];
        for (id, name, base_url, model, key) in legacy {
            let Some(key) = key.filter(|key| !key.trim().is_empty()) else {
                continue;
            };
            if self.custom_has(id) {
                continue;
            }
            self.custom_providers.push(CustomProvider {
                id: id.to_string(),
                name: Some(name.to_string()),
                base_url: base_url.to_string(),
                api_key: Some(key),
                model: model.map(str::to_string),
            });
        }

        // The oldest single-key field: `sk-apx…` belonged to APInex,
        // anything else to Dahl.
        let Some(key) = self.api_key.clone().filter(|key| !key.trim().is_empty()) else {
            return;
        };
        let (id, name, base_url, model): (&str, &str, &str, Option<&str>) =
            if key.starts_with("sk-apx") {
                ("apinex", "APInex", APINEX_URL, None)
            } else {
                ("dahl", "Dahl", DAHL_URL, Some(DAHL_MODEL))
            };
        if !self.custom_has(id) {
            self.custom_providers.push(CustomProvider {
                id: id.to_string(),
                name: Some(name.to_string()),
                base_url: base_url.to_string(),
                api_key: Some(key),
                model: model.map(str::to_string),
            });
        } else if self.api_key_for_provider(id).is_none() {
            self.set_provider_key(id, key);
        }
    }

    /// Build an [`ApiClient`] for the active provider: its protocol backend,
    /// bound to the configured API key and base URL.
    pub fn api_client(&self) -> ApiClient {
        let kind = self
            .find_provider(&self.provider)
            .map(|p| p.kind)
            .unwrap_or(ProviderKind::OpenAICompatible);
        ApiClient::new(kind.build(
            self.api_key.clone().unwrap_or_default(),
            self.base_url.clone(),
        ))
    }

    /// Store a provider's key: built-ins have dedicated fields; a custom
    /// provider's key lives on its `custom_providers` entry.
    fn set_provider_key(&mut self, provider_id: &str, key: String) {
        match provider_id {
            "openai" => self.openai_api_key = Some(key),
            "anthropic" => self.anthropic_api_key = Some(key),
            "gemini" => self.gemini_api_key = Some(key),
            other => {
                if let Some(custom) = self
                    .custom_providers
                    .iter_mut()
                    .find(|c| c.id.eq_ignore_ascii_case(other))
                {
                    custom.api_key = Some(key);
                }
            }
        }
    }

    pub fn api_key_for_provider(&self, provider_id: &str) -> Option<String> {
        // The dedicated field wins (the provider's `api_key` entry for
        // customs); the provider's `{ID}_API_KEY` environment variable is
        // the fallback.
        let provider = self.find_provider(provider_id);
        let stored = match provider.as_ref().map(|p| p.id.as_str()) {
            Some("openai") => self.openai_api_key.clone(),
            Some("anthropic") => self.anthropic_api_key.clone(),
            Some("gemini") => self.gemini_api_key.clone(),
            _ => self
                .custom_providers
                .iter()
                .find(|c| c.id.eq_ignore_ascii_case(provider_id))
                .and_then(|c| c.api_key.clone()),
        };
        let env_key = provider.map(|p| p.env_key);
        stored
            .or_else(|| env_key.and_then(|key| env::var(key).ok()))
            .filter(|s| !s.trim().is_empty())
    }

    pub fn set_provider(&mut self, provider_id: &str) -> Result<(), String> {
        let Some(provider) = self.find_provider(provider_id) else {
            return Err(format!("unknown provider: '{provider_id}'"));
        };
        self.provider = provider.id.clone();
        self.base_url = env::var(format!("{}_BASE_URL", provider.id.to_uppercase()))
            .unwrap_or_else(|_| provider.base_url.clone());
        self.model = env::var(format!("{}_MODEL", provider.id.to_uppercase()))
            .unwrap_or_else(|_| provider.default_model.clone());
        self.api_key = self.api_key_for_provider(&provider.id);
        Ok(())
    }

    pub fn load() -> Result<Self> {
        let executable_dir = env::current_exe()
            .ok()
            .and_then(|path| path.parent().map(PathBuf::from));
        let candidates = [
            executable_dir.as_ref().map(|dir| dir.join("config.json")),
            Some(PathBuf::from("config.json")),
            executable_dir.as_ref().map(|dir| dir.join("dahl.json")),
            Some(PathBuf::from("dahl.json")),
        ];
        for candidate in candidates {
            let Some(path) = candidate else { continue };
            if !path.exists() {
                continue;
            }
            let bytes =
                fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
            let mut config: Self = serde_json::from_slice(&bytes)
                .with_context(|| format!("parsing {}", path.display()))?;

            // Retire the removed Dahl/APInex built-ins into
            // `custom_providers` so old config files keep working.
            config.migrate_legacy_fields();

            // Environment keys always win over the file's values, both for
            // the built-ins and, via `{ID}_API_KEY`, for custom providers.
            for provider in builtin_providers() {
                if let Ok(key) = env::var(&provider.env_key) {
                    if !key.trim().is_empty() {
                        config.set_provider_key(&provider.id, key);
                    }
                }
            }
            let custom_ids: Vec<String> = config
                .custom_providers
                .iter()
                .map(|c| c.id.clone())
                .collect();
            for id in custom_ids {
                let env_name = format!("{}_API_KEY", id.to_uppercase());
                if let Ok(key) = env::var(&env_name) {
                    if !key.trim().is_empty() {
                        config.set_provider_key(&id, key);
                    }
                }
            }

            // No explicit provider? Use the first one that has a key,
            // falling back to OpenAI.
            let provider_id = if config.provider.is_empty() {
                config
                    .providers()
                    .iter()
                    .find(|p| config.api_key_for_provider(&p.id).is_some())
                    .map(|p| p.id.clone())
                    .unwrap_or_else(|| builtin_providers()[0].id.clone())
            } else {
                config.provider.clone()
            };
            let provider = config
                .find_provider(&provider_id)
                .unwrap_or_else(|| builtin_providers()[0].clone());
            config.provider = provider.id.clone();

            if config.base_url.is_empty() {
                config.base_url =
                    env::var(format!("{}_BASE_URL", provider.id.to_uppercase()))
                        .unwrap_or_else(|_| provider.base_url.clone());
            }

            if config.model.is_empty() {
                config.model = env::var(format!("{}_MODEL", provider.id.to_uppercase()))
                    .unwrap_or_else(|_| provider.default_model.clone());
            }

            // The legacy `api_key` field also acts as the active provider's
            // key when no dedicated field is present
            // (e.g. `{"provider": "openai", "api_key": "…"}`).
            if config.api_key_for_provider(&config.provider).is_none() {
                if let Some(key) = config.api_key.clone() {
                    if !key.trim().is_empty() {
                        let provider_id = config.provider.clone();
                        config.set_provider_key(&provider_id, key);
                    }
                }
            }

            config.api_key = config.api_key_for_provider(&config.provider);
            return Ok(config);
        }
        Ok(Self::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_registry_is_the_three_main_providers() {
        let config = Config::default();
        let providers = config.providers();
        let ids: Vec<&str> = providers.iter().map(|p| p.id.as_str()).collect();
        assert_eq!(ids, vec!["openai", "anthropic", "gemini"]);
        assert_eq!(providers[0].kind, ProviderKind::OpenAICompatible);
        assert_eq!(providers[0].base_url, "https://api.openai.com/v1");
        assert_eq!(providers[1].kind, ProviderKind::Anthropic);
        assert_eq!(providers[1].base_url, "https://api.anthropic.com/v1");
        assert_eq!(providers[2].kind, ProviderKind::Gemini);
        assert_eq!(
            providers[2].base_url,
            "https://generativelanguage.googleapis.com/v1beta"
        );
        assert!(providers.iter().all(|p| !p.is_custom));
        assert!(config.find_provider("dahl").is_none());
        assert!(config.find_provider("apinex").is_none());
    }

    #[test]
    fn custom_providers_extend_the_registry() {
        let config: Config = serde_json::from_str(
            r#"{
                "custom_providers": [
                    {
                        "id": "dahl",
                        "name": "Dahl",
                        "base_url": "https://inference.dahl.global/v1",
                        "api_key": "dahl-key",
                        "model": "MiniMaxAI/MiniMax-M2.7"
                    },
                    {"id": "groq", "base_url": "https://api.groq.com/openai/v1"}
                ]
            }"#,
        )
        .unwrap();
        let dahl = config.find_provider("dahl").unwrap();
        assert!(dahl.is_custom);
        assert_eq!(dahl.kind, ProviderKind::OpenAICompatible);
        assert_eq!(dahl.name, "Dahl");
        assert_eq!(dahl.base_url, "https://inference.dahl.global/v1");
        assert_eq!(dahl.default_model, "MiniMaxAI/MiniMax-M2.7");
        assert_eq!(dahl.env_key, "DAHL_API_KEY");
        assert_eq!(
            config.api_key_for_provider("dahl").as_deref(),
            Some("dahl-key")
        );
        // No name → falls back to the id; no model → availability-based.
        let groq = config.find_provider("groq").unwrap();
        assert_eq!(groq.name, "groq");
        assert_eq!(groq.default_model, "");
        assert!(groq.availability_based_model());
        // Custom ids may not shadow the built-ins.
        let shadowing: Config = serde_json::from_str(
            r#"{"custom_providers": [{"id": "openai", "base_url": "https://evil.example/v1"}]}"#,
        )
        .unwrap();
        let openai = shadowing.find_provider("openai").unwrap();
        assert!(!openai.is_custom);
        assert_eq!(openai.base_url, "https://api.openai.com/v1");
    }

    #[test]
    fn find_provider_matches_exact_and_prefix() {
        let config: Config = serde_json::from_str(
            r#"{
                "custom_providers": [
                    {"id": "dahl", "name": "Dahl", "base_url": "https://inference.dahl.global/v1"},
                    {"id": "apinex", "name": "APInex", "base_url": "https://api.apinex.bond/v1"}
                ]
            }"#,
        )
        .unwrap();
        assert_eq!(config.find_provider("dahl").unwrap().id, "dahl");
        assert_eq!(config.find_provider("DAHL").unwrap().id, "dahl");
        assert_eq!(config.find_provider("apinex").unwrap().id, "apinex");
        assert_eq!(config.find_provider("APInex").unwrap().id, "apinex");
        assert_eq!(config.find_provider("api").unwrap().id, "apinex");
        assert_eq!(config.find_provider("d").unwrap().id, "dahl");
        assert!(config.find_provider("unknown").is_none());
    }

    #[test]
    fn switching_provider_updates_base_url_and_model() {
        let mut config: Config = serde_json::from_str(
            r#"{
                "custom_providers": [
                    {"id": "apinex", "name": "APInex", "base_url": "https://api.apinex.bond/v1", "model": "gpt-5-6-terra"}
                ]
            }"#,
        )
        .unwrap();
        config.set_provider("anthropic").unwrap();
        assert_eq!(config.provider, "anthropic");
        assert_eq!(config.base_url, "https://api.anthropic.com/v1");
        assert_eq!(config.model, "claude-sonnet-4-5");

        config.set_provider("apinex").unwrap();
        assert_eq!(config.provider, "apinex");
        assert_eq!(config.base_url, "https://api.apinex.bond/v1");
        assert_eq!(config.model, "gpt-5-6-terra");

        // An empty `model` entry stays empty: it is resolved from the
        // endpoint's live model list at runtime.
        let mut config: Config = serde_json::from_str(
            r#"{"custom_providers": [{"id": "local", "base_url": "http://127.0.0.1:11434/v1"}]}"#,
        )
        .unwrap();
        config.set_provider("local").unwrap();
        assert_eq!(config.provider, "local");
        assert_eq!(config.base_url, "http://127.0.0.1:11434/v1");
        assert_eq!(config.model, "");
    }

    #[test]
    fn reads_native_provider_keys_from_json() {
        let config: Config = serde_json::from_str(
            r#"{
                "provider": "openai",
                "openai_api_key": "sk-openai",
                "anthropic_api_key": "sk-ant",
                "gemini_api_key": "AIza"
            }"#,
        )
        .unwrap();
        assert_eq!(config.provider, "openai");
        assert_eq!(config.openai_api_key.as_deref(), Some("sk-openai"));
        assert_eq!(config.anthropic_api_key.as_deref(), Some("sk-ant"));
        assert_eq!(config.gemini_api_key.as_deref(), Some("AIza"));
        assert_eq!(
            config.api_key_for_provider("openai"),
            Some("sk-openai".to_string())
        );
        assert_eq!(
            config.api_key_for_provider("anthropic"),
            Some("sk-ant".to_string())
        );
        assert_eq!(
            config.api_key_for_provider("gemini"),
            Some("AIza".to_string())
        );
    }

    #[test]
    fn set_provider_adopts_the_stored_key() {
        let mut config: Config = serde_json::from_str(
            "{\"gemini_api_key\": \"AIza\", \"provider\": \"gemini\"}",
        )
        .unwrap();
        config.set_provider("gemini").unwrap();
        assert_eq!(config.api_key.as_deref(), Some("AIza"));
        assert_eq!(
            config.base_url,
            "https://generativelanguage.googleapis.com/v1beta"
        );
    }

    #[test]
    fn legacy_dahl_and_apinex_fields_migrate_into_custom_providers() {
        let mut config: Config = serde_json::from_str(
            r#"{"dahl_api_key": "dahl-key", "apinex_api_key": "sk-apx-apinex-key"}"#,
        )
        .unwrap();
        config.migrate_legacy_fields();
        let dahl = config.find_provider("dahl").unwrap();
        assert!(dahl.is_custom);
        assert_eq!(dahl.base_url, "https://inference.dahl.global/v1");
        assert_eq!(dahl.default_model, "MiniMaxAI/MiniMax-M2.7");
        assert_eq!(
            config.api_key_for_provider("dahl").as_deref(),
            Some("dahl-key")
        );
        let apinex = config.find_provider("apinex").unwrap();
        assert_eq!(apinex.base_url, "https://api.apinex.bond/v1");
        // APInex keeps its availability-based default (no pinned model).
        assert_eq!(apinex.default_model, "");
        assert_eq!(
            config.api_key_for_provider("apinex").as_deref(),
            Some("sk-apx-apinex-key")
        );

        // The oldest single-key field routes by its `sk-apx` prefix.
        let mut config: Config =
            serde_json::from_str(r#"{"api_key": "sk-apx-single-key"}"#).unwrap();
        config.migrate_legacy_fields();
        assert_eq!(
            config.api_key_for_provider("apinex").as_deref(),
            Some("sk-apx-single-key")
        );
        assert!(config.find_provider("dahl").is_none());
    }

    #[test]
    fn explicit_custom_entries_win_over_the_legacy_migration() {
        let mut config: Config = serde_json::from_str(
            r#"{
                "apinex_api_key": "old-key",
                "custom_providers": [
                    {"id": "apinex", "base_url": "https://api.apinex.bond/v1", "api_key": "new-key"}
                ]
            }"#,
        )
        .unwrap();
        config.migrate_legacy_fields();
        assert_eq!(config.custom_providers.len(), 1);
        assert_eq!(
            config.api_key_for_provider("apinex").as_deref(),
            Some("new-key")
        );
    }

    #[test]
    fn sandbox_config_defaults_are_backward_compatible() {
        // A config without any sandbox keys keeps the historical behavior:
        // enabled, auto-approved, shell allowed, 30s timeout.
        let config: Config = serde_json::from_str("{}").unwrap();
        assert!(config.sandbox.enabled);
        assert!(config.sandbox.auto_approve);
        assert!(config.sandbox.allow_shell);
        assert_eq!(config.sandbox.shell_timeout_secs, 30);
        assert_eq!(config.sandbox.permission_mode, None);
        assert!(config.sandbox.extra_sensitive_names.is_empty());
        assert!(is_default_sandbox(&config.sandbox));
    }

    #[test]
    fn sandbox_config_reads_the_security_fields() {
        let config: Config = serde_json::from_str(
            r#"{
                "sandbox": {
                    "workspace_root": "/tmp/project",
                    "permission_mode": "workspace-write",
                    "shell_timeout_secs": 120,
                    "extra_sensitive_names": ["secrets.json", ".token"]
                }
            }"#,
        )
        .unwrap();
        assert_eq!(config.sandbox.workspace_root, "/tmp/project");
        assert_eq!(
            config.sandbox.permission_mode.as_deref(),
            Some("workspace-write")
        );
        assert_eq!(config.sandbox.shell_timeout_secs, 120);
        assert_eq!(config.sandbox.extra_sensitive_names.len(), 2);
        assert!(!is_default_sandbox(&config.sandbox));
    }

    #[test]
    fn sandbox_timeout_round_trips_zero_but_enforcement_clamps_it() {
        let config: Config =
            serde_json::from_str(r#"{"sandbox": {"shell_timeout_secs": 0}}"#).unwrap();
        assert_eq!(config.sandbox.shell_timeout_secs, 0);
    }
}
