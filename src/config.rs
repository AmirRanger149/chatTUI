use crate::api::client::ApiClient;
use crate::api::providers::{HttpTimeouts, ProviderKind};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use std::{env, fs, path::PathBuf};

/// A chat provider: an id/name, the wire protocol it speaks, its endpoint,
/// default model, and the environment variable that may hold its API key.
///
/// There are two kinds: the three built-ins (`openai`, `anthropic`, `gemini`)
/// and user-defined custom gateways declared under `custom_providers` in
/// `config.json` (Ollama, Groq, OpenRouter, and anything else that speaks
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
    /// OpenAI-compatible base URL, e.g. `https://api.groq.com/openai/v1`.
    pub base_url: String,
    /// API key; `{ID}_API_KEY` in the environment is the fallback.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    /// Default model. When omitted, chatTUI picks one from the endpoint's
    /// live model list (a `free/` model first).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

fn is_zero(value: &u64) -> bool {
    *value == 0
}

fn default_context_window_tokens() -> u64 {
    128_000
}

fn is_default_context_window_tokens(value: &u64) -> bool {
    *value == default_context_window_tokens()
}

fn default_compact_at_percent() -> u8 {
    80
}

fn is_default_compact_at_percent(value: &u8) -> bool {
    *value == default_compact_at_percent()
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
    /// Kernel-level isolation for shell commands: "auto" (default — apply
    /// it when the Linux kernel supports it, warn honestly otherwise),
    /// "require" (refuse shell commands when unavailable), or "off"
    /// (never apply; commands run with full user privileges).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub os_isolation: Option<String>,
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
            os_isolation: None,
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
    /// User-defined OpenAI-compatible gateways (Ollama, Groq, OpenRouter, …).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub custom_providers: Vec<CustomProvider>,
    /// The resolved API key of the active provider — runtime state set by
    /// `set_provider`, never read from config.json.
    #[serde(skip)]
    pub api_key: Option<String>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub base_url: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub model: String,
    #[serde(default = "default_temperature", skip_serializing_if = "is_default_temperature")]
    pub temperature: f32,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub provider: String,
    /// How a model's reasoning is treated on replay: `"strip"` (default)
    /// removes it before sending history back, `"opaque"` sends it back
    /// untouched, `"auto"` picks per provider. See `ReasoningReplay`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_replay: Option<String>,
    /// Context window size in tokens, used to decide when to compact. Only a
    /// fallback estimate is possible without it, so it is worth setting to
    /// your model's real number.
    #[serde(
        default = "default_context_window_tokens",
        skip_serializing_if = "is_default_context_window_tokens"
    )]
    pub context_window_tokens: u64,
    /// Compact the history once it passes this percentage of
    /// `context_window_tokens`.
    #[serde(
        default = "default_compact_at_percent",
        skip_serializing_if = "is_default_compact_at_percent"
    )]
    pub compact_at_percent: u8,
    /// Token budget for extended thinking, where the provider supports it
    /// (Anthropic). Unset means do not request it — and then there is no
    /// thinking block to round-trip either.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking_budget_tokens: Option<u32>,
    /// Cap on how many tokens the model may emit in ONE response. `0` (the
    /// default) sends no cap at all, so the provider applies its own default
    /// — and provider defaults are commonly only a few thousand tokens. That
    /// is the usual reason a large `write_file` call arrives with its
    /// arguments cut off mid-JSON: the response hit its output cap before the
    /// model finished writing the tool call. Set this to your model's real
    /// maximum output when large writes or long answers get truncated.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub max_output_tokens: u64,
    /// How long establishing an API connection may take, in seconds, before
    /// the attempt is abandoned (clamped to at least 1s at use).
    #[serde(
        default = "default_connect_timeout_secs",
        skip_serializing_if = "is_default_connect_timeout_secs"
    )]
    pub connect_timeout_secs: u64,
    /// How long an API stream may stay quiet between chunks, in seconds,
    /// before the connection is treated as dead (clamped to at least 1s at
    /// use). This is *not* a cap on the total generation time — a stream
    /// that keeps producing tokens may run as long as it needs.
    #[serde(
        default = "default_idle_timeout_secs",
        skip_serializing_if = "is_default_idle_timeout_secs"
    )]
    pub idle_timeout_secs: u64,
    /// How long to wait for a model's **first** output before treating the
    /// connection as dead, in seconds. Deliberately much longer than
    /// `idle_timeout_secs`: reasoning models can think for many minutes and
    /// some gateways buffer the whole chain of thought before sending a
    /// byte, so a short window here would kill a healthy request. Lower it
    /// if your provider fails fast and you would rather not wait.
    #[serde(
        default = "default_first_token_timeout_secs",
        skip_serializing_if = "is_default_first_token_timeout_secs"
    )]
    pub first_token_timeout_secs: u64,
    #[serde(default, skip_serializing_if = "is_default_sandbox")]
    pub sandbox: SandboxConfigFile,
    #[serde(default, skip_serializing_if = "is_default_agent")]
    pub agent: AgentConfigFile,
}

/// Agent-loop settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfigFile {
    /// Backstop ceiling for agent tool rounds. This is a safety net, not
    /// the normal stopping rule: the agent keeps working as long as it
    /// makes progress, and broken loops are stopped much earlier by the
    /// stagnation and consecutive-failure guards.
    #[serde(default = "default_agent_max_rounds")]
    pub max_rounds: u64,
}

impl Default for AgentConfigFile {
    fn default() -> Self {
        Self {
            max_rounds: default_agent_max_rounds(),
        }
    }
}

fn default_agent_max_rounds() -> u64 {
    crate::app::DEFAULT_AGENT_MAX_ROUNDS
}

fn is_default_agent(cfg: &AgentConfigFile) -> bool {
    cfg.max_rounds == default_agent_max_rounds()
}

fn is_default_sandbox(cfg: &SandboxConfigFile) -> bool {
    cfg.enabled
        && cfg.workspace_root.is_empty()
        && cfg.auto_approve
        && cfg.allow_shell
        && cfg.shell_timeout_secs == default_shell_timeout_secs()
        && cfg.permission_mode.is_none()
        && cfg.os_isolation.is_none()
        && cfg.extra_sensitive_names.is_empty()
}

fn default_connect_timeout_secs() -> u64 {
    15
}

fn is_default_connect_timeout_secs(value: &u64) -> bool {
    *value == default_connect_timeout_secs()
}

fn default_idle_timeout_secs() -> u64 {
    90
}

fn is_default_idle_timeout_secs(value: &u64) -> bool {
    *value == default_idle_timeout_secs()
}

fn default_first_token_timeout_secs() -> u64 {
    1800
}

fn is_default_first_token_timeout_secs(value: &u64) -> bool {
    *value == default_first_token_timeout_secs()
}

impl Default for Config {
    fn default() -> Self {
        let mut config = Self {
            openai_api_key: None,
            anthropic_api_key: None,
            gemini_api_key: None,
            custom_providers: Vec::new(),
            api_key: None,
            base_url: String::new(),
            model: String::new(),
            temperature: 0.7,
            provider: String::new(),
            connect_timeout_secs: default_connect_timeout_secs(),
            idle_timeout_secs: default_idle_timeout_secs(),
            first_token_timeout_secs: default_first_token_timeout_secs(),
            reasoning_replay: None,
            thinking_budget_tokens: None,
            max_output_tokens: 0,
            context_window_tokens: default_context_window_tokens(),
            compact_at_percent: default_compact_at_percent(),
            sandbox: SandboxConfigFile::default(),
            agent: AgentConfigFile::default(),
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

    /// Build an [`ApiClient`] for the active provider: its protocol backend,
    /// bound to the configured API key and base URL.
    pub fn api_client(&self) -> ApiClient {
        let kind = self
            .find_provider(&self.provider)
            .map(|p| p.kind)
            .unwrap_or(ProviderKind::OpenAICompatible);
        // Clamp away zero so a misconfiguration can never mean "no timeout".
        let timeouts = HttpTimeouts {
            connect: Duration::from_secs(self.connect_timeout_secs.max(1)),
            idle: Duration::from_secs(self.idle_timeout_secs.max(1)),
            first_token: Duration::from_secs(self.first_token_timeout_secs.max(1)),
        };
        ApiClient::new(kind.build(
            self.api_key.clone().unwrap_or_default(),
            self.base_url.clone(),
            timeouts,
            self.thinking_budget_tokens,
            self.max_output_tokens,
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
        assert!(config.find_provider("acme").is_none());
        assert!(config.find_provider("beta").is_none());
    }

    #[test]
    fn custom_providers_extend_the_registry() {
        let config: Config = serde_json::from_str(
            r#"{
                "custom_providers": [
                    {
                        "id": "acme",
                        "name": "Acme",
                        "base_url": "https://api.acme.example/v1",
                        "api_key": "acme-key",
                        "model": "AcmeAI/acme-model-1"
                    },
                    {"id": "beta", "base_url": "https://api.beta.example/v1"}
                ]
            }"#,
        )
        .unwrap();
        let acme = config.find_provider("acme").unwrap();
        assert!(acme.is_custom);
        assert_eq!(acme.kind, ProviderKind::OpenAICompatible);
        assert_eq!(acme.name, "Acme");
        assert_eq!(acme.base_url, "https://api.acme.example/v1");
        assert_eq!(acme.default_model, "AcmeAI/acme-model-1");
        assert_eq!(acme.env_key, "ACME_API_KEY");
        assert_eq!(
            config.api_key_for_provider("acme").as_deref(),
            Some("acme-key")
        );
        // No name → falls back to the id; no model → availability-based.
        let beta = config.find_provider("beta").unwrap();
        assert_eq!(beta.name, "beta");
        assert_eq!(beta.default_model, "");
        assert!(beta.availability_based_model());
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
                    {"id": "acme", "name": "Acme", "base_url": "https://api.acme.example/v1"},
                    {"id": "beta", "name": "Beta", "base_url": "https://api.beta.example/v1"}
                ]
            }"#,
        )
        .unwrap();
        assert_eq!(config.find_provider("acme").unwrap().id, "acme");
        assert_eq!(config.find_provider("ACME").unwrap().id, "acme");
        assert_eq!(config.find_provider("beta").unwrap().id, "beta");
        assert_eq!(config.find_provider("Beta").unwrap().id, "beta");
        assert_eq!(config.find_provider("be").unwrap().id, "beta");
        assert_eq!(config.find_provider("ac").unwrap().id, "acme");
        assert!(config.find_provider("unknown").is_none());
    }

    #[test]
    fn switching_provider_updates_base_url_and_model() {
        let mut config: Config = serde_json::from_str(
            r#"{
                "custom_providers": [
                    {"id": "acme", "name": "Acme", "base_url": "https://api.acme.example/v1", "model": "acme-model-9"}
                ]
            }"#,
        )
        .unwrap();
        config.set_provider("anthropic").unwrap();
        assert_eq!(config.provider, "anthropic");
        assert_eq!(config.base_url, "https://api.anthropic.com/v1");
        assert_eq!(config.model, "claude-sonnet-4-5");

        config.set_provider("acme").unwrap();
        assert_eq!(config.provider, "acme");
        assert_eq!(config.base_url, "https://api.acme.example/v1");
        assert_eq!(config.model, "acme-model-9");

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
    fn sandbox_config_defaults_are_backward_compatible() {
        // A config without any sandbox keys keeps the historical behavior:
        // enabled, auto-approved, shell allowed, 30s timeout.
        let config: Config = serde_json::from_str("{}").unwrap();
        assert!(config.sandbox.enabled);
        assert!(config.sandbox.auto_approve);
        assert!(config.sandbox.allow_shell);
        assert_eq!(config.sandbox.shell_timeout_secs, 30);
        assert_eq!(config.sandbox.permission_mode, None);
        assert_eq!(config.sandbox.os_isolation, None);
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
                    "os_isolation": "require",
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
        assert_eq!(config.sandbox.os_isolation.as_deref(), Some("require"));
        assert_eq!(config.sandbox.extra_sensitive_names.len(), 2);
        assert!(!is_default_sandbox(&config.sandbox));
    }

    #[test]
    fn agent_max_rounds_defaults_and_parses() {
        let config: Config = serde_json::from_str("{}").unwrap();
        assert_eq!(config.agent.max_rounds, crate::app::DEFAULT_AGENT_MAX_ROUNDS);
        let config: Config = serde_json::from_str(r#"{"agent": {"max_rounds": 120}}"#).unwrap();
        assert_eq!(config.agent.max_rounds, 120);
    }

    #[test]
    fn sandbox_timeout_round_trips_zero_but_enforcement_clamps_it() {
        let config: Config =
            serde_json::from_str(r#"{"sandbox": {"shell_timeout_secs": 0}}"#).unwrap();
        assert_eq!(config.sandbox.shell_timeout_secs, 0);
    }
}
