use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::{env, fs, path::PathBuf};

fn default_base_url() -> String {
    "https://generativelanguage.googleapis.com/v1beta".into()
}

fn is_default_base_url(value: &String) -> bool {
    value == "https://generativelanguage.googleapis.com/v1beta"
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub api_key: Option<String>,
    #[serde(
        default = "default_base_url",
        skip_serializing_if = "is_default_base_url"
    )]
    pub base_url: String,
    pub model: String,
    pub temperature: f32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            api_key: env::var("GEMINI_API_KEY").ok(),
            base_url: env::var("GEMINI_BASE_URL").unwrap_or_else(|_| default_base_url()),
            model: env::var("GEMINI_MODEL").unwrap_or_else(|_| "gemini-3.5-flash".into()),
            temperature: 0.7,
        }
    }
}

impl Config {
    pub fn load() -> Result<Self> {
        let executable_dir = env::current_exe()
            .ok()
            .and_then(|path| path.parent().map(PathBuf::from));
        let candidates = [
            executable_dir.map(|dir| dir.join("gemini.json")),
            Some(PathBuf::from("gemini.json")),
        ];
        for candidate in candidates {
            if let Some(path) = candidate {
                if path.exists() {
                    let bytes =
                        fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
                    let mut config: Self =
                        serde_json::from_slice(&bytes).context("parsing gemini.json")?;
                    if config.api_key.is_none() {
                        config.api_key = env::var("GEMINI_API_KEY").ok();
                    }
                    return Ok(config);
                }
            }
        }
        Ok(Self::default())
    }
}
