use anyhow::{anyhow, Context, Result};
use futures_util::StreamExt;
use reqwest::Client;
use serde_json::{json, Value};
use std::time::Duration;
use tokio::sync::mpsc::Sender;

pub struct ApiClient { http: Client, api_key: String, base_url: String }

impl ApiClient {
    pub fn new(api_key: String, base_url: String) -> Self {
        Self { http: Client::builder().timeout(Duration::from_secs(90)).build().expect("HTTP client"), api_key, base_url: base_url.trim_end_matches('/').into() }
    }

    pub async fn stream_chat(&self, messages: &[(String, String)], model: &str, temperature: f32, tx: Sender<Result<String>>) -> Result<()> {
        let mut system = String::new();
        let mut contents = Vec::new();
        for (role, content) in messages {
            if role == "system" { system = content.clone(); } else {
                contents.push(json!({ "role": if role == "assistant" { "model" } else { "user" }, "parts": [{ "text": content }] }));
            }
        }
        let mut body = json!({ "contents": contents, "generationConfig": { "temperature": temperature } });
        if !system.is_empty() { body["systemInstruction"] = json!({ "parts": [{ "text": system }] }); }
        let url = format!("{}/models/{}:streamGenerateContent", self.base_url, model);
        let response = self.http.post(url).query(&[("alt", "sse"), ("key", self.api_key.as_str())]).json(&body).send().await.context("sending Gemini request")?;
        let status = response.status();
        if !status.is_success() { return Err(anyhow!("Gemini request failed (HTTP {}): {}", status, response.text().await.unwrap_or_default())); }
        let mut stream = response.bytes_stream();
        let mut buffer = String::new();
        while let Some(chunk) = stream.next().await {
            buffer.push_str(&String::from_utf8_lossy(&chunk?));
            while let Some(end) = buffer.find('\n') {
                let line = buffer.drain(..=end).collect::<String>();
                let data = line.strip_prefix("data:").map(str::trim).unwrap_or_default();
                if data.is_empty() { continue; }
                let value: Value = serde_json::from_str(data).unwrap_or_default();
                if let Some(text) = value["candidates"][0]["content"]["parts"][0]["text"].as_str() { tx.send(Ok(text.into())).await.map_err(|_| anyhow!("stream receiver closed"))?; }
            }
        }
        Ok(())
    }
}
