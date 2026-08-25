use crate::{api::client::ApiClient, config::Config, session::manager::SessionManager};
use anyhow::Result;
use tokio::sync::mpsc::error::TryRecvError;
use tokio::sync::mpsc::{self, Receiver};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode { Normal, Insert }

pub enum Action { Quit, EnterInsert, EnterNormal, Submit(String), NewChat, ToggleHistory, ScrollUp, ScrollDown, Help, SelectHistory(usize) }

pub struct App { pub config: Config, pub sessions: SessionManager, pub mode: Mode, pub show_history: bool, pub scroll: u16, pub streaming: bool, pub error: Option<String>, pub response: String, pub tokens: Option<Receiver<Result<String>>> }

impl App {
    pub fn new(config: Config, sessions: SessionManager) -> Self { Self { config, sessions, mode: Mode::Normal, show_history: false, scroll: 0, streaming: false, error: None, response: String::new(), tokens: None } }
    pub fn dispatch(&mut self, action: Action) -> bool {
        match action {
            Action::Quit => return false,
            Action::EnterInsert => self.mode = Mode::Insert,
            Action::EnterNormal => self.mode = Mode::Normal,
            Action::NewChat => { self.sessions.new_session(); self.response.clear(); self.error = None; },
            Action::ToggleHistory => self.show_history = !self.show_history,
            Action::ScrollUp => self.scroll = self.scroll.saturating_add(2),
            Action::ScrollDown => self.scroll = self.scroll.saturating_sub(2),
            Action::Help => self.error = Some("i insert  Esc normal  Enter send  j/k scroll  h history  n new chat  q quit".into()),
            Action::SelectHistory(index) => { self.sessions.select(index); self.show_history = false; self.response.clear(); },
            Action::Submit(text) => { self.mode = Mode::Normal; self.response.clear(); self.sessions.add_message("user", text); },
        }
        true
    }

    pub async fn start_stream(&mut self) -> Result<()> {
        let Some(api_key) = self.config.api_key.clone() else {
            return Err(anyhow::anyhow!("GEMINI_API_KEY is not configured"));
        };
        let (tx, rx) = mpsc::channel(64);
        let messages: Vec<(String, String)> = self.sessions.current().messages.iter().map(|m| (m.role.clone(), m.content.clone())).collect();
        let model = self.config.model.clone();
        let temperature = self.config.temperature;
        let api_key_client = ApiClient::new(api_key, self.config.base_url.clone());
        tokio::spawn(async move {
            if let Err(error) = api_key_client.stream_chat(&messages, &model, temperature, tx.clone()).await {
                let _ = tx.send(Err(error)).await;
            }
        });
        self.tokens = Some(rx); self.streaming = true; Ok(())
    }

    pub async fn receive_token(&mut self) {
        let Some(mut rx) = self.tokens.take() else { return; };
        loop {
            match rx.try_recv() {
                Ok(Ok(token)) => self.response.push_str(&token),
                Ok(Err(error)) => self.error = Some(error.to_string()),
                Err(TryRecvError::Empty) => { self.tokens = Some(rx); return; }
                Err(TryRecvError::Disconnected) => break,
            }
        }
        self.streaming = false;
        if !self.response.is_empty() { self.sessions.add_message("assistant", self.response.clone()); }
    }
}
