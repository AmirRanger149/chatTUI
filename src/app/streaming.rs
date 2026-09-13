//! Running a chat request: spawning the background stream, consuming its
//! token events frame by frame, and reporting elapsed time while it runs.

use crate::api::types::{Message, Role, StreamEvent};
use crate::app::App;
use anyhow::Result;
use std::time::Instant;
use tokio::sync::mpsc::{self, error::TryRecvError};

impl App {
    pub fn elapsed_secs(&self) -> u64 {
        self.stream_started
            .map(|started| started.elapsed().as_secs())
            .unwrap_or(0)
    }

    pub fn elapsed_ms(&self) -> u64 {
        self.stream_started
            .map(|started| started.elapsed().as_millis() as u64)
            .unwrap_or(0)
    }

    pub(crate) fn start_stream(&mut self) -> Result<()> {
        if self.config.api_key.is_none() {
            let env_key = self
                .config
                .find_provider(&self.config.provider)
                .map(|p| p.env_key)
                .unwrap_or_else(|| "API_KEY".to_string());
            return Err(anyhow::anyhow!(
                "{env_key} is not configured — set it in config.json or the environment"
            ));
        }
        let (tx, rx) = mpsc::channel(64);
        let messages: Vec<Message> = self
            .sessions
            .current()
            .messages
            .iter()
            // Reasoning belongs to the turn that produced it, so never replay
            // `<think>` blocks back to the model.
            .map(|m| {
                Message::new(
                    Role::from(m.role.as_str()),
                    crate::ui::thinking::strip(&m.content),
                )
            })
            .collect();
        let model = self.config.model.clone();
        let temperature = self.config.temperature;
        let client = self.config.api_client();
        tokio::spawn(async move {
            if let Err(error) = client
                .stream_chat(&messages, &model, temperature, tx.clone())
                .await
            {
                let _ = tx.send(StreamEvent::Error(error.to_string())).await;
            }
        });
        self.tokens = Some(rx);
        self.streaming = true;
        self.stream_started = Some(Instant::now());
        Ok(())
    }

    pub async fn receive_token(&mut self) {
        let Some(mut rx) = self.tokens.take() else {
            return;
        };
        loop {
            match rx.try_recv() {
                Ok(StreamEvent::Delta(token)) => self.response.push_str(&token),
                // Fallback announcements and the like; the stream keeps going.
                Ok(StreamEvent::Notice(message)) => self.push_notice(message),
                Ok(StreamEvent::Error(error)) => {
                    self.finish_partial();
                    self.streaming = false;
                    self.stream_started = None;
                    self.push_error(error);
                    return;
                }
                Err(TryRecvError::Empty) => {
                    self.tokens = Some(rx);
                    return;
                }
                Err(TryRecvError::Disconnected) => break,
            }
        }
        self.streaming = false;
        self.stream_started = None;
        self.finish_partial();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::{App, Cell};
    use crate::config::Config;
    use crate::session::manager::SessionManager;

    fn test_app() -> App {
        App::new(Config::default(), SessionManager::for_tests())
    }

    #[test]
    fn stream_notices_become_transcript_rows() {
        let (tx, rx) = tokio::sync::mpsc::channel(4);
        tx.try_send(StreamEvent::Notice("switching to b/2".into()))
            .unwrap();
        tx.try_send(StreamEvent::Delta("hel".into())).unwrap();
        let mut app = test_app();
        app.streaming = true;
        app.tokens = Some(rx);
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        runtime.block_on(app.receive_token());
        assert_eq!(app.response, "hel");
        assert_eq!(app.cells.len(), 1);
        assert!(matches!(app.cells.last(), Some(Cell::Notice(_))));

        // Dropping the sender ends the stream and commits the partial answer.
        drop(tx);
        runtime.block_on(app.receive_token());
        assert!(!app.streaming);
        assert_eq!(app.cells.len(), 2);
        assert!(matches!(app.cells[0], Cell::Notice(_)));
        assert_eq!(app.cells[1], Cell::Assistant("hel".into()));
    }
}
