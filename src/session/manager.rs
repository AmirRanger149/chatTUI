use anyhow::Result;
use directories::ProjectDirs;
use serde::{Deserialize, Serialize};
use std::{fs, path::PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message { pub role: String, pub content: String }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    pub id: u64,
    pub title: String,
    pub messages: Vec<Message>,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct Store { sessions: Vec<Session>, current: usize }

pub struct SessionManager { store: Store, path: PathBuf }

impl SessionManager {
    pub fn load() -> Result<Self> {
        let path = ProjectDirs::from("", "chatTUI", "chat-tui").map(|d| d.data_dir().join("sessions.json")).unwrap_or_else(|| PathBuf::from("sessions.json"));
        let mut manager = Self { store: Store::default(), path };
        if manager.path.exists() { manager.store = serde_json::from_slice(&fs::read(&manager.path)?)?; }
        if manager.store.sessions.is_empty() { manager.new_session(); } else { manager.store.current = manager.store.current.min(manager.store.sessions.len() - 1); }
        Ok(manager)
    }

    pub fn save(&self) -> Result<()> {
        if let Some(parent) = self.path.parent() { fs::create_dir_all(parent)?; }
        fs::write(&self.path, serde_json::to_vec_pretty(&self.store)?)?;
        Ok(())
    }

    pub fn current(&self) -> &Session { &self.store.sessions[self.store.current] }
    pub fn current_mut(&mut self) -> &mut Session { &mut self.store.sessions[self.store.current] }
    pub fn sessions(&self) -> &[Session] { &self.store.sessions }

    pub fn new_session(&mut self) { let id = self.store.sessions.last().map(|s| s.id + 1).unwrap_or(1); self.store.sessions.insert(0, Session { id, title: "New conversation".into(), messages: Vec::new() }); self.store.current = 0; let _ = self.save(); }
    pub fn select(&mut self, index: usize) { if index < self.store.sessions.len() { self.store.current = index; let _ = self.save(); } }
    pub fn delete_current(&mut self) { if self.store.sessions.len() > 1 { self.store.sessions.remove(self.store.current); self.store.current = self.store.current.min(self.store.sessions.len() - 1); let _ = self.save(); } }
    pub fn add_message(&mut self, role: impl Into<String>, content: impl Into<String>) { let message = Message { role: role.into(), content: content.into() }; let session = self.current_mut(); if session.title == "New conversation" && message.role == "user" { session.title = message.content.lines().next().unwrap_or("New conversation").chars().take(42).collect(); } session.messages.push(message); let _ = self.save(); }
}
