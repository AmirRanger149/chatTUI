use anyhow::Result;
use directories::ProjectDirs;
use serde::{Deserialize, Serialize};
use std::{fs, path::PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallRecord {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallRecord>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    pub id: u64,
    pub title: String,
    pub messages: Vec<Message>,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct Store {
    sessions: Vec<Session>,
    current: usize,
}

pub struct SessionManager {
    store: Store,
    path: PathBuf,
}

impl SessionManager {
    pub fn load() -> Result<Self> {
        let path = ProjectDirs::from("", "chatTUI", "chat-tui")
            .map(|d| d.data_dir().join("sessions.json"))
            .unwrap_or_else(|| PathBuf::from("sessions.json"));
        let mut manager = Self {
            store: Store::default(),
            path,
        };
        if manager.path.exists() {
            manager.store = serde_json::from_slice(&fs::read(&manager.path)?)?;
        }
        if manager.store.sessions.is_empty() {
            manager.new_session();
        } else {
            manager.store.current = manager.store.current.min(manager.store.sessions.len() - 1);
        }
        Ok(manager)
    }

    pub fn save(&self) -> Result<()> {
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&self.path, serde_json::to_vec_pretty(&self.store)?)?;
        Ok(())
    }

    pub fn current(&self) -> &Session {
        &self.store.sessions[self.store.current]
    }
    pub fn current_mut(&mut self) -> &mut Session {
        &mut self.store.sessions[self.store.current]
    }
    pub fn sessions(&self) -> &[Session] {
        &self.store.sessions
    }

    pub fn current_index(&self) -> usize {
        self.store.current
    }

    pub fn len(&self) -> usize {
        self.store.sessions.len()
    }

    pub fn select(&mut self, index: usize) {
        if index < self.store.sessions.len() {
            self.store.current = index;
            let _ = self.save();
        }
    }

    pub fn delete_at(&mut self, index: usize) {
        if self.store.sessions.len() <= 1 || index >= self.store.sessions.len() {
            return;
        }
        self.store.sessions.remove(index);
        if self.store.current == index {
            self.store.current = self.store.current.min(self.store.sessions.len() - 1);
        } else if self.store.current > index {
            self.store.current -= 1;
        }
        let _ = self.save();
    }

    pub fn new_session(&mut self) {
        let id = self.store.sessions.last().map(|s| s.id + 1).unwrap_or(1);
        self.store.sessions.insert(
            0,
            Session {
                id,
                title: "New conversation".into(),
                messages: Vec::new(),
            },
        );
        self.store.current = 0;
        let _ = self.save();
    }

    pub fn add_message(&mut self, role: impl Into<String>, content: impl Into<String>) {
        let message = Message {
            role: role.into(),
            content: content.into(),
            tool_call_id: None,
            tool_calls: None,
        };
        let session = self.current_mut();
        if session.title == "New conversation" && message.role == "user" {
            session.title = message
                .content
                .lines()
                .next()
                .unwrap_or("New conversation")
                .chars()
                .take(42)
                .collect();
        }
        session.messages.push(message);
        let _ = self.save();
    }

    pub fn add_tool_result(&mut self, tool_call_id: impl Into<String>, content: impl Into<String>) {
        let message = Message {
            role: "tool".into(),
            content: content.into(),
            tool_call_id: Some(tool_call_id.into()),
            tool_calls: None,
        };
        self.current_mut().messages.push(message);
        let _ = self.save();
    }

    pub fn add_assistant_with_tools(&mut self, content: impl Into<String>, tool_calls: Vec<ToolCallRecord>) {
        let message = Message {
            role: "assistant".into(),
            content: content.into(),
            tool_call_id: None,
            tool_calls: Some(tool_calls),
        };
        self.current_mut().messages.push(message);
        let _ = self.save();
    }
}

#[cfg(test)]
impl SessionManager {
    pub fn for_tests() -> Self {
        let mut manager = Self {
            store: Store::default(),
            path: std::env::temp_dir().join("chat-tui-test-sessions.json"),
        };
        manager.new_session();
        manager
    }
}
