use anyhow::Result;
use directories::ProjectDirs;
use serde::{Deserialize, Serialize};
use std::{fs, path::PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallRecord {
    pub id: String,
    pub name: String,
    pub arguments: String,
    /// Gemini's opaque `thoughtSignature` for this call — must be echoed
    /// back verbatim on replay. Absent for other providers and for
    /// sessions recorded before signatures were preserved.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallRecord>>,
    /// Whether this tool result was an error. Absent on non-tool messages
    /// and on sessions saved before statuses were recorded — a replayed
    /// history has to be able to tell a failed tool call from a successful
    /// one, or every failure renders as a success after a restart.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub is_error: Option<bool>,
    /// The provider's signature over this assistant message's reasoning, when
    /// it sent one. Replaying a signed thinking block without its signature
    /// is rejected, so a reasoning round-trip is impossible without it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_signature: Option<String>,
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
            // A corrupt history must never brick startup: quarantine the
            // file and start fresh instead of failing.
            let bytes = fs::read(&manager.path)?;
            match serde_json::from_slice(&bytes) {
                Ok(store) => manager.store = store,
                Err(error) => {
                    let stamp = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_secs())
                        .unwrap_or(0);
                    let quarantine = manager
                        .path
                        .with_file_name(format!("sessions.json.corrupt-{stamp}"));
                    let _ = fs::rename(&manager.path, &quarantine);
                    eprintln!(
                        "chatTUI: sessions.json was unreadable ({error}); moved to {} and starting fresh",
                        quarantine.display()
                    );
                }
            }
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
        // Atomic save: write a temp file, then rename over the real one. A
        // crash mid-write can then at worst lose the save itself, never
        // leave a truncated sessions.json behind.
        let tmp = self.path.with_extension("json.tmp");
        fs::write(&tmp, serde_json::to_vec_pretty(&self.store)?)?;
        fs::rename(&tmp, &self.path)?;
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
            is_error: None,
            reasoning_signature: None,
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

    /// Record a tool result. `is_error` distinguishes a failure (including a
    /// permission denial, a timeout or an interrupt) from a success, and is
    /// what the transcript renders after the session is reloaded.
    pub fn add_tool_result(
        &mut self,
        tool_call_id: impl Into<String>,
        content: impl Into<String>,
        is_error: bool,
    ) {
        let message = Message {
            role: "tool".into(),
            content: content.into(),
            tool_call_id: Some(tool_call_id.into()),
            tool_calls: None,
            is_error: Some(is_error),
            reasoning_signature: None,
        };
        self.current_mut().messages.push(message);
        let _ = self.save();
    }

    /// Rough token estimate for the current session, used when the provider
    /// has not reported real counts yet. `chars / 4` is wrong by up to 2x,
    /// which is why it is only ever the fallback.
    pub fn estimate_tokens(&self) -> usize {
        self.current()
            .messages
            .iter()
            .map(|message| message.content.len())
            .sum::<usize>()
            / 4
    }

    /// Shrink the history so the next request fits in the window.
    ///
    /// Two passes, cheapest first. Old tool output is replaced by a one-line
    /// placeholder: a 20 kB build log from ten rounds ago is the biggest thing
    /// in the history and the least useful part of it. Only if that is not
    /// enough are whole messages dropped, and then only in groups — an
    /// assistant message that made tool calls must keep every one of its
    /// results, or the provider rejects the whole request.
    ///
    /// Returns what was removed, for the transcript, or `None` when there was
    /// nothing worth doing.
    pub fn compact(&mut self, keep_recent: usize, target_tokens: usize) -> Option<String> {
        let total = self.current().messages.len();
        if total <= keep_recent {
            return None;
        }
        let cut = total - keep_recent;

        // Pass 1: elide old tool output in place. Message count and pairing
        // are untouched, so this pass cannot break anything.
        let mut elided = 0usize;
        {
            let messages = &mut self.current_mut().messages;
            for message in messages.iter_mut().take(cut) {
                if message.role != "tool" || message.content.len() < 400 {
                    continue;
                }
                let was = message.content.len();
                message.content =
                    format!("[tool output elided during compaction — was {was} chars]");
                elided += 1;
            }
        }

        // Pass 2: drop whole groups from the front while still over budget.
        let mut dropped = 0usize;
        while self.estimate_tokens() > target_tokens {
            let total = self.current().messages.len();
            if total <= keep_recent {
                break;
            }
            // Find the end of the first droppable group: an assistant message
            // with tool calls, plus every result that answers it.
            let messages = &self.current().messages;
            let mut end = 1usize;
            if messages[0].role == "assistant" && messages[0].tool_calls.is_some() {
                while end < total && messages[end].role == "tool" {
                    end += 1;
                }
            } else if messages[0].role == "tool" {
                // An orphan result with no call to answer: it can never be
                // sent, so it goes.
                while end < total && messages[end].role == "tool" {
                    end += 1;
                }
            }
            let keep_from = end.min(total.saturating_sub(keep_recent).max(1));
            drop(messages);
            let removed: Vec<Message> = self.current_mut().messages.drain(..keep_from).collect();
            dropped += removed.len();
            if removed.is_empty() {
                break;
            }
        }

        if elided == 0 && dropped == 0 {
            return None;
        }
        if dropped > 0 {
            self.current_mut().messages.insert(
                0,
                Message {
                    role: "user".into(),
                    content: format!(
                        "[{dropped} earlier messages were removed to fit the context window.                          The recent conversation below is intact; ask again if you need                          something from before.]"
                    ),
                    tool_call_id: None,
                    tool_calls: None,
                    is_error: None,
                    reasoning_signature: None,
                },
            );
        }
        Some(format!(
            "compacted the history: {elided} old tool outputs elided, {dropped} messages dropped"
        ))
    }

    /// Attach a reasoning signature to the assistant message just stored.
    pub fn set_last_assistant_reasoning_signature(&mut self, signature: String) {
        if let Some(message) = self.current_mut().messages.last_mut() {
            if message.role == "assistant" {
                message.reasoning_signature = Some(signature);
            }
        }
    }

    pub fn add_assistant_with_tools(&mut self, content: impl Into<String>, tool_calls: Vec<ToolCallRecord>) {
        let message = Message {
            role: "assistant".into(),
            content: content.into(),
            tool_call_id: None,
            tool_calls: Some(tool_calls),
            is_error: None,
            reasoning_signature: None,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_call_record_round_trips_the_signature() {
        let record = ToolCallRecord {
            id: "c1".into(),
            name: "list_files".into(),
            arguments: "{}".into(),
            signature: Some("sig-bytes".into()),
        };
        let json = serde_json::to_string(&record).unwrap();
        let back: ToolCallRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(back.signature.as_deref(), Some("sig-bytes"));
    }

    #[test]
    fn legacy_tool_call_records_have_no_signature() {
        // Sessions written before signatures existed must load unchanged.
        let back: ToolCallRecord =
            serde_json::from_str(r#"{"id":"c1","name":"x","arguments":"{}"}"#).unwrap();
        assert_eq!(back.signature, None);
    }

    #[test]
    fn signature_free_records_serialize_without_the_field() {
        let record = ToolCallRecord {
            id: "c1".into(),
            name: "list_files".into(),
            arguments: "{}".into(),
            signature: None,
        };
        let json = serde_json::to_string(&record).unwrap();
        assert!(!json.contains("signature"));
    }
}
