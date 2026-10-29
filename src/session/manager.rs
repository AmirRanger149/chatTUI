use anyhow::{Context, Result};
use directories::ProjectDirs;
use serde::{Deserialize, Serialize};
use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

static SAVE_SEQUENCE: AtomicU64 = AtomicU64::new(0);
const MAX_HISTORY_BYTES: usize = 64 * 1024 * 1024;

fn is_false(value: &bool) -> bool {
    !*value
}

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
    /// The original failure status of a tool result, including after reload.
    #[serde(default, skip_serializing_if = "is_false")]
    pub is_error: bool,
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
            use std::io::Read;
            let mut bytes = Vec::new();
            fs::File::open(&manager.path)?
                .take(MAX_HISTORY_BYTES as u64 + 1)
                .read_to_end(&mut bytes)?;
            if bytes.len() > MAX_HISTORY_BYTES {
                anyhow::bail!("conversation history exceeds 64 MiB — preserve or archive sessions.json before continuing");
            }
            match serde_json::from_slice(&bytes) {
                Ok(store) => manager.store = store,
                Err(error) => {
                    // Hard-linking a unique name never replaces an earlier
                    // quarantine. If this or the unlink fails, keep the
                    // original file and refuse startup instead of saving a
                    // fresh store over the only recoverable history.
                    let quarantine = unique_quarantine_path(&manager.path);
                    fs::hard_link(&manager.path, &quarantine).with_context(|| {
                        format!("history is unreadable ({error}); preserving {}", manager.path.display())
                    })?;
                    fs::remove_file(&manager.path).with_context(|| {
                        format!("history is unreadable ({error}); preserving {}", manager.path.display())
                    })?;
                    eprintln!(
                        "chatTUI: sessions.json was unreadable ({error}); preserved at {} and starting fresh",
                        quarantine.display()
                    );
                }
            }
        }
        if manager.store.sessions.is_empty() {
            manager.new_session()?;
        } else {
            manager.store.current = manager.store.current.min(manager.store.sessions.len() - 1);
        }
        Ok(manager)
    }

    pub fn save(&self) -> Result<()> {
        let parent = self.path.parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        fs::create_dir_all(parent).context("creating history directory")?;
        let data = serde_json::to_vec_pretty(&self.store)?;
        if data.len() > MAX_HISTORY_BYTES {
            anyhow::bail!("conversation history exceeds the 64 MiB save limit");
        }
        let (tmp, mut file) = create_unique_temp(&self.path)?;
        // Remove an incomplete temp file on every error path; the original
        // store remains untouched until the completed snapshot is renamed.
        let mut guard = TempFile(tmp.clone());
        if let Ok(metadata) = fs::metadata(&self.path) {
            file.set_permissions(metadata.permissions())?;
        }
        file.write_all(&data).context("writing history snapshot")?;
        file.sync_all().context("syncing history snapshot")?;
        drop(file);
        fs::rename(&tmp, &self.path).context("replacing history snapshot")?;
        guard.0.clear();
        #[cfg(unix)]
        fs::File::open(parent)?.sync_all().context("syncing history directory")?;
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

    pub fn select(&mut self, index: usize) -> Result<()> {
        if index < self.store.sessions.len() {
            self.store.current = index;
            self.save()?;
        }
        Ok(())
    }

    pub fn delete_at(&mut self, index: usize) -> Result<()> {
        if self.store.sessions.len() <= 1 || index >= self.store.sessions.len() {
            return Ok(());
        }
        self.store.sessions.remove(index);
        if self.store.current == index {
            self.store.current = self.store.current.min(self.store.sessions.len() - 1);
        } else if self.store.current > index {
            self.store.current -= 1;
        }
        self.save()
    }

    pub fn new_session(&mut self) -> Result<()> {
        // Sessions are newest-first; looking at last() reused id 2 forever.
        let id = self.store.sessions.iter().map(|s| s.id).max().unwrap_or(0)
            .checked_add(1).context("session id overflow")?;
        self.store.sessions.insert(
            0,
            Session {
                id,
                title: "New conversation".into(),
                messages: Vec::new(),
            },
        );
        self.store.current = 0;
        self.save()
    }

    pub fn add_message(&mut self, role: impl Into<String>, content: impl Into<String>) -> Result<()> {
        let message = Message {
            role: role.into(),
            content: content.into(),
            tool_call_id: None,
            tool_calls: None,
            is_error: false,
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
        self.save()
    }

    pub fn add_tool_result(&mut self, tool_call_id: impl Into<String>, content: impl Into<String>, is_error: bool) -> Result<()> {
        let message = Message {
            role: "tool".into(),
            content: content.into(),
            tool_call_id: Some(tool_call_id.into()),
            tool_calls: None,
            is_error,
        };
        self.current_mut().messages.push(message);
        self.save()
    }

    pub fn add_assistant_with_tools(
        &mut self,
        content: impl Into<String>,
        tool_calls: Vec<ToolCallRecord>,
    ) -> Result<()> {
        let message = Message {
            role: "assistant".into(),
            content: content.into(),
            tool_call_id: None,
            tool_calls: Some(tool_calls),
            is_error: false,
        };
        self.current_mut().messages.push(message);
        self.save()
    }
}

/// An incomplete snapshot must not be mistaken for the real history on restart.
struct TempFile(PathBuf);

impl Drop for TempFile {
    fn drop(&mut self) {
        if !self.0.as_os_str().is_empty() {
            let _ = fs::remove_file(&self.0);
        }
    }
}

fn unique_quarantine_path(path: &Path) -> PathBuf {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|time| time.as_nanos())
        .unwrap_or(0);
    path.with_file_name(format!(
        "sessions.json.corrupt-{stamp}-{}-{}",
        std::process::id(),
        SAVE_SEQUENCE.fetch_add(1, Ordering::Relaxed)
    ))
}

fn create_unique_temp(path: &Path) -> Result<(PathBuf, fs::File)> {
    for _ in 0..16 {
        let tmp = path.with_file_name(format!(
            "sessions.json.{}.{}.tmp",
            std::process::id(),
            SAVE_SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        match options.open(&tmp) {
            Ok(file) => return Ok((tmp, file)),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error).context("creating history snapshot"),
        }
    }
    anyhow::bail!("could not create a unique history snapshot")
}

#[cfg(test)]
impl SessionManager {
    pub fn for_tests() -> Self {
        let mut manager = Self {
            store: Store::default(),
            path: std::env::temp_dir().join(format!(
                "chat-tui-test-sessions-{}-{}.json",
                std::process::id(),
                SAVE_SEQUENCE.fetch_add(1, Ordering::Relaxed)
            )),
        };
        manager.new_session().unwrap();
        manager
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_sessions_have_unique_ids_and_tool_failures_survive_reload() {
        let mut manager = SessionManager::for_tests();
        manager.new_session().unwrap();
        manager.new_session().unwrap();
        let ids: Vec<u64> = manager.sessions().iter().map(|session| session.id).collect();
        assert_eq!(ids, vec![3, 2, 1]);
        manager.add_tool_result("c1", "command failed", true).unwrap();
        let bytes = std::fs::read(&manager.path).unwrap();
        let store: Store = serde_json::from_slice(&bytes).unwrap();
        assert!(store.sessions[0].messages[0].is_error);
        let old: Message = serde_json::from_str(r#"{"role":"tool","content":"ok"}"#).unwrap();
        assert!(!old.is_error);
    }

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
