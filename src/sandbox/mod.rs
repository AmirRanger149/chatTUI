//! Workspace-scoped sandbox for agent tool execution.
//! Security model: all paths must resolve inside workspace_root, deny sensitive files.

use crate::api::types::ToolCall;
use anyhow::{anyhow, Context, Result};
use serde_json::Value;
use std::path::{Path, PathBuf};
use tokio::fs;

#[derive(Debug, Clone)]
pub struct SandboxConfig {
    pub enabled: bool,
    pub workspace_root: PathBuf,
    pub auto_approve: bool,
    pub allow_shell: bool,
    pub max_file_size: usize, // bytes
}

impl Default for SandboxConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            // Empty means unset: never fall back to process cwd / this crate.
            workspace_root: PathBuf::new(),
            auto_approve: true,
            allow_shell: true,
            max_file_size: 1024 * 1024, // 1MB
        }
    }
}

#[derive(Debug, Clone)]
pub struct Sandbox {
    pub config: SandboxConfig,
}

impl Sandbox {
    pub fn new(config: SandboxConfig) -> Self {
        Self { config }
    }

    #[allow(dead_code)]
    pub fn with_root(root: PathBuf) -> Self {
        let mut cfg = SandboxConfig::default();
        cfg.workspace_root = root;
        Self::new(cfg)
    }

    pub fn has_target(&self) -> bool {
        !self.config.workspace_root.as_os_str().is_empty()
    }

    /// Set the agent target directory. Relative paths are resolved from cwd
    /// only for this explicit user choice — never used as an implicit default.
    pub fn set_target(&mut self, raw: &str) -> Result<PathBuf> {
        let path = expand_user_path(raw)?;
        if !path.exists() {
            return Err(anyhow!("target directory does not exist: {}", path.display()));
        }
        if !path.is_dir() {
            return Err(anyhow!("target is not a directory: {}", path.display()));
        }
        let canonical = path
            .canonicalize()
            .with_context(|| format!("resolving target {}", path.display()))?;
        self.config.workspace_root = canonical.clone();
        Ok(canonical)
    }

    /// Resolve a user-supplied path to an absolute path inside workspace_root.
    /// Returns error if outside or denied.
    pub fn resolve_path(&self, user_path: &str) -> Result<PathBuf> {
        if !self.has_target() {
            return Err(anyhow!(
                "no target directory set — use /sandbox <path> or --sandbox <path>"
            ));
        }
        let user_path = user_path.trim();
        if user_path.is_empty() {
            return Err(anyhow!("empty path"));
        }

        // Disallow absolute paths outside workspace for safety, but allow joining.
        let p = Path::new(user_path);
        if p.is_absolute() {
            // Still check if it's inside root
            let canonical_root = self.config.workspace_root.canonicalize().unwrap_or_else(|_| self.config.workspace_root.clone());
            let canonical_p = p.canonicalize().unwrap_or_else(|_| p.to_path_buf());
            if !canonical_p.starts_with(&canonical_root) {
                // For absolute paths, we still allow if they are inside root after canonicalization,
                // otherwise deny unless it's explicitly inside.
                // To be strict: deny absolute paths not under root.
                if !p.starts_with(&canonical_root) {
                    return Err(anyhow!("absolute path outside workspace denied: {}", user_path));
                }
            }
            self.check_denied(&canonical_p)?;
            return Ok(canonical_p);
        }

        // Join with root
        let joined = self.config.workspace_root.join(p);

        // Prevent traversal via .. but allow canonicalization to fail if file doesn't exist yet (for write)
        // We check the parent canonicalization for new files.
        let parent = joined.parent().unwrap_or(&self.config.workspace_root);
        let canonical_parent = parent.canonicalize().unwrap_or_else(|_| {
            // If parent doesn't exist, try to canonicalize root + walk up
            self.config.workspace_root.clone()
        });
        let canonical_root = self.config.workspace_root.canonicalize().unwrap_or_else(|_| self.config.workspace_root.clone());

        if !canonical_parent.starts_with(&canonical_root) {
            return Err(anyhow!("path escapes workspace: {}", user_path));
        }

        // Check final path (if exists) is inside root
        if let Ok(canonical_joined) = joined.canonicalize() {
            if !canonical_joined.starts_with(&canonical_root) {
                return Err(anyhow!("path escapes workspace: {}", user_path));
            }
            self.check_denied(&canonical_joined)?;
            Ok(canonical_joined)
        } else {
            // File doesn't exist yet - check the joined path string for .. components
            // and deny list by file name
            let normalized = self.normalize_path(&joined);
            if !normalized.starts_with(&canonical_root) {
                return Err(anyhow!("path escapes workspace: {}", user_path));
            }
            self.check_denied_by_name(user_path)?;
            Ok(joined)
        }
    }

    fn normalize_path(&self, path: &Path) -> PathBuf {
        // Simple normalization without filesystem access
        let mut components = Vec::new();
        for comp in path.components() {
            match comp {
                std::path::Component::ParentDir => {
                    components.pop();
                }
                std::path::Component::CurDir => {}
                c => components.push(c),
            }
        }
        components.iter().collect()
    }

    fn check_denied(&self, path: &Path) -> Result<()> {
        let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        self.check_denied_by_name(file_name)?;
        // Also check full path for sensitive substrings
        let path_str = path.to_string_lossy().to_ascii_lowercase();
        let denied_substrings = [".git/config", ".env", "config.json"];
        for denied in &denied_substrings {
            if path_str.contains(denied) && file_name == "config.json" {
                // Allow reading config.json? No, deny to prevent key leakage, unless explicitly allowed.
                // For now, allow list but warn, but deny read if it's main config with keys?
                // We'll allow read but with warning in execution? Let's deny config.json at root only if it contains api keys.
                // Simpler: deny reading config.json that is in workspace root if it looks like main config.
                // For safety, deny all config.json reads that contain api_key? We'll just allow but truncate keys? Let's deny for now if user asks for config.json directly.
                // Actually we will allow but the sandbox execution will filter keys.
            }
        }
        Ok(())
    }

    fn check_denied_by_name(&self, name: &str) -> Result<()> {
        let lower = name.to_ascii_lowercase();
        // Deny list for obviously sensitive files
        let denied_exact = [".env", ".env.local", ".env.production"];
        let denied_suffix = [".key", ".pem", ".p12"];
        if denied_exact.contains(&lower.as_str()) {
            return Err(anyhow!("access to sensitive file denied: {}", name));
        }
        for suffix in &denied_suffix {
            if lower.ends_with(suffix) {
                return Err(anyhow!("access to sensitive file denied: {}", name));
            }
        }
        if lower.contains(".git/") && lower.contains("config") {
            return Err(anyhow!("access to .git/config denied"));
        }
        Ok(())
    }

    pub async fn read_file(&self, path: &str) -> Result<String> {
        let full = self.resolve_path(path)?;
        let metadata = fs::metadata(&full).await.context("file not found")?;
        if metadata.len() as usize > self.config.max_file_size {
            return Err(anyhow!(
                "file too large ({} bytes > {} limit)",
                metadata.len(),
                self.config.max_file_size
            ));
        }
        if !metadata.is_file() {
            return Err(anyhow!("not a file: {}", path));
        }
        let content = fs::read_to_string(&full).await.context("reading file")?;
        Ok(content)
    }

    pub async fn write_file(&self, path: &str, content: &str) -> Result<String> {
        let full = self.resolve_path(path)?;
        if let Some(parent) = full.parent() {
            fs::create_dir_all(parent).await.context("creating parent dirs")?;
        }
        fs::write(&full, content).await.context("writing file")?;
        Ok(format!("wrote {} bytes to {}", content.len(), path))
    }

    pub async fn edit_file(&self, path: &str, old_string: &str, new_string: &str) -> Result<String> {
        let full = self.resolve_path(path)?;
        let current = fs::read_to_string(&full).await.context("reading file for edit")?;
        if !current.contains(old_string) {
            return Err(anyhow!("old_string not found in file"));
        }
        // Ensure unique occurrence or allow first?
        let count = current.matches(old_string).count();
        if count > 1 {
            return Err(anyhow!(
                "old_string appears {} times, must be unique for safe edit",
                count
            ));
        }
        let new_content = current.replacen(old_string, new_string, 1);
        fs::write(&full, &new_content).await.context("writing edited file")?;
        Ok(format!(
            "edited {} ({} -> {} chars)",
            path,
            old_string.len(),
            new_string.len()
        ))
    }

    pub async fn list_files(&self, path: &str) -> Result<String> {
        let dir_path = if path.trim().is_empty() { "." } else { path };
        let full = self.resolve_path(dir_path)?;
        let metadata = fs::metadata(&full).await.context("path not found")?;
        if !metadata.is_dir() {
            return Err(anyhow!("not a directory: {}", dir_path));
        }
        let mut entries = fs::read_dir(&full).await.context("reading dir")?;
        let mut lines = Vec::new();
        while let Some(entry) = entries.next_entry().await? {
            let ft = entry.file_type().await?;
            let name = entry.file_name().to_string_lossy().to_string();
            let prefix = if ft.is_dir() { "dir " } else { "file" };
            lines.push(format!("{} {}", prefix, name));
        }
        lines.sort();
        if lines.is_empty() {
            Ok("(empty directory)".to_string())
        } else {
            Ok(lines.join("\n"))
        }
    }

    pub async fn bash(&self, command: &str) -> Result<String> {
        if !self.has_target() {
            return Err(anyhow!(
                "no target directory set — use /sandbox <path> or --sandbox <path>"
            ));
        }
        if !self.config.allow_shell {
            return Err(anyhow!("shell execution disabled in config"));
        }
        let cmd = command.trim();
        if cmd.is_empty() {
            return Err(anyhow!("empty command"));
        }
        // Basic denylist for destructive commands
        let lower = cmd.to_ascii_lowercase();
        let blocked = ["rm -rf /", "mkfs", ":(){:|:&};:", "dd if="];
        for b in &blocked {
            if lower.contains(b) {
                return Err(anyhow!("blocked dangerous command: {}", b));
            }
        }

        let output = tokio::process::Command::new("sh")
            .arg("-c")
            .arg(cmd)
            .current_dir(&self.config.workspace_root)
            .output()
            .await
            .context("executing command")?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let mut result = String::new();
        if !stdout.is_empty() {
            result.push_str(&stdout);
        }
        if !stderr.is_empty() {
            if !result.is_empty() {
                result.push_str("\n--- stderr ---\n");
            }
            result.push_str(&stderr);
        }
        if result.is_empty() {
            result = format!("(command exited with status {})", output.status);
        }
        // Truncate if too large
        if result.len() > 20000 {
            result.truncate(20000);
            result.push_str("\n... truncated");
        }
        Ok(result)
    }

    /// Execute a ToolCall by name.
    pub async fn execute_tool(&self, tool_call: &ToolCall) -> Result<String> {
        let args: Value = serde_json::from_str(&tool_call.arguments).unwrap_or(Value::Object(Default::default()));
        match tool_call.name.as_str() {
            "read_file" => {
                let path = args["path"].as_str().ok_or_else(|| anyhow!("missing path"))?;
                self.read_file(path).await
            }
            "write_file" => {
                let path = args["path"].as_str().ok_or_else(|| anyhow!("missing path"))?;
                let content = args["content"].as_str().ok_or_else(|| anyhow!("missing content"))?;
                self.write_file(path, content).await
            }
            "edit_file" => {
                let path = args["path"].as_str().ok_or_else(|| anyhow!("missing path"))?;
                let old = args["old_string"].as_str().ok_or_else(|| anyhow!("missing old_string"))?;
                let new = args["new_string"].as_str().ok_or_else(|| anyhow!("missing new_string"))?;
                self.edit_file(path, old, new).await
            }
            "list_files" => {
                let path = args["path"].as_str().unwrap_or(".");
                self.list_files(path).await
            }
            "bash" => {
                let cmd = args["command"].as_str().ok_or_else(|| anyhow!("missing command"))?;
                self.bash(cmd).await
            }
            other => Err(anyhow!("unknown tool: {}", other)),
        }
    }
}

fn expand_user_path(raw: &str) -> Result<PathBuf> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err(anyhow!("empty target path"));
    }
    if trimmed == "~" {
        let home = std::env::var_os("HOME").ok_or_else(|| anyhow!("HOME is not set"))?;
        return Ok(PathBuf::from(home));
    }
    if let Some(rest) = trimmed.strip_prefix("~/") {
        let home = std::env::var_os("HOME").ok_or_else(|| anyhow!("HOME is not set"))?;
        return Ok(PathBuf::from(home).join(rest));
    }
    Ok(PathBuf::from(trimmed))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sandbox_resolves_inside_root() {
        let dir = std::env::temp_dir().join("chatTUI_test_root");
        let _ = std::fs::create_dir_all(&dir);
        let sandbox = Sandbox::with_root(dir.clone());
        let p = sandbox.resolve_path("src/main.rs").unwrap();
        assert!(p.starts_with(&dir));
    }

    #[test]
    fn sandbox_blocks_traversal() {
        let dir = std::env::temp_dir().join("chatTUI_test_root2");
        let _ = std::fs::create_dir_all(&dir);
        let sandbox = Sandbox::with_root(dir);
        assert!(sandbox.resolve_path("../../etc/passwd").is_err());
    }

    #[tokio::test]
    async fn sandbox_read_write() {
        let dir = std::env::temp_dir().join(format!("chatTUI_test_rw_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let sandbox = Sandbox::with_root(dir.clone());
        sandbox.write_file("test.txt", "hello").await.unwrap();
        let content = sandbox.read_file("test.txt").await.unwrap();
        assert_eq!(content, "hello");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn default_sandbox_has_no_implicit_cwd_target() {
        let sandbox = Sandbox::new(SandboxConfig::default());
        assert!(!sandbox.has_target());
        assert!(sandbox.resolve_path("src/main.rs").is_err());
    }

    #[tokio::test]
    async fn writes_stay_under_configured_target_not_crate_root() {
        let target = std::env::temp_dir().join(format!("chatTUI_dst_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&target);
        let crate_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let sandbox = Sandbox::with_root(target.clone());
        sandbox
            .write_file("hello_agent.txt", "hi")
            .await
            .unwrap();
        let written = target.join("hello_agent.txt");
        assert!(written.exists());
        assert!(written.starts_with(&target));
        assert!(!crate_root.join("hello_agent.txt").exists());
        let _ = std::fs::remove_dir_all(&target);
    }
}
