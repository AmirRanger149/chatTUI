//! Workspace-scoped tool execution for agent mode.
//!
//! ## Security model — read this before trusting any of it
//!
//! Two independent layers protect shell commands, and both are documented
//! here with their real limits:
//!
//! 1. **Application-level restrictions** (always active):
//!    - The file tools (`read_file`, `write_file`, `edit_file`, `list_files`)
//!      resolve every path against the workspace root and refuse paths that
//!      escape it (directly, via `..`, or via symlinks) as well as sensitive
//!      files. These checks live in this process: a path validated here can
//!      still be swapped for a symlink between validation and I/O (TOCTOU),
//!      which only OS primitives such as `openat2` could close portably.
//!    - The permission layer ([`permissions::PermissionMode`]) gates which
//!      tool *classes* may run at all.
//!    - `bash` refuses commands that *name* sensitive files and keeps a tiny
//!      advisory denylist — both are string scans and both are trivially
//!      bypassable; they are foot-gun guards, never security controls.
//!
//! 2. **Kernel-level isolation for shell commands** (Linux, enabled by
//!    default via `os_isolation: auto`; see [`os_isolation`]): before the
//!    shell is spawned on a dedicated worker thread, the thread receives
//!    Landlock filesystem rules (reads everywhere, writes only under the
//!    workspace plus a small set of scratch roots) and a seccomp-bpf filter
//!    (network sockets, ptrace/process-injection, kernel-module loading,
//!    namespace/mount tricks, io_uring, the all-process `kill(-1)`, and
//!    more are denied). The two facilities are applied **independently**
//!    (one being unavailable on a kernel does not silently remove the
//!    other), the shell inherits whatever applied, and it cannot remove
//!    it. When a facility is unavailable, `auto` mode says exactly which
//!    one in the tool output and `require` mode refuses to run — the
//!    fallback is never silent.
//!
//!    Limits that remain even with isolation active, by design: reads
//!    outside the workspace are still possible (toolchains must read system
//!    files, so secret-*read* protection stays an application policy);
//!    `connect()` is denied even for Unix sockets; `/tmp`, `$TMPDIR`,
//!    `/dev/shm`, `$CARGO_HOME` and `$CARGO_TARGET_DIR` are writable;
//!    *targeted* signals (`kill <pid>`, `pkill`, `pidfd_send_signal`) can
//!    still reach same-uid processes including this session — an inherent
//!    limit of unprivileged same-uid sandboxing (only the all-process
//!    `kill(-1)` is denied, see below); and like every software sandbox
//!    this trusts the kernel.
//!
//! On platforms or configurations where kernel isolation is off, `bash` is
//! *not* confined at all: a command runs with the full privileges of the
//! chatTUI process and can read or write any file the user can, reach the
//! network, and spawn arbitrary child processes. Describe the system as
//! **workspace-restricted tool execution** — never as a sandbox — unless
//! kernel isolation is verified active.

pub mod os_isolation;
pub mod permissions;

use crate::api::types::ToolCall;
use crate::sandbox::os_isolation::OsIsolation;
use crate::sandbox::permissions::{authorize, PermissionMode, ToolKind};
use anyhow::{anyhow, Context, Result};
use serde_json::Value;
use std::ffi::OsStr;
use std::path::{Component, Path, PathBuf};
#[cfg(unix)]
use std::process::Stdio;
use std::sync::{atomic::{AtomicBool, AtomicU64, Ordering}, Arc};
use std::time::Duration;
use tokio::fs;
use tokio::io::AsyncReadExt;

pub const DEFAULT_SHELL_TIMEOUT_SECS: u64 = 30;
/// Lower bound for the enforced shell timeout so a tool call can never be
/// configured (or tricked) into blocking forever.
pub const MIN_SHELL_TIMEOUT_SECS: u64 = 1;
/// Tool output larger than this is truncated before it reaches the model.
const MAX_TOOL_OUTPUT_BYTES: usize = 20_000;
const MAX_DIRECTORY_ENTRIES: usize = 1_000;
static FILE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Default line window for ranged reads. Reading files window by window
/// keeps a 10,000-line file from flooding the conversation context — the
/// model pages through it instead of swallowing it whole.
pub const DEFAULT_READ_LIMIT: usize = 250;
/// Hard cap on a single read window.
pub const MAX_READ_LIMIT: usize = 1_000;
/// Display cap for one source line; minified files must not blow the
/// output budget on a single line.
const MAX_LINE_DISPLAY_CHARS: usize = 1_000;

#[derive(Debug, Clone)]
pub struct SandboxConfig {
    pub enabled: bool,
    pub workspace_root: PathBuf,
    pub auto_approve: bool,
    pub allow_shell: bool,
    pub max_file_size: usize, // bytes
    /// Hard timeout for the `bash` tool; enforced by killing the command's
    /// process group. Clamped to at least [`MIN_SHELL_TIMEOUT_SECS`].
    pub shell_timeout: Duration,
    /// Application-level permission mode (see [`permissions`]).
    pub permission_mode: PermissionMode,
    /// When to apply kernel-level isolation (Landlock filesystem
    /// confinement + seccomp syscall filter) to shell commands. See
    /// [`os_isolation`] for the exact guarantees and limits.
    pub os_isolation: OsIsolation,
    /// Extra sensitive file names/suffixes (from config.json) appended to
    /// the built-in sensitive-file policy. An entry starting with `.` is a
    /// suffix match, anything else an exact (case-insensitive) name match.
    pub extra_sensitive_names: Vec<String>,
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
            shell_timeout: Duration::from_secs(DEFAULT_SHELL_TIMEOUT_SECS),
            permission_mode: PermissionMode::FullAuto,
            os_isolation: OsIsolation::Auto,
            extra_sensitive_names: Vec::new(),
        }
    }
}

impl SandboxConfig {
    /// The timeout actually enforced by the `bash` tool — never zero.
    pub fn effective_shell_timeout(&self) -> Duration {
        if self.shell_timeout.is_zero() {
            Duration::from_secs(MIN_SHELL_TIMEOUT_SECS)
        } else {
            self.shell_timeout
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

    pub fn has_target(&self) -> bool {
        !self.config.workspace_root.as_os_str().is_empty()
    }

    /// Set the agent target directory. Relative paths are resolved from cwd
    /// only for this explicit user choice — never used as an implicit default.
    pub fn set_target(&mut self, raw: &str) -> Result<PathBuf> {
        let path = expand_user_path(raw)?;
        if !path.exists() {
            return Err(anyhow!(
                "target directory does not exist: {}",
                path.display()
            ));
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
    ///
    /// Guarantees about the returned path:
    /// - it is lexically normalized (no surviving `.` / `..` components), so
    ///   the kernel cannot later resolve a leftover `..` outside the root;
    /// - its deepest existing ancestor is symlink-resolved and still inside
    ///   the workspace, so a symlink cannot redirect I/O out of the root;
    /// - components that do not exist yet cannot be symlinks, so appending
    ///   them to the canonical ancestor is safe at resolution time.
    ///
    /// Remaining gap (documented, not solvable portably here): a symlink can
    /// appear *between* this check and the actual I/O (TOCTOU).
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
        if user_path.contains('\0') {
            return Err(anyhow!("path contains NUL byte"));
        }

        // `set_target` (and the test-only `with_root`) store an
        // already-canonical root, so this is the true workspace prefix,
        // not a stale look-alike.
        let root = self.config.workspace_root.clone();
        let requested = Path::new(user_path);
        let joined = if requested.is_absolute() {
            requested.to_path_buf()
        } else {
            root.join(requested)
        };

        // Normalize `..` and `.` lexically FIRST. Checking the raw path and
        // canonicalizing only when it exists is not enough: an absolute path
        // with `..` through a not-yet-existing directory would otherwise be
        // accepted unresolved and escape once the kernel resolves it.
        let normalized = normalize_lexical(&joined);
        if !normalized.starts_with(&root) {
            return Err(anyhow!("path escapes workspace: {}", user_path));
        }

        // Resolve symlinks through the deepest existing ancestor; a link
        // pointing outside the workspace is caught here.
        let resolved = canonicalize_deepest(&normalized);
        if !resolved.starts_with(&root) {
            return Err(anyhow!("path escapes workspace via symlink: {}", user_path));
        }
        Ok(resolved)
    }

    /// Enforce the sensitive-file policy for `path` (which must already be
    /// workspace-resolved). Every workspace-relative component is checked,
    /// so `secrets/.env` and `backup/.git/hooks/x` are caught, not just the
    /// final file name.
    pub fn check_access(&self, path: &Path, access: Access) -> Result<()> {
        let relative = path
            .strip_prefix(&self.config.workspace_root)
            .unwrap_or(path);
        let mut inside_git = false;
        for component in relative.components() {
            let name = component.as_os_str().to_string_lossy();
            if name == ".git" {
                inside_git = true;
            }
            if is_sensitive_name(&name, &self.config.extra_sensitive_names) {
                return Err(anyhow!("access to sensitive file denied: {}", name));
            }
            if inside_git {
                match access {
                    Access::Write => {
                        return Err(anyhow!(
                            "writes inside .git/ are denied: {}",
                            path.display()
                        ));
                    }
                    Access::Read if name == "config" || name == "credentials" => {
                        return Err(anyhow!(
                            "access to sensitive file denied: {name} (inside .git)"
                        ));
                    }
                    _ => {}
                }
            }
        }
        Ok(())
    }

    /// Read a window of a file: up to `limit` lines starting at the
    /// 1-based `offset`. The output is numbered lines with a header that
    /// says which range of the file was served, so large files are read in
    /// chunks instead of flooding the conversation context.
    pub async fn read_file(&self, path: &str, offset: usize, limit: usize) -> Result<String> {
        self.authorize(ToolKind::Read)?;
        let full = self.resolve_path(path)?;
        self.check_access(&full, Access::Read)?;
        let metadata = fs::metadata(&full).await.context("file not found")?;
        if !metadata.is_file() {
            return Err(anyhow!("not a file: {}", path));
        }
        if metadata.len() > self.config.max_file_size as u64 {
            return Err(anyhow!(
                "file too large ({} bytes > {} limit)",
                metadata.len(),
                self.config.max_file_size
            ));
        }
        let content = read_limited_file(&full, self.config.max_file_size)
            .await.context("reading file")?;
        let lines: Vec<&str> = content.lines().collect();
        let total = lines.len();

        if total == 0 {
            return Ok(format!("{path}: empty file"));
        }
        let offset = offset.max(1);
        if offset > total {
            return Ok(format!(
                "{path}: the file has {total} line(s); offset {offset} is past the end"
            ));
        }
        let limit = limit.clamp(1, MAX_READ_LIMIT);
        let start = offset - 1;
        let end = (start + limit).min(total);

        let mut out = format!("{path}: lines {}-{} of {total}\n", offset, end);
        for (index, line) in lines[start..end].iter().enumerate() {
            let display: String = if line.chars().count() > MAX_LINE_DISPLAY_CHARS {
                line.chars()
                    .take(MAX_LINE_DISPLAY_CHARS)
                    .collect::<String>()
                    + " …"
            } else {
                (*line).to_string()
            };
            let row = format!("{:>6}│{}\n", start + index + 1, display);
            if out.len().saturating_add(row.len()) > MAX_TOOL_OUTPUT_BYTES {
                out.push_str("… output limit reached — read a smaller line window\n");
                break;
            }
            out.push_str(&row);
        }
        if end < total {
            out.push_str(&format!(
                "… {} more line(s) — call read_file again with offset {}\n",
                total - end,
                end + 1
            ));
        }
        Ok(out)
    }

    pub async fn write_file(&self, path: &str, content: &str) -> Result<String> {
        self.authorize(ToolKind::Write)?;
        let full = self.resolve_path(path)?;
        self.check_access(&full, Access::Write)?;
        self.ensure_content_size(content)?;
        if let Some(parent) = full.parent() {
            fs::create_dir_all(parent).await.context("creating parent dirs")?;
            self.check_parent(parent)?;
        }
        let old_lines = match fs::metadata(&full).await {
            Ok(metadata) if metadata.is_file() => read_limited_file(&full, self.config.max_file_size)
                .await.context("reading old file")?.lines().count(),
            Ok(_) => return Err(anyhow!("not a file: {path}")),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => 0,
            Err(error) => return Err(error).context("checking old file"),
        };
        let new_lines = content.lines().count();
        self.atomic_write_file(full, content.to_string()).await?;
        if old_lines > 0 {
            Ok(format!("wrote {path} (+{new_lines} -{old_lines})"))
        } else {
            Ok(format!("wrote {path} (+{new_lines})"))
        }
    }

    pub async fn edit_file(
        &self,
        path: &str,
        old_string: &str,
        new_string: &str,
    ) -> Result<String> {
        self.authorize(ToolKind::Edit)?;
        let full = self.resolve_path(path)?;
        self.check_access(&full, Access::Write)?;
        let current = read_limited_file(&full, self.config.max_file_size)
            .await.context("reading file for edit")?;
        if old_string.is_empty() || !current.contains(old_string) {
            return Err(anyhow!("old_string not found in file"));
        }
        let count = current.matches(old_string).count();
        if count > 1 {
            return Err(anyhow!(
                "old_string appears {} times, must be unique for safe edit",
                count
            ));
        }
        let new_content = current.replacen(old_string, new_string, 1);
        self.ensure_content_size(&new_content)?;
        let removed = old_string.lines().count();
        let added = new_string.lines().count();
        self.atomic_write_file(full, new_content).await?;
        Ok(format!("edited {path} (+{added} -{removed})"))
    }

    fn ensure_content_size(&self, content: &str) -> Result<()> {
        if content.len() > self.config.max_file_size {
            return Err(anyhow!("file content too large ({} bytes > {} limit)",
                content.len(), self.config.max_file_size));
        }
        Ok(())
    }

    fn check_parent(&self, parent: &Path) -> Result<()> {
        let canonical = parent.canonicalize().context("resolving parent directory")?;
        if !canonical.starts_with(&self.config.workspace_root) {
            return Err(anyhow!("path escapes workspace: {}", parent.display()));
        }
        Ok(())
    }

    async fn atomic_write_file(&self, path: PathBuf, content: String) -> Result<()> {
        let workspace = self.config.workspace_root.clone();
        // A complete write/flush/rename operation on one worker cannot be
        // cancelled halfway through by dropping a Tokio filesystem future.
        // The UI waits for its result on interruption before changing chats.
        tokio::task::spawn_blocking(move || atomic_write_file(&path, content.as_bytes(), &workspace))
            .await.context("file writer failed")?
    }

    pub async fn list_files(&self, path: &str) -> Result<String> {
        self.authorize(ToolKind::Read)?;
        let dir_path = if path.trim().is_empty() { "." } else { path };
        let full = self.resolve_path(dir_path)?;
        self.check_access(&full, Access::Read)?;
        let metadata = fs::metadata(&full).await.context("path not found")?;
        if !metadata.is_dir() {
            return Err(anyhow!("not a directory: {}", dir_path));
        }
        let mut entries = fs::read_dir(&full).await.context("reading dir")?;
        let mut lines = Vec::new();
        let mut limited = false;
        while let Some(entry) = entries.next_entry().await? {
            if lines.len() >= MAX_DIRECTORY_ENTRIES {
                limited = true;
                break;
            }
            let name = entry.file_name().to_string_lossy().to_string();
            // Keep the sensitive-file inventory out of directory listings.
            if is_sensitive_name(&name, &self.config.extra_sensitive_names) {
                continue;
            }
            let is_dir = entry.file_type().await.map(|t| t.is_dir()).unwrap_or(false);
            let prefix = if is_dir { "dir " } else { "file" };
            if lines.iter().map(String::len).sum::<usize>() + name.len() + 16 > MAX_TOOL_OUTPUT_BYTES {
                limited = true;
                break;
            }
            lines.push(format!("{} {}", prefix, name));
        }
        lines.sort();
        if limited {
            lines.push("… directory listing limited; specify a narrower path".into());
        }
        if lines.is_empty() {
            Ok("(empty directory)".to_string())
        } else {
            Ok(lines.join("\n"))
        }
    }

    /// Run a shell command. **Read the module docs first.** What this method
    /// actually guarantees:
    /// - the workspace root is the command's current directory (cwd only);
    /// - the command runs in its own process group and is killed — group
    ///   included — when the configured timeout expires;
    /// - the environment is reduced to [`ENV_ALLOWLIST`], so API keys and
    ///   other credentials are not inherited;
    /// - commands that name sensitive files are refused (best effort);
    /// - **kernel-level isolation when available and enabled** (Linux,
    ///   `os_isolation` != off): the command cannot modify files outside the
    ///   workspace/scratch roots, open network connections, ptrace other
    ///   processes, load kernel modules, create namespaces or mounts, or
    ///   signal every process on the machine (`kill(-1)`) — enforced by the
    ///   kernel, not by command inspection. The restrictions are installed
    ///   in the forked child just before exec (see
    ///   [`os_isolation::confine_in_child`]), so the shell never runs
    ///   unconfined, and each facility is enforced independently of the
    ///   other. When the kernel lacks a facility, `auto` mode runs the
    ///   command with exactly the warning that names what is missing in the
    ///   output and `require` mode refuses to run; neither pretends the
    ///   command was confined, and neither can panic the worker.
    #[cfg(test)]
    pub async fn bash(&self, command: &str) -> Result<String> {
        self.bash_cancellable(command, Arc::new(AtomicBool::new(false))).await
    }

    async fn bash_cancellable(&self, command: &str, cancel: Arc<AtomicBool>) -> Result<String> {
        let cmd = self.prepare_shell(command)?;
        if cancel.load(Ordering::SeqCst) {
            return Err(anyhow!("command interrupted before it started"));
        }
        #[cfg(unix)]
        {
            let workspace = self.config.workspace_root.clone();
            let roots = os_isolation::writable_roots(&workspace);
            let isolation = self.config.os_isolation;
            let cmd = cmd.to_owned();
            let timeout = self.config.effective_shell_timeout();
            // The worker alone owns spawn, process-group signals and wait.
            // Dropping this future sets the flag even when the Tokio task is
            // aborted; it cannot miss a PID publication between threads.
            let _cancel_on_drop = CancelOnDrop(cancel.clone());
            let outcome = tokio::task::spawn_blocking(move || {
                run_shell_process(&cmd, &workspace, isolation, &roots, &cancel, timeout)
            })
            .await
            .context("shell worker failed")?
            .map_err(|message| anyhow!("{message}"))?;
            let text = format_shell_parts(
                &outcome.stdout,
                &outcome.stderr,
                &outcome.status.to_string(),
                !outcome.status.success(),
            );
            if !outcome.status.success() {
                return Err(anyhow!("command failed ({}):\n{text}", outcome.status));
            }
            Ok(text)
        }
        #[cfg(not(unix))]
        {
            let _ = cmd;
            // Windows has no process-group supervision or kernel isolation
            // in this backend. Never run a model-generated shell command
            // with inherited credentials and an unenforced `require` flag.
            Err(anyhow!(
                "shell execution is unavailable on this platform until bounded, isolated process-tree supervision is implemented"
            ))
        }
    }

    /// Shared pre-flight checks for shell execution; returns the trimmed
    /// command on success.
    fn prepare_shell<'a>(&self, command: &'a str) -> Result<&'a str> {
        if !self.has_target() {
            return Err(anyhow!(
                "no target directory set — use /sandbox <path> or --sandbox <path>"
            ));
        }
        if !self.config.allow_shell {
            return Err(anyhow!(
                "shell execution disabled in config (sandbox.allow_shell = false)"
            ));
        }
        self.authorize(ToolKind::Shell)?;
        let cmd = command.trim();
        if cmd.is_empty() {
            return Err(anyhow!("empty command"));
        }
        // Advisory only — a foot-gun guard, NOT a security boundary.
        let lower = cmd.to_ascii_lowercase();
        for blocked in ADVISORY_BLOCKED {
            if lower.contains(blocked) {
                return Err(anyhow!("blocked dangerous command: {}", blocked));
            }
        }
        // Defense in depth only — see `first_sensitive_reference`.
        if let Some(reference) = self.first_sensitive_reference(cmd) {
            return Err(anyhow!(
                "command references sensitive file '{}' — denied by the sensitive-file policy (best-effort scan)",
                reference
            ));
        }
        Ok(cmd)
    }

    /// Best-effort scan: refuse commands that *name* a sensitive file.
    /// Quoting (`cat .e''nv`), command substitution and many other shell
    /// features defeat this trivially — it exists only so the common
    /// `cat .env` cannot quietly succeed where `read_file(".env")` fails.
    fn first_sensitive_reference(&self, command: &str) -> Option<String> {
        const DELIMITERS: &[char] = &[
            ' ', '\t', '\n', '\r', ';', '|', '&', '(', ')', '<', '>', '"', '\'', '`',
        ];
        for token in command.split(|c| DELIMITERS.contains(&c)) {
            if token.is_empty() {
                continue;
            }
            let lower = token.to_ascii_lowercase();
            if lower.contains(".git/config") || lower.contains(".git/credentials") {
                return Some(token.to_string());
            }
            for segment in token.split('/') {
                if is_sensitive_name(segment, &self.config.extra_sensitive_names) {
                    return Some(segment.to_string());
                }
            }
        }
        None
    }

    /// Refuse `kind` unless the configured permission mode allows it.
    pub fn authorize(&self, kind: ToolKind) -> Result<()> {
        authorize(self.config.permission_mode, kind)
    }

    /// Execute a ToolCall by name, after the permission layer approves the
    /// tool's capability class.
    #[cfg(test)]
    pub async fn execute_tool(&self, tool_call: &ToolCall) -> Result<String> {
        self.execute_tool_cancellable(tool_call, false, Arc::new(AtomicBool::new(false)))
            .await
    }

    pub async fn execute_tool_cancellable(
        &self,
        tool_call: &ToolCall,
        approved: bool,
        cancel: Arc<AtomicBool>,
    ) -> Result<String> {
        let mut sandbox = self.clone();
        if approved {
            sandbox.config.permission_mode = sandbox.config.permission_mode.approved();
        }
        sandbox.execute_tool_inner(tool_call, cancel).await
    }

    async fn execute_tool_inner(&self, tool_call: &ToolCall, cancel: Arc<AtomicBool>) -> Result<String> {
        if cancel.load(Ordering::SeqCst) {
            return Err(anyhow!("tool interrupted before it started"));
        }
        let kind = ToolKind::from_tool_name(&tool_call.name)
            .ok_or_else(|| anyhow!("unknown tool: {}", tool_call.name))?;
        self.authorize(kind)?;
        // Never silently fall back to empty arguments: unparseable JSON
        // almost always means the call was truncated mid-stream, and the
        // model deserves an error that says so (and how to recover).
        let args: Value = match serde_json::from_str(&tool_call.arguments) {
            Ok(value) => value,
            Err(_) => {
                return Err(anyhow!(
                    "tool call arguments are not valid JSON — they most likely arrived truncated when the stream was cut mid-call; retry with smaller edits"
                ));
            }
        };
        match tool_call.name.as_str() {
            "read_file" => {
                let path = args["path"]
                    .as_str()
                    .ok_or_else(|| anyhow!("missing path"))?;
                let offset = args["offset"].as_u64().map(|v| v as usize).unwrap_or(1);
                let limit = args["limit"]
                    .as_u64()
                    .map(|v| v as usize)
                    .unwrap_or(DEFAULT_READ_LIMIT);
                self.read_file(path, offset, limit).await
            }
            "write_file" => {
                let path = args["path"]
                    .as_str()
                    .ok_or_else(|| anyhow!("missing path"))?;
                let content = args["content"]
                    .as_str()
                    .ok_or_else(|| anyhow!("missing content"))?;
                self.write_file(path, content).await
            }
            "edit_file" => {
                let path = args["path"]
                    .as_str()
                    .ok_or_else(|| anyhow!("missing path"))?;
                let old = args["old_string"]
                    .as_str()
                    .ok_or_else(|| anyhow!("missing old_string"))?;
                let new = args["new_string"]
                    .as_str()
                    .ok_or_else(|| anyhow!("missing new_string"))?;
                self.edit_file(path, old, new).await
            }
            "list_files" => {
                let path = args["path"].as_str().unwrap_or(".");
                self.list_files(path).await
            }
            "bash" => {
                let cmd = args["command"]
                    .as_str()
                    .ok_or_else(|| anyhow!("missing command"))?;
                self.bash_cancellable(cmd, cancel).await
            }
            other => Err(anyhow!("unknown tool: {}", other)),
        }
    }
}

#[cfg(test)]
impl Sandbox {
    /// Test helper: a sandbox rooted at `root` with every other setting at
    /// its default. Never used by the app itself — the real target comes
    /// from `set_target` (config.json or `--sandbox`), which refuses to
    /// default to anything implicit.
    pub fn with_root(root: PathBuf) -> Self {
        Self::new(SandboxConfig {
            // Canonicalize so the stored root is the true workspace prefix
            // (symlinks resolved); fall back to the raw path if that fails.
            workspace_root: root.canonicalize().unwrap_or(root),
            ..SandboxConfig::default()
        })
    }
}

/// Which direction an access check is for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Access {
    Read,
    Write,
}

/// Commands that are refused outright. **Advisory only** — trivial to
/// bypass and explicitly not a security control; kept as a foot-gun guard
/// against typo'd destructive commands (see module docs).
const ADVISORY_BLOCKED: &[&str] = &["rm -rf /", "mkfs", ":(){:|:&};:", "dd if="];

/// Environment variables passed through to shell commands. Everything else
/// is stripped so credentials (API keys, tokens, cloud credentials) held by
/// the chatTUI process are not exposed to model-generated commands. This is
/// hygiene, not isolation: a command can still read files containing
/// secrets if it can reach them (see module docs).
const ENV_ALLOWLIST: &[&str] = &[
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "SHELL",
    "TERM",
    "TZ",
    "TMPDIR",
    "TMP",
    "LANG",
    "CARGO_HOME",
    "RUSTUP_HOME",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "PYTHONUNBUFFERED",
    "VIRTUAL_ENV",
    "GOROOT",
    "GOPATH",
    "JAVA_HOME",
    "MAKEFLAGS",
    "CARGO_TARGET_DIR",
];

fn env_var_allowed(name: &OsStr) -> bool {
    let name = name.to_string_lossy();
    ENV_ALLOWLIST.iter().any(|allowed| *allowed == name) || name.starts_with("LC_")
}

/// Built-in sensitive-file policy. Deliberately small and principled —
/// extend it through `SandboxConfig::extra_sensitive_names` instead of
/// growing this list ad hoc:
/// - dotenv-style files: `.env`, `.env.*` and `*.env`
/// - key material: `*.pem`, `*.key`, `*.p12`, `*.pfx`, `*.jks` and the
///   common SSH private-key names (`id_rsa`, `id_ed25519`, …)
/// - git internals: handled in [`Sandbox::check_access`] (all writes under
///   `.git/` are denied — hooks are code execution — plus reads of
///   `.git/config` / `.git/credentials`)
///
/// Accepted trade-off: `.env.example` is caught by the `.env.*` rule and
/// denied too, rather than maintaining an exception list in this phase.
fn is_sensitive_name(name: &str, extra: &[String]) -> bool {
    let lower = name.to_ascii_lowercase();
    if lower == ".env" || lower.starts_with(".env.") || lower.ends_with(".env") {
        return true;
    }
    const KEY_SUFFIXES: &[&str] = &[".pem", ".key", ".p12", ".pfx", ".jks"];
    if KEY_SUFFIXES.iter().any(|suffix| lower.ends_with(suffix)) {
        return true;
    }
    const KEY_NAMES: &[&str] = &["id_rsa", "id_dsa", "id_ecdsa", "id_ed25519"];
    if KEY_NAMES.contains(&lower.as_str()) {
        return true;
    }
    for entry in extra {
        let entry = entry.trim().to_ascii_lowercase();
        if entry.is_empty() {
            continue;
        }
        if entry.starts_with('.') {
            if lower.ends_with(&entry) {
                return true;
            }
        } else if lower == entry {
            return true;
        }
    }
    false
}

/// Lexically resolve `.` and `..` without touching the filesystem. The walk
/// never escapes the root component: `..` at the root is a no-op.
fn normalize_lexical(path: &Path) -> PathBuf {
    let mut result: Vec<Component<'_>> = Vec::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => match result.last() {
                Some(Component::RootDir) | None => {}
                _ => {
                    result.pop();
                }
            },
            other => result.push(other),
        }
    }
    result.into_iter().collect()
}

/// Best-effort canonicalization that also works when the tail of `path`
/// does not exist yet: canonicalize the deepest existing ancestor, then
/// append the remaining components (which, not existing, cannot be
/// symlinks — at resolution time).
fn canonicalize_deepest(path: &Path) -> PathBuf {
    if let Ok(canonical) = path.canonicalize() {
        return canonical;
    }
    let mut suffix: Vec<std::ffi::OsString> = Vec::new();
    let mut current: &Path = path;
    while let Some(parent) = current.parent() {
        if let Some(name) = current.file_name() {
            suffix.push(name.to_os_string());
        }
        if let Ok(canonical) = parent.canonicalize() {
            let mut resolved = canonical;
            for part in suffix.iter().rev() {
                resolved.push(part);
            }
            return resolved;
        }
        current = parent;
    }
    path.to_path_buf()
}

/// Cancelling an async task must also stop its non-abortable shell worker.
#[cfg(unix)]
struct CancelOnDrop(Arc<AtomicBool>);

#[cfg(unix)]
impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

/// Raw result of one shell run, before rendering.
#[cfg(unix)]
struct ShellOutcome {
    stdout: Vec<u8>,
    stderr: Vec<u8>,
    status: std::process::ExitStatus,
}

/// Spawn `sh -c <cmd>` on the current (unrestricted) worker thread, in its
/// own process group, with the filtered environment, and collect the
/// output. Kernel isolation is installed **inside the forked child, just
/// before execve**, via a `pre_exec` hook — never on this thread — so the
/// process-spawn machinery runs normally and the shell still never executes
/// a single instruction unconfined. `require`-mode failures surface as
/// clean errors here (std plumbs `pre_exec` errors through its spawn-error
/// channel); auto-mode failures are announced by the child on stderr.
#[cfg(unix)]
fn run_shell_process(
    cmd: &str,
    workspace: &Path,
    isolation: OsIsolation,
    writable_roots: &[PathBuf],
    cancel: &AtomicBool,
    timeout: Duration,
) -> Result<ShellOutcome, String> {
    use std::sync::mpsc::{self, TryRecvError};
    use std::time::Instant;
    use std::os::unix::process::CommandExt;

    let started = Instant::now();
    if cancel.load(Ordering::SeqCst) {
        return Err("command interrupted before it started".into());
    }
    let mut builder = std::process::Command::new("sh");
    builder
        .arg("-c")
        .arg(cmd)
        .current_dir(workspace)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .env_clear();
    for (name, value) in std::env::vars_os() {
        if env_var_allowed(&name) {
            builder.env(name, value);
        }
    }
    // Own process group so a timeout can take down the whole tree
    // (background jobs, grandchildren), not just the `sh` process.
    //
    // DECISION FLAG — `kill -9 -1` from a tool command ends the whole
    // chatTUI session. Chosen approach: seccomp-deny `kill(-1, sig)` inside
    // the sandboxed child (see `seccomp::build_filter` in os_isolation) —
    // the one *broad* kill target, everything-at-once — plus this
    // documented note. Reasoning:
    // - The alternative wrapper/supervisor-as-group-leader idea does NOT
    //   actually help: `kill(-1)` on Linux signals every process the sender
    //   may signal except the sender itself, regardless of process-group
    //   membership, so a supervisor as pgid leader would still die — and so
    //   would chatTUI (same uid, same session). Re-shaping the group only
    //   moves the blast radius.
    // - Documentation alone was rejected because the footgun stays one
    //   accidental command away from ending the session; denying `kill(-1)`
    //   is one small BPF rule, cannot break the timeout kill (that runs
    //   from the unsandboxed parent via `kill_process_group`), and leaves
    //   targeted signals (`kill <pid>`, `kill -<pgid>`, `kill %job`) fully
    //   working for normal build/test flows.
    // - Residual, documented limits: targeted `kill <pid>`, `pkill` and
    //   `pidfd_send_signal` can still reach same-uid processes, chatTUI
    //   included — that is inherent to unprivileged same-uid sandboxing
    //   (the agent can always read /proc and find the pid) and is called
    //   out in the module security-model docs.
    unsafe {
        builder.pre_exec(|| {
            if libc::setpgid(0, 0) != 0 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
    if isolation != OsIsolation::Off {
        let roots = writable_roots.to_vec();
        let strict = isolation == OsIsolation::Require;
        unsafe {
            builder.pre_exec(move || {
                os_isolation::confine_in_child(&roots, os_isolation::WRITE_ONLY_DEVICES, strict)
            });
        }
    }

    let mut child = match builder.spawn() {
        Ok(child) => child,
        Err(error) => {
            if isolation == OsIsolation::Require
                && matches!(
                    error.kind(),
                    std::io::ErrorKind::PermissionDenied | std::io::ErrorKind::Unsupported
                )
            {
                return Err(
                    "OS-level isolation is required but unavailable on this system - refusing \
                     to run the command. Set sandbox.os_isolation = \"auto\" in config.json to \
                     allow an explicit unrestricted fallback."
                        .to_string(),
                );
            }
            return Err(format!("spawning sh: {error}"));
        }
    };
    // The child remains owned (and unreaped) until the worker knows both its
    // status and the fate of its pipes. Its PID cannot be recycled during a
    // timeout/cancellation signal. Background children that keep a pipe open
    // are subject to the same deadline as the group leader.
    let pid = child.id();
    let stdout_pipe = match child.stdout.take() {
        Some(pipe) => pipe,
        None => {
            kill_and_reap(&mut child, pid);
            return Err("stdout unavailable".into());
        }
    };
    let stderr_pipe = match child.stderr.take() {
        Some(pipe) => pipe,
        None => {
            kill_and_reap(&mut child, pid);
            return Err("stderr unavailable".into());
        }
    };
    let (stdout_tx, stdout_rx) = mpsc::sync_channel(1);
    let (stderr_tx, stderr_rx) = mpsc::sync_channel(1);
    if let Err(error) = std::thread::Builder::new()
        .spawn(move || { let _ = stdout_tx.send(read_capped(stdout_pipe)); })
    {
        kill_and_reap(&mut child, pid);
        return Err(format!("starting stdout reader: {error}"));
    }
    if let Err(error) = std::thread::Builder::new()
        .spawn(move || { let _ = stderr_tx.send(read_capped(stderr_pipe)); })
    {
        kill_and_reap(&mut child, pid);
        return Err(format!("starting stderr reader: {error}"));
    }
    let mut stdout = None;
    let mut stderr = None;
    loop {
        if cancel.load(Ordering::SeqCst) || started.elapsed() >= timeout {
            // libc is used from the owning worker: there is no separate PID
            // publication or unbounded `sh -c kill ...` helper to race with.
            kill_and_reap(&mut child, pid);
            if cancel.load(Ordering::SeqCst) {
                return Err("command interrupted; process group was killed".into());
            }
            return Err(format!("command timed out after {}s; process group was killed", timeout.as_secs()));
        }
        if stdout.is_none() {
            match stdout_rx.try_recv() {
                Ok(data) => stdout = Some(data),
                Err(TryRecvError::Disconnected) => {
                    kill_and_reap(&mut child, pid);
                    return Err("stdout reader stopped".into());
                },
                Err(TryRecvError::Empty) => {}
            }
        }
        if stderr.is_none() {
            match stderr_rx.try_recv() {
                Ok(data) => stderr = Some(data),
                Err(TryRecvError::Disconnected) => {
                    kill_and_reap(&mut child, pid);
                    return Err("stderr reader stopped".into());
                },
                Err(TryRecvError::Empty) => {}
            }
        }
        if stdout.is_some() && stderr.is_some() {
            // Poll/reap only after the pipes close, keeping the original
            // child PID reserved until any background descendants finish or
            // are killed on timeout. Never signal a recycled process group.
            let status = match child.try_wait() {
                Ok(status) => status,
                Err(error) => {
                    kill_and_reap(&mut child, pid);
                    return Err(format!("waiting on sh: {error}"));
                }
            };
            if let Some(status) = status {
                let (stdout, stdout_cut) = stdout.take().expect("checked above");
                let (stderr, stderr_cut) = stderr.take().expect("checked above");
                let mut result = ShellOutcome { stdout, stderr, status };
                if stdout_cut || stderr_cut {
                    result.stderr.extend_from_slice(b"\n(output truncated while reading)\n");
                }
                return Ok(result);
            }
        }
        // If one pipe is ready first, keep its bounded buffer until both
        // finish; never wait on an exited shell while descendants hold it.
        std::thread::sleep(Duration::from_millis(20));
    }
}

#[cfg(unix)]
fn read_capped(mut pipe: impl std::io::Read) -> (Vec<u8>, bool) {
    let mut captured = Vec::new();
    let mut buf = [0u8; 8192];
    let mut truncated = false;
    loop {
        match pipe.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => {
                let keep = n.min(MAX_TOOL_OUTPUT_BYTES.saturating_sub(captured.len()));
                captured.extend_from_slice(&buf[..keep]);
                truncated |= keep < n;
            }
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(_) => break,
        }
    }
    (captured, truncated)
}

#[cfg(unix)]
fn kill_and_reap(child: &mut std::process::Child, pid: u32) {
    kill_process_group(pid);
    let _ = child.kill();
    let _ = child.wait();
}

#[cfg(unix)]
fn kill_process_group(pid: u32) {
    if pid != 0 && pid <= i32::MAX as u32 {
        unsafe { libc::kill(-(pid as i32), libc::SIGKILL); }
    }
}

fn format_shell_parts(stdout: &[u8], stderr: &[u8], status_text: &str, failure: bool) -> String {
    let stdout = String::from_utf8_lossy(stdout);
    let stderr = String::from_utf8_lossy(stderr);
    let mut result = String::new();
    // On failure stderr is the diagnostic: do not truncate it away behind
    // thousands of bytes of noisy stdout.
    let parts = if failure {
        [(&*stderr, "stderr"), (&*stdout, "stdout")]
    } else {
        [(&*stdout, "stdout"), (&*stderr, "stderr")]
    };
    for (part, label) in parts {
        if !part.is_empty() {
            if !result.is_empty() {
                result.push_str(&format!("\n--- {label} ---\n"));
            }
            result.push_str(part);
        }
    }
    if result.is_empty() {
        result = format!("(command exited with status {status_text})");
    }
    if result.len() > MAX_TOOL_OUTPUT_BYTES {
        // Byte-truncating a String can split a multi-byte character and
        // panic; back off to the nearest char boundary.
        let mut cut = MAX_TOOL_OUTPUT_BYTES;
        while !result.is_char_boundary(cut) {
            cut -= 1;
        }
        result.truncate(cut);
        result.push_str("\n... truncated");
    }
    result
}

/// The metadata check is only a fast rejection: enforce the limit again as
/// bytes are read so a concurrently growing file cannot exhaust memory.
async fn read_limited_file(path: &Path, max: usize) -> Result<String> {
    let file = fs::File::open(path).await?;
    let mut bytes = Vec::new();
    file.take(max as u64 + 1).read_to_end(&mut bytes).await?;
    if bytes.len() > max {
        return Err(anyhow!("file exceeds the {max}-byte limit"));
    }
    Ok(String::from_utf8(bytes).context("file is not UTF-8")?)
}

struct FileTemp(PathBuf);
impl Drop for FileTemp {
    fn drop(&mut self) {
        if !self.0.as_os_str().is_empty() {
            let _ = std::fs::remove_file(&self.0);
        }
    }
}

/// Atomic replacement within the validated parent. Cross-process symlink
/// swaps remain a TOCTOU risk on platforms without handle-relative openat2;
/// this is an integrity improvement, not an OS sandbox. Reject hard links
/// instead of silently changing their semantics with a rename.
fn atomic_write_file(path: &Path, bytes: &[u8], workspace: &Path) -> Result<()> {
    use std::io::Write;
    let parent = path.parent().context("missing parent directory")?;
    if !parent.canonicalize()?.starts_with(workspace) {
        return Err(anyhow!("parent moved outside workspace"));
    }
    let original = match std::fs::symlink_metadata(path) {
        Ok(meta) => {
            if !meta.is_file() || meta.file_type().is_symlink() {
                return Err(anyhow!("refusing to replace a link or non-file"));
            }
            #[cfg(unix)] {
                use std::os::unix::fs::MetadataExt;
                if meta.nlink() > 1 {
                    return Err(anyhow!("refusing to replace a multiply linked file"));
                }
            }
            Some(meta)
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => return Err(error).context("checking target file"),
    };
    let mut tmp = None;
    for _ in 0..16 {
        let candidate = parent.join(format!(".chat-tui-{}-{}.tmp", std::process::id(),
            FILE_SEQUENCE.fetch_add(1, Ordering::Relaxed)));
        let mut options = std::fs::OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)] {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        match options.open(&candidate) {
            Ok(file) => { tmp = Some((candidate, file)); break; }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error).context("creating replacement file"),
        }
    }
    let (temp_path, mut file) = tmp.context("could not create unique replacement file")?;
    let mut guard = FileTemp(temp_path.clone());
    if let Some(meta) = &original {
        file.set_permissions(meta.permissions())?;
    }
    file.write_all(bytes).context("writing replacement file")?;
    file.sync_all().context("syncing replacement file")?;
    drop(file);
    if !parent.canonicalize()?.starts_with(workspace) {
        return Err(anyhow!("parent moved outside workspace before rename"));
    }
    // Check for a newly swapped symlink. A second change between this check
    // and rename is still possible without directory-handle syscalls.
    if std::fs::symlink_metadata(path).is_ok_and(|meta| meta.file_type().is_symlink()) {
        return Err(anyhow!("target became a symlink before rename"));
    }
    std::fs::rename(&temp_path, path).context("renaming replacement file")?;
    guard.0.clear();
    #[cfg(unix)]
    std::fs::File::open(parent)?.sync_all().context("syncing parent directory")?;
    Ok(())
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
    #[cfg(unix)]
    use std::time::Instant;

    fn unique_root(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("chatTUI_sec_{}_{}", name, std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[cfg(unix)]
    fn sandbox_with_timeout(root: PathBuf, secs: u64) -> Sandbox {
        let mut cfg = SandboxConfig::default();
        cfg.workspace_root = root.canonicalize().unwrap_or(root);
        cfg.shell_timeout = Duration::from_secs(secs);
        cfg.permission_mode = PermissionMode::FullAuto;
        Sandbox::new(cfg)
    }

    #[test]
    fn default_permissions_restore_full_auto() {
        let config = SandboxConfig::default();
        assert!(config.auto_approve && config.allow_shell);
        assert_eq!(config.permission_mode, PermissionMode::FullAuto);
    }

    #[test]
    fn sandbox_resolves_inside_root() {
        let dir = unique_root("resolve");
        let sandbox = Sandbox::with_root(dir.clone());
        let p = sandbox.resolve_path("src/main.rs").unwrap();
        assert!(p.starts_with(&dir));
    }

    #[test]
    fn sandbox_blocks_traversal() {
        let dir = unique_root("traversal");
        let sandbox = Sandbox::with_root(dir);
        assert!(sandbox.resolve_path("../../etc/passwd").is_err());
        assert!(sandbox.resolve_path("..").is_err());
        assert!(sandbox.resolve_path("a/../../..").is_err());
        assert!(sandbox.resolve_path("./sub/../../../x").is_err());
    }

    #[test]
    fn absolute_paths_outside_workspace_are_denied() {
        let dir = unique_root("abs-deny");
        let sandbox = Sandbox::with_root(dir);
        assert!(sandbox.resolve_path("/etc/passwd").is_err());
        assert!(sandbox.resolve_path("/usr/bin/env").is_err());
    }

    #[test]
    fn absolute_path_with_dotdot_through_missing_dir_cannot_escape() {
        // Regression: canonicalize() fails for the missing directory, and the
        // old fallback accepted the raw unresolved path — the kernel would
        // then resolve the `..` outside the workspace.
        let dir = unique_root("abs-dotdot");
        let sandbox = Sandbox::with_root(dir.clone());
        let outside = dir.join("missing-dir").join("..").join("..").join("etc");
        let escape = outside.join("passwd");
        assert!(escape.is_absolute());
        assert!(sandbox.resolve_path(&escape.to_string_lossy()).is_err());
    }

    #[test]
    fn absolute_path_inside_workspace_is_allowed_even_if_missing() {
        let dir = unique_root("abs-allow");
        let sandbox = Sandbox::with_root(dir.clone());
        let root = dir.canonicalize().unwrap();
        let target = root.join("sub").join("new-file.txt");
        let resolved = sandbox.resolve_path(&target.to_string_lossy()).unwrap();
        assert!(resolved.starts_with(&root));
        assert_eq!(resolved, target);
    }

    #[test]
    fn dotdot_that_stays_inside_the_workspace_is_allowed() {
        let dir = unique_root("dotdot-inside");
        std::fs::create_dir_all(dir.join("sub")).unwrap();
        let sandbox = Sandbox::with_root(dir.clone());
        let root = dir.canonicalize().unwrap();
        let resolved = sandbox.resolve_path("sub/../keep.txt").unwrap();
        assert_eq!(resolved, root.join("keep.txt"));
    }

    #[test]
    fn nul_bytes_and_empty_paths_are_rejected() {
        let dir = unique_root("nul");
        let sandbox = Sandbox::with_root(dir);
        assert!(sandbox.resolve_path("").is_err());
        assert!(sandbox.resolve_path("   ").is_err());
        assert!(sandbox.resolve_path("a\0b").is_err());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn symlink_escape_is_denied() {
        let dir = unique_root("symlink");
        let sandbox = Sandbox::with_root(dir.clone());
        std::os::unix::fs::symlink("/etc/hosts", dir.join("escape.txt")).unwrap();
        std::os::unix::fs::symlink("/", dir.join("out")).unwrap();
        // Symlinked file pointing outside.
        assert!(sandbox.read_file("escape.txt", 1, DEFAULT_READ_LIMIT).await.is_err());
        // Directory symlink: read through it.
        assert!(sandbox.read_file("out/etc/hosts", 1, DEFAULT_READ_LIMIT).await.is_err());
        // Directory symlink: write through it.
        assert!(sandbox
            .write_file("out/escape-write.txt", "x")
            .await
            .is_err());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn nested_symlink_escape_is_denied() {
        let dir = unique_root("nested-symlink");
        std::fs::create_dir_all(dir.join("a")).unwrap();
        std::os::unix::fs::symlink("/etc", dir.join("a").join("b")).unwrap();
        let sandbox = Sandbox::with_root(dir);
        assert!(sandbox.read_file("a/b/passwd", 1, DEFAULT_READ_LIMIT).await.is_err());
    }

    #[tokio::test]
    async fn reading_nonexistent_paths_fails_cleanly() {
        let dir = unique_root("missing");
        let sandbox = Sandbox::with_root(dir);
        let err = sandbox.read_file("does/not/exist.txt", 1, DEFAULT_READ_LIMIT).await.unwrap_err();
        assert!(!err.to_string().contains("panic"));
        // And writing into a non-existent nested directory still works.
        sandbox
            .write_file("does/new/file.txt", "created")
            .await
            .unwrap();
        assert!(sandbox
            .read_file("does/new/file.txt", 1, DEFAULT_READ_LIMIT)
            .await
            .unwrap()
            .contains("created"));
    }

    #[tokio::test]
    async fn sensitive_files_are_denied_for_every_file_tool() {
        let dir = unique_root("sensitive");
        let sandbox = Sandbox::with_root(dir.clone());
        // dotenv-style names
        assert!(sandbox.read_file(".env", 1, DEFAULT_READ_LIMIT).await.is_err());
        assert!(sandbox.write_file(".env", "X=1").await.is_err());
        assert!(sandbox.edit_file(".env", "A", "B").await.is_err());
        assert!(sandbox.read_file("config/.env.local", 1, DEFAULT_READ_LIMIT).await.is_err());
        assert!(sandbox.read_file("production.env", 1, DEFAULT_READ_LIMIT).await.is_err());
        // key material
        assert!(sandbox.read_file("server.pem", 1, DEFAULT_READ_LIMIT).await.is_err());
        assert!(sandbox.write_file("tls/host.key", "x").await.is_err());
        assert!(sandbox.read_file("id_ed25519", 1, DEFAULT_READ_LIMIT).await.is_err());
        // git internals: reads of config/credentials, all writes
        assert!(sandbox.read_file(".git/config", 1, DEFAULT_READ_LIMIT).await.is_err());
        assert!(sandbox
            .write_file(".git/hooks/pre-commit", "#!/bin/sh")
            .await
            .is_err());
        // non-sensitive git internals stay readable
        std::fs::create_dir_all(dir.join(".git")).unwrap();
        std::fs::write(dir.join(".git").join("HEAD"), "ref: refs/heads/main").unwrap();
        assert!(sandbox.read_file(".git/HEAD", 1, DEFAULT_READ_LIMIT).await.is_ok());
        // normal files keep working
        sandbox.write_file("normal.txt", "fine").await.unwrap();
        assert!(sandbox.read_file("normal.txt", 1, DEFAULT_READ_LIMIT).await.unwrap().contains("fine"));
    }

    #[tokio::test]
    async fn list_files_hides_sensitive_entries() {
        let dir = unique_root("list");
        std::fs::write(dir.join(".env"), "SECRET=1").unwrap();
        std::fs::write(dir.join("id_rsa"), "key").unwrap();
        std::fs::write(dir.join("readme.md"), "hi").unwrap();
        let sandbox = Sandbox::with_root(dir);
        let listing = sandbox.list_files(".").await.unwrap();
        assert!(listing.contains("readme.md"));
        assert!(!listing.contains(".env"));
        assert!(!listing.contains("id_rsa"));
    }

    #[tokio::test]
    async fn extra_sensitive_names_from_config_are_enforced() {
        let dir = unique_root("extra");
        std::fs::write(dir.join("secrets.json"), "{}").unwrap();
        std::fs::write(dir.join("token.txt"), "t").unwrap();
        std::fs::write(dir.join("anything-else.txt"), "ok").unwrap();
        let mut cfg = SandboxConfig::default();
        cfg.workspace_root = dir.canonicalize().unwrap();
        cfg.extra_sensitive_names = vec!["secrets.json".into(), ".token".into()];
        let sandbox = Sandbox::new(cfg);
        assert!(sandbox.read_file("secrets.json", 1, DEFAULT_READ_LIMIT).await.is_err());
        assert!(sandbox.read_file("token.txt", 1, DEFAULT_READ_LIMIT).await.is_err()); // suffix match
        assert!(sandbox.read_file("anything-else.txt", 1, DEFAULT_READ_LIMIT).await.is_ok());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn write_through_parent_symlink_outside_workspace_is_denied() {
        let dir = unique_root("parent-symlink");
        let outside = unique_root("parent-symlink-outside");
        std::os::unix::fs::symlink(&outside, dir.join("link")).unwrap();
        let sandbox = Sandbox::with_root(dir);
        assert!(sandbox.write_file("link/child.txt", "x").await.is_err());
        assert!(!outside.join("child.txt").exists());
    }

    #[tokio::test]
    async fn sandbox_read_write() {
        let dir = unique_root("rw");
        let sandbox = Sandbox::with_root(dir.clone());
        sandbox.write_file("test.txt", "hello").await.unwrap();
        let content = sandbox.read_file("test.txt", 1, DEFAULT_READ_LIMIT).await.unwrap();
        assert!(content.contains("hello"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn default_sandbox_has_no_implicit_cwd_target() {
        let sandbox = Sandbox::new(SandboxConfig::default());
        assert!(!sandbox.has_target());
        assert!(sandbox.resolve_path("src/main.rs").is_err());
        // No target also means no shell, regardless of mode.
        let rt = tokio::runtime::Runtime::new().unwrap();
        assert!(rt.block_on(sandbox.bash("echo hi")).is_err());
    }

    #[tokio::test]
    async fn writes_stay_under_configured_target_not_crate_root() {
        let target = unique_root("dst");
        let crate_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let sandbox = Sandbox::with_root(target.clone());
        sandbox.write_file("hello_agent.txt", "hi").await.unwrap();
        let written = target.join("hello_agent.txt");
        assert!(written.exists());
        assert!(written.starts_with(&target));
        assert!(!crate_root.join("hello_agent.txt").exists());
        let _ = std::fs::remove_dir_all(&target);
    }

    #[test]
    fn shell_failure_keeps_stderr_even_if_stdout_floods() {
        let formatted = format_shell_parts(&vec![b'x'; 25_000], b"important failure", "7", true);
        assert!(formatted.contains("important failure"));
        assert!(formatted.contains("truncated"));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn nonzero_shell_exit_is_an_error_even_with_output() {
        let sandbox = Sandbox::with_root(unique_root("nonzero-shell"));
        let err = sandbox.bash("echo broken >&2; exit 7").await.unwrap_err();
        assert!(err.to_string().contains("exit status: 7"));
        assert!(err.to_string().contains("broken"));
    }

    #[tokio::test]
    async fn write_and_edit_refuse_oversize_content_without_replacing_the_file() {
        let root = unique_root("write-limit");
        let mut sandbox = Sandbox::with_root(root.clone());
        sandbox.config.max_file_size = 8;
        sandbox.write_file("a.txt", "ok").await.unwrap();
        assert!(sandbox.write_file("a.txt", "123456789").await.is_err());
        assert!(sandbox.edit_file("a.txt", "ok", "123456789").await.is_err());
        assert_eq!(std::fs::read_to_string(root.join("a.txt")).unwrap(), "ok");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn bash_completes_within_timeout() {
        let dir = unique_root("bash-ok");
        let sandbox = sandbox_with_timeout(dir, 10);
        let out = sandbox.bash("echo ok").await.unwrap();
        assert!(out.contains("ok"));
    }

    #[test]
    fn shell_timeout_is_clamped_away_from_zero() {
        let mut cfg = SandboxConfig::default();
        cfg.shell_timeout = Duration::ZERO;
        assert_eq!(
            cfg.effective_shell_timeout(),
            Duration::from_secs(MIN_SHELL_TIMEOUT_SECS)
        );
        cfg.shell_timeout = Duration::from_secs(45);
        assert_eq!(cfg.effective_shell_timeout(), Duration::from_secs(45));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn bash_timeout_is_enforced() {
        let dir = unique_root("timeout");
        let sandbox = sandbox_with_timeout(dir, 1);
        let started = Instant::now();
        let err = sandbox.bash("sleep 5").await.unwrap_err();
        let elapsed = started.elapsed();
        assert!(
            err.to_string().contains("timed out"),
            "unexpected error: {err}"
        );
        assert!(elapsed.as_secs() < 4, "timeout took too long: {elapsed:?}");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn bash_timeout_kills_background_processes() {
        // `sh` exits only after `wait`; the backgrounded sleep keeps the
        // output pipes open, so a naive implementation would hang forever
        // here. The timeout must fire AND the process group must die.
        let dir = unique_root("timeout-group");
        let sandbox = sandbox_with_timeout(dir, 1);
        let result = sandbox.bash("sleep 37 & wait").await;
        assert!(result.is_err());
        let mut gone = false;
        for _ in 0..40 {
            if !process_running("sleep 37") {
                gone = true;
                break;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        assert!(gone, "backgrounded `sleep 37` survived the timeout kill");
    }

    #[cfg(unix)]
    fn process_running(needle: &str) -> bool {
        let entries = match std::fs::read_dir("/proc") {
            Ok(entries) => entries,
            Err(_) => return false,
        };
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if !name.bytes().all(|b| b.is_ascii_digit()) {
                continue;
            }
            if let Ok(cmdline) = std::fs::read(entry.path().join("cmdline")) {
                let text = String::from_utf8_lossy(&cmdline).replace('\0', " ");
                if text.contains(needle) {
                    return true;
                }
            }
        }
        false
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn bash_does_not_inherit_sensitive_environment() {
        let dir = unique_root("env");
        let sandbox = Sandbox::with_root(dir);
        std::env::set_var("CHATTUI_SECRET_TEST", "top-secret-value");
        let secret = sandbox.bash(r#"printf '%s' "$CHATTUI_SECRET_TEST""#).await.unwrap();
        assert!(
            !secret.contains("top-secret-value"),
            "credentials leaked into the shell environment"
        );
        // The allowlist keeps the shell usable for builds.
        let path = sandbox.bash("printenv PATH").await.unwrap();
        assert!(!path.trim().is_empty(), "allowlisted PATH must survive");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn bash_denies_commands_naming_sensitive_files() {
        let dir = unique_root("shell-sensitive");
        let sandbox = Sandbox::with_root(dir);
        let err = sandbox.bash("cat .env").await.unwrap_err();
        assert!(err.to_string().contains("sensitive"), "{err}");
        assert!(sandbox.bash("cat config/id_rsa").await.is_err());
        assert!(sandbox.bash("less .git/config").await.is_err());
        // Documented trade-off: even harmless uses of a sensitive name are
        // refused (the scan is textual by design).
        assert!(sandbox.bash("echo .env").await.is_err());
        assert!(sandbox.bash("echo hello world").await.is_ok());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn shell_reads_outside_the_workspace_are_allowed_by_design() {
        // Pins the honest behavior in BOTH modes: even with kernel isolation
        // active, reads outside the workspace stay allowed (toolchains must
        // read system files; secret-read protection is application policy).
        // Writes outside are covered by the isolation-specific tests below.
        let dir = unique_root("no-sandbox");
        let sandbox = Sandbox::with_root(dir);
        let out = sandbox.bash("cat /etc/hosts").await.unwrap();
        assert!(!out.trim().is_empty());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn bash_kernel_isolation_denies_outside_writes_when_supported() {
        if !os_isolation::os_isolation_supported() {
            eprintln!("skipping: kernel lacks Landlock/seccomp support");
            return;
        }
        let Some(home) = std::env::var_os("HOME")
            .map(PathBuf::from)
            .filter(|p| !p.as_os_str().is_empty())
        else {
            eprintln!("skipping: HOME is not set");
            return;
        };
        let dir = unique_root("iso-write");
        let mut cfg = SandboxConfig::default();
        cfg.workspace_root = dir.canonicalize().unwrap();
        cfg.os_isolation = OsIsolation::Require;
        cfg.permission_mode = PermissionMode::FullAuto;
        let sandbox = Sandbox::new(cfg);

        let escape = home.join("chattui_iso_escape_probe");
        let _ = std::fs::remove_file(&escape);

        // Write outside the workspace: the kernel must refuse it, and
        // nonzero shell status must propagate instead of faking success.
        let err = sandbox
            .bash(&format!("touch '{}'", escape.display()))
            .await
            .unwrap_err();
        let lower = err.to_string().to_ascii_lowercase();
        assert!(
            lower.contains("permission denied") || lower.contains("operation not permitted"),
            "expected a filesystem denial, got: {err}"
        );
        assert!(
            !escape.exists(),
            "the sandboxed shell must not be able to write outside the workspace"
        );

        // The workspace itself stays writable, and normal output flows.
        let out = sandbox
            .bash("echo ok > inside.txt && cat inside.txt")
            .await
            .unwrap();
        assert!(out.contains("ok"), "unexpected output: {out}");
        assert!(dir.join("inside.txt").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn bash_network_access_is_denied_when_isolation_is_supported() {
        if !os_isolation::os_isolation_supported() {
            eprintln!("skipping: kernel lacks Landlock/seccomp support");
            return;
        }
        let dir = unique_root("iso-net");
        let sandbox = Sandbox::with_root(dir);
        // `ping` needs a socket; under isolation the kernel refuses it.
        let err = sandbox.bash("ping -c1 -W1 127.0.0.1").await.unwrap_err();
        let lower = err.to_string().to_ascii_lowercase();
        if lower.contains("not found") {
            eprintln!("skipping: ping is not installed");
            return;
        }
        assert!(
            lower.contains("permitted") || lower.contains("denied"),
            "expected a socket permission failure, got: {err}"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn bash_auto_mode_fallback_is_explicit_not_silent() {
        let dir = unique_root("iso-auto");
        let sandbox = Sandbox::with_root(dir);
        let out = sandbox.bash("echo hi").await.unwrap();
        let supported = os_isolation::os_isolation_supported();
        if supported {
            assert!(
                !out.contains("os-level isolation unavailable"),
                "no warning may appear when isolation applied: {out}"
            );
        } else {
            assert!(
                out.contains("os-level isolation unavailable"),
                "the unrestricted fallback must say so explicitly: {out}"
            );
        }
        assert!(out.contains("hi"));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn bash_require_mode_fails_closed_when_isolation_unsupported() {
        // Only meaningful on machines without kernel support; elsewhere the
        // command simply runs (confined), so the test asserts the invariant
        // that holds in both worlds.
        let dir = unique_root("iso-require");
        let mut cfg = SandboxConfig::default();
        cfg.workspace_root = dir.canonicalize().unwrap();
        cfg.os_isolation = OsIsolation::Require;
        cfg.permission_mode = PermissionMode::FullAuto;
        let sandbox = Sandbox::new(cfg);
        let result = sandbox.bash("echo hi").await;
        if os_isolation::os_isolation_supported() {
            assert!(result.unwrap().contains("hi"));
        } else {
            let error = result.unwrap_err().to_string();
            assert!(error.contains("required but unavailable"), "{error}");
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn advisory_denylist_still_blocks_obvious_footguns() {
        let dir = unique_root("advisory");
        let sandbox = Sandbox::with_root(dir);
        let err = sandbox.bash("rm -rf /").await.unwrap_err();
        assert!(err.to_string().contains("blocked dangerous command"));
    }

    #[tokio::test]
    async fn permission_modes_gate_tools() {
        let dir = unique_root("perm");

        // read-only: reads pass the policy (file-not-found, not permission),
        // writes/edits/shell are refused.
        let mut cfg = SandboxConfig::default();
        cfg.workspace_root = dir.canonicalize().unwrap();
        cfg.permission_mode = PermissionMode::ReadOnly;
        let sandbox = Sandbox::new(cfg);
        let err = sandbox.read_file("missing.txt", 1, DEFAULT_READ_LIMIT).await.unwrap_err();
        assert!(err.to_string().contains("file not found"));
        let err = sandbox.write_file("x.txt", "y").await.unwrap_err();
        assert!(err.to_string().contains("permission denied"), "{err}");
        let err = sandbox.edit_file("x.txt", "a", "b").await.unwrap_err();
        assert!(err.to_string().contains("permission denied"));
        let err = sandbox.bash("echo hi").await.unwrap_err();
        assert!(err.to_string().contains("permission denied"));

        // workspace-write: files yes, shell no.
        let mut cfg = SandboxConfig::default();
        cfg.workspace_root = dir.canonicalize().unwrap();
        cfg.permission_mode = PermissionMode::WorkspaceWrite;
        let sandbox = Sandbox::new(cfg);
        sandbox.write_file("x.txt", "y").await.unwrap();
        assert!(sandbox.read_file("x.txt", 1, DEFAULT_READ_LIMIT).await.unwrap().contains("y"));
        let err = sandbox.bash("echo hi").await.unwrap_err();
        assert!(err.to_string().contains("permission denied"));

        // ask-before-write denies a write that was not approved (the app asks
        // first and runs approved calls via `execute_tool_cancellable`).
        let mut cfg = SandboxConfig::default();
        cfg.workspace_root = dir.canonicalize().unwrap();
        cfg.permission_mode = PermissionMode::AskBeforeWrite;
        let sandbox = Sandbox::new(cfg);
        let err = sandbox.write_file("y.txt", "y").await.unwrap_err();
        assert!(err.to_string().contains("permission denied"));
        assert!(err.to_string().contains("ask-before-write"));

        // The legacy master switch still wins over full-auto for shell.
        let mut cfg = SandboxConfig::default();
        cfg.workspace_root = dir.canonicalize().unwrap();
        cfg.permission_mode = PermissionMode::FullAuto;
        cfg.allow_shell = false;
        let sandbox = Sandbox::new(cfg);
        let err = sandbox.bash("echo hi").await.unwrap_err();
        assert!(err.to_string().contains("allow_shell"));
    }

    #[tokio::test]
    async fn execute_tool_refuses_unknown_tools_and_denied_kinds() {
        let dir = unique_root("dispatch");
        let mut cfg = SandboxConfig::default();
        cfg.workspace_root = dir.canonicalize().unwrap();
        cfg.permission_mode = PermissionMode::ReadOnly;
        let sandbox = Sandbox::new(cfg);

        let unknown = ToolCall::new("1", "rm_file", "{}");
        let err = sandbox.execute_tool(&unknown).await.unwrap_err();
        assert!(err.to_string().contains("unknown tool"));

        let bash = ToolCall::new("2", "bash", r#"{"command":"echo hi"}"#);
        let err = sandbox.execute_tool(&bash).await.unwrap_err();
        assert!(err.to_string().contains("permission denied"));
    }

    #[tokio::test]
    async fn execute_tool_runs_allowed_kinds_end_to_end() {
        let dir = unique_root("dispatch-ok");
        let sandbox = Sandbox::with_root(dir);
        let write = ToolCall::new("1", "write_file", r#"{"path":"a.txt","content":"hello"}"#);
        sandbox.execute_tool(&write).await.unwrap();
        let read = ToolCall::new("2", "read_file", r#"{"path":"a.txt"}"#);
        assert!(sandbox.execute_tool(&read).await.unwrap().contains("hello"));
        let list = ToolCall::new("3", "list_files", "{}");
        assert!(sandbox.execute_tool(&list).await.unwrap().contains("a.txt"));
    }

    #[tokio::test]
    async fn execute_tool_reports_truncated_arguments_explicitly() {
        let dir = unique_root("trunc-args");
        let sandbox = Sandbox::with_root(dir);
        // Unterminated JSON — what a stream cut mid-arguments leaves behind.
        let call = ToolCall::new("1", "read_file", r#"{"path": "#);
        let err = sandbox.execute_tool(&call).await.unwrap_err();
        assert!(err.to_string().contains("truncated"), "unexpected: {err}");
    }

    #[tokio::test]
    async fn read_file_serves_numbered_windows() {
        let dir = unique_root("read-window");
        let body: String = (1..=600).map(|i| format!("line {i}\n")).collect();
        fs::write(dir.join("big.txt"), body).await.unwrap();
        let sandbox = Sandbox::with_root(dir);

        // First window: 250 numbered lines plus a pointer onward.
        let out = sandbox.read_file("big.txt", 1, 250).await.unwrap();
        assert!(out.contains("lines 1-250 of 600"), "header: {out}");
        assert!(out.contains("1│line 1"));
        assert!(out.contains("250│line 250"));
        assert!(!out.contains("line 251"));
        assert!(
            out.contains("offset 251"),
            "footer must point onward: {out}"
        );

        // A later window; the tail carries no onward pointer.
        let out = sandbox.read_file("big.txt", 551, 250).await.unwrap();
        assert!(out.contains("lines 551-600 of 600"), "header: {out}");
        assert!(!out.contains("more line"));

        // Past the end: informative, not a crash.
        let out = sandbox.read_file("big.txt", 700, 250).await.unwrap();
        assert!(out.contains("past the end"), "{out}");

        // Empty file: plain and simple.
        let out = sandbox.read_file("empty-missing", 1, 250).await;
        assert!(out.is_err(), "a missing file is still an error");
    }

    #[tokio::test]
    async fn write_and_edit_report_line_stats() {
        let dir = unique_root("stats");
        let sandbox = Sandbox::with_root(dir);

        // New file: additions only.
        let out = sandbox.write_file("a.txt", "one\ntwo\n").await.unwrap();
        assert_eq!(out, "wrote a.txt (+2)");

        // Overwrite: additions and removals.
        let out = sandbox
            .write_file("a.txt", "one\ntwo\nthree\nfour\nfive\n")
            .await
            .unwrap();
        assert_eq!(out, "wrote a.txt (+5 -2)");

        // Edit: one line replaced by two.
        let out = sandbox
            .edit_file("a.txt", "three", "three-a\nthree-b")
            .await
            .unwrap();
        assert_eq!(out, "edited a.txt (+2 -1)");
    }

    #[test]
    fn output_truncation_never_panics_on_multibyte_boundaries() {
        let mut stdout = Vec::new();
        // Multi-byte characters straddling the truncation point.
        while stdout.len() < MAX_TOOL_OUTPUT_BYTES + 16 {
            stdout.extend_from_slice("é".as_bytes());
        }
        let truncated = format_shell_parts(&stdout, &[], "0", false);
        assert!(truncated.len() <= MAX_TOOL_OUTPUT_BYTES + "\n... truncated".len());
        assert!(truncated.ends_with("... truncated"));
    }
}
