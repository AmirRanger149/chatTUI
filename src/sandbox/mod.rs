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
use std::process::Stdio;
#[cfg(unix)] // used by the sandboxed worker thread only
use std::sync::atomic::{AtomicI32, Ordering};
use std::time::Duration;
use tokio::fs;

pub const DEFAULT_SHELL_TIMEOUT_SECS: u64 = 30;
/// Lower bound for the enforced shell timeout so a tool call can never be
/// configured (or tricked) into blocking forever.
pub const MIN_SHELL_TIMEOUT_SECS: u64 = 1;
/// Tool output larger than this is truncated before it reaches the model.
const MAX_TOOL_OUTPUT_BYTES: usize = 20_000;

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

    #[allow(dead_code)] // test helper; not used by the app itself
    pub fn with_root(root: PathBuf) -> Self {
        let mut cfg = SandboxConfig::default();
        // Canonicalize so the stored root is the true workspace prefix
        // (symlinks resolved); fall back to the raw path if that fails.
        cfg.workspace_root = root.canonicalize().unwrap_or(root);
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

        // `set_target` / `with_root` store an already-canonical root, so
        // this is the true workspace prefix, not a stale look-alike.
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
                        return Err(anyhow!("access to sensitive file denied: {name} (inside .git)"));
                    }
                    _ => {}
                }
            }
        }
        Ok(())
    }

    pub async fn read_file(&self, path: &str) -> Result<String> {
        self.authorize(ToolKind::Read)?;
        let full = self.resolve_path(path)?;
        self.check_access(&full, Access::Read)?;
        let metadata = fs::metadata(&full).await.context("file not found")?;
        if !metadata.is_file() {
            return Err(anyhow!("not a file: {}", path));
        }
        if metadata.len() as usize > self.config.max_file_size {
            return Err(anyhow!(
                "file too large ({} bytes > {} limit)",
                metadata.len(),
                self.config.max_file_size
            ));
        }
        let content = fs::read_to_string(&full).await.context("reading file")?;
        Ok(content)
    }

    pub async fn write_file(&self, path: &str, content: &str) -> Result<String> {
        self.authorize(ToolKind::Write)?;
        let full = self.resolve_path(path)?;
        self.check_access(&full, Access::Write)?;
        if let Some(parent) = full.parent() {
            fs::create_dir_all(parent).await.context("creating parent dirs")?;
            // Re-validate after creating the parents: create_dir_all follows
            // symlinks, so make sure the directory we are about to write
            // into still resolves inside the workspace. (A symlink swapped
            // in after this check remains a TOCTOU race — see module docs.)
            let canonical_parent = parent.canonicalize().context("resolving parent")?;
            if !canonical_parent.starts_with(&self.config.workspace_root) {
                return Err(anyhow!("path escapes workspace: {}", path));
            }
        }
        fs::write(&full, content).await.context("writing file")?;
        Ok(format!("wrote {} bytes to {}", content.len(), path))
    }

    pub async fn edit_file(&self, path: &str, old_string: &str, new_string: &str) -> Result<String> {
        self.authorize(ToolKind::Edit)?;
        let full = self.resolve_path(path)?;
        self.check_access(&full, Access::Write)?;
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
        while let Some(entry) = entries.next_entry().await? {
            let name = entry.file_name().to_string_lossy().to_string();
            // Keep the sensitive-file inventory out of directory listings.
            if is_sensitive_name(&name, &self.config.extra_sensitive_names) {
                continue;
            }
            let is_dir = entry.file_type().await.map(|t| t.is_dir()).unwrap_or(false);
            let prefix = if is_dir { "dir " } else { "file" };
            lines.push(format!("{} {}", prefix, name));
        }
        lines.sort();
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
    pub async fn bash(&self, command: &str) -> Result<String> {
        let cmd = self.prepare_shell(command)?;
        let timeout = self.config.effective_shell_timeout();

        #[cfg(unix)]
        {
            let workspace = self.config.workspace_root.clone();
            let roots = os_isolation::writable_roots(&workspace);
            let mode = self.config.os_isolation;
            let cmd_owned = cmd.to_string();

            let (otx, orx) = tokio::sync::oneshot::channel::<Result<ShellOutcome, String>>();
            let pid_slot = std::sync::Arc::new(AtomicI32::new(-1));
            let pid_for_thread = pid_slot.clone();

            let worker = std::thread::Builder::new()
                .name("sandboxed-shell".into())
                .spawn(move || {
                    let outcome = run_shell_process(
                        &cmd_owned,
                        &workspace,
                        &pid_for_thread,
                        mode,
                        &roots,
                    );
                    let _ = otx.send(outcome);
                });

            if let Err(error) = worker {
                return Err(anyhow!("could not start the sandboxed shell worker: {error}"));
            }

            let outcome = match tokio::time::timeout(timeout, orx).await {
                Ok(Ok(inner)) => inner,
                Ok(Err(_worker_gone)) => Err("shell worker exited without reporting".to_string()),
                Err(_elapsed) => {
                    let pid = pid_slot.load(Ordering::SeqCst);
                    if pid > 0 {
                        kill_process_group(pid as u32).await;
                    }
                    return Err(anyhow!(
                        "command timed out after {}s and was killed: {}",
                        timeout.as_secs(),
                        cmd
                    ));
                }
            };

            let outcome = outcome.map_err(|error| anyhow!("{error}"))?;
            Ok(format_shell_parts(&outcome.stdout, &outcome.stderr, &outcome.status_text))
        }

        #[cfg(not(unix))]
        {
            let output = tokio::process::Command::new("sh")
                .arg("-c")
                .arg(cmd)
                .current_dir(&self.config.workspace_root)
                .stdin(Stdio::null())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .kill_on_drop(true)
                .output()
                .await
                .context("executing command")?;
            let _ = timeout; // no timeout on this platform yet; documented gap
            Ok(format_shell_output(output))
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
    pub async fn execute_tool(&self, tool_call: &ToolCall) -> Result<String> {
        let kind = ToolKind::from_tool_name(&tool_call.name)
            .ok_or_else(|| anyhow!("unknown tool: {}", tool_call.name))?;
        self.authorize(kind)?;
        let args: Value = serde_json::from_str(&tool_call.arguments)
            .unwrap_or(Value::Object(Default::default()));
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

#[cfg(not(unix))] // the unix path renders through `ShellOutcome` instead
fn format_shell_output(output: std::process::Output) -> String {
    format_shell_parts(&output.stdout, &output.stderr, &output.status.to_string())
}

/// Raw result of one sandboxed shell run, before rendering.
#[cfg(unix)]
struct ShellOutcome {
    stdout: Vec<u8>,
    stderr: Vec<u8>,
    status_text: String,
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
    pid_slot: &AtomicI32,
    isolation: OsIsolation,
    writable_roots: &[PathBuf],
) -> Result<ShellOutcome, String> {
    use std::io::Read;
    use std::os::unix::process::CommandExt;

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
    pid_slot.store(child.id() as i32, Ordering::SeqCst);

    // Drain both pipes concurrently so a chatty child can never fill them
    // and deadlock while `wait()` runs.
    let mut stdout_pipe = child.stdout.take().ok_or_else(|| "stdout unavailable".to_string())?;
    let mut stderr_pipe = child.stderr.take().ok_or_else(|| "stderr unavailable".to_string())?;
    let stdout_reader = std::thread::spawn(move || {
        let mut buffer = Vec::new();
        let _ = stdout_pipe.read_to_end(&mut buffer);
        buffer
    });
    let stderr_reader = std::thread::spawn(move || {
        let mut buffer = Vec::new();
        let _ = stderr_pipe.read_to_end(&mut buffer);
        buffer
    });

    let status = child.wait().map_err(|error| format!("waiting on sh: {error}"))?;
    let stdout = stdout_reader.join().unwrap_or_default();
    let stderr = stderr_reader.join().unwrap_or_default();
    Ok(ShellOutcome { stdout, stderr, status_text: status.to_string() })
}

fn format_shell_parts(stdout: &[u8], stderr: &[u8], status_text: &str) -> String {
    let stdout = String::from_utf8_lossy(stdout);
    let stderr = String::from_utf8_lossy(stderr);
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

/// Best-effort SIGKILL for a whole process group. Uses the shell builtin so
/// no extra crates are needed. Group members that escaped into their own
/// session (double-fork / setsid) survive — a documented limitation.
#[cfg(unix)]
async fn kill_process_group(pid: u32) {
    if pid == 0 {
        // `-0` would mean "my own process group" — never kill that.
        return;
    }
    let script = format!("kill -9 -{pid} 2>/dev/null || true");
    let kill = tokio::process::Command::new("sh")
        .arg("-c")
        .arg(&script)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .kill_on_drop(true)
        .output();
    let _ = tokio::time::timeout(Duration::from_secs(2), kill).await;
}

#[cfg(not(unix))]
async fn kill_process_group(_pid: u32) {}

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
        Sandbox::new(cfg)
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
        assert!(sandbox.read_file("escape.txt").await.is_err());
        // Directory symlink: read through it.
        assert!(sandbox.read_file("out/etc/hosts").await.is_err());
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
        assert!(sandbox.read_file("a/b/passwd").await.is_err());
    }

    #[tokio::test]
    async fn reading_nonexistent_paths_fails_cleanly() {
        let dir = unique_root("missing");
        let sandbox = Sandbox::with_root(dir);
        let err = sandbox.read_file("does/not/exist.txt").await.unwrap_err();
        assert!(!err.to_string().contains("panic"));
        // And writing into a non-existent nested directory still works.
        sandbox
            .write_file("does/new/file.txt", "created")
            .await
            .unwrap();
        assert_eq!(
            sandbox
                .read_file("does/new/file.txt")
                .await
                .unwrap(),
            "created"
        );
    }

    #[tokio::test]
    async fn sensitive_files_are_denied_for_every_file_tool() {
        let dir = unique_root("sensitive");
        let sandbox = Sandbox::with_root(dir.clone());
        // dotenv-style names
        assert!(sandbox.read_file(".env").await.is_err());
        assert!(sandbox.write_file(".env", "X=1").await.is_err());
        assert!(sandbox.edit_file(".env", "A", "B").await.is_err());
        assert!(sandbox.read_file("config/.env.local").await.is_err());
        assert!(sandbox.read_file("production.env").await.is_err());
        // key material
        assert!(sandbox.read_file("server.pem").await.is_err());
        assert!(sandbox.write_file("tls/host.key", "x").await.is_err());
        assert!(sandbox.read_file("id_ed25519").await.is_err());
        // git internals: reads of config/credentials, all writes
        assert!(sandbox.read_file(".git/config").await.is_err());
        assert!(sandbox.write_file(".git/hooks/pre-commit", "#!/bin/sh").await.is_err());
        // non-sensitive git internals stay readable
        std::fs::create_dir_all(dir.join(".git")).unwrap();
        std::fs::write(dir.join(".git").join("HEAD"), "ref: refs/heads/main").unwrap();
        assert!(sandbox.read_file(".git/HEAD").await.is_ok());
        // normal files keep working
        sandbox.write_file("normal.txt", "fine").await.unwrap();
        assert_eq!(sandbox.read_file("normal.txt").await.unwrap(), "fine");
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
        assert!(sandbox.read_file("secrets.json").await.is_err());
        assert!(sandbox.read_file("token.txt").await.is_err()); // suffix match
        assert!(sandbox.read_file("anything-else.txt").await.is_ok());
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
        let content = sandbox.read_file("test.txt").await.unwrap();
        assert_eq!(content, "hello");
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
        assert_eq!(cfg.effective_shell_timeout(), Duration::from_secs(MIN_SHELL_TIMEOUT_SECS));
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
        let secret = sandbox.bash("printenv CHATTUI_SECRET_TEST").await.unwrap();
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
        let sandbox = Sandbox::new(cfg);

        let escape = home.join("chattui_iso_escape_probe");
        let _ = std::fs::remove_file(&escape);

        // Write outside the workspace: the kernel must refuse it. The shell
        // still exits normally, so the denial shows up in stderr output.
        let out = sandbox
            .bash(&format!("touch '{}'", escape.display()))
            .await
            .expect("the shell itself must run");
        let lower = out.to_ascii_lowercase();
        assert!(
            lower.contains("permission denied") || lower.contains("operation not permitted"),
            "expected a filesystem denial, got: {out}"
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
        let out = sandbox
            .bash("ping -c1 -W1 127.0.0.1")
            .await
            .expect("the shell itself must run");
        let lower = out.to_ascii_lowercase();
        if lower.contains("not found") {
            eprintln!("skipping: ping is not installed");
            return;
        }
        assert!(
            lower.contains("permitted") || lower.contains("denied"),
            "expected a socket permission failure, got: {out}"
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

    #[tokio::test]
    async fn bash_require_mode_fails_closed_when_isolation_unsupported() {
        // Only meaningful on machines without kernel support; elsewhere the
        // command simply runs (confined), so the test asserts the invariant
        // that holds in both worlds.
        let dir = unique_root("iso-require");
        let mut cfg = SandboxConfig::default();
        cfg.workspace_root = dir.canonicalize().unwrap();
        cfg.os_isolation = OsIsolation::Require;
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
        let err = sandbox.read_file("missing.txt").await.unwrap_err();
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
        assert_eq!(sandbox.read_file("x.txt").await.unwrap(), "y");
        let err = sandbox.bash("echo hi").await.unwrap_err();
        assert!(err.to_string().contains("permission denied"));

        // ask-before-write denies until interactive approval exists.
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
        assert_eq!(sandbox.execute_tool(&read).await.unwrap(), "hello");
        let list = ToolCall::new("3", "list_files", "{}");
        assert!(sandbox.execute_tool(&list).await.unwrap().contains("a.txt"));
    }

    #[test]
    fn output_truncation_never_panics_on_multibyte_boundaries() {
        let mut stdout = Vec::new();
        // Multi-byte characters straddling the truncation point.
        while stdout.len() < MAX_TOOL_OUTPUT_BYTES + 16 {
            stdout.extend_from_slice("é".as_bytes());
        }
        let truncated = format_shell_parts(&stdout, &[], "0");
        assert!(truncated.len() <= MAX_TOOL_OUTPUT_BYTES + "\n... truncated".len());
        assert!(truncated.ends_with("... truncated"));
    }
}
