//! Kernel-enforced process isolation for agent shell commands (Linux only;
//! every other platform is a no-op that reports itself as unsupported).
//!
//! Two independent, unprivileged kernel facilities are applied to a dedicated
//! worker thread *before* the shell is spawned. The shell and every process
//! it creates inherit both, and an unprivileged process cannot remove them:
//!
//! - **Filesystem (Landlock LSM, kernel >= 5.13):** read/execute stay
//!   allowed everywhere — shells, compilers and linkers must read system
//!   files — but *modifying* the filesystem is only permitted under the
//!   workspace root, `/tmp` (plus `$TMPDIR`), `/dev/shm`, `$CARGO_HOME` and
//!   `$CARGO_TARGET_DIR` when set, and write-only character devices
//!   (`/dev/null`, `/dev/zero`, `/dev/full`, `/dev/random`, `/dev/urandom`).
//! - **Syscall filter (seccomp-bpf):** creating non-Unix sockets and the
//!   whole connect/listen/send/receive family is denied; likewise ptrace and
//!   cross-process memory injection (`process_vm_*`), kernel-module loading,
//!   `kexec`, `bpf`, `perf_event_open`, keyring syscalls, namespace creation
//!   (`clone` with namespace flags, `unshare`, `setns`, mount-family
//!   syscalls, `chroot`), `pidfd_getfd`, io_uring, `swapon`/`reboot`.
//!   `clone3` returns `ENOSYS` so libc falls back to plain `clone`, whose
//!   flags *are* inspectable.
//!
//! Enforcement happens inside the kernel against the actual syscalls — it
//! does not depend on inspecting the command string. The remaining limits
//! are documented, not hidden: reading files outside the workspace stays
//! possible by design (secret-read protection remains an application-level
//! policy); `connect()` is denied even for Unix sockets, so tools that talk
//! to local daemons fail; and like every in-process sandbox this trusts the
//! kernel itself.
//!
//! Raw syscalls are used through the existing `libc` dependency instead of
//! higher-level crates, so enabling this feature adds nothing new to the
//! dependency tree.

use std::path::PathBuf;

/// When to apply kernel-level isolation to shell commands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OsIsolation {
    /// Apply it when the kernel supports it; run with a warning when not.
    Auto,
    /// Only run shell commands if isolation can be applied; otherwise
    /// return a structured error (fail closed).
    Require,
    /// Never apply kernel isolation (previous behavior).
    Off,
}

impl OsIsolation {
    pub fn parse(raw: &str) -> Option<OsIsolation> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "auto" => Some(OsIsolation::Auto),
            "require" => Some(OsIsolation::Require),
            "off" => Some(OsIsolation::Off),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            OsIsolation::Auto => "auto",
            OsIsolation::Require => "require",
            OsIsolation::Off => "off",
        }
    }
}

impl Default for OsIsolation {
    fn default() -> Self {
        OsIsolation::Auto
    }
}

/// Outcome of trying to confine the current thread.
#[derive(Debug, Clone)]
pub struct IsolationReport {
    /// Filesystem confinement (Landlock) applied.
    pub fs: bool,
    /// Syscall filter (seccomp) applied.
    pub syscall: bool,
    /// Human-readable reasons for whatever did not apply.
    pub notes: Vec<String>,
}

/// Cheap check used by tests and error messages: probes both facilities on
/// a throwaway thread (restrictions die with the thread).
pub fn os_isolation_supported() -> bool {
    let handle = std::thread::Builder::new()
        .name("isolation-probe".into())
        .spawn(|| restrict_current_thread(&[], &[]));
    match handle {
        Ok(handle) => handle
            .join()
            .map(|report| report.fs && report.syscall)
            .unwrap_or(false),
        Err(_) => false,
    }
}

/// Apply both facilities to the **current thread**. Callers must run this on
/// a thread that was spawned for the purpose: the restrictions are
/// irreversible for that thread, and everything it spawns afterwards
/// inherits them. `writable_roots` get full filesystem access; the entries
/// of `write_only_paths` get write access only.
#[cfg(target_os = "linux")]
pub(crate) fn restrict_current_thread(
    writable_roots: &[PathBuf],
    write_only_paths: &[&'static str],
) -> IsolationReport {
    // Required before both Landlock's restrict_self and seccomp filters; it
    // also stops the sandboxed command from gaining privileges through
    // setuid binaries.
    if let Err(error) = set_no_new_privs() {
        return IsolationReport {
            fs: false,
            syscall: false,
            notes: vec![format!("PR_SET_NO_NEW_PRIVS failed: {error}")],
        };
    }

    let mut report = IsolationReport { fs: true, syscall: true, notes: Vec::new() };
    if let Err(note) = seccomp::apply() {
        report.syscall = false;
        report.notes.push(note);
    }
    if let Err(note) = landlock::restrict(writable_roots, write_only_paths) {
        report.fs = false;
        report.notes.push(note);
    }
    report
}

#[cfg(not(target_os = "linux"))]
pub(crate) fn restrict_current_thread(
    _writable_roots: &[PathBuf],
    _write_only_paths: &[&'static str],
) -> IsolationReport {
    IsolationReport {
        fs: false,
        syscall: false,
        notes: vec!["OS-level isolation is only implemented on Linux".into()],
    }
}

/// Directories the sandboxed shell may *modify*, on top of the workspace.
/// `/tmp` and `/dev/shm` are granted because ordinary builds need scratch
/// space; `CARGO_HOME` / `CARGO_TARGET_DIR` are granted when exported so
/// Rust builds keep working. Documented as part of the security model.
#[cfg(target_os = "linux")]
pub(crate) fn writable_roots(workspace: &std::path::Path) -> Vec<PathBuf> {
    let mut roots = vec![workspace.to_path_buf()];
    roots.push(PathBuf::from("/tmp"));
    roots.push(PathBuf::from("/dev/shm"));
    for name in ["TMPDIR", "CARGO_TARGET_DIR", "CARGO_HOME"] {
        if let Some(value) = std::env::var_os(name) {
            let path = PathBuf::from(value);
            if !path.as_os_str().is_empty() {
                roots.push(path);
            }
        }
    }
    roots
}

#[cfg(not(target_os = "linux"))]
pub(crate) fn writable_roots(_workspace: &std::path::Path) -> Vec<PathBuf> {
    Vec::new()
}

#[cfg(target_os = "linux")]
pub(crate) const WRITE_ONLY_DEVICES: &[&str] = &[
    "/dev/null",
    "/dev/full",
    "/dev/zero",
    "/dev/random",
    "/dev/urandom",
];

/// Non-Linux platforms have no Landlock layer, so there is nothing to
/// grant; the constant exists so callers compile unchanged.
#[cfg(not(target_os = "linux"))]
pub(crate) const WRITE_ONLY_DEVICES: &[&str] = &[];

// ---------------------------------------------------------------------------
// libc plumbing shared by both facilities
// ---------------------------------------------------------------------------

#[cfg(target_os = "linux")]
fn set_no_new_privs() -> Result<(), std::io::Error> {
    const PR_SET_NO_NEW_PRIVS: libc::c_int = 38;
    let rc = unsafe {
        libc::prctl(PR_SET_NO_NEW_PRIVS, 1 as libc::c_ulong, 0, 0, 0)
    };
    if rc != 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Landlock: filesystem confinement via three dedicated syscalls
// ---------------------------------------------------------------------------

#[cfg(target_os = "linux")]
mod landlock {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;
    use std::path::{Path, PathBuf};

    const SYS_LANDLOCK_CREATE_RULESET: libc::c_long = 444;
    const SYS_LANDLOCK_ADD_RULE: libc::c_long = 445;
    const SYS_LANDLOCK_RESTRICT_SELF: libc::c_long = 446;

    const LANDLOCK_CREATE_RULESET_VERSION: libc::c_ulong = 1;
    const LANDLOCK_RULE_PATH_BENEATH: libc::c_ulong = 1;

    // UAPI access-rights bits (stable, include/uapi/linux/landlock.h).
    const EXECUTE: u64 = 1 << 0;
    const WRITE_FILE: u64 = 1 << 1;
    const READ_FILE: u64 = 1 << 2;
    const READ_DIR: u64 = 1 << 3;
    const REMOVE_DIR: u64 = 1 << 4;
    const REMOVE_FILE: u64 = 1 << 5;
    const MAKE_CHAR: u64 = 1 << 6;
    const MAKE_DIR: u64 = 1 << 7;
    const MAKE_REG: u64 = 1 << 8;
    const MAKE_SOCK: u64 = 1 << 9;
    const MAKE_FIFO: u64 = 1 << 10;
    const MAKE_BLOCK: u64 = 1 << 11;
    const MAKE_SYM: u64 = 1 << 12;
    /// Rename/link across directories (kernel >= 5.19, ABI v2).
    const REFER: u64 = 1 << 13;
    /// truncate()/ftruncate() (kernel >= 6.2, ABI v3); without handling it,
    /// those two calls would stay allowed everywhere — a quiet escape hatch.
    const TRUNCATE: u64 = 1 << 14;

    /// Read/execute everywhere: what shells, compilers and linkers need.
    const READ_ONLY_ACCESS: u64 = EXECUTE | READ_FILE | READ_DIR;

    /// All rights up to the kernel's ABI version. Rights that are
    /// "handled but not granted" are the ones Landlock denies, so the
    /// handled set must be as complete as the kernel supports.
    fn handled_access_fs(abi: u64) -> u64 {
        let mut bits = EXECUTE
            | WRITE_FILE
            | READ_FILE
            | READ_DIR
            | REMOVE_DIR
            | REMOVE_FILE
            | MAKE_CHAR
            | MAKE_DIR
            | MAKE_REG
            | MAKE_SOCK
            | MAKE_FIFO
            | MAKE_BLOCK
            | MAKE_SYM;
        if abi >= 2 {
            bits |= REFER;
        }
        if abi >= 3 {
            bits |= TRUNCATE;
        }
        bits
    }

    #[repr(C)]
    struct RulesetAttr {
        handled_access_fs: u64,
    }

    #[repr(C)]
    struct PathBeneathAttr {
        allowed_access: u64,
        parent_fd: libc::c_int,
        _reserved: u32,
    }

    /// Ask the kernel which Landlock ABI it implements. An error means the
    /// facility is unavailable (pre-5.13 kernel, or Landlock not compiled
    /// in / disabled at boot); the caller turns that into a plain note.
    fn detect_abi() -> Result<u64, String> {
        let rc = unsafe {
            libc::syscall(
                SYS_LANDLOCK_CREATE_RULESET,
                std::ptr::null::<RulesetAttr>(),
                0 as libc::c_ulong,
                LANDLOCK_CREATE_RULESET_VERSION,
            )
        };
        if rc < 0 {
            let error = std::io::Error::last_os_error();
            return Err(match error.raw_os_error() {
                Some(libc::ENOSYS) => {
                    "Landlock unsupported: kernel is older than 5.13".to_string()
                }
                Some(libc::EOPNOTSUPP) | Some(libc::EINVAL) => {
                    "Landlock unsupported: not enabled in this kernel".to_string()
                }
                _ => format!("Landlock unsupported: {error}"),
            });
        }
        Ok(rc as u64)
    }

    fn open_path_fd(path: &Path) -> Result<libc::c_int, String> {
        let c_path = CString::new(path.as_os_str().as_bytes().to_vec())
            .map_err(|_| format!("path contains NUL: {}", path.display()))?;
        let fd = unsafe { libc::open(c_path.as_ptr(), libc::O_PATH | libc::O_CLOEXEC) };
        if fd < 0 {
            return Err(format!("{}: {}", path.display(), std::io::Error::last_os_error()));
        }
        Ok(fd)
    }

    fn add_rule(ruleset_fd: libc::c_int, parent_fd: libc::c_int, access: u64) -> Result<(), String> {
        let attr = PathBeneathAttr {
            allowed_access: access,
            parent_fd,
            _reserved: 0,
        };
        let rc = unsafe {
            libc::syscall(
                SYS_LANDLOCK_ADD_RULE,
                ruleset_fd,
                LANDLOCK_RULE_PATH_BENEATH,
                &attr as *const PathBeneathAttr,
                0 as libc::c_ulong,
            )
        };
        if rc < 0 {
            return Err(format!("Landlock add_rule failed: {}", std::io::Error::last_os_error()));
        }
        Ok(())
    }

    /// Confine the current thread: read/execute everywhere, modifications
    /// only under `writable_roots`; `write_only_paths` get write access
    /// without directory rights.
    pub(super) fn restrict(
        writable_roots: &[PathBuf],
        write_only_paths: &[&'static str],
    ) -> Result<(), String> {
        let abi = detect_abi()?;
        let handled = handled_access_fs(abi);
        let ruleset_attr = RulesetAttr { handled_access_fs: handled };

        let ruleset_fd = unsafe {
            libc::syscall(
                SYS_LANDLOCK_CREATE_RULESET,
                &ruleset_attr as *const RulesetAttr,
                std::mem::size_of::<RulesetAttr>() as libc::c_ulong,
                0 as libc::c_ulong,
            )
        };
        if ruleset_fd < 0 {
            let error = std::io::Error::last_os_error();
            return Err(match error.raw_os_error() {
                Some(libc::E2BIG) => "Landlock ruleset too large for this kernel".to_string(),
                _ => format!("Landlock ruleset creation failed: {error}"),
            });
        }
        let ruleset_fd = ruleset_fd as libc::c_int;

        // Read/execute everywhere.
        let result = open_path_fd(Path::new("/"))
            .and_then(|root_fd| add_rule(ruleset_fd, root_fd, READ_ONLY_ACCESS));

        // Full access under each writable root.
        let result = result.and_then(|_| {
            for root in writable_roots {
                let fd = open_path_fd(root)?;
                add_rule(ruleset_fd, fd, handled)?;
            }
            Ok(())
        });

        // Write-only device nodes.
        let result = result.and_then(|_| {
            for device in write_only_paths {
                let fd = open_path_fd(Path::new(device))?;
                add_rule(ruleset_fd, fd, WRITE_FILE)?;
            }
            Ok(())
        });

        let restrict_result = result.and_then(|_| unsafe {
            let rc = libc::syscall(SYS_LANDLOCK_RESTRICT_SELF, ruleset_fd, 0 as libc::c_ulong);
            if rc < 0 {
                Err(format!("Landlock restrict failed: {}", std::io::Error::last_os_error()))
            } else {
                Ok(())
            }
        });

        unsafe { libc::close(ruleset_fd) };
        restrict_result
    }
}

// ---------------------------------------------------------------------------
// seccomp-bpf: deny the dangerous syscall surface
// ---------------------------------------------------------------------------

#[cfg(target_os = "linux")]
mod seccomp {
    /// Syscall numbers that differ between architectures (classic-BPF
    /// `seccomp_data` offsets: nr @ 0, arch @ 4, args[0] low word @ 16).
    struct SysNums {
        socket: i64,
        connect: i64,
        accept: i64,
        accept4: i64,
        sendto: i64,
        recvfrom: i64,
        sendmsg: i64,
        recvmsg: i64,
        recvmmsg: i64,
        sendmmsg: i64,
        shutdown: i64,
        bind: i64,
        listen: i64,
        clone: i64,
        ptrace: i64,
        mount: i64,
        umount2: i64,
        chroot: i64,
        swapon: i64,
        swapoff: i64,
        reboot: i64,
        init_module: i64,
        delete_module: i64,
        kexec_load: i64,
        add_key: i64,
        request_key: i64,
        keyctl: i64,
        perf_event_open: i64,
        process_vm_readv: i64,
        process_vm_writev: i64,
        unshare: i64,
        setns: i64,
        bpf: i64,
        userfaultfd: i64,
    }

    #[cfg(target_arch = "x86_64")]
    const AUDIT_ARCH: u32 = 0xC000_003E; // x86-64, little-endian, 64-bit
    #[cfg(target_arch = "x86_64")]
    const NR: SysNums = SysNums {
        socket: 41,
        connect: 42,
        accept: 43,
        sendto: 44,
        recvfrom: 45,
        sendmsg: 46,
        recvmsg: 47,
        shutdown: 48,
        bind: 49,
        listen: 50,
        clone: 56,
        ptrace: 101,
        mount: 165,
        umount2: 166,
        chroot: 161,
        swapon: 167,
        swapoff: 168,
        reboot: 169,
        init_module: 175,
        delete_module: 176,
        kexec_load: 246,
        add_key: 248,
        request_key: 249,
        keyctl: 250,
        unshare: 272,
        accept4: 288,
        perf_event_open: 298,
        recvmmsg: 299,
        process_vm_readv: 310,
        process_vm_writev: 311,
        bpf: 321,
        userfaultfd: 323,
        sendmmsg: 307,
        setns: 308,
    };

    #[cfg(target_arch = "aarch64")]
    const AUDIT_ARCH: u32 = 0xC000_00B7; // arm64, little-endian, 64-bit
    #[cfg(target_arch = "aarch64")]
    const NR: SysNums = SysNums {
        socket: 198,
        connect: 203,
        accept: 202,
        sendto: 206,
        recvfrom: 207,
        sendmsg: 211,
        recvmsg: 212,
        shutdown: 210,
        bind: 200,
        listen: 201,
        clone: 220,
        ptrace: 117,
        mount: 40,
        umount2: 39,
        chroot: 51,
        swapon: 224,
        swapoff: 225,
        reboot: 142,
        init_module: 105,
        delete_module: 106,
        kexec_load: 104,
        add_key: 217,
        request_key: 218,
        keyctl: 219,
        unshare: 266,
        accept4: 242,
        perf_event_open: 241,
        recvmmsg: 243,
        process_vm_readv: 270,
        process_vm_writev: 271,
        bpf: 386,
        userfaultfd: 282,
        sendmmsg: 269,
        setns: 268,
    };

    // Syscalls >= 424 share one table across architectures.
    const UNIFIED_DENY: &[i64] = &[
        425, // io_uring_setup
        426, // io_uring_enter
        428, // open_tree
        429, // move_mount
        430, // fsopen
        431, // fsconfig
        432, // fsmount
        433, // fspick
        438, // pidfd_getfd (steal fds from other processes)
        442, // mount_setattr
    ];
    const CLONE3: i64 = 435;

    const AF_UNIX: u32 = 1;
    const AF_NETLINK: u32 = 16;

    /// Namespace-creation flags for clone(): CLONE_NEWNS | NEWUTS | NEWIPC
    /// | NEWUSER | NEWPID | NEWNET | NEWCGROUP.
    const CLONE_NAMESPACE_FLAGS: u32 = 0x7E02_0000;

    // Classic BPF instruction encoding.
    const BPF_LD_ABS_W: u16 = 0x20; // LD | W | ABS
    const BPF_JEQ: u16 = 0x15; // JMP | JEQ | K
    const BPF_JSET: u16 = 0x45; // JMP | JSET | K
    const BPF_RET_K: u16 = 0x06; // RET | K
    const SECCOMP_RET_ERRNO: u32 = 0x0005_0000;
    const SECCOMP_RET_ALLOW: u32 = 0x7FFF_0000;
    const EPERM: u32 = 1;
    const ENOSYS: u32 = 38;

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct SockFilter {
        code: u16,
        jt: u8,
        jf: u8,
        k: u32,
    }

    const fn ld_abs(k: u32) -> SockFilter {
        SockFilter { code: BPF_LD_ABS_W, jt: 0, jf: 0, k }
    }
    const fn jeq(k: u32, jt: u8, jf: u8) -> SockFilter {
        SockFilter { code: BPF_JEQ, jt, jf, k }
    }
    const fn jset(k: u32, jt: u8, jf: u8) -> SockFilter {
        SockFilter { code: BPF_JSET, jt, jf, k }
    }
    const fn ret_errno(errno: u32) -> SockFilter {
        SockFilter { code: BPF_RET_K, jt: 0, jf: 0, k: SECCOMP_RET_ERRNO | errno }
    }
    const fn ret_allow() -> SockFilter {
        SockFilter { code: BPF_RET_K, jt: 0, jf: 0, k: SECCOMP_RET_ALLOW }
    }

    fn build_filter() -> Vec<SockFilter> {
        let mut f: Vec<SockFilter> = Vec::new();
        // if (arch != AUDIT_ARCH) deny
        f.push(ld_abs(4)); // arch
        f.push(jeq(AUDIT_ARCH, 1, 0)); // match -> skip the deny ret
        f.push(ret_errno(EPERM));
        // load syscall number
        f.push(ld_abs(0)); // nr

        // Simple denies: JEQ nr -> RET EPERM (each pair, jf skips the ret).
        let denies: Vec<i64> = [
            NR.connect,
            NR.bind,
            NR.listen,
            NR.accept,
            NR.accept4,
            NR.sendto,
            NR.recvfrom,
            NR.sendmsg,
            NR.recvmsg,
            NR.recvmmsg,
            NR.sendmmsg,
            NR.shutdown,
            NR.ptrace,
            NR.mount,
            NR.umount2,
            NR.chroot,
            NR.swapon,
            NR.swapoff,
            NR.reboot,
            NR.init_module,
            NR.delete_module,
            NR.kexec_load,
            NR.add_key,
            NR.request_key,
            NR.keyctl,
            NR.perf_event_open,
            NR.process_vm_readv,
            NR.process_vm_writev,
            NR.unshare,
            NR.setns,
            NR.bpf,
            NR.userfaultfd,
        ]
        .iter()
        .chain(UNIFIED_DENY.iter())
        .copied()
        .collect();
        for nr in denies {
            f.push(jeq(nr as u32, 0, 1));
            f.push(ret_errno(EPERM));
        }

        // clone with namespace-creation flags -> EPERM (fork/threads stay fine)
        f.push(jeq(NR.clone as u32, 0, 3)); // miss -> skip 3 insns
        f.push(ld_abs(16)); // args[0] low 32 bits = flags
        f.push(jset(CLONE_NAMESPACE_FLAGS, 0, 1));
        f.push(ret_errno(EPERM));

        // clone3 -> ENOSYS so libc falls back to clone() whose flags we see
        f.push(jeq(CLONE3 as u32, 0, 1));
        f.push(ret_errno(ENOSYS));

        // socket(): allow only AF_UNIX / AF_NETLINK
        //   JEQ socket: hit -> fall through; miss -> jump 5 to RET_ALLOW
        f.push(jeq(NR.socket as u32, 0, 4));
        f.push(ld_abs(16)); // args[0] low 32 bits = address family
        f.push(jeq(AF_UNIX, 2, 0));
        f.push(jeq(AF_NETLINK, 1, 0));
        f.push(ret_errno(EPERM));

        f.push(ret_allow());
        f
    }

    pub(super) fn apply() -> Result<(), String> {
        let filter = build_filter();
        let fprog = libc::sock_fprog {
            len: filter.len() as u16,
            filter: filter.as_ptr() as *mut libc::sock_filter,
        };
        // The filter's lifetime covers the prctl call; the kernel copies it.
        let rc = unsafe {
            libc::prctl(
                libc::PR_SET_SECCOMP, // 22
                libc::SECCOMP_MODE_FILTER as libc::c_ulong, // 2
                &fprog as *const libc::sock_fprog,
                0 as libc::c_ulong,
                0 as libc::c_ulong,
            )
        };
        if rc != 0 {
            let error = std::io::Error::last_os_error();
            return Err(match error.raw_os_error() {
                Some(libc::ENOSYS) => "seccomp unsupported: kernel lacks it".to_string(),
                Some(libc::EINVAL) => "seccomp filter rejected by kernel".to_string(),
                Some(libc::EACCES) => {
                    "seccomp denied: NO_NEW_PRIVS could not be set".to_string()
                }
                _ => format!("seccomp apply failed: {error}"),
            });
        }
        Ok(())
    }
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn unique_root(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("chatTUI_iso_{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir.canonicalize().unwrap()
    }

    /// Runs `body` on a fresh thread that gets confined first; restrictions
    /// die with the thread so the rest of the test process is unaffected.
    fn assert_inside_confined_thread<F: FnOnce() + Send + 'static>(
        workspace: PathBuf,
        body: F,
    ) -> Result<(), String> {
        let (tx, rx) = std::sync::mpsc::channel::<Result<(), String>>();
        let writer = std::thread::Builder::new()
            .name("confined-probe".into())
            .spawn(move || {
                let writable = vec![workspace];
                let report = restrict_current_thread(&writable, WRITE_ONLY_DEVICES);
                if !report.fs || !report.syscall {
                    tx.send(Err(format!(
                        "isolation did not apply: {}",
                        report.notes.join("; ")
                    )))
                    .ok();
                    return;
                }
                body();
                tx.send(Ok(())).ok();
            })
            .map_err(|e| e.to_string())?;
        writer.join().map_err(|_| "confined thread panicked".to_string())?;
        rx.recv().map_err(|_| "confined thread hung".to_string())
    }

    #[test]
    fn isolation_probe_reports_supported_kernels() {
        // No assertion about the result — this machine either supports the
        // facilities or it does not; `supported()` must not crash and must
        // agree with a direct probe.
        let supported = os_isolation_supported();
        let direct = {
            let handle = std::thread::Builder::new()
                .spawn(|| {
                    let report = restrict_current_thread(&[], &[]);
                    report.fs && report.syscall
                })
                .unwrap();
            handle.join().unwrap()
        };
        assert_eq!(supported, direct);
    }

    #[test]
    fn confined_thread_can_write_workspace_and_read_system_files() {
        let ws = unique_root("allow");
        assert_inside_confined_thread(ws.clone(), move || {
            std::fs::write(ws.join("inside.txt"), b"ok").expect("workspace write must work");
            assert_eq!(std::fs::read_to_string(ws.join("inside.txt")).unwrap(), "ok");
            std::fs::read_to_string("/etc/hosts").expect("reads must stay allowed");
        })
        .unwrap();
        let _ = std::fs::remove_dir_all(&ws);
    }

    #[test]
    fn confined_thread_cannot_write_outside_granted_roots() {
        let ws = unique_root("deny");
        assert_inside_confined_thread(ws.clone(), move || {
            // Only the workspace is granted in this probe, so /tmp —
            // normally writable in real runs — is the ungranted target.
            let outside = std::env::temp_dir().join(format!(
                "chattui_iso_escape_{}",
                std::process::id()
            ));
            let result = std::fs::write(&outside, b"escape");
            assert!(result.is_err(), "write outside granted roots must be denied");
            assert!(!outside.exists());
        })
        .unwrap();
        let _ = std::fs::remove_dir_all(&ws);
    }

    #[test]
    fn confined_thread_cannot_create_network_sockets() {
        let ws = unique_root("net");
        assert_inside_confined_thread(ws, move || {
            let result = std::net::TcpStream::connect("1.1.1.1:80");
            assert!(result.is_err(), "network connect must be denied");
        })
        .unwrap();
    }

    #[test]
    fn isolation_config_parses_all_modes() {
        assert_eq!(OsIsolation::parse("auto"), Some(OsIsolation::Auto));
        assert_eq!(OsIsolation::parse("require"), Some(OsIsolation::Require));
        assert_eq!(OsIsolation::parse("off"), Some(OsIsolation::Off));
        assert_eq!(OsIsolation::parse("REQUIRE"), Some(OsIsolation::Require));
        assert_eq!(OsIsolation::parse("yolo"), None);
        assert_eq!(OsIsolation::parse(""), None);
        assert_eq!(OsIsolation::default(), OsIsolation::Auto);
    }
}
