# Improving chatTUI's agent — incremental, not a rewrite

*Companion to the code in `src/`, references current line numbers.*

**The short version:** the useful parts of a well-built terminal agent are mechanisms, not
crates. Every one that matters here can be grafted onto the architecture chatTUI already has —
the `StreamEvent` channel, `App`'s state, the `Sandbox`, the provider backends — without
replacing any of it. Each change is independently shippable and leaves the app working.

| # | Change | Benefit in one line | Status |
|---|---|---|---|
| 0 | Report an incomplete response instead of hiding it | A truncated answer no longer looks finished | **done** |
| 0b | Separate the first-token wait from the between-chunks idle limit | A 30-minute think is not mistaken for a dead connection | **done** |
| 12 | Remove the `/target` duplicate of `/sandbox` | One way to set the workspace | **done** |
| 13 | A command can never touch the user's terminal | Running a TUI no longer destroys the session | **done** |
| 3 | Persist a real tool status, not just text | Failures survive a restart instead of rendering as successes | **done** |
| 1 | Run tools in a spawned task, results back through the existing channel | TUI never freezes; `Esc` stops the round | **done** |
| 2 | `apply_patch` tool (patch format, atomic, multi-file) | Edits stop failing on whitespace; multi-file changes are transactional | **done** |
| 4 | Head **+ tail** output truncation | The error at the end of a build is no longer the part we cut | **done** |
| 5 | Make `ask-before-*` actually ask, plus `/approve auto\|manual` | A safe mode people will use, instead of `full-auto` by default | **done** |
| 6 | Escalate on sandbox denial instead of erroring | A denial becomes a question, not a 3-round loop the guard has to kill | **done** |
| 11 | `ask_user` tool so the model can ask up to 5 questions mid-task | The agent stops guessing at decisions only the user can make | **done** |
| 7 | Real token accounting + auto-compaction | The "no fixed step cap" claim stops being a context-overflow time bomb | **done** |
| 8 | Generate the system prompt from policy | The prompt cannot drift from what is actually enforced | **done** |
| 9 | Round-trip reasoning per provider | Fixes silent quality loss on thinking models during tool use | **done** |
| 10 | `exec_command` sessions + `yield_time_ms` | Long builds, dev servers, REPLs instead of a 30 s kill timer | **done** |

The rest of this document is the evidence: §1 is what breaks today and where, §2 is the set of
mechanisms worth borrowing, §3 is the incremental plan, §4 is what to deliberately skip, §5 is
verification.

---

## 1. What we have today (and where it breaks)

The agent mechanism is spread across four files: `src/app/streaming.rs`, `src/sandbox/mod.rs`,
`src/tools/mod.rs`, plus agent state on `App` (`src/app/mod.rs`).

```text
main::run ──> App::receive_token()                    src/main.rs:66
                 ├── drain StreamEvent{Delta,ToolCall,Notice,Retry,Error}
                 ├── execute tools inline, on the UI task   streaming.rs:295
                 ├── write tool results to SessionManager    streaming.rs:310
                 ├── run agent health guards                 streaming.rs:340-390
                 └── self.start_stream()  ← the "loop"       streaming.rs:400
```

### Defects that follow from that shape

| # | Defect | Where | Consequence |
|---|---|---|---|
| 1 | **Tools run on the UI task.** `sandbox.execute_tool(tc).await` is awaited inside `receive_token()`, which `main::run` awaits before drawing. | `streaming.rs:295`, `main.rs:66-74` | A 30 s `bash` (default `sandbox.shell_timeout_secs`) freezes the whole TUI. `Esc` cannot cancel it — it only aborts the *model* task (`app/mod.rs:337`). |
| 2 | ~~A response that stops early is indistinguishable from one that finished.~~ | `api/providers/*` | **Fixed** — see change 0. |
| 3 | ~~A long "thinking" pause before the first token trips the idle timeout, and the retry restarts the wait.~~ | `api/providers/*` | **Fixed** — see change 0b. |
| 4 | ~~A command the agent runs could take over the user's terminal.~~ | `sandbox/mod.rs` | **Fixed** — see change 13. |
| 5 | ~~Tool results were stored without a status, and replay hardcoded `is_error: false`.~~ | `session/manager.rs`, `app/mod.rs:287` | **Fixed** — see change 3. |
| 6 | ~~**No prompt-level approval.** `AskBeforeWrite` / `AskBeforeShell` *deny*.~~ | `sandbox/permissions.rs` | **Fixed** — see change 5. |
| 7 | **No escalation.** A sandbox denial is returned to the model as an error string and the model retries the same thing. | `sandbox/mod.rs`, `streaming.rs:299` | Exactly the loop the stagnation guard (`AGENT_STAGNATION_LIMIT = 3`, `app/mod.rs:51`) was invented to stop. The guard is a symptom, not a feature. |
| 8 | **Edits are exact-match string surgery.** `edit_file` requires `old_string` to match byte-for-byte and to be unique; `write_file` replaces whole files. | `tools/mod.rs`, `sandbox/mod.rs:372` | Whitespace drift, repeated identical lines and large files all fail or cost wasted round trips. Nothing is transactional: a multi-file change is N independent writes, each able to fail halfway. |
| 9 | **Output truncation is head-only, 20 kB.** | `sandbox/mod.rs:84` | The failing assertion at the *end* of a build is exactly what gets cut. |
| 10 | **No context accounting and no compaction.** `context_summary()` is `chars / 4`. | `app/mod.rs:442` | The README's "no fixed step cap — the agent keeps working while it makes progress" only holds until the history overflows the model's window, after which every turn fails. |
| 11 | **Reasoning is stripped on replay, unconditionally.** | `streaming.rs:45` | Wrong for providers whose thinking blocks must be echoed back during tool use; silent quality loss and/or 400s. |
| 12 | **The system prompt is hand-written and duplicated.** It restates facts the enforcement code also encodes, and keeps them in sync by hand. | `streaming.rs:143` | The prompt can disagree with what is actually enforced; a test asserts the tool prose still contains the right words. |
| 13 | **Tool surface is prose + string dispatch.** Args pulled out with `args["path"].as_str().ok_or("missing path")`; two parallel `ToolDefinition` types. | `tools/mod.rs`, `sandbox/mod.rs:575-605` | No type checking, no shared metadata (read-only? parallel-safe? approval class?). |
| 14 | **Every round re-serializes everything.** The "loop" is a `start_stream()` rebuild: re-read the session, re-run repairs, re-send the whole tool list. | `streaming.rs:27-160` | Cost grows quadratically over a long run. |

None of this is bad code. It is a chat client that grew an agent, one reasonable patch at a
time.

---

## 2. The mechanisms worth borrowing

Nine ideas, of which I'd take five, adapt three, and skip the rest.

### 2.1 A protocol, not a function call

A submission queue (UI → engine) and an event queue (engine → UI) as the only interface, with
the engine running in its own task and the TUI as one client among several. Vocabulary is
precise: a **turn** is one model request plus the tool work it causes; a **task** is a series
of turns in response to one user input; at most one task runs at a time. The same engine then
drives a TUI, a headless `--json` mode, and an IDE extension.

### 2.2 Items, not messages

Three representations with one job each: a provider-neutral **wire** history, a canonical
**item** type for the UI and for persistence (user message, assistant message, reasoning,
command execution, file change, plan, compaction), and provider payloads built at the boundary
and nowhere else. Anything stored or rendered is an item with an id, so the UI can stream a
started/completed pair instead of inferring state from text.

### 2.3 Two tools that matter, both typed

One shell tool — `cmd`, `workdir`, `tty`, `yield_time_ms`, `max_output_tokens`, a per-command
sandbox override, plus a way to write to a still-running session — and one edit tool that takes
a **patch**, not JSON: `*** Begin Patch`, `*** Add File`, `*** Update File`, `@@` context
hunks, `*** End Patch`, applied atomically across many files with context-anchored matching.

Three properties matter more than the tools:

- **Commands outlive the round.** `yield_time_ms` returns partial output plus a session id for a
  still-running process, so a long build, a dev server or a REPL is possible without a kill
  timer. Output is buffered head **and tail**, 50/50, with an omission marker.
- **Edits are patches.** The model gets a diff back, so the transcript can show what changed
  and an approval prompt can preview it.
- **The tool surface is per-model metadata**, not a global: capable models get patches, weaker
  ones get simpler edits.

### 2.4 Approval and sandbox are separate axes, and denial is not the end

A *sandbox policy* says what the kernel will allow (read-only / workspace-write with writable
roots and a network switch / full access). An *approval policy* says when a human is asked. The
two collapse into one verdict — auto-approve, ask, or reject with a reason — and the
orchestration is always the same:

```text
assess → approval (maybe) → run sandboxed → if the kernel denied:
        ask once, run again unsandboxed, cache the decision for this command prefix
```

That single rule is why a restrictive default can still be pleasant: the boundary is enforced
first, and crossing it is a question rather than a failure.

### 2.5 A context budget that is managed, not hoped for

Token counts come back as events; compaction triggers before the window fills and replaces old
history with a summary **item**, so it is visible, replayable and attributable. The model can
also ask how much budget it has left, which lets it wrap up instead of hitting the wall
mid-edit.

### 2.6 A per-turn diff

Every file change in a turn accumulates into one diff event. "Here is what this turn changed"
is first-class, not something you reconstruct.

### 2.7 Append-only session files

One JSON line per item: crash cost is the last line, `jq` works on them, and resume replays
items rather than trusting a migration path.

### 2.8 Instructions and environment as injected context

A project instructions file discovered from the working directory upward, plus an environment
block (cwd, git state, platform, sandbox and approval policy), composed per turn. Prompt text
lives in one place, generated from the same values the enforcement reads.

### 2.9 Command policy for *fewer prompts*, not for safety

Parsed commands matched against allow/prompt/forbid prefix rules, so `git status` can be
auto-approved forever while `git push` still asks. Explicitly a usability layer, never a
security boundary.

---

## 3. The changes, in dependency order

Each is independently shippable: the app builds, runs and keeps its tests green after every
one. Nothing here needs a new crate, a protocol layer, or a rewrite of `App`.

### 0. Report an incomplete response — *done*

**Was:** a provider that closed the socket mid-answer made the backend return success. The
OpenAI-compatible path set `stream_ended = true` on `Ok(None)`, drained leftover tool calls and
returned `Ok(())`; `finish_reason` was only ever inspected for `"tool_calls"`, so `"length"`
(the model hit its output cap — the commonest cause of "it just stopped mid-sentence") and
`"content_filter"` were ignored. Anthropic never looked at `message_stop` at all. A truncated
reply was therefore indistinguishable from a complete one.

**Now:** each backend tracks whether the provider ended the stream the way it intended — its
end-of-stream sentinel *or* a finish reason, since gateways differ in which they send — and one
shared classifier (`api/sse.rs::abnormal_stream_end`) turns the ending into a transcript notice:
connection closed early, output limit reached, or content filter. If *nothing at all* arrived,
that is a transient failure worth retrying; if output arrived, it is reported and the partial
answer is kept. Tool calls are emitted from a single exit path so the ending is classified in
exactly one place.

### 0b. First-token wait ≠ idle wait — *done*

**Was:** one 90 s idle limit applied to every chunk, including the first. A gateway that buffers
a long chain of thought and sends nothing was declared dead at 90 s, classified transient, and
**retried three times** — each retry restarting the entire think — before falling back to a
different model mid-task.

**Now:** two windows. Until the model produces anything, the limit is
`first_token_timeout_secs` (1800 by default, configurable); once output starts, the tight
`idle_timeout_secs` applies between chunks. Neither caps total response time. A first-token
timeout after a long wait is reported as final rather than retried, because a retry would throw
the thinking away and start the same wait again.

### 12. Remove `/target` — *done*

It was defined in the command list and dispatched to the identical handler as `/sandbox`
(`"/sandbox" | "/target" =>`). Two lines removed; no test or README reference existed.

### 13. A command can never touch the user's terminal — *done*

**Was:** the spawn did `setpgid(0, 0)` — a new process *group*, but the same **session**, so the
child kept chatTUI's controlling terminal. stdin was `/dev/null` and stdout/stderr were pipes,
which looks safe and isn't: terminal libraries enable raw mode through `/dev/tty`, not stdin,
and that open succeeded. So a model-run TUI put the *user's* terminal into raw mode and the
alternate screen, and the SIGKILL at the timeout meant it never restored either. There was also
no control-sequence stripping anywhere, so captured escape bytes could be re-emitted from the
transcript.

**Now, in four layers:**

1. `setsid()` in the existing `pre_exec` — the child gets its own session and **no controlling
   terminal**, so `/dev/tty` cannot be opened. Interactive programs fail in milliseconds instead
   of hijacking the session. `setpgid` stays as the fallback, and the child is still its own
   process-group leader (`pgid == pid`), so the timeout kill takes down the whole tree.
2. `sandbox/ansi.rs` strips CSI/OSC/DCS sequences and control characters from captured output
   before it becomes a tool result — so the transcript physically cannot inject terminal
   control. Character-based, not byte-based: `0x9B` (8-bit CSI) is also a valid UTF-8
   continuation byte, and a byte scan would corrupt multi-byte text.
3. `main.rs::reassert_terminal()` re-applies raw mode, cursor visibility, bracketed paste,
   mouse-capture-off and colour reset after any tool call. Idempotent, and the next frame
   repaints everything.
4. The tool description and the timeout error both state that there is no terminal, so the model
   stops reaching for interactive programs — and a timeout with no output says so explicitly.

### 3. Persist a real tool status — *done*

**Was:** a tool result was stored as `{role:"tool", content, tool_call_id}` with no status, and
`rebuild_cells` hardcoded `is_error: false` — so after a restart every failed tool call rendered
as a success.

**Now:** `Message.is_error: Option<bool>`, saved with the result (including the "interrupted"
backfill) and used on replay. `Option` with a serde default, so sessions saved before this
change still load and are treated as successes rather than marked broken.

### 1. Run tools off the UI task — *done*

A tool round now runs on its own task and reports back through the same event channel the
tokens use, via two new events: `ToolResult { id, content, is_error }` per call and
`ToolRoundDone { round_key, any_error, all_error }` when the round finishes. `receive_token`
returns as soon as the round is handed over, so the draw loop never waits on a command.

The health guards stayed on `App`, where their counters live: the task fingerprints the round
(sorted `name\0arguments`) and sends the verdicts back, and `finish_tool_round` applies the
stagnation, consecutive-failure and ceiling guards exactly as before.

Interrupt is now real: `interrupt()` aborts the tool task and backfills an explicit
"interrupted" result for every call that never reported, so the assistant/tool pairing in
history stays valid. The one honest limit — a command *already running* is not killed by
`Esc`; it stays bounded by the shell timeout and its process group — is documented in the
code and the README.

### 2. `apply_patch` — *done*

A single `patch` string argument in the format above: parse → validate **all** hunks → write all
files or none. Keep `read_file`/`list_files`; keep `write_file` behind a config flag for models
that fumble patches. Context-anchored edits survive drift, a multi-file change becomes one
atomic operation, and the parsed diff feeds both the transcript and change 5's approval preview.

*Adaptation:* a grammar-constrained freeform tool is a single-vendor API feature. chatTUI speaks
Chat Completions for OpenAI *and every gateway*, plus two other vendors, so the patch travels as
one string field on a normal function call: same format, no grammar enforcement.

### 4. Head + tail truncation — *done*

`keep_head_and_tail` in `sandbox/mod.rs` splits the 20 kB budget in half, cuts both windows on
line boundaries where it can, and states how many bytes were dropped. Both cut points are
backed off to char boundaries, because tool output is arbitrary bytes from an arbitrary program.
The failing assertion is at the end of a log; head-only truncation deleted it.

### 5. Make `ask-before-*` actually ask — *done*

The permission layer now returns three answers instead of two: `Gate::Allow`, `Gate::Deny`
(the mode rules it out — read-only cannot write, workspace-write cannot shell) and `Gate::Ask`
(the user decides). Nothing runs before that verdict, so a call can never start and then be
refused.

How a prompt works:

* `run_tool_round` asks the gate about each call *before* executing it. On `Ask` it sends
  `StreamEvent::ApprovalNeeded` and stops, then reports the round as
  `ToolRoundDone { suspended: true }` — parked, not finished.
* The status row replaces the spinner with `⚠ Approve? <what it would do> (y allow • a always •
  n deny)`. The full call is already in the transcript, so the row stays one line. `Esc` means
  no. While a prompt is up, `y`/`a`/`n` are answers, not typing.
* Answering calls `resolve_approval`, which records the decision and **resumes the same round**:
  `run_tool_round` is spawned again with the round's calls plus the ids that already reported, so
  saying yes to the third call does not re-run the first two.
* A denial is written back as a tool result ("the user denied this… do not retry, ask instead")
  so the model hears about it instead of looping on the same call.
* `a` remembers the decision under an *allowance key* — the tool name for file tools, the first
  two words for shell. Approving one `git push` covers the next one, which is what keeps a
  careful mode usable; approving `rm -rf` does not approve `cargo`. `/approve reset` forgets them
  all, and switching modes forgets them too.
* `/approve` toggles, `/approve auto` = `full-auto`, `/approve manual` = `ask-before-write`
  (reads free, writes and shell ask). Changing the mode clears the remembered approvals.

*Adaptation:* the row shows a one-line summary rather than a rendered diff. A diff needs a
scrollable region and its own key handling; the transcript already shows the call, and the
patch itself lands there as a tool cell. The parsed patch from change 2 is what makes that cell
readable.

### 6. Escalate on sandbox denial — *done*

Enforce first, then ask: run sandboxed → on kernel denial, ask once → if allowed, re-run that
call unrestricted and cache the decision. A restrictive sandbox stops being a dead end, and the
stagnation guard goes back to catching real loops.

The denial is detected inside `bash()` — the only place that still knows both the exit status and
whether isolation was actually applied — and marked on the output. `run_tool_round` sees the
marker, strips it, and parks the round with an escalated prompt (`Unconfined?`) instead of
recording a result. Yes re-runs that one call through `Sandbox::without_isolation()`: Landlock
and seccomp off, workspace restriction and sensitive-file policy still on. No records the
sandboxed output as the tool result plus a line telling the model to work inside the workspace
rather than retry.

*Adaptation:* the detector is a phrase list (filesystem denials, plus what a blocked `socket()`
looks like from cargo/npm/curl/git), not a command-policy engine. Deliberately heuristic: a false
positive costs one question, a false negative costs the retry loop.

### 11. `ask_user` — *done*

A tool the model calls when only the user can decide: up to 5 questions, each with an id, a
short header, the question, and 2–3 labelled options, plus an automatic free-form "other".
Needs the same block-and-resume machinery as change 5, so build them together; the picker reuses
the overlay/selection code the `/model`, `/history` and `/code` popups already use.

### 7. Token accounting and compaction — *done*

Surface the usage the providers already report, track input/cached/output per turn, and
auto-compact past ~80 % of a configurable window by summarizing the older half into a
`Compaction` cell. This is what makes "no fixed step cap" safe.

### 8. Generate the system prompt from policy — *done*

New module `src/instructions.rs`. `agent_system_prompt(&SandboxConfig)` builds the whole prompt
from the values the enforcement reads — workspace, permission mode, shell timeout, isolation
mode, output budget, and whether the shell is reachable at all in this mode — so the text can no
longer drift from the behaviour. `describe_isolation` is the part that matters most: `off`,
`auto` and `require` each get their own accurate sentence, where before the prompt claimed
kernel confinement unconditionally regardless of the setting.

*An earlier version of this also read project rules from an `AGENTS.md` file. That was removed:
it only ever helped if the user wrote and maintained the file themselves, which is work they can
do once in the conversation instead.*

### 9. Round-trip reasoning per provider — *done*

Store each provider's reasoning payload opaquely and replay it verbatim where the protocol
requires it, instead of stripping it from every message on replay.

Three answers, one config field (`reasoning_replay`): `strip` (the old behaviour — most
endpoints have no contract about reasoning on input and resent reasoning is wasted tokens),
`opaque` (send it back with the message it came from), and `auto` (Anthropic opaque, everything
else strip). The payload is never interpreted: stored as it arrived, sent back as it arrived.

Anthropic needed the other half too. Thinking blocks are captured on the way in, their text
streamed as `<think>` so the existing transcript and status row handle them unchanged, and their
signature kept; on the way out a signed block is rebuilt and placed before the text and
`tool_use` blocks it was issued with. An unsigned block is not sent at all — the API rejects it,
and a 400 is worse than a duller answer. Extended thinking is opt-in via
`thinking_budget_tokens`, and enabling it drops `temperature` from the request because the API
refuses a non-default temperature alongside thinking.

### 10. `exec_command` sessions — *done*

`yield_time_ms` returns partial output plus a session id, with a way to continue or kill it.
Needs a small process table and a PTY decision (plain pipes are probably enough). The largest
change and the only one that argues for a bigger refactor — hence last.

---

## 4. What to deliberately *not* copy

| Mechanism | Why it doesn't pay off here |
|---|---|
| Submission/event protocol + separate engine process | Its value is multiple front ends (TUI, headless, IDE). We have one. Change 1 gets the responsiveness benefit in 60 lines. Revisit only if a headless/CI mode appears. |
| Append-only session files, fork/resume | Genuinely nicer than `sessions.json`, but a migration with real risk and no day-one benefit. Defer until someone asks to resume or fork. |
| Parsed-command policy engine | It exists to reduce *prompts*, not to add safety — and 5/6 already reduce prompts. A parser dependency plus a rule language is a lot of surface for that. |
| Tool servers, sub-agents, hooks, plugins, skills | Separate features with their own subsystems, not architecture. |
| Build-system and workspace sprawl | Product surface for a 100-crate repo; irrelevant at ~13 kLOC. |

---

## 5. Order, effort, verification

**Order:** 0 → 13 → 0b → 12 → 3 (all done) → 1 → 2 → 4 → 5 → 11 → 6 → 8 → 9 → 7 → 10.

**Effort:** the remaining changes total ~1,900 lines, roughly a third of it tests for the patch
parser and the adapters. Agent-related code today is ~4.7 kLOC (`streaming.rs`, `sandbox/mod.rs`,
`os_isolation.rs`, `permissions.rs`, `tools/mod.rs`) — this is additive, not a reshape.

**What stays untouched:** the three provider backends and their retry/model-fallback logic, the
whole `src/ui/` layer except a couple of new cell renderers, `os_isolation.rs` (Landlock +
seccomp — already good, and change 6 makes it *more* useful rather than replacing it), config
compatibility (every new field optional, today's behaviour as the default), and `sessions.json`
(extended, not replaced).

**Verification.** Testing is done by hand against real providers; the sandbox this was written
in has no Rust toolchain and no network route to install one, so nothing here was compiled
before handoff. The pure parts carry unit tests that need no network — the stream-ending
classifier and the control-sequence stripper — and the rest has a manual checklist:

- ask a model for a long answer over a gateway that drops connections → an incomplete-response
  notice appears instead of a clean-looking truncation;
- use a reasoning model that buffers → no spurious "went quiet" retry, and a long wait is
  reported rather than silently retried;
- have the agent run a TUI (`cargo run` on a ratatui app) → it fails fast, chatTUI's input and
  screen survive, and the timeout error explains why;
- let a tool fail, restart chatTUI, open the session → the failure still shows as a failure;
- `/target` is gone from the palette; `/sandbox` still sets the workspace.
