# chatTUI

`chatTUI` is a small, fast terminal chat client for OpenAI, Anthropic, Google
Gemini, and any custom OpenAI-compatible endpoint you add yourself. It is built
in Rust and designed for people who prefer a focused keyboard workflow over a
browser window.

The app uses `ratatui` for the interface, `crossterm` for terminal input,
`tokio` for asynchronous work, and `reqwest` for native SSE streaming.

## What You Get

- Live responses as the selected model generates them
- An always-on composer with a `Working` / `Thinking` status row while streaming
- Scrollable conversation view
- Local conversation history saved as JSON
- History drawer for returning to previous chats
- Three built-in providers — OpenAI, Anthropic, Google Gemini
- Custom OpenAI-compatible endpoints (Ollama, Groq, OpenRouter, …) configured
  directly in `config.json` with your own base URL and key
- `/model` picker that lists the models the API actually offers (`GET /models`)
  and marks the active one
- Automatic fallback: if a model rejects the request or is at high demand,
  chatTUI switches to another available model and announces the switch
- Markdown-friendly response output with shaded code boxes
- Copy any code block from a response to the clipboard (`ctrl+g` or `/code`)
- Animated reasoning view for thinking models
  (see [Reasoning Models](#reasoning-models))
- Agent tools that never block the interface — commands run in the
  background and their results appear as they land
  (see [Project Instructions](#project-instructions-agentsmd))
- A single native Rust binary with no Python or OpenAI SDK dependency

## Before You Start

You need:

- Rust and Cargo from [rustup.rs](https://rustup.rs/)
- An API key for at least one provider: [OpenAI](https://platform.openai.com/),
  [Anthropic](https://www.anthropic.com/), [Google Gemini](https://ai.google.dev/),
  or any OpenAI-compatible gateway (e.g. Groq, OpenRouter, a local Ollama)

Check your Rust installation:

```bash
rustc --version
cargo --version
```

## Install

Clone the repository and enter the project directory:

```bash
git clone https://github.com/AmirRanger149/chatTUI.git
cd chatTUI
```

Build the optimized release binary:

```bash
cargo build --release
```

Or run directly while developing:

```bash
cargo run
```

## Configuration (`config.json`)

`chatTUI` looks for `config.json` in the application directory or the current working directory.

There are two kinds of providers:

- **The 3 main providers** — `openai`, `anthropic`, `gemini` — are built in.
  You only set their API keys.
- **Custom providers** — any other OpenAI-compatible endpoint — are defined by
  you under `custom_providers`, with the base URL and API key right in
  `config.json`.

### Full & Complete `config.json` Template

Create a `config.json` file in your root folder:

```json
{
  "openai_api_key": "sk-openai-your-key-here",
  "anthropic_api_key": "sk-ant-your-key-here",
  "gemini_api_key": "AIza-your-key-here",
  "custom_providers": [
    {
      "id": "groq",
      "name": "Groq",
      "base_url": "https://api.groq.com/openai/v1",
      "api_key": "gsk-your-groq-key-here",
      "model": "your-model-id"
    },
    {
      "id": "ollama",
      "name": "Ollama (local)",
      "base_url": "http://127.0.0.1:11434/v1"
    }
  ],
  "provider": "groq",
  "temperature": 0.7
}
```

### The 3 Main Providers

Each of these only needs a key — the endpoints and default models are built in.

**For OpenAI:**
```json
{
  "provider": "openai",
  "openai_api_key": "sk-openai-your-key-here"
}
```

**For Anthropic:**
```json
{
  "provider": "anthropic",
  "anthropic_api_key": "sk-ant-your-key-here"
}
```

**For Gemini:**
```json
{
  "provider": "gemini",
  "gemini_api_key": "AIza-your-key-here"
}
```

### Custom Providers (`custom_providers`)

Any OpenAI-compatible API can be added under `custom_providers`. Each entry is
a `POST /chat/completions` endpoint with your base URL and key:

| Field | Description |
| --- | --- |
| `id` | Unique id used with `/provider <id>` and the `provider` field (lowercase recommended) |
| `name` | Optional display name (defaults to the id) |
| `base_url` | OpenAI-compatible base URL, e.g. `https://api.groq.com/openai/v1` |
| `api_key` | API key; the `{ID}_API_KEY` environment variable is the fallback |
| `model` | Optional default model; when omitted, chatTUI picks one from the endpoint's live model list (a `free/` model first) |

**Example — one custom provider:**
```json
{
  "custom_providers": [
    {
      "id": "ollama",
      "name": "Ollama (local)",
      "base_url": "http://127.0.0.1:11434/v1",
      "model": "your-model-id"
    }
  ],
  "provider": "ollama"
}
```

**Example — several custom providers side by side:**
```json
{
  "custom_providers": [
    {
      "id": "groq",
      "name": "Groq",
      "base_url": "https://api.groq.com/openai/v1",
      "api_key": "gsk-your-groq-key-here",
      "model": "your-model-id"
    },
    {
      "id": "ollama",
      "name": "Ollama (local)",
      "base_url": "http://127.0.0.1:11434/v1",
      "model": "your-model-id"
    },
    {
      "id": "openrouter",
      "name": "OpenRouter",
      "base_url": "https://openrouter.ai/api/v1",
      "api_key": "sk-or-your-openrouter-key-here"
    }
  ]
}
```

Switch between all providers with the `/provider` popup (custom entries are
marked `· custom`) or directly with `/provider openai|anthropic|gemini|<custom-id>`.

> **Note:** If only one provider has an API key in `config.json`, chatTUI will
> automatically set that provider as the active default on startup. If several
> keys are present, the first provider in the list that has a key (OpenAI,
> Anthropic, Gemini, then your custom providers in file order) is selected by
> default and you can switch between them anytime using the `/provider`
> command — or set `provider` explicitly.

### Configuration Fields

| Field | Description | Default |
| --- | --- | --- |
| `openai_api_key` | API key for OpenAI | `None` (or `OPENAI_API_KEY` env) |
| `anthropic_api_key` | API key for Anthropic | `None` (or `ANTHROPIC_API_KEY` env) |
| `gemini_api_key` | API key for Google Gemini | `None` (or `GEMINI_API_KEY` env) |
| `custom_providers` | Your own OpenAI-compatible endpoints (see table above) | `[]` |
| `provider` | Active provider id (`openai`, `anthropic`, `gemini`, or a custom id) | First provider with a key |
| `temperature` | Sampling temperature for responses | `0.7` |
| `model` | Optional model name override for the active provider | Provider default |
| `base_url` | Optional endpoint override for the active provider | Provider default |
| `connect_timeout_secs` | Max seconds to wait while establishing an API connection | `15` |
| `idle_timeout_secs` | Max seconds a stream may stay quiet between chunks **after output has started** before the connection counts as dead — this is *not* a cap on total generation time | `90` |
| `first_token_timeout_secs` | Max seconds to wait for a model's **first** output. Deliberately long: reasoning models can think for many minutes and some gateways buffer the whole chain of thought before sending a byte | `1800` |
| `agent.max_rounds` | Backstop ceiling for agent tool rounds. A safety net only — healthy runs keep going as long as they make progress; broken loops are stopped earlier by the stagnation and consecutive-failure guards | `50` |
| `context_window_tokens` | Size of the model's context window, used to decide when to compact history. Worth setting to your model's real number | `128000` |
| `compact_at_percent` | Compact the history once it passes this percentage of the window | `80` |
| `reasoning_replay` | What happens to a model's reasoning when history is sent back: `auto` (per provider), `strip`, or `opaque` | `auto` |
| `thinking_budget_tokens` | Ask for extended thinking with this token budget (Anthropic). Unset means don't ask — and then there is nothing to replay either | *unset* |
| `max_output_tokens` | Cap on tokens per **single response**, sent as `max_tokens` (Gemini: `maxOutputTokens`). `0` sends no cap, so the provider's own default applies — often only a few thousand tokens, which is the usual reason a big `write_file` arrives with its arguments cut off mid-JSON, or an answer just stops. Set it to your model's real maximum output | `0` (provider default) |

### Large writes arriving truncated

A `write_file` call carries the entire file as one string in the tool call's
arguments. If the provider's output cap is reached before that string is
finished, the JSON never closes and chatTUI reports:

```
tool call arguments arrived truncated: the response hit its output token
limit before the tool call's JSON was complete
```

That is a *cap*, not a broken connection — raising `max_output_tokens` to your
model's real maximum output fixes it. If your model genuinely cannot emit the
whole file in one response, tell the model to build the file in pieces; the
tool description already asks it to, and `apply_patch` keeps each request
small.

> **The context window is managed, not hoped for.** chatTUI reads the token
> counts the provider reports (all three protocols send them) and compacts the
> history *before* the request that would overflow, not after it is rejected.
> Compaction is deterministic and needs no extra API call: old tool output is
> replaced by a one-line placeholder first — a 20 kB build log from ten rounds
> ago is the biggest and least useful thing in the history — and only if that
> is not enough are whole messages dropped, always in groups so a tool call
> never loses its results. The last 12 messages are always kept verbatim, and
> the transcript says what was removed. The status bar shows the real count
> when the provider gave one (`~` when chatTUI estimated).
>> **Reasoning is not silently thrown away.** Most endpoints have no
> contract about reasoning on input, so by default it is stripped before
> history is resent — resending it would double the token bill for text the
> model will not use. Anthropic is the exception: its thinking blocks carry a
> signature the next turn is validated against, so `auto` keeps them and sends
> them back with the message they came from. Dropping them mid-tool-loop is
> what quietly degrades an answer. Force either behaviour with
> `reasoning_replay: "strip"` or `"opaque"`.
>> **Incomplete answers are reported, not hidden.** A response is only
> treated as finished when the provider sent its end-of-stream marker or a
> finish reason. If the connection closed mid-answer, if the model hit its
> output limit (`length` / `max_tokens`), or if the provider's content filter
> cut it short, chatTUI says so in the transcript instead of showing a
> truncated reply that looks complete.
>
> **Long thinking is not a dead connection.** Until a model produces its
> first output the quiet-connection limit is `first_token_timeout_secs`
> (30 minutes by default), not `idle_timeout_secs`; after output starts, the
> short limit applies between chunks. When a long wait does end in a timeout
> it is reported rather than retried, because a retry would throw the
> thinking away and restart the same wait.
>
> **Retries & timeouts.** Transient failures (network errors, timeouts,
> HTTP 408/429/5xx) retry the *same* model up to three times with
> exponential backoff — announced in the transcript and shown as an animated
> `Retrying in …s` countdown above the composer — before falling back to
> another model as described below. Model-specific failures (missing or
> overloaded models) skip straight to the model fallback. Streams themselves
> are only bounded by the idle timeout, so long generations are never cut
> off as long as tokens keep arriving, and a lone malformed SSE line from a
> gateway no longer kills the stream.

> **Endpoints & models.** Each provider's endpoint/model can be overridden
> with `{ID}_BASE_URL` / `{ID}_MODEL` environment variables (e.g.
> `ANTHROPIC_BASE_URL`, `GEMINI_MODEL`, `GROQ_BASE_URL`) — this works for
> custom providers too.

### Environment Variables (Alternative)

You can also export environment variables instead of creating a `config.json`:

```bash
# The 3 main providers
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GEMINI_API_KEY="AIza..."

# Custom providers: {ID}_API_KEY, built from the uppercased id
export GROQ_API_KEY="gsk-..."
export OPENROUTER_API_KEY="sk-or-..."

# Optional endpoint/model overrides for any provider
export GROQ_BASE_URL="https://api.groq.com/openai/v1"
export GEMINI_MODEL="your-model-id"

cargo run --release
```

## Using chatTUI

The composer is always focused — just start typing and press `Enter` to send.

### Keyboard

| Key | Action |
| --- | --- |
| `Enter` | Send the message |
| `Shift+Enter` | Newline in the composer |
| `Esc` | Close a popup, then interrupt a running stream, then clear the composer |
| `Ctrl+H` | Conversation history |
| `Ctrl+T` | Inspect tool activity — write/edit diffs, bash command + output |
| `Ctrl+G` | Browse and copy code blocks |
| `Ctrl+R` | Show / hide model reasoning |
| `PgUp` / `PgDn` | Scroll the transcript |
| `Up` / `Down` | Recall previous prompts |
| `←` / `→` | Move the cursor (`Ctrl` jumps whole words) |
| `Ctrl+U` | Clear the composer |
| `?` | Keyboard shortcuts |
| `Ctrl+C` ×2 | Quit |

### Slash commands

| Command | Action |
| --- | --- |
| `/help` | Show keyboard shortcuts |
| `/new` | Start a new conversation |
| `/history` | Browse saved conversations |
| `/code` | Browse and copy code blocks |
| `/model` | Pick a model from the API's live list (`/model <id>` sets one directly) |
| `/provider` | Select API provider (`/provider <name>` sets one directly) |
| `/agent` | Toggle agent mode (gives the model file tools and a shell) |
| `/sandbox` | Show or set the directory the agent works in (`/sandbox <dir>`) |
| `/approve` | Toggle auto-approve, or make tools ask first (`/approve auto\|manual\|reset`) |
| `/quit` | Exit chatTUI |

Type `/` to open the command palette, then `Tab` to complete.

### Clipboard

`ctrl+g` / `/code` copies through your system clipboard when a clipboard tool
is available — `pbcopy` on macOS, `clip` on Windows, `wl-copy` / `xclip` /
`xsel` on Linux — so the copy is verified. Otherwise chatTUI falls back to the
terminal's OSC 52 sequence (with tmux/screen passthrough, and automatically
over SSH), which is best-effort: if pasting comes up empty, install `wl-copy`
(Wayland) or `xclip` (X11), or enable OSC 52 clipboard support in your
terminal.

## Reasoning Models

Some models expose their private chain of thought by wrapping it in
`<think> … </think>` before the actual answer (others stream it in a
`reasoning_content` field). chatTUI understands both shapes and gives them
their own animated treatment instead of dumping raw tags into the transcript.

**While the model is thinking**, the status row above the composer turns into a
shimmering indicator with a live preview of the thought being written:

```text
✻ Thinking (7s • esc to interrupt)  comparing the two approaches
```

The transcript shows the reasoning as a dim, italic block behind a `┃` rule,
kept to the last few lines so it never pushes the answer off screen.

**Once the answer starts**, the block collapses into a single quiet summary line:

```text
✻ Thought for 84 words  ▸ ctrl+r
```

Press `Ctrl+R` at any time to expand or collapse completed reasoning blocks.

Notes:

- Reasoning is never replayed back to the API on later turns, so it does not
  consume context or confuse the model.
- Interrupting a stream mid-thought (`Esc`) still leaves a tidy, collapsible block.
- Models that do not emit `<think>` tags are unaffected — you get the usual
  `• Working` indicator and plain markdown output.

## Models And Endpoints

Each provider has a default model, and you can change it with `{ID}_MODEL`,
the `model` value in your config (or config file `model` override for the
active provider), or `/model`. The model name must be available through the
provider; retrieve current model IDs from its `GET /models` endpoint.

A custom endpoint must support:

```text
POST /chat/completions
```

and, for the `/model` picker and automatic fallback:

```text
GET /models
```

### Availability-Based Default (Custom Providers)

Some gateways serve a rotating list of models — including free models
published under a `free/` namespace — so a hardcoded default model can
disappear or be replaced. When a
custom provider's entry omits `model` (at startup, or after switching to it
with `/provider`), chatTUI fetches its live `GET /models` list in the
background and sets the default model from what is actually available:

1. the first free model (`free/…`), or
2. the first model in the live list.

The pick is announced in the transcript (e.g.
`mygateway default set to available free model: free/your-model`).
An explicit choice always wins — a model set with `{ID}_MODEL`, the entry's
`model` field, or `/model` is never overridden — and if the fetch fails the
send-time fallback still covers it. The fetched list doubles as the cached
list shown by the `/model` picker. Pin a `model` in the entry whenever you
want a stable default instead.

### The `/model` Picker

Run `/model` with no argument and chatTUI queries the endpoint's `GET /models`,
then shows what is actually available in a popup: your current model is marked
`· active` and preselected, `↑↓` (or `PgUp` / `PgDn`) move through the list,
and `Enter` switches to the highlighted model. Press `r` to refetch the list
from the API and `esc` to close. The list is cached for five minutes so
reopening the picker is instant.

`/model <id>` still sets a model directly. When a fetched list is cached, the
id is resolved against it — exact match (case-insensitive), then a unique
prefix, then a unique suffix — so both `/model acmeai/acme-model-1` and the
shorthand `/model acme-model-1` find `AcmeAI/acme-model-1`. Anything
ambiguous or unknown is set exactly as typed.

### Automatic Model Fallback

When a send fails because of the model — a bad request against it, a
model that no longer exists, throttling, or the classic "currently
experiencing high demand" overload — chatTUI fetches the endpoint's model
list, picks another available model (preferring one from the same family,
e.g. another `AcmeAI/…`), and retries. Every switch is announced in the
transcript, so you always know which model answered:

```text
⚠ AcmeAI/acme-model-2 is unavailable — the model rejected the request (HTTP 429: rate limit exceeded)
  switching to AcmeAI/acme-model-1
```

Up to three fallbacks are tried per message, and the request is only retried
before any output has been written — a stream that breaks mid-answer is
reported as-is instead of being spliced onto a second model. Authentication
failures are reported immediately, since a different model cannot fix those.

## Agent Tools: Security Model

Agent mode (`/sandbox <dir>` or `--sandbox <dir>`) gives the model file tools
and a shell. What is actually enforced — and what is not:

- **File tools (`read_file`, `write_file`, `edit_file`, `apply_patch`,
  `list_files`) are workspace-restricted.** Paths must resolve inside the target directory,
  including through symlinks; `..` traversal and absolute escapes are
  refused. Sensitive files are refused for reading and writing: dotenv files
  (`.env`, `*.env`), key material (`*.pem`, `*.key`, `*.p12`, `*.pfx`,
  `*.jks`, SSH private keys), `.git/config`, and every write under `.git/`.
  Add names via `sandbox.extra_sensitive_names` in `config.json`.
- **A command never gets your terminal.** `stdin` is `/dev/null`,
  `stdout`/`stderr` are pipes, and the command runs in its own **session**
  (`setsid`), so it has no controlling terminal and cannot open `/dev/tty`.
  Without that, one `cargo run` of a TUI — or `vim`, `htop`, `ssh`, `sudo` —
  would switch your real terminal to the alternate screen, hide the cursor
  and put it into raw mode, and being killed at the timeout would leave it
  that way. Interactive programs therefore fail instead of taking the session
  over. Captured output is additionally stripped of terminal control
  sequences before it reaches the transcript, so escape bytes cannot be
  replayed at your terminal either (the cost: no colour in build output).
  As a last line of defence chatTUI re-asserts its own terminal state after
  every tool call.
- **Long and non-exiting commands are supported, without weakening the
  sandbox.** `bash` takes a per-call `timeout_secs` (1-1800) for a build you
  know is slow, and `session: true` for anything that does not exit on its
  own — a dev server, a watch loop, a REPL. A session returns whatever it has
  printed plus a `session_id`; `write_stdin` sends it input and reads more,
  `kill_session` kills its whole process tree. Sessions are spawned through
  the same path as a one-shot command, so they get the same filtered
  environment, the same `setsid` (still no controlling terminal — a TUI still
  cannot run) and the same kernel isolation. They are capped at 8 alive and
  30 minutes each, and every one is killed when chatTUI exits, because a
  session is its own process session and would otherwise outlive it.
- **The `bash` tool has a hard timeout** (default 30s,
  `sandbox.shell_timeout_secs`; the command's whole process tree is killed
  when it expires) and a **reduced environment** — credential-like variables
  such as API keys are not passed to shell commands.
- **Kernel-level isolation for shell commands** (Linux, on by default via
  `sandbox.os_isolation: "auto"`): before a command runs, the kernel is
  told to confine it — Landlock rules allow reads everywhere but restrict
  *writes* to the workspace plus a small set of scratch roots (`/tmp`,
  `$TMPDIR`, `/dev/shm`, `$CARGO_HOME`, `$CARGO_TARGET_DIR`), and a seccomp
  filter denies network sockets, `ptrace`/process-memory injection,
  kernel-module loading, and namespace/mount tricks. The restrictions
  cannot be undone by the command. On kernels older than 5.13 (or with the
  facilities unavailable), `"auto"` runs the command unrestricted and says
  so in the tool output; `"require"` refuses to run instead; `"off"`
  restores the previous, unrestricted behavior.
- **What isolation does not cover, by design:** reads outside the workspace
  stay possible (toolchains must read system files), so secret-*read*
  protection remains the application-level sensitive-file policy described
  above; Unix-socket `connect()` is denied, so tools that talk to local
  daemons fail; a kernel vulnerability could defeat any in-process
  sandbox; and if your user account can access privileged host resources
  (e.g. a container-daemon socket), remove that access on the host — no
  in-process sandbox can close it.
- **Permissions:** `sandbox.permission_mode` selects `read-only`,
  `workspace-write`, `ask-before-write`, `ask-before-shell` or `full-auto`.
  When unset, the legacy `auto_approve` / `allow_shell` flags decide.
  The `ask-*` modes really do ask: before a gated call runs, the spinner
  turns into `⚠ Approve? <what it would do> (y allow • a always • n deny)`.
  `y` allows that one call, `a` allows it and everything like it for the
  rest of the session, `n` (or `Esc`) refuses it and tells the model to ask
  you instead of retrying. "Like it" means the same tool, or for shell the
  same first two words — approving one `git push` covers the next, and does
  not cover `rm`. The prompt pauses the round; nothing else runs while it is
  up, and answering resumes exactly where it stopped. `/approve` toggles
  between auto and manual without editing `config.json`; `/approve reset`
  forgets what you have allowed, and switching modes forgets it too.
- **Context-aware file tools:** `read_file` serves numbered windows of at
  most 1,000 lines (250 by default), so large files are paged through
  instead of flooding the conversation; `write_file` / `edit_file` report
  their change as `+added -removed` line counts.
- **The agent can ask you.** `ask_user` lets the model stop and ask up to 5
  questions — with options you can pick by number, or free text — instead of
  guessing at a decision only you can make. The round pauses until you answer;
  `Esc` dismisses the rest and the model is told you did, so it picks the
  conservative option rather than waiting forever.
- **A command the sandbox blocks becomes a question, not a dead end.** When
  the kernel refuses a command, chatTUI asks whether to run that one call
  unconfined (`y`) or leave it blocked (`n`) instead of handing the model an
  error it will retry three times. Unconfined means Landlock/seccomp off for
  that call only — the workspace restriction and the sensitive-file policy
  still apply.
- **`apply_patch` edits by context, not by exact string.** The model sends
  one patch that can touch several files at once:

  ```
  *** Begin Patch
  *** Update File: src/main.rs
  @@ fn main
       let x = 1;
  -    let y = 2;
  +    let y = 3;
  *** Add File: notes.txt
  +hello
  *** End Patch
  ```

  Lines keep a leading space (context), `+` (add) or `-` (remove); `@@` is an
  optional hint that narrows where to look. Every path is checked against the
  workspace rules *before* anything is written, and the whole patch is applied
  or none of it is — so a five-file change cannot land half-finished. Unlike
  `edit_file`, it does not need the old text to match byte-for-byte, which is
  why it survives re-indented files.
- **Tools never block the interface.** A round of tool calls runs on its own
  task, so a command that takes the full timeout no longer freezes the
  terminal: the transcript fills in as each result lands, the status row
  keeps counting, and `Esc` stops the round. (A command already running is
  not killed by `Esc` — it is still bounded by the shell timeout and its
  process group.)
- **Command output keeps both ends.** Past 20,000 bytes the middle is
  dropped, not the end: a build or test run reports its failure last, and
  that is the part worth keeping. The marker says how many bytes were
  dropped, so an incomplete log is never mistaken for a complete one.
- **Agent loop:** the agent keeps working for as long as it makes
  progress — there is no fixed step cap. It is stopped by specific guards
  instead: the same failing tool call 3 rounds in a row (stagnation), 4
  consecutive rounds where every tool errored, or a generous backstop
  ceiling (`agent.max_rounds` in `config.json`, default 50). Every stop is
  announced in the transcript with its reason, and `Esc` aborts the
  in-flight model request.

When kernel isolation is unavailable or disabled, this system is accurately
described as *workspace-restricted tool execution*, not a sandbox; with
isolation active, the shell boundary is kernel-enforced for writes, network,
and process access.

## Saved Data

Conversation history is stored in the platform data directory, normally:

```text
~/.local/share/chatTUI/chat-tui/sessions.json
```

The history file contains your saved messages. Back it up if you need to keep
your conversations, and protect it if they contain private information.
Saves are atomic (temp file + rename), so a crash cannot leave a truncated
history behind; if the file is ever unreadable at startup, chatTUI moves it
aside as `sessions.json.corrupt-<timestamp>` and starts fresh instead of
refusing to launch. Corrupted tool calls from an interrupted or truncated
stream are repaired automatically on replay (truncated arguments become
`{}`, missing tool results are backfilled), and each repair is announced in
the transcript.

## Troubleshooting

**The app says the API key is missing**

Set the active provider's key — `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or
`GEMINI_API_KEY` for the built-ins, or the `api_key` of your custom provider
in `config.json` (its `{ID}_API_KEY` environment variable works too).

**The model is rejected**

Type `/model` to pick from the list the API actually offers. If a request is
rejected by a model that is overloaded or no longer available, chatTUI
automatically retries with another available model and tells you in the
transcript.

**The request fails or times out**

Check your network connection, API quota, endpoint URL (`base_url` in the
custom provider's entry), and API key. A custom endpoint must support
OpenAI-compatible streaming responses.

**A key was exposed**

Revoke it at the provider it belongs to and create a replacement.

## License

See [LICENSE](LICENSE).
