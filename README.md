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
- Custom OpenAI-compatible endpoints (Dahl, APInex, Ollama, Groq, OpenRouter,
  …) configured directly in `config.json` with your own base URL and key
- `/model` picker that lists the models the API actually offers (`GET /models`)
  and marks the active one
- Automatic fallback: if a model rejects the request or is at high demand,
  chatTUI switches to another available model and announces the switch
- Markdown-friendly response output with shaded code boxes
- Copy any code block from a response to the clipboard (`ctrl+g` or `/code`)
- Animated reasoning view for thinking models such as `MiniMaxAI/MiniMax-M2.7`
  (see [Reasoning Models](#reasoning-models))
- A single native Rust binary with no Python or OpenAI SDK dependency

## Before You Start

You need:

- Rust and Cargo from [rustup.rs](https://rustup.rs/)
- An API key for at least one provider: [OpenAI](https://platform.openai.com/),
  [Anthropic](https://www.anthropic.com/), [Google Gemini](https://ai.google.dev/),
  or any OpenAI-compatible gateway (e.g. [Dahl Inference](https://inference.dahl.global/),
  [APInex](https://api.apinex.bond/v1), a local Ollama)

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
      "id": "dahl",
      "name": "Dahl",
      "base_url": "https://inference.dahl.global/v1",
      "api_key": "your-dahl-key-here",
      "model": "MiniMaxAI/MiniMax-M2.7"
    },
    {
      "id": "apinex",
      "name": "APInex",
      "base_url": "https://api.apinex.bond/v1",
      "api_key": "sk-apx-your-apinex-key-here"
    }
  ],
  "provider": "dahl",
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
| `base_url` | OpenAI-compatible base URL, e.g. `https://inference.dahl.global/v1` |
| `api_key` | API key; the `{ID}_API_KEY` environment variable is the fallback |
| `model` | Optional default model; when omitted, chatTUI picks one from the endpoint's live model list (a `free/` model first) |

**Example — one custom provider (Dahl):**
```json
{
  "custom_providers": [
    {
      "id": "dahl",
      "name": "Dahl",
      "base_url": "https://inference.dahl.global/v1",
      "api_key": "your-dahl-key-here",
      "model": "MiniMaxAI/MiniMax-M2.7"
    }
  ],
  "provider": "dahl"
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
      "model": "llama-3.3-70b-versatile"
    },
    {
      "id": "ollama",
      "name": "Ollama (local)",
      "base_url": "http://127.0.0.1:11434/v1",
      "model": "llama3.2"
    },
    {
      "id": "apinex",
      "name": "APInex",
      "base_url": "https://api.apinex.bond/v1",
      "api_key": "sk-apx-your-apinex-key-here"
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

> **Deprecated fields still work.** Older configs that use `dahl_api_key`,
> `apinex_api_key`, or the legacy single `api_key` field keep working: those
> fields are automatically migrated into equivalent `custom_providers` entries
> on startup (Dahl → `https://inference.dahl.global/v1`, APInex →
> `https://api.apinex.bond/v1`). New configs should use `custom_providers`.

> **Endpoints & models.** Each provider's endpoint/model can be overridden
> with `{ID}_BASE_URL` / `{ID}_MODEL` environment variables (e.g.
> `ANTHROPIC_BASE_URL`, `GEMINI_MODEL`, `DAHL_BASE_URL`) — this works for
> custom providers too.

### Environment Variables (Alternative)

You can also export environment variables instead of creating a `config.json`:

```bash
# The 3 main providers
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GEMINI_API_KEY="AIza..."

# Custom providers: {ID}_API_KEY, built from the uppercased id
export DAHL_API_KEY="your-dahl-key"
export APINEX_API_KEY="sk-apx..."

# Optional endpoint/model overrides for any provider
export DAHL_BASE_URL="https://inference.dahl.global/v1"
export GEMINI_MODEL="gemini-2.5-flash"

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
| `Ctrl+T` | Conversation history |
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
`<think> … </think>` before the actual answer. `MiniMaxAI/MiniMax-M2.7` — the
Dahl example's default model — is one of them. chatTUI understands that format
and gives it its own animated treatment instead of dumping raw tags into the
transcript.

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
published under the `free/` namespace, e.g. `free/deepseek-v4-flash-0731` on
APInex — so a hardcoded default model can disappear or be replaced. When a
custom provider's entry omits `model` (at startup, or after switching to it
with `/provider`), chatTUI fetches its live `GET /models` list in the
background and sets the default model from what is actually available:

1. the first free model (`free/…`), or
2. the first model in the live list.

The pick is announced in the transcript (e.g.
`APInex default set to available free model: free/deepseek-v4-flash-0731`).
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
prefix, then a unique suffix — so both `/model minimaxai/minimax-m2.7` and the
shorthand `/model minimax-m2.7` find `MiniMaxAI/MiniMax-M2.7`. Anything
ambiguous or unknown is set exactly as typed.

### Automatic Model Fallback

When a send fails because of the model — a bad request against it, a
model that no longer exists, throttling, or the classic "currently
experiencing high demand" overload — chatTUI fetches the endpoint's model
list, picks another available model (preferring one from the same family,
e.g. another `MiniMaxAI/…`), and retries. Every switch is announced in the
transcript, so you always know which model answered:

```text
⚠ MiniMaxAI/MiniMax-M2.7 is unavailable — the model rejected the request (HTTP 429: rate limit exceeded)
  switching to MiniMaxAI/MiniMax-M1
```

Up to three fallbacks are tried per message, and the request is only retried
before any output has been written — a stream that breaks mid-answer is
reported as-is instead of being spliced onto a second model. Authentication
failures are reported immediately, since a different model cannot fix those.

## Agent Tools: Security Model

Agent mode (`/sandbox <dir>` or `--sandbox <dir>`) gives the model file tools
and a shell. What is actually enforced — and what is not:

- **File tools (`read_file`, `write_file`, `edit_file`, `list_files`) are
  workspace-restricted.** Paths must resolve inside the target directory,
  including through symlinks; `..` traversal and absolute escapes are
  refused. Sensitive files are refused for reading and writing: dotenv files
  (`.env`, `*.env`), key material (`*.pem`, `*.key`, `*.p12`, `*.pfx`,
  `*.jks`, SSH private keys), `.git/config`, and every write under `.git/`.
  Add names via `sandbox.extra_sensitive_names` in `config.json`.
- **The `bash` tool is NOT a sandbox.** Commands run via `sh -c` with the
  workspace as the current directory only. They run with your full user
  privileges and can read or write files outside the workspace, reach the
  network, and spawn processes. Treat model-generated shell commands with
  the same caution you would treat any script from an LLM.
- **Shell hardening that is applied:** a hard timeout (default 30s,
  `sandbox.shell_timeout_secs`; the command's process group is killed when
  it expires) and a reduced environment — credential-like variables such as
  API keys are not passed to shell commands.
- **Permissions:** `sandbox.permission_mode` selects `read-only`,
  `workspace-write`, `ask-before-write`, `ask-before-shell` or `full-auto`.
  When unset, the legacy `auto_approve` / `allow_shell` flags decide.
  `ask-*` modes currently deny the gated action (interactive approval is
  not implemented yet) instead of allowing it silently.
- **Agent loop:** tool-call rounds are capped (`MAX_AGENT_ITERATIONS = 10`),
  and `Esc` aborts the in-flight model request.

True isolation (seccomp/landlock, containers, VMs) is not implemented; until
it is, this system is accurately described as *workspace-restricted tool
execution*, not a sandbox.

## Saved Data

Conversation history is stored in the platform data directory, normally:

```text
~/.local/share/chatTUI/chat-tui/sessions.json
```

The history file contains your saved messages. Back it up if you need to keep
your conversations, and protect it if they contain private information.

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
