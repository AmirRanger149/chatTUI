# chatTUI

A fast, keyboard-first terminal chat client written in Rust. It uses
`ratatui`, `crossterm`, `tokio`, and `reqwest` to stream Gemini-compatible
responses without blocking the terminal.

## Features

- Vim-style `NORMAL` and `INSERT` modes
- Live asynchronous SSE streaming
- Scrollable chat history
- Local JSON conversation persistence
- History drawer and session switching
- Configurable Gemini model, endpoint, and temperature
- Markdown-friendly response rendering
- No OpenAI SDK or Python runtime required

## Requirements

- Rust stable toolchain
- A Gemini API key

## Install And Run

```bash
cargo run --release
```

Set the API key before launching:

```bash
export GEMINI_API_KEY="AIza-your-key"
cargo run --release
```

The binary can be built with:

```bash
cargo build --release
```

## Configuration

Configuration is stored in the platform config directory under
`chat-tui/config.json`. The default values are:

```json
{
  "api_key": "AIza-your-key",
  "base_url": "https://generativelanguage.googleapis.com/v1beta",
  "model": "gemini-3.5-flash",
  "temperature": 0.7
}
```

Environment variables override the key settings:

```text
GEMINI_API_KEY
GEMINI_BASE_URL
GEMINI_MODEL
```

Conversation history is stored in the platform data directory under
`chat-tui/sessions.json`.

## Controls

### Normal Mode

| Key | Action |
| --- | --- |
| `i` | Enter Insert mode |
| `j` / `Down` | Scroll down |
| `k` / `Up` | Scroll up |
| `h` | Toggle history drawer |
| `n` | Start a new chat |
| `?` | Show help |
| `q` | Quit |

### Insert Mode

| Key | Action |
| --- | --- |
| Any character | Add to prompt |
| `Enter` | Send prompt |
| `Backspace` | Delete a character |
| `Esc` | Return to Normal mode |

## API Compatibility

The client targets Gemini's native streaming endpoint. `base_url` may be
changed for a compatible Gemini gateway or proxy, but the endpoint must support
`models/{model}:streamGenerateContent` with SSE responses.

Keep API keys private. If a key is exposed, revoke it and create a replacement.
