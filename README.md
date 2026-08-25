# chatTUI

`chatTUI` is a small, fast terminal chat client for Google Gemini. It is built
in Rust and designed for people who prefer a focused keyboard workflow over a
browser window.

The app uses `ratatui` for the interface, `crossterm` for terminal input,
`tokio` for asynchronous work, and `reqwest` for native Gemini SSE streaming.

## What You Get

- Live responses as Gemini generates them
- Vim-style `NORMAL` and `INSERT` modes
- Scrollable conversation view
- Local conversation history saved as JSON
- History drawer for returning to previous chats
- Configurable Gemini model, temperature, and API endpoint
- Markdown-friendly response output
- A single native Rust binary with no Python or OpenAI SDK dependency

## Before You Start

You need:

- Rust and Cargo from [rustup.rs](https://rustup.rs/)
- A Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey)

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

## Configure Your Gemini Key

### Environment variable

This is the quickest option:

```bash
export GEMINI_API_KEY="AIza-your-key"
cargo run --release
```

You can put the export in your shell profile if you use the app regularly.

### `gemini.json`

The app looks for `gemini.json` beside the compiled application first. When
running with `cargo run`, it also checks the current working directory. Its
contents can look like this:

```json
{
  "api_key": "AIza-your-key",
  "model": "gemini-3.5-flash",
  "temperature": 0.7
}
```

The API key is read from this file so you do not need to enter it each time.
Keep this file private and never commit it.

Environment variables take priority for the key-related values:

```text
GEMINI_API_KEY
GEMINI_BASE_URL
GEMINI_MODEL
```

### Example file

```json
{
  "api_key": "AIza-your-key",
  "model": "gemini-3.5-flash"
}
```


## Using chatTUI

The app starts in `NORMAL` mode. Press `i` to begin writing a prompt.

### Normal mode

| Key | Action |
| --- | --- |
| `i` | Enter Insert mode |
| `j` or `Down` | Scroll down |
| `k` or `Up` | Scroll up |
| `h` | Toggle the history drawer |
| `Shift+H` | Switch to the next saved conversation |
| `n` | Start a new conversation |
| `?` | Show a help hint |
| `q` | Quit |

### Insert mode

| Key | Action |
| --- | --- |
| Any character | Add it to the prompt |
| `Enter` | Submit the prompt |
| `Backspace` | Delete the previous character |
| `Esc` | Return to Normal mode |

## Models And Endpoints

The default model is `gemini-3.5-flash`. Change it with `GEMINI_MODEL` or the
`model` value in your config file. The model name must be available to your
Gemini account.

The client uses this endpoint internally by default, so it does not need to be
present in `config.json`:

```text
https://generativelanguage.googleapis.com/v1beta
```

You may set `GEMINI_BASE_URL` for a compatible gateway or proxy. The endpoint
must support:

```text
POST /models/{model}:streamGenerateContent?alt=sse&key={api_key}
```

## Saved Data

Conversation history is stored in the platform data directory, normally:

```text
~/.local/share/chatTUI/chat-tui/sessions.json
```

The history file contains your saved messages. Back it up if you need to keep
your conversations, and protect it if they contain private information.

## Troubleshooting

**The app says the API key is missing**

Set `GEMINI_API_KEY` or create the JSON configuration file in the location
above.

**The model is rejected**

Check the spelling and confirm that the model is available to your Gemini API
account.

**The request fails or times out**

Check your network connection, API quota, endpoint URL, and API key. A proxy
must support Gemini's streaming response format.

**A key was exposed**

Revoke it immediately in Google AI Studio and create a replacement.

## License

See [LICENSE](LICENSE).
