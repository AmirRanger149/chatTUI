# chatTUI

`chatTUI` is a lightweight, keyboard-friendly terminal chat application for
Google Gemini. It is written in Python with Textual and uses Gemini's REST API
directly, so it does not depend on the OpenAI Python library.

## Features

- Live streaming Gemini responses
- Markdown rendering for formatted answers and code
- Local conversation history stored as JSON
- Create, switch, and delete conversations
- Gemini model selection per conversation
- System prompts per conversation
- Slash commands for common actions
- API key setup through environment variables, JSON files, plain-text files, or the Settings screen
- Custom Gemini-compatible base URLs for supported proxies or gateways

## Requirements

- Python 3.11 or newer
- A Google Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey)

## Installation

Clone the project and create a virtual environment:

```bash
git clone https://github.com/AmirRanger149/chatTUI.git
cd chatTUI
python3 -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

You can also install the project itself:

```bash
python -m pip install -e .
```

## Start The App

```bash
python app.py
```

The app will display a warning if no Gemini key is configured. You can add a
key using any of the methods below.

## API Key Setup

### Option 1: Environment Variable

Set the key before starting the app:

```bash
export GEMINI_API_KEY="AIza-your-key"
python app.py
```

You may also create a `.env` file in the project directory:

```env
GEMINI_API_KEY=AIza-your-key
GEMINI_MODEL=gemini-3.5-flash
```

### Option 2: Paste The Key In Settings

1. Start the app with `python app.py`.
2. Press `Ctrl+S` or select **Settings**.
3. Paste the key into the **API key** field.
4. Optionally change the model or base URL.
5. Select **Apply**.

The key is kept in memory for the current run. Do not paste keys into source
files or commit them to Git.

### Option 3: Import A JSON File

Create a file such as `gemini.json`:

```json
{
	"api_key": "AIza-your-key",
	"model": "gemini-3.5-flash"
}
```

Then open **Settings**, enter the path to the file under **API file**, select
**Import API**, and select **Apply**.

The longer field names are also accepted:

```json
{
	"gemini_api_key": "AIza-your-key",
	"gemini_base_url": "https://generativelanguage.googleapis.com/v1beta",
	"default_model": "gemini-3.5-flash"
}
```

### Option 4: Import A Plain-Text Key File

Put only the key in a text file, for example `gemini-key.txt`:

```text
AIza-your-key
```

Enter that file's path in **Settings** and select **Import API**. JSON files
containing only a string are supported too:

```json
"AIza-your-key"
```

## Models

The default model is `gemini-3.5-flash`. You can enter another available Gemini
model in Settings or change it during a conversation:

```text
/model gemini-2.5-flash
```

The model name must be available to your Gemini API account.

## Commands And Controls

| Control | Action |
| --- | --- |
| `Enter` | Send the message |
| `Shift+Enter` | Add a new line |
| `Ctrl+N` | Create a new conversation |
| `Ctrl+D` | Delete the current conversation |
| `Ctrl+S` | Open Settings |
| `Ctrl+C` | Quit |
| `/model <name>` | Change the active Gemini model |
| `/system <prompt>` | Change the current system prompt |
| `/clear` | Clear the current conversation context |

You can also click the **Send** button instead of pressing Enter.

## Custom Base URL

Gemini normally uses:

```text
https://generativelanguage.googleapis.com/v1beta
```

To use a compatible gateway or proxy, add `gemini_base_url` to the JSON file:

```json
{
	"api_key": "AIza-your-key",
	"base_url": "https://your-gemini-gateway.example/v1beta",
	"model": "gemini-3.5-flash"
}
```

## Conversation Storage

Conversations are stored locally at:

```text
~/.chat-tui/sessions.json
```

The file is created automatically. It contains your conversation history, so
protect it if your chats contain private information.

## Troubleshooting

**`GEMINI_API_KEY is not configured`**

Set `GEMINI_API_KEY`, paste a key in Settings, or import a key file.

**Authentication or invalid model errors**

Check that the API key is active and that the model name is available to your
account.

**Network or timeout errors**

Check your connection and confirm that the configured base URL is correct.

**Never share an API key**

If a key is accidentally posted publicly or committed, revoke it in Google AI
Studio and create a replacement.
