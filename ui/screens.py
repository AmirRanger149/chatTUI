from __future__ import annotations

from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Header, Input, Label, Markdown

from config import Settings
from ui.widgets import ChatInput, MessageBubble, Sidebar, WelcomePanel


class SettingsScreen(ModalScreen[None]):
    def __init__(self, model: str, system_prompt: str) -> None:
        super().__init__()
        self.model = model
        self.system_prompt = system_prompt

    def compose(self) -> ComposeResult:
        with Container(id="settings-dialog"):
            yield Label("Session settings", id="settings-title")
            yield Label("Model")
            yield Input(value=self.model, id="settings-model")
            yield Label("API file (.json or plain text)")
            yield Input(placeholder="/path/to/gemini.json or key.txt", id="settings-api-file")
            yield Button("Import API", id="settings-import-api")
            yield Label("API key (or paste it directly)")
            yield Input(value="", password=True, placeholder="Imported key is kept in memory", id="settings-api-key")
            yield Label("Base URL (optional)")
            yield Input(value="", id="settings-base-url")
            yield Label("System prompt")
            yield Input(value=self.system_prompt, id="settings-system")
            with Horizontal(classes="modal-actions"):
                yield Button("Cancel", id="settings-cancel")
                yield Button("Apply", id="settings-apply", variant="primary")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "settings-import-api":
            path = self.query_one("#settings-api-file", Input).value.strip()
            try:
                imported = Settings.from_json(Path(path).expanduser())
                self.query_one("#settings-api-key", Input).value = imported.active_api_key or ""
                self.query_one("#settings-base-url", Input).value = (
                    imported.gemini_base_url
                ) or ""
                if imported.default_model:
                    self.query_one("#settings-model", Input).value = imported.default_model
                self.notify("API settings imported", severity="information")
            except (OSError, ValueError, TypeError, KeyError) as exc:
                self.notify(f"Could not import API JSON: {exc}", severity="error")
        elif event.button.id == "settings-apply":
            self.dismiss({
                "model": self.query_one("#settings-model", Input).value,
                "system": self.query_one("#settings-system", Input).value,
                "api_key": self.query_one("#settings-api-key", Input).value,
                "base_url": self.query_one("#settings-base-url", Input).value,
            })
        elif event.button.id == "settings-cancel":
            self.dismiss(None)


class ChatScreen(Container):
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="workspace"):
            yield Sidebar(id="sidebar")
            with Vertical(id="chat-column"):
                with Horizontal(id="chat-heading"):
                    with Vertical(id="heading-copy"):
                        yield Label("New conversation", id="session-title")
                        yield Label("A private space for your next good idea", id="session-subtitle")
                    with Horizontal(id="heading-actions"):
                        yield Label("gemini-3.5-flash", id="model-label")
                        yield Button("Settings", id="open-settings")
                yield VerticalScroll(id="conversation")
                with Horizontal(id="status-row"):
                    yield Label("Ready when you are", id="status-line")
                    yield Label("Enter to send  |  Shift+Enter for a new line", id="input-hint")
                with Horizontal(id="composer"):
                    yield ChatInput("", placeholder="Write a message to Gemini...", id="chat-input")
                    yield Button("Send", id="send-message", variant="primary")
        yield Footer()

    def render_messages(self, messages: list[tuple[str, str]]) -> None:
        conversation = self.query_one("#conversation", VerticalScroll)
        conversation.remove_children()
        if not messages:
            conversation.mount(WelcomePanel(id="welcome-panel"))
        else:
            for role, content in messages:
                conversation.mount(MessageBubble(role, content))

    def add_message(self, role: str, content: str = "") -> MessageBubble:
        bubble = MessageBubble(role, content)
        conversation = self.query_one("#conversation", VerticalScroll)
        welcome = conversation.query("#welcome-panel")
        if welcome:
            welcome.first().remove()
        conversation.mount(bubble)
        return bubble

    def scroll_to_latest(self) -> None:
        self.query_one("#conversation", VerticalScroll).scroll_end(animate=False)
