from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Header, Input, Label, Markdown, Select

from ui.widgets import ChatInput, MessageBubble, Sidebar


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
            yield Label("System prompt")
            yield Input(value=self.system_prompt, id="settings-system")
            with Horizontal(classes="modal-actions"):
                yield Button("Cancel", id="settings-cancel")
                yield Button("Apply", id="settings-apply", variant="primary")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "settings-apply":
            self.dismiss((self.query_one("#settings-model", Input).value, self.query_one("#settings-system", Input).value))
        elif event.button.id == "settings-cancel":
            self.dismiss(None)


class ChatScreen(Container):
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="workspace"):
            yield Sidebar(id="sidebar")
            with Vertical(id="chat-column"):
                with Horizontal(id="chat-heading"):
                    yield Label("New conversation", id="session-title")
                    yield Button("Settings", id="open-settings")
                yield VerticalScroll(id="conversation")
                yield Label("", id="status-line")
                yield ChatInput("", placeholder="Message ChatGPT...", id="chat-input")
        yield Footer()

    def render_messages(self, messages: list[tuple[str, str]]) -> None:
        conversation = self.query_one("#conversation", VerticalScroll)
        conversation.remove_children()
        for role, content in messages:
            conversation.mount(MessageBubble(role, content))

    def add_message(self, role: str, content: str = "") -> MessageBubble:
        bubble = MessageBubble(role, content)
        self.query_one("#conversation", VerticalScroll).mount(bubble)
        return bubble
