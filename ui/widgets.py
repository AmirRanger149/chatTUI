from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.message import Message
from textual.widgets import Button, Label, ListItem, ListView, Markdown, Select, Static, TextArea


class ChatInput(TextArea):
    class Submitted(Message):
        def __init__(self, value: str) -> None:
            super().__init__()
            self.value = value

    async def _on_key(self, event: TextArea.Key) -> None:
        if event.key in ("enter", "ctrl+enter"):
            event.prevent_default()
            event.stop()
            value = self.text.strip()
            if value:
                self.post_message(self.Submitted(value))
                self.clear()
        else:
            await super()._on_key(event)


class ModelPicker(Select[str]):
    def __init__(self, value: str = "gemini-3.5-flash") -> None:
        super().__init__(
            [(model, model) for model in ("gemini-3.5-flash", "gemini-2.5-flash", "gemini-2.0-flash")],
            value=value,
            id="model-picker",
        )


class ExpandControl(Button):
    def __init__(self, toggle: object) -> None:
        super().__init__("Show more", id="expand-message", classes="expand-message")
        self.toggle = toggle

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.toggle()


class MessageBubble(Vertical):
    def __init__(self, role: str, content: str = "", **kwargs: object) -> None:
        self.role = role
        super().__init__(classes=f"message {role}", **kwargs)
        self.content = content
        self.is_expanded = False

    def compose(self) -> ComposeResult:
        yield Label("YOU" if self.role == "user" else "ASSISTANT", classes="message-label")
        yield ExpandControl(self.toggle_expanded)
        markdown = Markdown(self.content or "", id="message-content")
        markdown.disabled = True
        yield markdown

    def on_mount(self) -> None:
        self.query_one("#message-content", Markdown).update(self.content or " ")
        self._update_expand_control()

    def _update_expand_control(self) -> None:
        button = self.query_one("#expand-message", Button)
        if self.role != "assistant" or len(self.content) < 1200:
            button.display = False
        else:
            button.display = True
            button.label = "Show less" if self.is_expanded else "Show more"
            self.set_class(self.is_expanded, "expanded")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "expand-message":
            self.toggle_expanded()

    def toggle_expanded(self) -> None:
        self.is_expanded = not self.is_expanded
        self._update_expand_control()

    def update_content(self, content: str) -> None:
        self.content = content
        markdown = self.query("#message-content")
        if markdown:
            markdown.first().update(content or " ")
        if self.is_mounted:
            self._update_expand_control()


class SessionItem(ListItem):
    def __init__(self, session_id: str, title: str) -> None:
        self.session_id = session_id
        super().__init__(Label(title))


class Sidebar(Vertical):
    class Selected(Message):
        def __init__(self, session_id: str) -> None:
            super().__init__()
            self.session_id = session_id

    class NewRequested(Message):
        pass

    class DeleteRequested(Message):
        pass

    def compose(self) -> ComposeResult:
        yield Static("CHAT TUI", classes="brand")
        yield Button("+  New chat", id="new-chat", variant="primary")
        yield Label("CONVERSATIONS", classes="section-label")
        yield ListView(id="session-list")
        yield Button("Delete current", id="delete-chat", variant="error")
        yield Static("Ctrl+N  new chat\nCtrl+D  delete chat\nCtrl+Enter  send", classes="shortcuts")

    def refresh_sessions(self, sessions: list[tuple[str, str]], current_id: str) -> None:
        session_list = self.query_one("#session-list", ListView)
        session_list.clear()
        for session_id, title in sessions:
            item = SessionItem(session_id, title)
            if session_id == current_id:
                item.add_class("active")
            session_list.append(item)

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        item = event.item
        if isinstance(item, SessionItem):
            self.post_message(self.Selected(item.session_id))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "new-chat":
            self.post_message(self.NewRequested())
        elif event.button.id == "delete-chat":
            self.post_message(self.DeleteRequested())
