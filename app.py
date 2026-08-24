from __future__ import annotations

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Button, Input, Label

from api.client import ChatAPIError, ChatAPIClient
from config import Settings
from state.session_manager import SessionManager
from ui.screens import ChatScreen, SettingsScreen
from ui.widgets import ChatInput, Sidebar, MessageBubble


class ChatTUI(App[None]):
    TITLE = "Chat TUI"
    CSS_PATH = "ui/styles.tcss"
    BINDINGS = [
        Binding("ctrl+n", "new_session", "New chat"),
        Binding("ctrl+d", "delete_session", "Delete chat"),
        Binding("ctrl+s", "settings", "Settings"),
        Binding("ctrl+c", "quit", "Quit", show=False),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.settings = Settings.load()
        self.sessions = SessionManager(self.settings.sessions_path, self.settings.default_model)
        self.client: ChatAPIClient | None = None
        if self.settings.has_api_key:
            self.client = ChatAPIClient(self.settings.gemini_api_key, self.settings.gemini_base_url)
        self.streaming_bubble: MessageBubble | None = None

    def compose(self) -> ComposeResult:
        yield ChatScreen()

    def on_mount(self) -> None:
        self.refresh_view()
        if not self.settings.has_api_key:
            self.notify("GEMINI_API_KEY is not configured. Add it to your settings.", severity="warning", timeout=8)

    def refresh_view(self) -> None:
        screen = self.query_one(ChatScreen)
        session = self.sessions.current
        screen.query_one("#session-title", Label).update(session.title)
        screen.render_messages([(message.role, message.content) for message in session.messages])
        screen.query_one("#sidebar", Sidebar).refresh_sessions(
            [(item.id, item.title) for item in self.sessions.sessions], session.id
        )

    def action_new_session(self) -> None:
        self.sessions.new_session()
        self.refresh_view()

    def action_delete_session(self) -> None:
        self.sessions.delete_current()
        self.refresh_view()
        self.notify("Conversation deleted", severity="information")

    def action_settings(self) -> None:
        session = self.sessions.current
        self.push_screen(SettingsScreen(session.model, session.system_prompt), self.apply_settings)

    def apply_settings(self, result: dict[str, str] | None) -> None:
        if result:
            self.sessions.current.model = result["model"].strip() or self.settings.default_model
            self.sessions.current.system_prompt = result["system"].strip() or "You are a helpful assistant."
            if result["api_key"].strip():
                self.settings.gemini_api_key = result["api_key"].strip()
            if result["base_url"].strip():
                self.settings.gemini_base_url = result["base_url"].strip()
            if self.settings.has_api_key:
                self.client = ChatAPIClient(self.settings.gemini_api_key, self.settings.gemini_base_url)
            self.sessions.save()
            self.notify(f"Using {self.sessions.current.model}", severity="information")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "open-settings":
            self.action_settings()
        elif event.button.id == "send-message":
            input_widget = self.query_one("#chat-input", ChatInput)
            value = input_widget.text.strip()
            if value:
                input_widget.clear()
                self.handle_input(value)

    def on_sidebar_new_requested(self, event: Sidebar.NewRequested) -> None:
        self.action_new_session()

    def on_sidebar_delete_requested(self, event: Sidebar.DeleteRequested) -> None:
        self.action_delete_session()

    def on_sidebar_selected(self, event: Sidebar.Selected) -> None:
        self.sessions.select(event.session_id)
        self.refresh_view()

    def on_chat_input_submitted(self, event: ChatInput.Submitted) -> None:
        self.handle_input(event.value)

    def handle_input(self, value: str) -> None:
        if value.startswith("/model "):
            self.sessions.current.model = value[7:].strip() or self.settings.default_model
            self.sessions.save()
            self.notify(f"Model set to {self.sessions.current.model}")
            return
        if value.startswith("/system "):
            self.sessions.current.system_prompt = value[8:].strip() or "You are a helpful assistant."
            self.sessions.save()
            self.notify("System prompt updated")
            return
        if value == "/clear":
            self.sessions.clear_context()
            self.refresh_view()
            self.notify("Context cleared")
            return
        self.send_message(value)

    def send_message(self, value: str) -> None:
        if self.client is None:
            self.notify("Configure GEMINI_API_KEY before sending messages.", severity="error")
            return
        self.sessions.add_message("user", value)
        self.query_one(ChatScreen).add_message("user", value)
        self.streaming_bubble = self.query_one(ChatScreen).add_message("assistant")
        self.query_one("#status-line", Label).update("Thinking...")
        self.run_worker(self.stream_response(), exclusive=True, group="chat")

    async def stream_response(self) -> None:
        session = self.sessions.current
        messages = [{"role": "system", "content": session.system_prompt}]
        messages.extend({"role": message.role, "content": message.content} for message in session.messages)
        response = ""
        try:
            async for token in self.client.stream_chat(messages, session.model, self.settings.temperature):
                response += token
                if self.streaming_bubble:
                    self.streaming_bubble.update_content(response)
                    self.query_one("#conversation").scroll_end(animate=False)
            self.sessions.add_message("assistant", response)
            self.query_one("#status-line", Label).update("")
        except ChatAPIError as exc:
            if self.streaming_bubble:
                self.streaming_bubble.update_content(f"**Error:** {exc}")
            self.query_one("#status-line", Label).update("Request failed")
            self.notify(str(exc), severity="error", timeout=8)

    async def on_unmount(self) -> None:
        if self.client:
            await self.client.close()


def main() -> None:
    ChatTUI().run()


if __name__ == "__main__":
    main()
