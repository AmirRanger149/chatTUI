from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class Message:
    role: str
    content: str


@dataclass
class Session:
    id: str
    title: str
    model: str
    system_prompt: str
    messages: list[Message] = field(default_factory=list)
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @classmethod
    def create(cls, model: str, system_prompt: str) -> Session:
        return cls(uuid.uuid4().hex, "New conversation", model, system_prompt)


class SessionManager:
    def __init__(self, path: Path, default_model: str) -> None:
        self.path = path
        self.default_model = default_model
        self.sessions: list[Session] = []
        self.current_id: str | None = None
        self.load()
        if not self.sessions:
            self.new_session()
        elif self.current_id is None:
            self.current_id = self.sessions[0].id

    @property
    def current(self) -> Session:
        if self.current_id is None:
            self.new_session()
        return next(session for session in self.sessions if session.id == self.current_id)

    def load(self) -> None:
        if not self.path.is_file():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self.sessions = [
                Session(
                    id=item["id"],
                    title=item.get("title", "New conversation"),
                    model=item.get("model", self.default_model),
                    system_prompt=item.get("system_prompt", "You are a helpful assistant."),
                    messages=[Message(**message) for message in item.get("messages", [])],
                    updated_at=item.get("updated_at", ""),
                )
                for item in raw.get("sessions", [])
            ]
            self.current_id = raw.get("current_id")
        except (OSError, KeyError, TypeError, json.JSONDecodeError):
            self.sessions = []
            self.current_id = None

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"current_id": self.current_id, "sessions": [asdict(session) for session in self.sessions]}
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def new_session(self) -> Session:
        session = Session.create(self.default_model, "You are a helpful assistant.")
        self.sessions.insert(0, session)
        self.current_id = session.id
        self.save()
        return session

    def select(self, session_id: str) -> Session:
        if any(session.id == session_id for session in self.sessions):
            self.current_id = session_id
            self.save()
        return self.current

    def delete_current(self) -> Session:
        if len(self.sessions) == 1:
            self.sessions[0] = Session.create(self.default_model, "You are a helpful assistant.")
            self.current_id = self.sessions[0].id
        else:
            self.sessions = [session for session in self.sessions if session.id != self.current_id]
            self.current_id = self.sessions[0].id
        self.save()
        return self.current

    def clear_context(self) -> None:
        self.current.messages.clear()
        self.current.title = "New conversation"
        self.current.updated_at = datetime.now(timezone.utc).isoformat()
        self.save()

    def add_message(self, role: str, content: str) -> None:
        session = self.current
        session.messages.append(Message(role, content))
        if role == "user" and session.title == "New conversation":
            session.title = content.strip().replace("\n", " ")[:42] or session.title
        session.updated_at = datetime.now(timezone.utc).isoformat()
        self.save()
