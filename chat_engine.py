from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime


SESSION_MAX_AGE_MINUTES = 30


@dataclass
class ChatSession:
    session_id: str
    library: str
    messages: list[dict] = field(default_factory=list)
    created_at: str = ""
    last_active: str = ""


class ChatSessionStore:
    """Thread-safe in-memory session store with auto-expiry."""

    def __init__(self) -> None:
        self._sessions: dict[str, ChatSession] = {}
        self._lock = threading.Lock()

    def create_session(self, library: str = "all") -> str:
        session_id = uuid.uuid4().hex[:12]
        now = datetime.now().isoformat(timespec="seconds")
        session = ChatSession(
            session_id=session_id,
            library=library,
            messages=[],
            created_at=now,
            last_active=now,
        )
        with self._lock:
            self._sessions[session_id] = session
        return session_id

    def get_session(self, session_id: str) -> ChatSession | None:
        with self._lock:
            return self._sessions.get(session_id)

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        plan: dict | None = None,
    ) -> ChatSession | None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            msg: dict = {"role": role, "content": content}
            if plan:
                msg["plan"] = plan
            session.messages.append(msg)
            session.last_active = datetime.now().isoformat(timespec="seconds")
            return session

    def get_context(self, session_id: str) -> list[dict]:
        """Return conversation history suitable for LLM context."""
        session = self.get_session(session_id)
        if session is None:
            return []
        return [
            {"role": msg["role"], "content": msg["content"]}
            for msg in session.messages
        ]

    def delete_session(self, session_id: str) -> bool:
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                return True
            return False

    def list_sessions(self) -> list[dict]:
        with self._lock:
            return [
                {
                    "session_id": s.session_id,
                    "library": s.library,
                    "message_count": len(s.messages),
                    "created_at": s.created_at,
                    "last_active": s.last_active,
                }
                for s in self._sessions.values()
            ]

    def cleanup_expired(self, max_age_minutes: int = SESSION_MAX_AGE_MINUTES) -> int:
        now = datetime.now()
        expired = []
        with self._lock:
            for sid, s in self._sessions.items():
                try:
                    last = datetime.fromisoformat(s.last_active)
                    if (now - last).total_seconds() > max_age_minutes * 60:
                        expired.append(sid)
                except (ValueError, TypeError):
                    expired.append(sid)
            for sid in expired:
                del self._sessions[sid]
        return len(expired)


# Singleton store shared across requests
session_store = ChatSessionStore()


def route_chat(
    query: str,
    session_id: str | None,
    library: str,
    engine: object,
) -> dict:
    """Main entry point for multi-turn chat.

    Args:
        query: user's message text
        session_id: existing session ID or None for new session
        library: which library to search
        engine: VectorEngine instance

    Returns:
        dict with keys: session_id, results, plan, should_clarify, clarification_text
    """
    if not session_id or not session_store.get_session(session_id):
        session_id = session_store.create_session(library)

    chat_history = session_store.get_context(session_id)

    # Build plan with conversation context
    expanded_query = engine.expand_person_names(query, library=library)
    plan = engine.agent.build_plan(expanded_query, chat_history=chat_history)

    # Execute search
    results = engine.search_with_plan(plan, limit=36, library=library)

    # Add user message to session
    session_store.add_message(session_id, "user", query)

    # Determine if clarification is needed
    quality = plan.query_quality
    should_clarify = quality.get("level") == "weak" and len(results) < 20

    clarification_text = ""
    hints = plan.clarification_hints
    if should_clarify and hints:
        clarification_text = hints[0]
    elif len(results) == 0:
        should_clarify = True
        clarification_text = "没有找到匹配的结果。可以试试换个描述方式？"

    # Add agent response to session
    agent_content = f"找到 {len(results)} 个结果"
    if clarification_text:
        agent_content += f"。{clarification_text}"
    session_store.add_message(
        session_id,
        "agent",
        agent_content,
        plan=plan.to_dict(),
    )

    # Periodic cleanup
    session_store.cleanup_expired()

    return {
        "session_id": session_id,
        "results": results,
        "plan": plan.to_dict(),
        "should_clarify": should_clarify,
        "clarification_text": clarification_text,
    }
