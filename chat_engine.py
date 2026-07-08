from __future__ import annotations

import json
import os
import threading
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


SESSION_MAX_AGE_MINUTES = 30
ROOT = Path(__file__).parent.resolve()


def load_local_env() -> None:
    for env_path in [ROOT / ".env.local", ROOT / ".env"]:
        if not env_path.exists():
            continue
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_local_env()

LLM_API_BASE = os.environ.get("LLM_API_BASE", "https://api.deepseek.com/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek-chat")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_TIMEOUT = int(os.environ.get("LLM_TIMEOUT", "25"))


CHAT_SYSTEM_PROMPT = """你是一个多模态素材库智能体，负责把用户的多轮自然语言聊天转化为可执行检索。
素材库包含图片、视频、文档、OCR字幕、ASR转写、人脸人物分组、标签和元数据。

你必须返回 JSON，不要返回 Markdown。

JSON 格式：
{
  "should_search": true,
  "search_query": "用于检索系统的完整中文查询，必须结合上下文补全",
  "answer": "给用户看的自然语言回复，简短说明你理解了什么、将如何检索",
  "reason": "一句话说明上下文合并或意图判断"
}

规则：
- 如果用户是在继续上一轮，比如“只要视频”“不要风景”“换成海边”“个人库里的”，必须结合历史生成完整 search_query。
- 如果用户问的是系统能力或闲聊，可将 should_search 设为 false，并直接 answer。
- search_query 应该保留人物、地点、动作、时间、素材类型等关键条件。
- 如果用户说“不要/排除/去掉某类内容”，search_query 必须原样保留“不要/排除/去掉”这种否定词，不要改写成“非某某”。
- 不要编造素材库里不存在的具体文件名。
"""


def llm_available() -> bool:
    return bool(LLM_API_KEY)


def call_chat_llm(query: str, chat_history: list[dict]) -> dict | None:
    if not llm_available():
        return None
    messages = [{"role": "system", "content": CHAT_SYSTEM_PROMPT}]
    for message in chat_history[-10:]:
        role = "assistant" if message.get("role") == "agent" else "user"
        messages.append({"role": role, "content": str(message.get("content") or "")})
    messages.append({"role": "user", "content": query})
    payload = json.dumps(
        {
            "model": LLM_MODEL,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 800,
            "response_format": {"type": "json_object"},
        },
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{LLM_API_BASE.rstrip('/')}/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json; charset=utf-8",
            "Authorization": f"Bearer {LLM_API_KEY}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=LLM_TIMEOUT) as response:
            body = json.loads(response.read().decode("utf-8"))
        content = body["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            return parsed
    except (KeyError, json.JSONDecodeError, TimeoutError, urllib.error.URLError, urllib.error.HTTPError, OSError):
        return None
    return None


@dataclass
class ChatSession:
    session_id: str
    library: str
    messages: list[dict] = field(default_factory=list)
    created_at: str = ""
    last_active: str = ""


class ChatSessionStore:
    """Thread-safe in-memory session store for multi-turn search."""

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
            message: dict = {"role": role, "content": content}
            if plan:
                message["plan"] = plan
            session.messages.append(message)
            session.last_active = datetime.now().isoformat(timespec="seconds")
            return session

    def get_context(self, session_id: str) -> list[dict]:
        session = self.get_session(session_id)
        if session is None:
            return []
        return [
            {"role": message["role"], "content": message["content"]}
            for message in session.messages
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
                    "session_id": session.session_id,
                    "library": session.library,
                    "message_count": len(session.messages),
                    "created_at": session.created_at,
                    "last_active": session.last_active,
                }
                for session in self._sessions.values()
            ]

    def cleanup_expired(self, max_age_minutes: int = SESSION_MAX_AGE_MINUTES) -> int:
        now = datetime.now()
        expired: list[str] = []
        with self._lock:
            for session_id, session in self._sessions.items():
                try:
                    last_active = datetime.fromisoformat(session.last_active)
                    if (now - last_active).total_seconds() > max_age_minutes * 60:
                        expired.append(session_id)
                except (TypeError, ValueError):
                    expired.append(session_id)
            for session_id in expired:
                del self._sessions[session_id]
        return len(expired)


session_store = ChatSessionStore()


def route_chat(
    query: str,
    session_id: str | None,
    library: str,
    engine: object,
    limit: int = 36,
) -> dict:
    if not session_id or not session_store.get_session(session_id):
        session_id = session_store.create_session(library)

    chat_history = session_store.get_context(session_id)
    llm_result = call_chat_llm(query, chat_history)
    should_search = True if llm_result is None else bool(llm_result.get("should_search", True))
    search_query = str((llm_result or {}).get("search_query") or query).strip()
    if has_negative_request(query) and not has_negative_request(search_query):
        search_query = f"{search_query} {query}".strip()
    llm_answer = str((llm_result or {}).get("answer") or "").strip()
    llm_reason = str((llm_result or {}).get("reason") or "").strip()

    results = []
    plan_dict: dict = {
        "raw_query": search_query,
        "intent": "chat",
        "agent_summary": llm_answer or "当前为多轮对话。",
        "recall_routes": ["chat_only"],
        "trace": [],
    }
    plan = None
    if should_search and search_query:
        expanded_query = engine.expand_person_names(search_query, library=library)
        plan = engine.agent.build_plan(expanded_query)
        result_payload = engine.agent_search_with_plan(plan, limit=limit, library=library)
        results = result_payload.get("results", [])
        results = apply_chat_negative_filters(search_query, results)
        plan_dict = result_payload.get("plan", plan.to_dict())
        plan_dict["chat_agent"] = {
            "provider": "deepseek" if llm_result is not None else "rule_fallback",
            "model": LLM_MODEL if llm_result is not None else "rule-context-chat",
            "search_query": search_query,
            "reason": llm_reason,
        }

    session_store.add_message(session_id, "user", query)

    quality = plan.query_quality if plan is not None else {"level": "good"}
    should_clarify = quality.get("level") == "weak" and len(results) < max(8, limit // 3)
    hints = plan.clarification_hints if plan is not None else []
    clarification_text = hints[0] if should_clarify and hints else ""
    if should_search and not results:
        should_clarify = True
        clarification_text = clarification_text or "没有找到匹配结果，可以补充人物、地点、动作、时间或素材类型。"

    if llm_answer:
        agent_content = llm_answer
        if should_search:
            agent_content += f" 找到 {len(results)} 个结果。"
    else:
        agent_content = f"找到 {len(results)} 个结果"
    if clarification_text:
        agent_content += f"。{clarification_text}"
    session_store.add_message(session_id, "agent", agent_content, plan=plan_dict)
    session_store.cleanup_expired()

    return {
        "session_id": session_id,
        "results": results,
        "plan": plan_dict,
        "should_clarify": should_clarify,
        "clarification_text": clarification_text,
        "answer": agent_content,
        "search_query": search_query,
        "llm_used": llm_result is not None,
    }


def has_negative_request(text: str) -> bool:
    text = str(text or "")
    return any(marker in text for marker in ["不要", "排除", "去掉", "不包含", "别要"])


def apply_chat_negative_filters(query: str, results: list[dict]) -> list[dict]:
    terms = extract_negative_terms(query)
    if not terms:
        return results
    filtered = []
    for item in results:
        haystack = " ".join(
            [
                str(item.get("filename") or ""),
                str(item.get("best_prompt") or ""),
                " ".join(str(tag) for tag in item.get("tags") or []),
                " ".join(str(hit) for hit in item.get("tag_hits") or []),
                " ".join(str(hit) for hit in item.get("ocr_hits") or []),
                " ".join(str(line) for line in item.get("explanation") or []),
            ]
        ).lower()
        if any(alias.lower() in haystack for term in terms for alias in negative_aliases(term)):
            continue
        filtered.append(item)
    return filtered


def extract_negative_terms(query: str) -> list[str]:
    terms: list[str] = []
    for marker in ["不要", "排除", "去掉", "不包含", "别要"]:
        if marker not in query:
            continue
        tail = query.split(marker, 1)[1]
        tail = tail.replace("，", " ").replace(",", " ").replace("并且", " ").replace("和", " ")
        for token in tail.split():
            clean = token.strip(" 的了素材内容结果")
            if 1 < len(clean) <= 12:
                terms.append(clean)
    return terms[:6]


def negative_aliases(term: str) -> set[str]:
    aliases = {term}
    alias_map = {
        "访谈": {"访谈", "采访", "interview", "talk show"},
        "采访": {"访谈", "采访", "interview", "talk show"},
        "文档": {"document", "文档", "文章", "报告"},
        "图片": {"image", "photo", "jpg", "png"},
        "视频": {"video", "mp4", "webm"},
    }
    aliases.update(alias_map.get(term, set()))
    return aliases
