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


CHAT_SYSTEM_PROMPT = """你是一个多模态素材库智能体，负责深度理解用户的多轮自然语言输入并转化为精确的检索语义。

## 素材库能力
- 图片/视频/文档及其缩略图
- CLIP 跨模态视觉语义检索（中英文）
- OCR 画面文字识别、ASR 音频转写、字幕文本
- 人脸检测与人物聚类分组（个人库）
- 标签系统（自动场景标签 + 人工标签）
- 元数据过滤（年份、季节、素材类型、库别）

## 你的核心任务：语义分析
对每条用户输入，你必须完成以下分析步骤：

### 1. 意图分类
- **search**：用户想找素材（初次检索或新话题）
- **refine**：用户在上一轮基础上追加/排除条件（如"只要视频""不要风景""换成海边"）
- **clarify**：用户的问题含糊，需要追问（如"找点东西"）
- **chat**：纯闲聊或系统能力询问（如"你能做什么"）

### 2. 实体与条件提取
从用户输入和对话历史中提取：
- **人物**：姓名、称呼（小明、爸爸、person_0001）、"我""我们"
- **地点**：故宫、海边、学校、办公室等
- **场景**：合影、自拍、风景、演讲、访谈、烹饪等
- **物体**：车、猫、手机、咖啡、书等
- **动作**：吃、跑、打电话、拍照、骑等
- **时间**：去年、今年、春天、2024年、晚上等
- **素材类型**：图片/照片、视频/短片、文档/报告
- **素材库**：个人库/我的素材、公共库
- **质量**：清晰、不要重复、主体明确
- **否定条件**：不要XX、排除XX、去掉XX

### 3. 上下文合并
如果用户说"只要视频""不要风景""换成海边""个人库里的"等简短补充：
- 必须取上一轮完整的 search_query，叠加本次新增条件
- 如果本次是"换成XX"，应替换对应条件而非简单拼接
- 如果本次是"不要XX"，应将否定条件追加到完整查询末尾

### 4. 歧义检测
遇到以下情况应标记 should_clarify：
- 查询极短（≤3字）且无上下文可补全
- 人物名未绑定（如"找小明的照片"但系统可能没有小明的面部索引）
- 条件矛盾（如"白天的夜景"）
- 指代不明（如"那个地方的"——哪个地方？）

## 输出 JSON 格式
{
  "should_search": true,
  "intent": "search|refine|clarify|chat",
  "search_query": "结合历史和实体提取后的完整中文检索语句",
  "answer": "给用户的自然语言回复，用一两句话说明你的理解",
  "reason": "简短说明意图判断和上下文合并逻辑",
  "entities": {
    "people": ["提取到的人物名"],
    "locations": ["地点"],
    "scenes": ["场景"],
    "objects": ["物体"],
    "actions": ["动作"],
    "time": ["时间表达"],
    "media_type": "image|video|document|null",
    "library": "personal|public|all",
    "negatives": ["否定条件列表"]
  },
  "should_clarify": false,
  "clarify_question": ""
}

## 重要规则
- search_query 必须是一句完整、自然的中文，包含所有已提取的关键条件
- 否定词（不要/排除/去掉）必须原样保留在 search_query 中，不要改写成"非XX"
- 如果用户提供的是英文或中英混合，search_query 可包含英文关键词
- 对于 refine 意图，search_query 必须在历史基础上叠加新条件，不能丢失历史约束
- 不要编造素材库里不存在的具体文件名
- entities 字段有助于下游的结构化检索，请尽量准确提取
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


def _enrich_plan_with_llm_entities(plan, llm_entities: dict) -> None:
    """将 LLM 提取的实体补充到规则解析得到的 QueryPlan 中，弥补规则解析的不足。"""
    if not llm_entities or not plan:
        return
    # 补充人物
    llm_people = [str(p) for p in llm_entities.get("people", []) if p]
    existing_people = set(plan.unresolved_conditions.get("people", []))
    for person in llm_people:
        if person not in existing_people:
            plan.unresolved_conditions.setdefault("people", []).append(person)
    # 补充地点
    llm_locations = [str(loc) for loc in llm_entities.get("locations", []) if loc]
    existing_locations = set(plan.unresolved_conditions.get("locations", []))
    for loc in llm_locations:
        if loc not in existing_locations:
            plan.unresolved_conditions.setdefault("locations", []).append(loc)
    # 补充场景
    llm_scenes = [str(s) for s in llm_entities.get("scenes", []) if s]
    existing_scenes = set(plan.unresolved_conditions.get("scenes", []))
    for scene in llm_scenes:
        if scene not in existing_scenes:
            plan.unresolved_conditions.setdefault("scenes", []).append(scene)
    # 补充物体
    llm_objects = [str(o) for o in llm_entities.get("objects", []) if o]
    existing_objects = set(plan.unresolved_conditions.get("objects", []))
    for obj in llm_objects:
        if obj not in existing_objects:
            plan.unresolved_conditions.setdefault("objects", []).append(obj)
    # 补充动作
    llm_actions = [str(a) for a in llm_entities.get("actions", []) if a]
    existing_actions = set(plan.unresolved_conditions.get("actions", []))
    for action in llm_actions:
        if action not in existing_actions:
            plan.unresolved_conditions.setdefault("actions", []).append(action)
    # 补充时间
    llm_time = [str(t) for t in llm_entities.get("time", []) if t]
    existing_time = set(plan.unresolved_conditions.get("time_words", []))
    for t in llm_time:
        if t not in existing_time:
            plan.unresolved_conditions.setdefault("time_words", []).append(t)
    # 补充媒体类型
    llm_media = str(llm_entities.get("media_type") or "").strip()
    if llm_media and llm_media not in {"null", "none", ""}:
        existing_media = set(plan.unresolved_conditions.get("media", []))
        if llm_media not in existing_media:
            plan.unresolved_conditions.setdefault("media", []).append(llm_media)
        if llm_media in ("video", "image", "document") and not plan.executable_filters.get("kind"):
            plan.executable_filters["kind"] = llm_media
    # 补充素材库
    llm_library = str(llm_entities.get("library") or "").strip()
    if llm_library in ("personal", "public") and "library" not in plan.strict_conditions:
        plan.strict_conditions["library"] = llm_library
    # 补充否定条件
    llm_negatives = [str(n) for n in llm_entities.get("negatives", []) if n]
    if llm_negatives:
        existing_neg = set(plan.negative_conditions.get("raw", []))
        for neg in llm_negatives:
            if neg not in existing_neg:
                plan.negative_conditions.setdefault("raw", []).append(neg)


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
    llm_entities = (llm_result or {}).get("entities") or {}
    llm_intent = str((llm_result or {}).get("intent") or "chat")
    plan_dict: dict = {
        "raw_query": search_query,
        "intent": "chat",
        "agent_summary": llm_answer or "当前为多轮对话。",
        "recall_routes": ["chat_only"],
        "trace": [],
        "chat_intent": llm_intent,
        "chat_entities": llm_entities,
    }
    plan = None
    if should_search and search_query:
        expanded_query = engine.expand_person_names(search_query, library=library)
        plan = engine.agent.build_plan(expanded_query)
        # 将 LLM 提取的实体注入 plan 中，补充规则解析结果
        if llm_entities:
            _enrich_plan_with_llm_entities(plan, llm_entities)
        result_payload = engine.agent_search_with_plan(plan, limit=limit, library=library)
        results = result_payload.get("results", [])
        results = apply_chat_negative_filters(search_query, results)
        plan_dict = result_payload.get("plan", plan.to_dict())
        plan_dict["chat_agent"] = {
            "provider": "deepseek" if llm_result is not None else "rule_fallback",
            "model": LLM_MODEL if llm_result is not None else "rule-context-chat",
            "search_query": search_query,
            "reason": llm_reason,
            "llm_intent": llm_intent,
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
