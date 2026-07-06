from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Iterable


PERSON_MARKERS = ["\u548c", "\u4e0e", "\u8ddf"]
PERSON_BOUNDARY_PATTERN = r"\u5728|\u62cd|\u7684|\u5408\u5f71|\u7167\u7247|\u56fe\u7247|\u7d20\u6750|\u4e00\u8d77|\s"

TIME_WORDS = [
    "\u53bb\u5e74",
    "\u4eca\u5e74",
    "\u524d\u5e74",
    "\u6625\u5929",
    "\u590f\u5929",
    "\u79cb\u5929",
    "\u51ac\u5929",
    "\u4e0a\u5348",
    "\u4e2d\u5348",
    "\u4e0b\u5348",
    "\u665a\u4e0a",
    "\u591c\u665a",
    "\u5468\u672b",
]

LOCATION_WORDS = [
    "\u6545\u5bab",
    "\u957f\u57ce",
    "\u5929\u5b89\u95e8",
    "\u9890\u548c\u56ed",
    "\u5b66\u6821",
    "\u6559\u5ba4",
    "\u529e\u516c\u5ba4",
    "\u6d77\u8fb9",
    "\u6c99\u6ee9",
    "\u516c\u56ed",
    "\u5546\u573a",
    "\u9910\u5385",
    "\u57ce\u5e02",
    "\u8857\u9053",
    "\u5c71",
    "\u6e56",
]

SCENE_WORDS = [
    "\u5408\u5f71",
    "\u81ea\u62cd",
    "\u4eba\u50cf",
    "\u98ce\u666f",
    "\u5efa\u7b51",
    "\u96ea\u666f",
    "\u591c\u666f",
    "\u84dd\u5929",
    "\u8349\u5730",
    "\u7011\u5e03",
    "\u6d77\u6d6a",
    "\u7f8e\u98df",
    "\u505a\u996d",
    "\u70f9\u996a",
    "\u5496\u5561",
    "\u805a\u9910",
    "\u8fd0\u52a8",
    "\u7bee\u7403",
    "\u8db3\u7403",
    "\u8df3\u821e",
    "\u4f1a\u8bae",
    "\u8bbf\u8c08",
    "\u91c7\u8bbf",
    "\u5bf9\u8bdd",
    "\u6f14\u8bb2",
    "\u8bb2\u5ea7",
    "\u6c47\u62a5",
    "\u5b57\u5e55",
    "\u8bed\u97f3",
    "\u8f6c\u5199",
    "\u89e3\u8bf4",
    "\u65c5\u884c",
    "\u6bd5\u4e1a",
    "\u5ba0\u7269",
    "\u8f66\u8f86",
    "\u82b1",
]

OBJECT_WORDS = [
    "\u72d7",
    "\u732b",
    "\u516c\u4ea4\u8f66",
    "\u5df4\u58eb",
    "\u6c7d\u8f66",
    "\u8f66",
    "\u6469\u6258\u8f66",
    "\u81ea\u884c\u8f66",
    "\u98de\u673a",
    "\u706b\u8f66",
    "\u9e1f",
    "\u52a8\u7269",
    "\u9a6c",
    "\u725b",
    "\u7f8a",
    "\u62ab\u8428",
    "\u4e09\u660e\u6cbb",
    "\u86cb\u7cd5",
    "\u9999\u8549",
    "\u82f9\u679c",
    "\u897f\u5170\u82b1",
    "\u80e1\u841d\u535c",
    "\u4eba",
    "\u5b69\u5b50",
    "\u7537\u4eba",
    "\u5973\u4eba",
    "\u7bee\u7403",
    "\u8db3\u7403",
    "\u5496\u5561",
    "\u6c34\u679c",
    "\u65b9\u5757",
    "\u7403",
    "\u624b\u673a",
    "\u7535\u8bdd",
    "\u667a\u80fd\u624b\u673a",
    "\u5e73\u677f",
    "\u7535\u8111",
    "\u7b14\u8bb0\u672c\u7535\u8111",
]

EN_LOCATION_PROMPTS = {
    "\u6545\u5bab": ["Forbidden City", "Chinese imperial palace", "traditional Chinese palace architecture"],
    "\u957f\u57ce": ["Great Wall of China"],
    "\u5929\u5b89\u95e8": ["Tiananmen Square"],
    "\u9890\u548c\u56ed": ["Summer Palace in Beijing", "Chinese garden"],
    "\u5b66\u6821": ["school campus"],
    "\u6559\u5ba4": ["classroom"],
    "\u529e\u516c\u5ba4": ["office"],
    "\u6d77\u8fb9": ["beach", "seaside", "ocean coast"],
    "\u6c99\u6ee9": ["sand beach"],
    "\u516c\u56ed": ["park"],
    "\u5546\u573a": ["shopping mall"],
    "\u9910\u5385": ["restaurant"],
    "\u57ce\u5e02": ["city street", "urban scene"],
    "\u8857\u9053": ["street"],
    "\u5c71": ["mountain"],
    "\u6e56": ["lake"],
}

EN_SCENE_PROMPTS = {
    "\u5408\u5f71": ["group photo", "people posing together", "portrait of several people"],
    "\u81ea\u62cd": ["selfie", "self portrait"],
    "\u4eba\u50cf": ["portrait photo", "person"],
    "\u98ce\u666f": ["landscape photo", "scenery"],
    "\u5efa\u7b51": ["architecture", "building"],
    "\u96ea\u666f": ["snowy scene", "snow landscape"],
    "\u591c\u666f": ["night scene", "city lights at night"],
    "\u84dd\u5929": ["blue sky"],
    "\u8349\u5730": ["grass field"],
    "\u7011\u5e03": ["waterfall"],
    "\u6d77\u6d6a": ["ocean waves", "sea waves"],
    "\u7f8e\u98df": ["food", "meal"],
    "\u505a\u996d": ["cooking"],
    "\u70f9\u996a": ["cooking"],
    "\u5496\u5561": ["coffee"],
    "\u805a\u9910": ["people eating together", "dinner gathering"],
    "\u8fd0\u52a8": ["sports", "running"],
    "\u7bee\u7403": ["basketball"],
    "\u8db3\u7403": ["football", "soccer"],
    "\u5496\u5561": ["coffee"],
    "\u6c34\u679c": ["fruit"],
    "\u8df3\u821e": ["dance"],
    "\u4f1a\u8bae": ["meeting", "conference room"],
    "\u8bbf\u8c08": ["interview", "talk show", "speaker conversation"],
    "\u91c7\u8bbf": ["interview", "journalism interview", "reporter talking to guest"],
    "\u5bf9\u8bdd": ["conversation", "people talking"],
    "\u6f14\u8bb2": ["lecture", "presentation", "speaker on stage"],
    "\u8bb2\u5ea7": ["lecture", "education talk", "speaker presentation"],
    "\u6c47\u62a5": ["presentation", "conference talk", "slides"],
    "\u5b57\u5e55": ["subtitle", "caption text", "text on screen"],
    "\u8bed\u97f3": ["speech", "audio transcript"],
    "\u8f6c\u5199": ["transcript", "speech text"],
    "\u89e3\u8bf4": ["narration", "voiceover", "speaker"],
    "\u65c5\u884c": ["travel photo", "tourist photo"],
    "\u6bd5\u4e1a": ["graduation photo"],
    "\u5ba0\u7269": ["pet"],
    "\u8f66\u8f86": ["vehicle", "car"],
    "\u82b1": ["flower"],
}

EN_OBJECT_PROMPTS = {
    "\u72d7": ["dog"],
    "\u732b": ["cat"],
    "\u516c\u4ea4\u8f66": ["bus"],
    "\u5df4\u58eb": ["bus"],
    "\u6c7d\u8f66": ["car"],
    "\u8f66": ["car", "vehicle"],
    "\u6469\u6258\u8f66": ["motorcycle"],
    "\u81ea\u884c\u8f66": ["bicycle"],
    "\u98de\u673a": ["airplane"],
    "\u706b\u8f66": ["train"],
    "\u9e1f": ["bird"],
    "\u52a8\u7269": ["animal"],
    "\u9a6c": ["horse"],
    "\u725b": ["cow"],
    "\u7f8a": ["sheep"],
    "\u62ab\u8428": ["pizza"],
    "\u4e09\u660e\u6cbb": ["sandwich"],
    "\u86cb\u7cd5": ["cake"],
    "\u9999\u8549": ["banana"],
    "\u82f9\u679c": ["apple"],
    "\u897f\u5170\u82b1": ["broccoli"],
    "\u80e1\u841d\u535c": ["carrot"],
    "\u4eba": ["person"],
    "\u5b69\u5b50": ["child", "person"],
    "\u7537\u4eba": ["man", "person"],
    "\u5973\u4eba": ["woman", "person"],
    "\u65b9\u5757": ["square", "block"],
    "\u7403": ["ball"],
    "\u5496\u5561": ["coffee"],
    "\u6c34\u679c": ["fruit"],
    "\u624b\u673a": ["mobile phone", "cell phone", "smartphone"],
    "\u667a\u80fd\u624b\u673a": ["smartphone", "mobile phone"],
    "\u7535\u8bdd": ["phone", "telephone"],
    "\u5e73\u677f": ["tablet computer"],
    "\u7535\u8111": ["computer", "laptop"],
    "\u7b14\u8bb0\u672c\u7535\u8111": ["laptop computer"],
}

COLOR_WORDS = {
    "\u7ea2\u8272": ["red"],
    "\u7ea2": ["red"],
    "\u84dd\u8272": ["blue"],
    "\u84dd": ["blue"],
    "\u7eff\u8272": ["green"],
    "\u7eff": ["green"],
    "\u9ec4\u8272": ["yellow"],
    "\u9ec4": ["yellow"],
    "\u767d\u8272": ["white"],
    "\u767d": ["white"],
    "\u9ed1\u8272": ["black"],
    "\u9ed1": ["black"],
}

ACTION_WORDS = {
    "\u79fb\u52a8": ["moving"],
    "\u8dd1": ["running"],
    "\u884c\u9a76": ["driving"],
    "\u9a91": ["riding"],
    "\u98de": ["flying"],
    "\u5403": ["eating"],
    "\u505a\u996d": ["cooking"],
    "\u70f9\u996a": ["cooking"],
    "\u8d70": ["walking"],
    "\u8df3\u821e": ["dancing"],
    "\u62ff\u7740": ["holding"],
    "\u624b\u6301": ["holding"],
    "\u4e3e\u7740": ["holding"],
    "\u6253\u7535\u8bdd": ["talking on the phone", "calling"],
    "\u901a\u8bdd": ["talking on the phone", "phone call"],
    "\u63a5\u7535\u8bdd": ["answering a phone call"],
    "\u770b\u624b\u673a": ["looking at a phone"],
    "\u73a9\u624b\u673a": ["using a phone"],
    "\u62cd\u7167": ["taking a photo"],
    "\u6444\u5f71": ["taking a photo"],
}

SPATIAL_RELATIONS = {
    "\u5de6\u8fb9": "left_of",
    "\u5de6\u4fa7": "left_of",
    "\u5de6": "left_of",
    "\u53f3\u8fb9": "right_of",
    "\u53f3\u4fa7": "right_of",
    "\u53f3": "right_of",
    "\u4e0a": "on",
    "\u4e0a\u9762": "on",
    "\u8f66\u4e0a": "on",
    "\u4e0b": "under",
    "\u4e0b\u9762": "under",
    "\u5e95": "under",
    "\u8f66\u5e95": "under",
    "\u91cc": "inside",
    "\u91cc\u9762": "inside",
    "\u5185": "inside",
    "\u65c1\u8fb9": "beside",
    "\u65c1": "beside",
    "\u9644\u8fd1": "near",
    "\u540e\u9762": "behind",
    "\u524d\u9762": "in front of",
}

EN_TIME_PROMPTS = {
    "\u6625\u5929": ["spring"],
    "\u590f\u5929": ["summer"],
    "\u79cb\u5929": ["autumn", "fall"],
    "\u51ac\u5929": ["winter", "cold weather"],
    "\u665a\u4e0a": ["night"],
    "\u591c\u665a": ["night"],
}


@dataclass
class QueryPlan:
    raw_query: str
    intent: str
    semantic_queries: list[str]
    executable_filters: dict[str, str | int | None]
    unresolved_conditions: dict[str, list[str]]
    recall_routes: list[str]
    rerank_policy: str
    downstream_requirements: list[str]
    execution_steps: list[str]
    query_rewrites: list[str]
    clarification_hints: list[str]
    agent_summary: str
    query_quality: dict[str, int | str]
    trace: list[dict[str, object]]
    relation_queries: list[str]
    negative_relation_queries: list[str]
    strict_conditions: dict[str, object]
    negative_conditions: dict[str, list[str]]

    def to_dict(self) -> dict:
        return asdict(self)


class SearchAgent:
    """Agent planner owned by role A.

    It parses the query and schedules retrieval, but it does not pretend to
    solve capabilities owned by other roles. For example, a person name such as
    "Xiao Ming" is an unresolved person constraint until role B provides a
    face/person index.
    """

    def build_plan(self, query: str, now: datetime | None = None) -> QueryPlan:
        now = now or datetime.now()
        normalized = " ".join(query.strip().split())
        unresolved = {
            "people": extract_people(normalized),
            "locations": extract_location_words(normalized),
            "scenes": extract_scene_words(normalized),
            "objects": extract_object_words(normalized),
            "relations": extract_spatial_relations(normalized),
            "colors": extract_mapped_words(normalized, COLOR_WORDS),
            "actions": extract_mapped_words(normalized, ACTION_WORDS),
            "media": extract_media_words(normalized),
            "time_words": extract_time_words(normalized),
            "text_signals": extract_text_signal_words(normalized),
            "quality": extract_quality_words(normalized),
        }
        negative_conditions = extract_negative_conditions(normalized)
        unresolved = remove_negative_from_positive(unresolved, negative_conditions)
        strict_conditions = infer_strict_conditions(normalized)
        executable_filters = infer_executable_filters(normalized, now)
        semantic_queries = build_semantic_queries(normalized, unresolved)
        relation_queries = build_relation_queries(unresolved)
        negative_relation_queries = build_negative_relation_queries(unresolved)
        recall_routes = ["clip_text_image"]

        if executable_filters.get("kind") or executable_filters.get("year") or executable_filters.get("season"):
            recall_routes.append("metadata_filter")
        if normalized:
            recall_routes.append("filename_keyword")
        if has_non_empty(unresolved):
            recall_routes.append("deferred_structured_recall")
        recall_routes.append("tag_text_recall")

        return QueryPlan(
            raw_query=normalized,
            intent=infer_intent(normalized),
            semantic_queries=semantic_queries,
            executable_filters=executable_filters,
            unresolved_conditions={key: value for key, value in unresolved.items() if value},
            recall_routes=recall_routes,
            rerank_policy="0.63*clip + 0.14*tags + 0.06*metadata + 0.09*filename + 0.03*ocr + 0.05*relation",
            downstream_requirements=build_downstream_requirements(unresolved),
            execution_steps=build_execution_steps(normalized, unresolved, executable_filters),
            query_rewrites=semantic_queries[:6],
            clarification_hints=build_clarification_hints(normalized, unresolved, executable_filters),
            agent_summary=build_agent_summary(normalized, unresolved, executable_filters),
            query_quality=score_query_quality(normalized, unresolved, executable_filters),
            trace=build_agent_trace(
                normalized,
                unresolved,
                executable_filters,
                recall_routes,
                semantic_queries,
                relation_queries,
                negative_relation_queries,
                strict_conditions,
                negative_conditions,
            ),
            relation_queries=relation_queries,
            negative_relation_queries=negative_relation_queries,
            strict_conditions=strict_conditions,
            negative_conditions=negative_conditions,
        )


def infer_intent(query: str) -> str:
    search_words = [
        "\u627e",
        "\u641c\u7d22",
        "\u68c0\u7d22",
        "\u67e5",
        "\u7b5b\u9009",
        "\u7167\u7247",
        "\u56fe\u7247",
        "\u7d20\u6750",
        "\u5408\u5f71",
    ]
    if any(word in query for word in search_words):
        return "asset_search"
    if any(word in query.lower() for word in ["search", "find", "photo", "image", "asset"]):
        return "asset_search"
    return "semantic_retrieval"


def extract_people(query: str) -> list[str]:
    people = []
    for marker in PERSON_MARKERS:
        if marker in query:
            tail = query.split(marker, 1)[1]
            name = re.split(PERSON_BOUNDARY_PATTERN, tail, maxsplit=1)[0]
            if 1 <= len(name) <= 8:
                people.append(name)
    for match in re.finditer(r"\u5c0f[\u4e00-\u9fffA-Za-z0-9_]{1,8}", query):
        name = re.split(PERSON_BOUNDARY_PATTERN, match.group(0), maxsplit=1)[0]
        if 2 <= len(name) <= 8:
            people.append(name)
    for match in re.finditer(r"person[_-]?\d+", query, flags=re.IGNORECASE):
        people.append(normalize_person_id(match.group(0)))
    stop_words = {
        "\u5408\u5f71",
        "\u7167\u7247",
        "\u56fe\u7247",
        "\u7d20\u6750",
        "\u4e00\u8d77",
        "\u53bb\u5e74",
        "\u4eca\u5e74",
        "\u51ac\u5929",
        "\u590f\u5929",
        "clip",
        "agent",
        "ocr",
        "asr",
        "ppt",
        "pdf",
    }
    return unique([item for item in people if item.lower() not in stop_words])


def extract_time_words(query: str) -> list[str]:
    found = [word for word in TIME_WORDS if word in query]
    found.extend(
        match.group(0)
        for match in re.finditer(r"\d{4}\u5e74?", query)
        if not re.search(r"person[_-]?$", query[: match.start()], flags=re.IGNORECASE)
    )
    return unique(found)


def extract_location_words(query: str) -> list[str]:
    found = [word for word in sorted(LOCATION_WORDS, key=len, reverse=True) if word in query]
    found.extend(re.findall(r"\u5728([\u4e00-\u9fffA-Za-z0-9_]{2,12}?)(?:\u62cd|\u7684|\u5408\u5f71|\u7167\u7247|\u56fe\u7247)", query))
    invalid = {"\u8f66\u5e95", "\u8f66\u4e0a", "\u5de6\u8fb9", "\u53f3\u8fb9"}
    return unique([item for item in found if item not in invalid and not re.match(r"person[_-]?\d+", item, flags=re.IGNORECASE)])


def extract_scene_words(query: str) -> list[str]:
    return unique([word for word in sorted(SCENE_WORDS, key=len, reverse=True) if word in query])


def extract_object_words(query: str) -> list[str]:
    return unique([word for word in sorted(OBJECT_WORDS, key=len, reverse=True) if word in query])


def extract_spatial_relations(query: str) -> list[str]:
    entities = extract_relation_entities(query)
    if len(entities) < 2:
        return []
    query_for_match = normalize_person_mentions(query)
    relations = []
    for subject_key, subject_prompt in entities:
        for target_key, target_prompt in entities:
            if subject_key == target_key:
                continue
            for marker, relation in sorted(SPATIAL_RELATIONS.items(), key=lambda item: len(item[0]), reverse=True):
                patterns = [
                    f"{re.escape(subject_key)}.*?{re.escape(marker)}.*?{re.escape(target_key)}",
                    f"{re.escape(subject_key)}.*?{re.escape(target_key)}{re.escape(marker)}",
                ]
                if any(re.search(pattern, query_for_match) for pattern in patterns):
                    relations.append(f"{subject_prompt} {relation} {target_prompt}")
    return unique(relations)


def extract_relation_entities(query: str) -> list[tuple[str, str]]:
    entities = [(word, first_prompt(word, EN_OBJECT_PROMPTS)) for word in extract_object_words(query)]
    entities.extend((person, person_relation_prompt(person)) for person in extract_people(query))
    return unique_pairs(entities)


def extract_mapped_words(query: str, mapping: dict[str, list[str]]) -> list[str]:
    return unique([word for word in sorted(mapping, key=len, reverse=True) if word in query])


def extract_media_words(query: str) -> list[str]:
    media = []
    if any(word in query for word in ["\u77ed\u89c6\u9891", "\u89c6\u9891", "\u7247\u6bb5"]):
        media.append("video")
    if any(word in query for word in ["\u7167\u7247", "\u56fe\u7247", "\u56fe\u50cf", "\u76f8\u7247"]):
        media.append("image")
    if any(word in query.lower() for word in ["\u6587\u6863", "\u62a5\u544a", "pdf", "docx", "pptx", "md"]):
        media.append("document")
    return media


def extract_text_signal_words(query: str) -> list[str]:
    signals = []
    lowered = query.lower()
    if any(word in lowered for word in ["ocr", "\u753b\u9762\u6587\u5b57", "\u56fe\u7247\u91cc\u7684\u5b57", "\u5e26\u6587\u5b57"]):
        signals.append("ocr")
    if any(word in lowered for word in ["asr", "\u97f3\u9891", "\u8bed\u97f3", "\u8bf4\u5230", "\u8bb2\u5230", "\u8f6c\u5199"]):
        signals.append("asr")
    if any(word in query for word in ["\u5b57\u5e55", "\u5e26\u5b57\u5e55"]):
        signals.append("subtitle")
    if any(word in query for word in ["\u6587\u6863", "\u6b63\u6587", "\u62a5\u544a"]):
        signals.append("document")
    return unique(signals)


def extract_quality_words(query: str) -> list[str]:
    quality = []
    if any(word in query for word in ["\u4e0d\u8981\u592a\u6697", "\u6392\u9664\u592a\u6697", "\u4e0d\u592a\u6697"]):
        quality.append("exclude_dark")
    if any(word in query for word in ["\u6e05\u6670", "\u9ad8\u6e05", "\u6e05\u695a"]):
        quality.append("clear")
    if any(word in query for word in ["\u4e0d\u8981\u91cd\u590d", "\u6392\u9664\u91cd\u590d", "\u53bb\u91cd"]):
        quality.append("deduplicate")
    if any(word in query for word in ["\u4e3b\u4f53\u660e\u786e", "\u6709\u4e3b\u4f53"]):
        quality.append("clear_subject")
    return unique(quality)


def extract_negative_conditions(query: str) -> dict[str, list[str]]:
    negative_text = []
    for marker in ["\u4e0d\u8981", "\u4e0d\u542b", "\u4e0d\u5305\u542b", "\u6392\u9664", "\u4e0d\u662f", "\u975e"]:
        start = 0
        while True:
            index = query.find(marker, start)
            if index < 0:
                break
            fragment = query[index + len(marker): index + len(marker) + 18]
            fragment = re.split(r"[,，。；;\s]|\u7684|\u4e14|\u5e76\u4e14|\u6216|\u4f46", fragment, maxsplit=1)[0]
            if fragment:
                negative_text.append(fragment)
            start = index + len(marker)
    joined = " ".join(negative_text)
    return {
        "objects": extract_object_words(joined),
        "scenes": extract_scene_words(joined),
        "media": extract_media_words(joined),
        "raw": unique(negative_text),
    }


def remove_negative_from_positive(
    unresolved: dict[str, list[str]],
    negative_conditions: dict[str, list[str]],
) -> dict[str, list[str]]:
    cleaned = {key: list(value) for key, value in unresolved.items()}
    for key in ["objects", "scenes", "media"]:
        negative_values = set(negative_conditions.get(key, []))
        if negative_values:
            cleaned[key] = [item for item in cleaned.get(key, []) if item not in negative_values]
    negative_object_prompts = {
        first_prompt(item, EN_OBJECT_PROMPTS)
        for item in negative_conditions.get("objects", [])
    }
    if negative_object_prompts:
        cleaned["relations"] = [
            relation
            for relation in cleaned.get("relations", [])
            if not any(prompt and prompt in relation for prompt in negative_object_prompts)
        ]
    return cleaned


def infer_strict_conditions(query: str) -> dict[str, object]:
    conditions: dict[str, object] = {}
    if any(word in query for word in ["\u4e2a\u4eba\u5e93", "\u6211\u7684\u7d20\u6750", "\u6211\u7684\u76f8\u518c", "\u79c1\u4eba"]):
        conditions["library"] = "personal"
    elif any(word in query for word in ["\u516c\u5171\u5e93", "\u516c\u5171\u7d20\u6750"]):
        conditions["library"] = "public"

    people_count = infer_people_count(query)
    if people_count:
        conditions["people_count"] = people_count

    if any(word in query for word in ["\u5fc5\u987b", "\u53ea\u8981", "\u4e00\u5b9a\u8981", "\u4e25\u683c"]):
        conditions["strict_mode"] = True
    return conditions


def infer_people_count(query: str) -> int | None:
    for count, words in [
        (2, ["\u4e24\u4e2a\u4eba", "\u4e8c\u4eba", "\u4e24\u4eba", "\u53cc\u4eba"]),
        (3, ["\u4e09\u4e2a\u4eba", "\u4e09\u4eba"]),
        (4, ["\u56db\u4e2a\u4eba", "\u56db\u4eba"]),
    ]:
        if any(word in query for word in words):
            return count
    match = re.search(r"(\d+)\s*\u4e2a?\u4eba", query)
    return int(match.group(1)) if match else None


def infer_executable_filters(query: str, now: datetime) -> dict[str, str | int | None]:
    year: int | None = None
    if "\u53bb\u5e74" in query:
        year = now.year - 1
    elif "\u524d\u5e74" in query:
        year = now.year - 2
    elif "\u4eca\u5e74" in query:
        year = now.year
    else:
        match = next(
            (
                item
                for item in re.finditer(r"(\d{4})\u5e74?", query)
                if not re.search(r"person[_-]?$", query[: item.start()], flags=re.IGNORECASE)
            ),
            None,
        )
        if match:
            year = int(match.group(1))

    season = None
    for word in ["\u6625\u5929", "\u590f\u5929", "\u79cb\u5929", "\u51ac\u5929"]:
        if word in query:
            season = word
            break

    media_kind = None
    if any(word in query for word in ["\u77ed\u89c6\u9891", "\u89c6\u9891", "\u7247\u6bb5"]):
        media_kind = "video"
    elif any(word in query for word in ["\u7167\u7247", "\u56fe\u7247", "\u56fe\u50cf", "\u5408\u5f71", "\u76f8\u7247"]):
        media_kind = "image"
    elif any(word in query.lower() for word in ["\u6587\u6863", "\u62a5\u544a", "pdf", "docx", "pptx", "md"]):
        media_kind = "document"

    return {"year": year, "season": season, "kind": media_kind}


def build_semantic_queries(query: str, unresolved: dict[str, list[str]]) -> list[str]:
    queries = [query] if query else []
    location_text = " ".join(unresolved.get("locations", []))
    scene_text = " ".join(unresolved.get("scenes", []))
    if location_text or scene_text:
        queries.append(" ".join(item for item in [location_text, scene_text] if item))
    queries.extend(build_visual_prompts(unresolved))
    return unique([item.strip() for item in queries if item.strip()])


def build_visual_prompts(unresolved: dict[str, list[str]]) -> list[str]:
    locations = flatten_prompts(unresolved.get("locations", []), EN_LOCATION_PROMPTS)
    scenes = flatten_prompts(unresolved.get("scenes", []), EN_SCENE_PROMPTS)
    objects = flatten_prompts(unresolved.get("objects", []), EN_OBJECT_PROMPTS)
    colors = flatten_prompts(unresolved.get("colors", []), COLOR_WORDS)
    actions = flatten_prompts(unresolved.get("actions", []), ACTION_WORDS)
    times = flatten_prompts(unresolved.get("time_words", []), EN_TIME_PROMPTS)
    relations = unresolved.get("relations", [])
    wants_video = "video" in unresolved.get("media", [])
    prompts = []
    for relation in relations:
        relation_prompt = relation_to_prompt(relation)
        if wants_video:
            prompts.append(f"a video of {relation_prompt}")
        prompts.append(f"a photo of {relation_prompt}")
        prompts.append(relation_prompt)
    descriptor = " ".join(unique([*colors[:1], *actions[:1], *objects[:1]])).strip()
    if descriptor:
        if wants_video:
            prompts.append(f"a video of {descriptor}")
        prompts.append(f"a photo of {descriptor}")
    interaction_prompts = build_interaction_prompts(objects, actions, wants_video)
    prompts.extend(interaction_prompts)
    scene_descriptor = " ".join(unique([*colors[:1], *scenes[:2]])).strip()
    if wants_video and scene_descriptor:
        prompts.append(f"a video of {scene_descriptor}")
    if scene_descriptor:
        prompts.append(f"a photo of {scene_descriptor}")
    if objects and scenes:
        prompts.append(f"a photo of {objects[0]} in {scenes[0]}")
    if objects and locations:
        prompts.append(f"a photo of {objects[0]} at {locations[0]}")
    if objects:
        prompts.append(f"a photo of {objects[0]}")
    if locations and scenes:
        prompts.append(f"a {scenes[0]} at {locations[0]}")
    if scenes:
        prompts.append(f"a photo of {scenes[0]}")
    if locations:
        prompts.append(f"a photo of {locations[0]}")
    if locations and scenes:
        prompts.append(" ".join(unique([*colors[:1], *actions[:1], *objects[:2], *scenes[:2], *locations[:2], *times[:1]])))
    if times and (scenes or objects):
        prompts.append(f"a {times[0]} {(objects or scenes)[0]}")
    return prompts


def build_interaction_prompts(objects: list[str], actions: list[str], wants_video: bool) -> list[str]:
    prompts = []
    object_set = set(objects)
    action_set = set(actions)
    phone_objects = {"mobile phone", "cell phone", "smartphone", "phone", "telephone"}
    holding_actions = {"holding"}
    calling_actions = {"talking on the phone", "calling", "phone call", "answering a phone call"}
    if object_set & phone_objects and action_set & (holding_actions | calling_actions):
        base_prompts = [
            "a person holding a mobile phone",
            "a person talking on the phone",
            "a person making a phone call",
            "a person using a smartphone",
        ]
        for prompt in base_prompts:
            if wants_video:
                prompts.append(prompt.replace("a person", "a video of a person", 1))
            prompts.append(prompt.replace("a person", "a photo of a person", 1))
    if {"computer", "laptop", "laptop computer"} & object_set:
        if {"using", "looking", "working"} & action_set:
            prompts.append("a person using a laptop computer")
    return unique(prompts)


def build_relation_queries(unresolved: dict[str, list[str]]) -> list[str]:
    queries = []
    for relation in unresolved.get("relations", []):
        prompt = relation_to_prompt(relation)
        queries.extend([f"a photo of {prompt}", prompt])
    return unique(queries)


def build_negative_relation_queries(unresolved: dict[str, list[str]]) -> list[str]:
    negatives = []
    for relation in unresolved.get("relations", []):
        parts = relation.split()
        if len(parts) < 3:
            continue
        subject = parts[0]
        predicate = " ".join(parts[1:-1])
        target = parts[-1]
        for negative_predicate in relation_contrasts(predicate):
            negative = f"{subject} {negative_predicate} {target}"
            prompt = relation_to_prompt(negative)
            negatives.extend([f"a photo of {prompt}", prompt])
    return unique(negatives)


def relation_contrasts(predicate: str) -> list[str]:
    contrasts = {
        "on": ["under", "inside", "beside"],
        "under": ["on", "inside", "beside"],
        "left_of": ["right_of", "on", "under"],
        "right_of": ["left_of", "on", "under"],
        "inside": ["on", "under", "beside"],
        "beside": ["on", "under", "inside"],
        "near": ["on", "under", "inside"],
        "behind": ["in front of", "on", "beside"],
        "in front of": ["behind", "on", "beside"],
    }
    return contrasts.get(predicate, ["on", "under", "inside", "beside"])


def flatten_prompts(words: list[str], mapping: dict[str, list[str]]) -> list[str]:
    prompts = []
    for word in words:
        prompts.extend(mapping.get(word, []))
    return unique(prompts)


def first_prompt(word: str, mapping: dict[str, list[str]]) -> str:
    prompts = mapping.get(word, [])
    return prompts[0] if prompts else word


def normalize_person_id(value: str) -> str:
    match = re.search(r"person[_-]?(\d+)", value, flags=re.IGNORECASE)
    if not match:
        return value
    number = match.group(1)
    if len(number) < 4:
        number = number.zfill(4)
    return f"person_{number}"


def normalize_person_mentions(text: str) -> str:
    return re.sub(
        r"person[_-]?(\d+)",
        lambda match: normalize_person_id(match.group(0)),
        text,
        flags=re.IGNORECASE,
    )


def person_relation_prompt(person_id: str) -> str:
    return person_id


def relation_to_prompt(relation: str) -> str:
    prompt = re.sub(
        r"person_(\d+)",
        lambda match: f"person {int(match.group(1))}",
        relation,
        flags=re.IGNORECASE,
    )
    return prompt.replace("left_of", "left of").replace("right_of", "right of")


def unique_pairs(values: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    seen = set()
    result = []
    for key, label in values:
        pair = (key, label)
        if key and pair not in seen:
            seen.add(pair)
            result.append(pair)
    return result


def build_downstream_requirements(unresolved: dict[str, list[str]]) -> list[str]:
    requirements = []
    if unresolved.get("people"):
        requirements.append("B.face_person_index: resolve person names to person_id")
    if unresolved.get("relations"):
        requirements.append("B.vlm_or_detection_relation_check: verify object spatial relations")
    if unresolved.get("locations") or unresolved.get("scenes"):
        requirements.append("B.vlm_scene_tags and C.semantic_tags: provide structured labels")
    if unresolved.get("time_words"):
        requirements.append("D.sqlite_metadata: provide reliable capture_time metadata")
    return requirements


def build_execution_steps(
    query: str,
    unresolved: dict[str, list[str]],
    executable_filters: dict[str, str | int | None],
) -> list[str]:
    steps = ["解析用户检索意图"]
    if has_non_empty(unresolved):
        steps.append("拆解人物、场景、物体、动作、地点、时间、空间关系等条件")
    if unresolved.get("relations"):
        steps.append("生成关系感知提示词，区分 on / under / inside / beside 等组合语义")
    if executable_filters.get("kind") or executable_filters.get("year") or executable_filters.get("season"):
        steps.append("把素材类型、年份、季节转成可执行过滤条件")
    if query:
        steps.append("生成中英文语义提示词并调用 CLIP 召回")
    if unresolved.get("locations") or unresolved.get("scenes") or unresolved.get("objects"):
        steps.append("结合标签、OCR/ASR 文本和文件名做补充命中")
    steps.append("按 CLIP、标签、元数据、文件名、文本信号融合重排")
    return steps


def build_clarification_hints(
    query: str,
    unresolved: dict[str, list[str]],
    executable_filters: dict[str, str | int | None],
) -> list[str]:
    hints = []
    if not query:
        return ["请输入想找的画面、人物、地点或素材类型。"]
    if unresolved.get("people"):
        names = "、".join(unresolved["people"])
        hints.append(f"检测到人物条件：{names}。如果这些名字还没有绑定到人物相册，系统只能先按“有人/合影”语义检索，不能真正知道名字对应谁。")
    if unresolved.get("time_words") and not executable_filters.get("year"):
        hints.append("检测到时间描述，但素材需要有拍摄时间元数据，才能做稳定的年份或季节过滤。")
    if unresolved.get("relations"):
        hints.append("检测到物体空间关系。当前先用关系提示词增强 CLIP 检索，若要严格区分“在车上/车底”，后续应接入目标检测或 VLM 关系复核。")
    if not (unresolved.get("scenes") or unresolved.get("objects") or unresolved.get("locations") or unresolved.get("actions")):
        hints.append("可以加入更具体的画面条件，例如“室内采访视频”“带字幕的演讲”“海边人物合影”。")
    if "video" in unresolved.get("media", []) and not (unresolved.get("actions") or unresolved.get("scenes")):
        hints.append("视频检索建议补充动作或场景，例如“做饭视频”“访谈视频”“城市街道行驶视频”。")
    if len(query) <= 2:
        hints.append("当前查询很短，结果会更依赖 CLIP 的大概语义，相似但不精确的素材可能会靠前。")
    return unique(hints)


def build_agent_summary(
    query: str,
    unresolved: dict[str, list[str]],
    executable_filters: dict[str, str | int | None],
) -> str:
    if not query:
        return "等待用户输入检索目标。"
    parts = []
    if executable_filters.get("kind"):
        parts.append("只看视频" if executable_filters["kind"] == "video" else "只看图片")
    if unresolved.get("people"):
        parts.append("包含人物条件")
    if unresolved.get("locations"):
        parts.append("包含地点条件")
    if unresolved.get("scenes"):
        parts.append("包含场景条件")
    if unresolved.get("objects"):
        parts.append("包含物体条件")
    if unresolved.get("actions"):
        parts.append("包含动作条件")
    if unresolved.get("relations"):
        parts.append("包含物体空间关系")
    if unresolved.get("time_words"):
        parts.append("包含时间条件")
    if not parts:
        return "这是一个较宽泛的语义检索，主要依赖 CLIP 视觉语义召回。"
    return "本次查询将按“" + "、".join(parts) + "”组合检索。"


def score_query_quality(
    query: str,
    unresolved: dict[str, list[str]],
    executable_filters: dict[str, str | int | None],
) -> dict[str, int | str]:
    if not query:
        return {"score": 0, "level": "empty", "message": "尚未输入查询。"}
    signal_count = sum(1 for values in unresolved.values() if values)
    if executable_filters.get("kind"):
        signal_count += 1
    if executable_filters.get("year") or executable_filters.get("season"):
        signal_count += 1
    length_bonus = 1 if len(query) >= 4 else 0
    score = min(100, 28 + signal_count * 14 + length_bonus * 10)
    if score >= 76:
        level = "good"
        message = "查询条件比较完整，适合多路召回和融合重排。"
    elif score >= 52:
        level = "medium"
        message = "查询可用，但增加场景、动作、地点或素材类型会更稳定。"
    else:
        level = "weak"
        message = "查询较短或条件较少，结果更容易偏泛化。"
    return {"score": score, "level": level, "message": message}


def build_agent_trace(
    query: str,
    unresolved: dict[str, list[str]],
    executable_filters: dict[str, str | int | None],
    recall_routes: list[str],
    semantic_queries: list[str],
    relation_queries: list[str],
    negative_relation_queries: list[str],
    strict_conditions: dict[str, object],
    negative_conditions: dict[str, list[str]],
) -> list[dict[str, object]]:
    trace = [
        {
            "stage": "intent_parse",
            "title": "意图解析",
            "detail": infer_intent(query),
        },
        {
            "stage": "condition_split",
            "title": "条件拆解",
            "detail": {
                "positive": {key: value for key, value in unresolved.items() if value},
                "strict": strict_conditions,
                "negative": {key: value for key, value in negative_conditions.items() if value},
            },
        },
        {
            "stage": "query_rewrite",
            "title": "语义改写",
            "detail": semantic_queries[:8],
        },
    ]
    if relation_queries:
        trace.append(
            {
                "stage": "relation_contrast",
                "title": "关系对比",
                "detail": {
                    "positive": relation_queries[:6],
                    "negative": negative_relation_queries[:8],
                },
            }
        )
    trace.extend(
        [
        {
            "stage": "route_plan",
            "title": "召回路线",
            "detail": recall_routes,
        },
        {
            "stage": "filter_plan",
            "title": "可执行过滤",
            "detail": {
                **{key: value for key, value in executable_filters.items() if value},
                **strict_conditions,
            },
        },
        ]
    )
    return trace


def has_non_empty(values: dict[str, Iterable[str]]) -> bool:
    return any(bool(list(item)) for item in values.values())


def unique(values: Iterable[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        value = value.strip()
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result
