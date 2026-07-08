from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Protocol


class AssetLike(Protocol):
    id: str
    filename: str
    path: str
    kind: str
    library: str
    created_at: str
    width: int
    height: int
    vector_model: str


class MetadataStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_schema()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_schema(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS assets (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    path TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    library TEXT NOT NULL DEFAULT 'public',
                    created_at TEXT NOT NULL,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    vector_model TEXT NOT NULL,
                    sha256 TEXT,
                    indexed_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_assets_kind ON assets(kind);
                CREATE INDEX IF NOT EXISTS idx_assets_vector_model ON assets(vector_model);
                CREATE INDEX IF NOT EXISTS idx_assets_created_at ON assets(created_at);
                CREATE INDEX IF NOT EXISTS idx_assets_sha256 ON assets(sha256);

                CREATE TABLE IF NOT EXISTS tags (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    source TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS asset_tags (
                    asset_id TEXT NOT NULL,
                    tag_id INTEGER NOT NULL,
                    confidence REAL NOT NULL DEFAULT 1.0,
                    source TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(asset_id, tag_id, source),
                    FOREIGN KEY(asset_id) REFERENCES assets(id),
                    FOREIGN KEY(tag_id) REFERENCES tags(id)
                );

                CREATE INDEX IF NOT EXISTS idx_asset_tags_asset ON asset_tags(asset_id);
                CREATE INDEX IF NOT EXISTS idx_asset_tags_tag ON asset_tags(tag_id);

                CREATE TABLE IF NOT EXISTS ocr_texts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    asset_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    engine TEXT NOT NULL,
                    confidence REAL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(asset_id) REFERENCES assets(id)
                );

                CREATE INDEX IF NOT EXISTS idx_ocr_texts_asset ON ocr_texts(asset_id);

                CREATE TABLE IF NOT EXISTS document_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    asset_id TEXT NOT NULL,
                    chunk_id TEXT NOT NULL UNIQUE,
                    chunk_index INTEGER NOT NULL,
                    page_start INTEGER,
                    page_end INTEGER,
                    section_title TEXT,
                    heading_path TEXT,
                    chunk_type TEXT NOT NULL DEFAULT 'text',
                    text TEXT NOT NULL,
                    summary TEXT,
                    embedding_text TEXT,
                    keywords TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(asset_id) REFERENCES assets(id)
                );

                CREATE INDEX IF NOT EXISTS idx_document_chunks_asset ON document_chunks(asset_id);
                CREATE INDEX IF NOT EXISTS idx_document_chunks_chunk ON document_chunks(chunk_id);

                CREATE TABLE IF NOT EXISTS search_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    plan_json TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    backend TEXT NOT NULL,
                    top_score REAL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_search_logs_created_at ON search_logs(created_at);

                CREATE TABLE IF NOT EXISTS persons (
                    person_id TEXT PRIMARY KEY,
                    library TEXT NOT NULL DEFAULT 'personal',
                    alias TEXT,
                    display_name TEXT,
                    face_count INTEGER NOT NULL DEFAULT 0,
                    prototype_face_id TEXT,
                    confidence REAL NOT NULL DEFAULT 1.0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_persons_library ON persons(library);

                CREATE TABLE IF NOT EXISTS asset_faces (
                    face_id TEXT PRIMARY KEY,
                    asset_id TEXT NOT NULL,
                    person_id TEXT NOT NULL,
                    bbox_x1 INTEGER NOT NULL,
                    bbox_y1 INTEGER NOT NULL,
                    bbox_x2 INTEGER NOT NULL,
                    bbox_y2 INTEGER NOT NULL,
                    confidence REAL NOT NULL,
                    quality REAL NOT NULL,
                    source TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(asset_id) REFERENCES assets(id),
                    FOREIGN KEY(person_id) REFERENCES persons(person_id)
                );

                CREATE INDEX IF NOT EXISTS idx_asset_faces_asset ON asset_faces(asset_id);
                CREATE INDEX IF NOT EXISTS idx_asset_faces_person ON asset_faces(person_id);
                """
            )
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(assets)").fetchall()}
            if "library" not in columns:
                conn.execute("ALTER TABLE assets ADD COLUMN library TEXT NOT NULL DEFAULT 'public'")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_assets_library ON assets(library)")

    def sync_assets(self, assets: list[AssetLike], root: Path | None = None) -> None:
        seen_ids = {asset.id for asset in assets}
        with self.connect() as conn:
            for asset in assets:
                self.upsert_asset(conn, asset, root)
            if seen_ids:
                placeholders = ",".join("?" for _ in seen_ids)
                conn.execute(f"DELETE FROM assets WHERE id NOT IN ({placeholders})", tuple(seen_ids))
            else:
                conn.execute("DELETE FROM assets")

    def sync_asset(self, asset: AssetLike, root: Path | None = None) -> None:
        with self.connect() as conn:
            self.upsert_asset(conn, asset, root)

    def upsert_asset(self, conn: sqlite3.Connection, asset: AssetLike, root: Path | None = None) -> None:
        path = Path(asset.path)
        if not path.is_absolute() and root is not None:
            path = root / path
        file_hash = sha256_file(path) if path.exists() else None
        conn.execute(
            """
            INSERT INTO assets (
                id, filename, path, kind, created_at, width, height,
                library, vector_model, sha256, indexed_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                filename=excluded.filename,
                path=excluded.path,
                kind=excluded.kind,
                library=excluded.library,
                created_at=excluded.created_at,
                width=excluded.width,
                height=excluded.height,
                vector_model=excluded.vector_model,
                sha256=excluded.sha256,
                indexed_at=excluded.indexed_at
            """,
            (
                asset.id,
                asset.filename,
                asset.path,
                asset.kind,
                asset.created_at,
                asset.width,
                asset.height,
                asset.library,
                asset.vector_model,
                file_hash,
                datetime.now().isoformat(timespec="seconds"),
            ),
        )

    def stats(self) -> dict:
        with self.connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM assets").fetchone()[0]
            by_kind = rows_to_dict(conn.execute("SELECT kind, COUNT(*) AS count FROM assets GROUP BY kind"))
            by_library = rows_to_dict(conn.execute("SELECT library, COUNT(*) AS count FROM assets GROUP BY library"))
            by_model = rows_to_dict(
                conn.execute("SELECT vector_model, COUNT(*) AS count FROM assets GROUP BY vector_model")
            )
            duplicate_groups = conn.execute(
                """
                SELECT COUNT(*) FROM (
                    SELECT sha256 FROM assets
                    WHERE sha256 IS NOT NULL
                    GROUP BY sha256
                    HAVING COUNT(*) > 1
                )
                """
            ).fetchone()[0]
            duplicate_assets = conn.execute(
                """
                SELECT COALESCE(SUM(count - 1), 0) FROM (
                    SELECT COUNT(*) AS count FROM assets
                    WHERE sha256 IS NOT NULL
                    GROUP BY sha256
                    HAVING COUNT(*) > 1
                )
                """
            ).fetchone()[0]
            search_log_count = conn.execute("SELECT COUNT(*) FROM search_logs").fetchone()[0]
            tag_count = conn.execute("SELECT COUNT(*) FROM tags").fetchone()[0]
            asset_tag_count = conn.execute("SELECT COUNT(*) FROM asset_tags").fetchone()[0]
            ocr_text_count = conn.execute("SELECT COUNT(*) FROM ocr_texts").fetchone()[0]
            document_chunk_count = conn.execute("SELECT COUNT(*) FROM document_chunks").fetchone()[0]
            person_count = conn.execute("SELECT COUNT(*) FROM persons").fetchone()[0]
            face_count = conn.execute("SELECT COUNT(*) FROM asset_faces").fetchone()[0]
        return {
            "total_assets": total,
            "by_kind": by_kind,
            "by_library": by_library,
            "by_vector_model": by_model,
            "duplicate_groups": duplicate_groups,
            "duplicate_assets": duplicate_assets,
            "search_log_count": search_log_count,
            "tag_count": tag_count,
            "asset_tag_count": asset_tag_count,
            "ocr_text_count": ocr_text_count,
            "document_chunk_count": document_chunk_count,
            "person_count": person_count,
            "face_count": face_count,
            "db_path": str(self.db_path),
        }

    def replace_person_clusters(self, persons: list[dict], faces: list[dict], library: str = "personal") -> None:
        now = datetime.now().isoformat(timespec="seconds")
        with self.connect() as conn:
            existing_names = {
                row["person_id"]: {"alias": row["alias"], "display_name": row["display_name"]}
                for row in conn.execute(
                    "SELECT person_id, alias, display_name FROM persons WHERE library = ?",
                    (library,),
                ).fetchall()
            }
            conn.execute("DELETE FROM asset_faces WHERE person_id IN (SELECT person_id FROM persons WHERE library = ?)", (library,))
            conn.execute("DELETE FROM persons WHERE library = ?", (library,))
            for person in persons:
                old_name = existing_names.get(person["person_id"], {})
                conn.execute(
                    """
                    INSERT INTO persons(
                        person_id, library, alias, display_name, face_count,
                        prototype_face_id, confidence, created_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        person["person_id"],
                        library,
                        person.get("alias") or old_name.get("alias"),
                        person.get("display_name") or old_name.get("display_name"),
                        person.get("face_count", 0),
                        person.get("prototype_face_id"),
                        person.get("confidence", 1.0),
                        now,
                        now,
                    ),
                )
            for face in faces:
                x1, y1, x2, y2 = face["bbox"]
                conn.execute(
                    """
                    INSERT INTO asset_faces(
                        face_id, asset_id, person_id,
                        bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                        confidence, quality, source, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        face["face_id"],
                        face["asset_id"],
                        face["person_id"],
                        int(x1),
                        int(y1),
                        int(x2),
                        int(y2),
                        float(face.get("confidence", 1.0)),
                        float(face.get("quality", 1.0)),
                        face.get("source", "opencv_haar_clip"),
                        now,
                    ),
                )

    def list_persons(self, library: str = "personal") -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT person_id, library, alias, display_name, face_count,
                       prototype_face_id, confidence, created_at, updated_at
                FROM persons
                WHERE library = ?
                ORDER BY face_count DESC, person_id
                """,
                (library,),
            ).fetchall()
            persons = []
            for row in rows:
                item = dict(row)
                assets = conn.execute(
                    """
                    SELECT DISTINCT assets.id, assets.filename, assets.kind, assets.width, assets.height
                    FROM asset_faces
                    JOIN assets ON assets.id = asset_faces.asset_id
                    WHERE asset_faces.person_id = ?
                    ORDER BY assets.filename
                    """,
                    (item["person_id"],),
                ).fetchall()
                item["assets"] = [dict(asset) for asset in assets]
                item["asset_count"] = len(item["assets"])
                persons.append(item)
            return persons

    def update_person_name(self, person_id: str, display_name: str, library: str = "personal") -> None:
        name = display_name.strip()
        now = datetime.now().isoformat(timespec="seconds")
        with self.connect() as conn:
            result = conn.execute(
                """
                UPDATE persons
                SET display_name = ?, alias = ?, updated_at = ?
                WHERE person_id = ? AND library = ?
                """,
                (name or None, name or None, now, person_id, library),
            )
            if result.rowcount == 0:
                raise ValueError(f"unknown person_id: {person_id}")

    def person_name_map(self, library: str = "personal") -> dict[str, dict[str, str | None]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT person_id, alias, display_name
                FROM persons
                WHERE library = ?
                """,
                (library,),
            ).fetchall()
            return {
                row["person_id"]: {
                    "alias": row["alias"],
                    "display_name": row["display_name"],
                }
                for row in rows
            }

    def named_person_ids(self, library: str = "personal") -> set[str]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT person_id
                FROM persons
                WHERE library = ?
                  AND (display_name IS NOT NULL OR alias IS NOT NULL)
                """,
                (library,),
            ).fetchall()
            return {row["person_id"] for row in rows}

    def face_asset_ids(self, library: str = "personal") -> list[str]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT asset_faces.asset_id
                FROM asset_faces
                JOIN persons ON persons.person_id = asset_faces.person_id
                WHERE persons.library = ?
                ORDER BY asset_faces.asset_id
                """,
                (library,),
            ).fetchall()
            return [row["asset_id"] for row in rows]

    def get_asset_faces(self, asset_id: str) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT face_id, asset_id, person_id,
                       bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                       confidence, quality, source, created_at
                FROM asset_faces
                WHERE asset_id = ?
                ORDER BY person_id, face_id
                """,
                (asset_id,),
            ).fetchall()
            result = []
            for row in rows:
                item = dict(row)
                item["bbox"] = [item.pop("bbox_x1"), item.pop("bbox_y1"), item.pop("bbox_x2"), item.pop("bbox_y2")]
                result.append(item)
            return result

    def delete_asset(self, asset_id: str) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM asset_faces WHERE asset_id = ?", (asset_id,))
            conn.execute("DELETE FROM asset_tags WHERE asset_id = ?", (asset_id,))
            conn.execute("DELETE FROM ocr_texts WHERE asset_id = ?", (asset_id,))
            conn.execute("DELETE FROM document_chunks WHERE asset_id = ?", (asset_id,))
            conn.execute("DELETE FROM assets WHERE id = ?", (asset_id,))

    def merge_persons(self, target_person_id: str, source_person_ids: list[str], library: str = "personal") -> list[str]:
        source_person_ids = [item for item in source_person_ids if item and item != target_person_id]
        if not source_person_ids:
            return []
        now = datetime.now().isoformat(timespec="seconds")
        with self.connect() as conn:
            placeholders = ",".join("?" for _ in source_person_ids)
            affected_rows = conn.execute(
                f"""
                SELECT DISTINCT asset_id FROM asset_faces
                WHERE person_id = ? OR person_id IN ({placeholders})
                """,
                (target_person_id, *source_person_ids),
            ).fetchall()
            conn.execute(
                f"UPDATE asset_faces SET person_id = ? WHERE person_id IN ({placeholders})",
                (target_person_id, *source_person_ids),
            )
            conn.execute(
                f"DELETE FROM persons WHERE library = ? AND person_id IN ({placeholders})",
                (library, *source_person_ids),
            )
            face_count = conn.execute(
                "SELECT COUNT(*) FROM asset_faces WHERE person_id = ?",
                (target_person_id,),
            ).fetchone()[0]
            prototype = conn.execute(
                "SELECT face_id FROM asset_faces WHERE person_id = ? ORDER BY quality DESC, face_id LIMIT 1",
                (target_person_id,),
            ).fetchone()
            conn.execute(
                """
                UPDATE persons
                SET face_count = ?, prototype_face_id = ?, updated_at = ?
                WHERE person_id = ? AND library = ?
                """,
                (face_count, prototype["face_id"] if prototype else None, now, target_person_id, library),
            )
            return [row["asset_id"] for row in affected_rows]

    def get_person_asset_ids(self, person_ids: list[str], library: str = "personal") -> list[str]:
        clean_ids = [person_id for person_id in person_ids if person_id]
        if not clean_ids:
            return []
        placeholders = ",".join("?" for _ in clean_ids)
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT DISTINCT asset_faces.asset_id
                FROM asset_faces
                JOIN persons ON persons.person_id = asset_faces.person_id
                WHERE persons.library = ? AND asset_faces.person_id IN ({placeholders})
                ORDER BY asset_faces.asset_id
                """,
                (library, *clean_ids),
            ).fetchall()
            return [row["asset_id"] for row in rows]

    def get_asset_face_person_ids(self, asset_id: str) -> list[str]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT person_id FROM asset_faces WHERE asset_id = ? ORDER BY person_id",
                (asset_id,),
            ).fetchall()
            return [row["person_id"] for row in rows]

    def delete_person(self, person_id: str, library: str = "personal") -> list[str]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT asset_id FROM asset_faces WHERE person_id = ?",
                (person_id,),
            ).fetchall()
            conn.execute("DELETE FROM asset_faces WHERE person_id = ?", (person_id,))
            conn.execute("DELETE FROM persons WHERE person_id = ? AND library = ?", (person_id, library))
            return [row["asset_id"] for row in rows]

    def remove_asset_person(self, asset_id: str, person_id: str, library: str = "personal") -> list[str]:
        now = datetime.now().isoformat(timespec="seconds")
        with self.connect() as conn:
            conn.execute("DELETE FROM asset_faces WHERE asset_id = ? AND person_id = ?", (asset_id, person_id))
            face_count = conn.execute(
                "SELECT COUNT(*) FROM asset_faces WHERE person_id = ?",
                (person_id,),
            ).fetchone()[0]
            if face_count:
                prototype = conn.execute(
                    "SELECT face_id FROM asset_faces WHERE person_id = ? ORDER BY quality DESC, face_id LIMIT 1",
                    (person_id,),
                ).fetchone()
                conn.execute(
                    """
                    UPDATE persons
                    SET face_count = ?, prototype_face_id = ?, updated_at = ?
                    WHERE person_id = ? AND library = ?
                    """,
                    (face_count, prototype["face_id"] if prototype else None, now, person_id, library),
                )
            else:
                conn.execute("DELETE FROM persons WHERE person_id = ? AND library = ?", (person_id, library))
            return [asset_id]

    def cleanup_person_clusters(self, library: str = "personal") -> list[str]:
        now = datetime.now().isoformat(timespec="seconds")
        with self.connect() as conn:
            affected = conn.execute(
                """
                SELECT DISTINCT asset_faces.asset_id
                FROM asset_faces
                LEFT JOIN assets ON assets.id = asset_faces.asset_id
                WHERE assets.id IS NULL
                """
            ).fetchall()
            conn.execute(
                """
                DELETE FROM asset_faces
                WHERE asset_id NOT IN (SELECT id FROM assets)
                """
            )
            persons = conn.execute(
                "SELECT person_id FROM persons WHERE library = ?",
                (library,),
            ).fetchall()
            for person in persons:
                person_id = person["person_id"]
                face_count = conn.execute(
                    "SELECT COUNT(*) FROM asset_faces WHERE person_id = ?",
                    (person_id,),
                ).fetchone()[0]
                if not face_count:
                    conn.execute("DELETE FROM persons WHERE person_id = ? AND library = ?", (person_id, library))
                    continue
                prototype = conn.execute(
                    "SELECT face_id FROM asset_faces WHERE person_id = ? ORDER BY quality DESC, face_id LIMIT 1",
                    (person_id,),
                ).fetchone()
                conn.execute(
                    """
                    UPDATE persons
                    SET face_count = ?, prototype_face_id = ?, updated_at = ?
                    WHERE person_id = ? AND library = ?
                    """,
                    (face_count, prototype["face_id"] if prototype else None, now, person_id, library),
                )
            live_assets = conn.execute(
                """
                SELECT DISTINCT asset_id FROM asset_faces
                WHERE person_id IN (SELECT person_id FROM persons WHERE library = ?)
                """,
                (library,),
            ).fetchall()
            return sorted({row["asset_id"] for row in affected} | {row["asset_id"] for row in live_assets})

    def duplicate_groups(self, limit: int = 20) -> list[dict]:
        with self.connect() as conn:
            groups = conn.execute(
                """
                SELECT sha256, COUNT(*) AS count
                FROM assets
                WHERE sha256 IS NOT NULL
                GROUP BY sha256
                HAVING COUNT(*) > 1
                ORDER BY count DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            result = []
            for group in groups:
                files = conn.execute(
                    "SELECT id, filename, path FROM assets WHERE sha256 = ? ORDER BY filename",
                    (group["sha256"],),
                ).fetchall()
                result.append(
                    {
                        "sha256": group["sha256"],
                        "count": group["count"],
                        "files": [dict(file) for file in files],
                    }
                )
            return result

    def log_search(self, query: str, plan: dict, results: list[dict], backend: str) -> None:
        import json

        top_score = results[0]["score"] if results else None
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO search_logs (
                    query, plan_json, result_json, backend, top_score, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    query,
                    json.dumps(plan, ensure_ascii=False),
                    json.dumps(results, ensure_ascii=False),
                    backend,
                    top_score,
                    datetime.now().isoformat(timespec="seconds"),
                ),
            )

    def recent_search_logs(self, limit: int = 20) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT id, query, backend, top_score, created_at
                FROM search_logs
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [dict(row) for row in rows]

    def set_asset_tags(self, asset_id: str, tags: list[str], source: str, confidence: float = 1.0) -> None:
        now = datetime.now().isoformat(timespec="seconds")
        with self.connect() as conn:
            for tag in tags:
                tag = tag.strip().lower()
                if not tag:
                    continue
                conn.execute(
                    """
                    INSERT INTO tags(name, source, created_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(name) DO NOTHING
                    """,
                    (tag, source, now),
                )
                row = conn.execute("SELECT id FROM tags WHERE name = ?", (tag,)).fetchone()
                conn.execute(
                    """
                    INSERT INTO asset_tags(asset_id, tag_id, confidence, source, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(asset_id, tag_id, source) DO UPDATE SET
                        confidence=excluded.confidence,
                        created_at=excluded.created_at
                    """,
                    (asset_id, row["id"], confidence, source, now),
                )

    def get_asset_tags(self, asset_id: str) -> list[str]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT tags.name
                FROM asset_tags
                JOIN tags ON tags.id = asset_tags.tag_id
                WHERE asset_tags.asset_id = ?
                ORDER BY tags.name
                """,
                (asset_id,),
            ).fetchall()
            return [row["name"] for row in rows]

    def get_asset_tag_rows(self, asset_id: str) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT tags.name, asset_tags.source, asset_tags.confidence, asset_tags.created_at
                FROM asset_tags
                JOIN tags ON tags.id = asset_tags.tag_id
                WHERE asset_tags.asset_id = ?
                ORDER BY asset_tags.source, asset_tags.confidence DESC, tags.name
                """,
                (asset_id,),
            ).fetchall()
            return [dict(row) for row in rows]

    def has_asset_tag(self, asset_id: str, tag_name: str, source: str | None = None) -> bool:
        query = """
            SELECT 1
            FROM asset_tags
            JOIN tags ON tags.id = asset_tags.tag_id
            WHERE asset_tags.asset_id = ? AND tags.name = ?
        """
        params: list[str] = [asset_id, tag_name]
        if source is not None:
            query += " AND asset_tags.source = ?"
            params.append(source)
        query += " LIMIT 1"
        with self.connect() as conn:
            return conn.execute(query, tuple(params)).fetchone() is not None

    def tag_facets(self, limit: int = 120, library: str = "all") -> list[dict]:
        library_clause = "" if library == "all" else "AND assets.library = ?"
        params: tuple = () if library == "all" else (library,)
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT tags.name, COUNT(DISTINCT asset_tags.asset_id) AS asset_count,
                       GROUP_CONCAT(DISTINCT asset_tags.source) AS sources
                FROM tags
                JOIN asset_tags ON asset_tags.tag_id = tags.id
                JOIN assets ON assets.id = asset_tags.asset_id
                WHERE tags.name NOT IN ('image_material', 'video_material')
                  {library_clause}
                GROUP BY tags.name
                ORDER BY asset_count DESC, tags.name
                LIMIT ?
                """,
                (*params, limit),
            ).fetchall()
            return [dict(row) for row in rows]

    def asset_ids_for_tag(self, tag_name: str, library: str = "all", limit: int = 160) -> list[str]:
        library_clause = "" if library == "all" else "AND assets.library = ?"
        params: list = [tag_name]
        if library != "all":
            params.append(library)
        params.append(limit)
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT DISTINCT assets.id, assets.indexed_at
                FROM assets
                JOIN asset_tags ON asset_tags.asset_id = assets.id
                JOIN tags ON tags.id = asset_tags.tag_id
                WHERE tags.name = ?
                  {library_clause}
                ORDER BY assets.indexed_at DESC
                LIMIT ?
                """,
                tuple(params),
            ).fetchall()
            return [row["id"] for row in rows]

    def delete_asset_tags(self, asset_id: str, source: str) -> int:
        with self.connect() as conn:
            cursor = conn.execute(
                "DELETE FROM asset_tags WHERE asset_id = ? AND source = ?",
                (asset_id, source),
            )
            return cursor.rowcount

    def delete_asset_tags_by_source(self, source: str) -> int:
        with self.connect() as conn:
            cursor = conn.execute("DELETE FROM asset_tags WHERE source = ?", (source,))
            return cursor.rowcount

    def upsert_ocr_text(self, asset_id: str, text: str, engine: str, confidence: float | None = None) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO ocr_texts(asset_id, text, engine, confidence, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (asset_id, text, engine, confidence, datetime.now().isoformat(timespec="seconds")),
            )

    def get_ocr_text(self, asset_id: str) -> str:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT text FROM ocr_texts WHERE asset_id = ? ORDER BY id DESC",
                (asset_id,),
            ).fetchall()
            return " ".join(row["text"] for row in rows)

    def list_text_signals(self, asset_id: str) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT id, asset_id, text, engine, confidence, created_at
                FROM ocr_texts
                WHERE asset_id = ?
                ORDER BY id DESC
                """,
                (asset_id,),
            ).fetchall()
        signals = []
        for row in rows:
            item = dict(row)
            text_type, engine = split_text_engine(item.get("engine") or "")
            item["text_type"] = text_type
            item["engine"] = engine
            item["summary"] = summarize_text(item.get("text") or "")
            signals.append(item)
        return signals

    def get_text_by_type(self, asset_id: str, text_types: set[str]) -> str:
        signals = self.list_text_signals(asset_id)
        return " ".join(item["text"] for item in signals if item["text_type"] in text_types)

    def replace_document_chunks(self, asset_id: str, chunks: list[dict]) -> None:
        now = datetime.now().isoformat(timespec="seconds")
        with self.connect() as conn:
            conn.execute("DELETE FROM document_chunks WHERE asset_id = ?", (asset_id,))
            for chunk in chunks:
                conn.execute(
                    """
                    INSERT INTO document_chunks (
                        asset_id, chunk_id, chunk_index, page_start, page_end,
                        section_title, heading_path, chunk_type, text, summary,
                        embedding_text, keywords, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        asset_id,
                        str(chunk.get("chunk_id") or ""),
                        int(chunk.get("chunk_index") or 0),
                        chunk.get("page_start"),
                        chunk.get("page_end"),
                        str(chunk.get("section_title") or ""),
                        json.dumps(chunk.get("heading_path") or [], ensure_ascii=False),
                        str(chunk.get("chunk_type") or "text"),
                        str(chunk.get("text") or ""),
                        str(chunk.get("summary") or ""),
                        str(chunk.get("embedding_text") or chunk.get("text") or ""),
                        json.dumps(chunk.get("keywords") or [], ensure_ascii=False),
                        now,
                    ),
                )

    def list_document_chunks(self, asset_id: str | None = None) -> list[dict]:
        with self.connect() as conn:
            if asset_id:
                rows = conn.execute(
                    """
                    SELECT * FROM document_chunks
                    WHERE asset_id = ?
                    ORDER BY chunk_index
                    """,
                    (asset_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM document_chunks
                    ORDER BY asset_id, chunk_index
                    """
                ).fetchall()
        return [decode_document_chunk(row) for row in rows]

    def get_document_chunk(self, chunk_id: str) -> dict | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM document_chunks WHERE chunk_id = ?", (chunk_id,)).fetchone()
        return decode_document_chunk(row) if row else None

    def count_document_chunks(self, asset_id: str | None = None) -> int:
        with self.connect() as conn:
            if asset_id:
                return int(conn.execute("SELECT COUNT(*) FROM document_chunks WHERE asset_id = ?", (asset_id,)).fetchone()[0])
            return int(conn.execute("SELECT COUNT(*) FROM document_chunks").fetchone()[0])


def rows_to_dict(rows: sqlite3.Cursor) -> dict[str, int]:
    return {row[0]: row[1] for row in rows.fetchall()}


def decode_document_chunk(row: sqlite3.Row) -> dict:
    item = dict(row)
    for key in ["heading_path", "keywords"]:
        try:
            item[key] = json.loads(item.get(key) or "[]")
        except Exception:
            item[key] = []
    return item


def split_text_engine(engine: str) -> tuple[str, str]:
    if ":" in engine:
        text_type, raw_engine = engine.split(":", 1)
    else:
        text_type, raw_engine = "ocr", engine
    text_type = text_type.strip().lower() or "ocr"
    if text_type not in {"ocr", "subtitle_ocr", "asr", "manual_text", "document_text"}:
        text_type = "ocr"
    return text_type, raw_engine.strip() or "unknown"


def summarize_text(text: str, max_chars: int = 220) -> str:
    clean = " ".join((text or "").split())
    if len(clean) <= max_chars:
        return clean
    sentence_end = max(clean.rfind(".", 0, max_chars), clean.rfind("。", 0, max_chars), clean.rfind("!", 0, max_chars), clean.rfind("?", 0, max_chars))
    if sentence_end >= 80:
        return clean[: sentence_end + 1]
    return clean[:max_chars].rstrip() + "..."


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
