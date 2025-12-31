import asyncio
import json
import os
from typing import List, Dict, Any, Optional
import requests
from langchain_core.documents import Document

from backend.entity_registry import RULE_LINK_PRESETS


def _get_graph_deps():
    from backend.core.graph_client import graphiti_app, vector_store, QDRANT_URL, langchain_embeddings
    return graphiti_app, vector_store, QDRANT_URL, langchain_embeddings


class GenericEntityService:
    BUILD_MARKER = "2025-12-31-stdwrite-v1"
    """
    通用实体服务 (Generic Entity Service)

    职责：
    1. 提供对任意实体的 增(Ingest)、查(Search)、改(Update) 能力。
    2. 维护 Graph (Neo4j) 与 Vector (Qdrant) 的数据一致性。
    3. 支持读写分离策略：大字段只存向量库，不存图数据库。
    """

    @staticmethod
    def _validate_cypher_identifier(value: str) -> str:
        if not value or not isinstance(value, str):
            raise ValueError("Invalid identifier")
        ok = all((c.isalnum() or c == "_") for c in value)
        if not ok:
            raise ValueError(f"Invalid identifier: {value}")
        return value

    @staticmethod
    def _extract_standard_keywords(source_item: Dict[str, Any]) -> List[str]:
        tags_val = source_item.get("tags") or []
        if isinstance(tags_val, str):
            try:
                parsed = json.loads(tags_val)
                tags = parsed if isinstance(parsed, list) else [tags_val]
            except Exception:
                tags = [tags_val]
        else:
            tags = tags_val

        if not isinstance(tags, list):
            tags = [tags]

        raw_items: List[str] = []
        for t in tags:
            s = str(t).strip()
            if s:
                raw_items.append(s)

        for sf in ["title", "summary"]:
            v = source_item.get(sf)
            if v:
                s = str(v).strip()
                if s:
                    raw_items.append(s)

        stop_exact = {"针织", "混纺"}
        stop_contains = ("标准", "纺织", "行业标准")

        keywords: List[str] = []
        seen: set[str] = set()
        for raw in raw_items:
            text = str(raw).strip()
            if not text:
                continue
            if any(c.isdigit() for c in text):
                continue
            if any("a" <= c.lower() <= "z" for c in text):
                continue
            if text in stop_exact:
                continue
            if any(w in text for w in stop_contains):
                continue
            if text not in seen:
                seen.add(text)
                keywords.append(text)

        return keywords[:3]

    @staticmethod
    def _standard_mentions_keyword(source_item: Dict[str, Any], keyword: str) -> bool:
        if not keyword:
            return False
        try:
            tags_val = source_item.get("tags")
            title = source_item.get("title")
            summary = source_item.get("summary")
            main_content = source_item.get("main_content")
            extracted_tables = source_item.get("extracted_tables")
            hay = " ".join(
                str(x)
                for x in [
                    tags_val if tags_val is not None else "",
                    title if title is not None else "",
                    summary if summary is not None else "",
                    main_content if main_content is not None else "",
                    extracted_tables if extracted_tables is not None else "",
                ]
                if x is not None
            )
            return keyword in hay
        except Exception:
            return False

    @staticmethod
    async def link_entities_by_rules(
            source_label: str,
            source_id_field: str,
            target_label: str,
            target_id_field: str,
            source_list_fields: Optional[List[str]] = None,
            source_text_fields: Optional[List[str]] = None,
            target_text_fields: Optional[List[str]] = None,
            rel_type: str = "RELATED_TO",
            mode: str = "contains_any",
    ) -> int:
        graphiti_app, _, _, _ = _get_graph_deps()
        source_list_fields = source_list_fields or []
        source_text_fields = source_text_fields or []
        target_text_fields = target_text_fields or []

        src_lbl = GenericEntityService._validate_cypher_identifier(source_label)
        tgt_lbl = GenericEntityService._validate_cypher_identifier(target_label)
        src_id = GenericEntityService._validate_cypher_identifier(source_id_field)
        tgt_id = GenericEntityService._validate_cypher_identifier(target_id_field)
        rel = GenericEntityService._validate_cypher_identifier(rel_type)

        for f in source_list_fields + source_text_fields + target_text_fields:
            GenericEntityService._validate_cypher_identifier(f)

        if mode not in {"contains_any", "equals_any"}:
            raise ValueError("Invalid mode")

        conditions: List[str] = []

        for sf in source_list_fields:
            per_target = []
            for tf in target_text_fields:
                if mode == "equals_any":
                    per_target.append(
                        f"coalesce(trim(toString(p.{tf})), '') = coalesce(trim(toString(k)), '')"
                    )
                else:
                    per_target.append(f"coalesce(toString(p.{tf}), '') CONTAINS toString(k)")
            if per_target:
                conditions.append(
                    f"ANY(k IN coalesce(s.{sf}, []) WHERE coalesce(trim(toString(k)), '') <> '' AND ({' OR '.join(per_target)}))"
                )

        for sf in source_text_fields:
            for tf in target_text_fields:
                if mode == "equals_any":
                    conditions.append(
                        f"(coalesce(trim(toString(s.{sf})), '') <> '' AND "
                        f"coalesce(trim(toString(p.{tf})), '') <> '' AND "
                        f"coalesce(trim(toString(p.{tf})), '') = coalesce(trim(toString(s.{sf})), ''))"
                    )
                else:
                    conditions.append(
                        f"(coalesce(trim(toString(s.{sf})), '') <> '' AND "
                        f"coalesce(trim(toString(p.{tf})), '') <> '' AND "
                        f"coalesce(toString(p.{tf}), '') CONTAINS coalesce(toString(s.{sf}), ''))"
                    )

        if not conditions:
            return 0

        cypher = f"""
        MATCH (s:{src_lbl})
        MATCH (p:{tgt_lbl})
        WHERE ({' OR '.join(conditions)})
        MERGE (s)-[:{rel}]->(p)
        RETURN count(*) as edges_created
        """

        async with graphiti_app.driver.session() as session:
            result = await session.run(cypher)
            record = await result.single()
            edges_val = record.get("edges_created") if record else None
            edges_created = int(edges_val) if edges_val is not None else 0
            await result.consume()
            return edges_created

    @staticmethod
    async def link_rule_preset_for_source(
            preset_name: str,
            source_id: Any,
            sample_limit: int = 10,
            keyword: Optional[str] = None,
    ) -> Dict[str, Any]:
        graphiti_app, _, _, _ = _get_graph_deps()
        preset = RULE_LINK_PRESETS.get(preset_name)
        if not preset:
            return {"status": "error", "message": "Preset not found", "preset": preset_name}

        def _compact_cypher(s: str) -> str:
            return " ".join(line.strip() for line in s.splitlines() if line.strip())

        source_label = GenericEntityService._validate_cypher_identifier(preset["source_label"])
        source_id_field = GenericEntityService._validate_cypher_identifier(preset["source_id_field"])
        target_label = GenericEntityService._validate_cypher_identifier(preset["target_label"])
        target_id_field = GenericEntityService._validate_cypher_identifier(preset["target_id_field"])
        rel_type = GenericEntityService._validate_cypher_identifier(preset.get("rel_type") or "RELATED_TO")

        target_text_fields: List[str] = preset.get("target_text_fields") or []
        mode: str = preset.get("mode") or "contains_any"

        for f in target_text_fields:
            GenericEntityService._validate_cypher_identifier(f)

        if mode not in {"contains_any", "equals_any"}:
            return {"status": "error", "message": "Invalid mode", "mode": mode}

        if preset_name == "standard_to_product":
            fetch_cypher = f"""
            MATCH (s:{source_label} {{ {source_id_field}: $sid }})
            RETURN s as s
            """
            keyword_override = (keyword or "").strip()

            per_target = []
            for tf in target_text_fields:
                if mode == "equals_any":
                    per_target.append(
                        f"coalesce(trim(toString(p.{tf})), '') = coalesce(trim(toString($keyword)), '')"
                    )
                else:
                    per_target.append(f"coalesce(toString(p.{tf}), '') CONTAINS toString($keyword)")

            where_clause = f"({' OR '.join(per_target)})"

            sample_limit = max(0, min(int(sample_limit), 50))

            count_cypher = f"""
            MATCH (p:{target_label})
            WHERE ({where_clause})
            RETURN count(p) as matched_count
            """

            list_cypher = f"""
            MATCH (p:{target_label})
            WHERE ({where_clause})
            RETURN p.{target_id_field} as target_id
            ORDER BY toString(p.{target_id_field})
            """

            link_cypher = f"""
            MATCH (s:{source_label} {{ {source_id_field}: $sid }})
            WITH s
            UNWIND $target_ids as tid
            MATCH (p:{target_label})
            WHERE toString(p.{target_id_field}) = toString(tid)
            MERGE (s)-[:{rel_type}]->(p)
            RETURN count(DISTINCT p) as edges_created
            """

            sample_cypher = f"""
            MATCH (s:{source_label} {{ {source_id_field}: $sid }})-[:{rel_type}]->(p:{target_label})
            RETURN p.{target_id_field} as target_id
            LIMIT $limit
            """

            async with graphiti_app.driver.session() as session:
                print(
                    "[STD_LINK_SOURCE_FETCH] preset=", preset_name,
                    "source_id=", str(source_id),
                    "fetch_cypher=", fetch_cypher,
                    "params=", json.dumps({"sid": str(source_id)}, ensure_ascii=False),
                    flush=True,
                )
                fetch_res = await session.run(fetch_cypher, sid=source_id)
                fetch_rec = await fetch_res.single()
                await fetch_res.consume()
                src_node = fetch_rec.get("s") if fetch_rec else None
                if src_node is None:
                    print(
                        "[STD_LINK_SKIP] preset=", preset_name,
                        "source_id=", str(source_id),
                        "reason=source_not_found_in_neo4j",
                        "keyword_override=", keyword_override if keyword_override else None,
                        flush=True,
                    )
                    return {
                        "status": "success",
                        "preset": preset_name,
                        "source_id": str(source_id),
                        "edges_created": 0,
                        "targets_sample": [],
                        "keywords": [],
                        "matched_count": 0,
                        "linked": False,
                        "cypher": {},
                    }
                src_props = dict(src_node)
                keywords = [keyword_override] if keyword_override else GenericEntityService._extract_standard_keywords(src_props)
                keywords = [k for k in keywords if isinstance(k, str) and k.strip()]
                if not keywords:
                    print(
                        "[STD_LINK_SKIP] preset=", preset_name,
                        "source_id=", str(source_id),
                        "reason=no_keyword_candidates",
                        "standard_tags=", json.dumps(src_props.get("tags"), ensure_ascii=False),
                        "standard_title=", src_props.get("title"),
                        "standard_summary=", src_props.get("summary"),
                        "standard_has_main_content=", "main_content" in src_props,
                        "standard_has_extracted_tables=", "extracted_tables" in src_props,
                        flush=True,
                    )
                    return {
                        "status": "success",
                        "preset": preset_name,
                        "source_id": str(source_id),
                        "edges_created": 0,
                        "targets_sample": [],
                        "keywords": keywords,
                        "matched_count": 0,
                        "linked": False,
                        "cypher": {},
                    }

                using_keyword = str(keywords[0]).strip()
                if not GenericEntityService._standard_mentions_keyword(src_props, using_keyword):
                    literal_keyword = using_keyword.replace("\\", "\\\\").replace("'", "\\'")
                    literal_where_clause = where_clause.replace("$keyword", f"'{literal_keyword}'")
                    print(
                        "[STD_LINK_SKIP] preset=", preset_name,
                        "source_id=", str(source_id),
                        "reason=standard_not_mention_keyword",
                        "keyword=", using_keyword,
                        "standard_tags=", json.dumps(src_props.get("tags"), ensure_ascii=False),
                        "standard_title=", src_props.get("title"),
                        "standard_summary=", src_props.get("summary"),
                        "standard_has_main_content=", "main_content" in src_props,
                        "standard_has_extracted_tables=", "extracted_tables" in src_props,
                        "count_literal=", f"""
                        MATCH (p:{target_label})
                        WHERE ({literal_where_clause})
                        RETURN count(p) as matched_count
                        """,
                        "list_literal=", f"""
                        MATCH (p:{target_label})
                        WHERE ({literal_where_clause})
                        RETURN p.{target_id_field} as target_id
                        ORDER BY toString(p.{target_id_field})
                        """,
                        flush=True,
                    )
                    return {
                        "status": "success",
                        "preset": preset_name,
                        "source_id": str(source_id),
                        "edges_created": 0,
                        "targets_sample": [],
                        "keywords": keywords,
                        "matched_count": 0,
                        "linked": False,
                        "cypher": {
                            "count_param": _compact_cypher(count_cypher),
                            "count_literal": _compact_cypher(
                                f"""
                                MATCH (p:{target_label})
                                WHERE ({literal_where_clause})
                                RETURN count(p) as matched_count
                                """
                            ),
                            "list_param": _compact_cypher(list_cypher),
                            "list_literal": _compact_cypher(
                                f"""
                                MATCH (p:{target_label})
                                WHERE ({literal_where_clause})
                                RETURN p.{target_id_field} as target_id
                                ORDER BY toString(p.{target_id_field})
                                """
                            ),
                            "link_param": _compact_cypher(link_cypher),
                            "sample_params": {"sid": str(source_id), "keyword": using_keyword, "target_ids": []},
                        },
                    }

                print(
                    "[STD_LINK_CYPHER_PARAM] preset=", preset_name,
                    "source_id=", str(source_id),
                    "keyword=", using_keyword,
                    "count_cypher=", count_cypher,
                    "params=", json.dumps({"keyword": using_keyword}, ensure_ascii=False),
                    flush=True,
                )
                literal_keyword = using_keyword.replace("\\", "\\\\").replace("'", "\\'")
                literal_where_clause = where_clause.replace("$keyword", f"'{literal_keyword}'")
                count_literal = f"""
                MATCH (p:{target_label})
                WHERE ({literal_where_clause})
                RETURN count(p) as matched_count
                """
                print(
                    "[STD_LINK_CYPHER_LITERAL] preset=", preset_name,
                    "source_id=", str(source_id),
                    "keyword=", using_keyword,
                    "count_cypher=", count_literal,
                    flush=True,
                )
                count_res = await session.run(count_cypher, keyword=using_keyword)
                count_rec = await count_res.single()
                matched_val = count_rec.get("matched_count") if count_rec else None
                matched_count = int(matched_val) if matched_val is not None else 0
                await count_res.consume()

                if matched_count <= 0:
                    print(
                        "[STD_LINK_RESULT] preset=", preset_name,
                        "source_id=", str(source_id),
                        "keyword=", using_keyword,
                        "matched_count=", matched_count,
                        "linked=", False,
                        "count_literal=", count_literal,
                        "list_literal=", f"""
                        MATCH (p:{target_label})
                        WHERE ({literal_where_clause})
                        RETURN p.{target_id_field} as target_id
                        ORDER BY toString(p.{target_id_field})
                        """,
                        flush=True,
                    )
                    return {
                        "status": "success",
                        "preset": preset_name,
                        "source_id": str(source_id),
                        "edges_created": 0,
                        "targets_sample": [],
                        "keywords": keywords,
                        "matched_count": matched_count,
                        "linked": False,
                        "cypher": {
                            "count_param": _compact_cypher(count_cypher),
                            "count_literal": _compact_cypher(
                                f"""
                                MATCH (p:{target_label})
                                WHERE ({literal_where_clause})
                                RETURN count(p) as matched_count
                                """
                            ),
                            "list_param": _compact_cypher(list_cypher),
                            "list_literal": _compact_cypher(
                                f"""
                                MATCH (p:{target_label})
                                WHERE ({literal_where_clause})
                                RETURN p.{target_id_field} as target_id
                                ORDER BY toString(p.{target_id_field})
                                """
                            ),
                            "link_param": _compact_cypher(link_cypher),
                            "sample_params": {"sid": str(source_id), "keyword": using_keyword, "target_ids": []},
                        },
                    }

                print(
                    "[STD_LINK_CYPHER_PARAM] preset=", preset_name,
                    "source_id=", str(source_id),
                    "keyword=", using_keyword,
                    "list_cypher=", list_cypher,
                    "params=", json.dumps({"keyword": using_keyword}, ensure_ascii=False),
                    flush=True,
                )
                list_literal = f"""
                MATCH (p:{target_label})
                WHERE ({literal_where_clause})
                RETURN p.{target_id_field} as target_id
                ORDER BY toString(p.{target_id_field})
                """
                print(
                    "[STD_LINK_CYPHER_LITERAL] preset=", preset_name,
                    "source_id=", str(source_id),
                    "keyword=", using_keyword,
                    "list_cypher=", list_literal,
                    flush=True,
                )
                list_res = await session.run(list_cypher, keyword=using_keyword)
                list_rows = await list_res.data()
                await list_res.consume()
                target_ids = [
                    str(r.get("target_id"))
                    for r in list_rows
                    if isinstance(r, dict) and r.get("target_id") is not None
                ]

                print(
                    "[STD_LINK_TARGET_IDS] preset=", preset_name,
                    "source_id=", str(source_id),
                    "keyword=", using_keyword,
                    "target_ids=", json.dumps(target_ids, ensure_ascii=False),
                    flush=True,
                )

                print(
                    "[STD_LINK_CYPHER_PARAM] preset=", preset_name,
                    "source_id=", str(source_id),
                    "keyword=", using_keyword,
                    "link_cypher=", link_cypher,
                    "params=", json.dumps({"sid": str(source_id), "target_ids": target_ids}, ensure_ascii=False),
                    flush=True,
                )
                link_res = await session.run(link_cypher, sid=source_id, target_ids=target_ids)
                link_rec = await link_res.single()
                edges_val = link_rec.get("edges_created") if link_rec else None
                edges_created = int(edges_val) if edges_val is not None else 0
                await link_res.consume()

                print(
                    "[STD_LINK_RESULT] preset=", preset_name,
                    "source_id=", str(source_id),
                    "keyword=", using_keyword,
                    "matched_count=", matched_count,
                    "edges_created=", edges_created,
                    "linked=", True,
                    flush=True,
                )

                sample_res = await session.run(sample_cypher, sid=source_id, limit=sample_limit)
                sample_rows = await sample_res.data()
                await sample_res.consume()

            targets_sample = [r.get("target_id") for r in sample_rows if isinstance(r, dict)]
            literal_keyword = using_keyword.replace("\\", "\\\\").replace("'", "\\'")
            literal_where_clause = where_clause.replace("$keyword", f"'{literal_keyword}'")
            return {
                "status": "success",
                "preset": preset_name,
                "source_id": str(source_id),
                "edges_created": edges_created,
                "targets_sample": targets_sample,
                "keywords": keywords,
                "matched_count": matched_count,
                "linked": True,
                "cypher": {
                    "count_param": _compact_cypher(count_cypher),
                    "count_literal": _compact_cypher(
                        f"""
                        MATCH (p:{target_label})
                        WHERE ({literal_where_clause})
                        RETURN count(p) as matched_count
                        """
                    ),
                    "list_param": _compact_cypher(list_cypher),
                    "list_literal": _compact_cypher(
                        f"""
                        MATCH (p:{target_label})
                        WHERE ({literal_where_clause})
                        RETURN p.{target_id_field} as target_id
                        ORDER BY toString(p.{target_id_field})
                        """
                    ),
                    "link_param": _compact_cypher(link_cypher),
                    "sample_params": {"sid": str(source_id), "keyword": using_keyword, "target_ids": target_ids},
                },
            }

        source_list_fields: List[str] = preset.get("source_list_fields") or []
        source_text_fields: List[str] = preset.get("source_text_fields") or []

        for f in source_list_fields + source_text_fields:
            GenericEntityService._validate_cypher_identifier(f)

        conditions: List[str] = []

        for sf in source_list_fields:
            per_target = []
            for tf in target_text_fields:
                if mode == "equals_any":
                    per_target.append(
                        f"coalesce(trim(toString(p.{tf})), '') = coalesce(trim(toString(k)), '')"
                    )
                else:
                    per_target.append(f"coalesce(toString(p.{tf}), '') CONTAINS toString(k)")
            if per_target:
                conditions.append(
                    f"ANY(k IN coalesce(s.{sf}, []) WHERE coalesce(trim(toString(k)), '') <> '' AND ({' OR '.join(per_target)}))"
                )

        for sf in source_text_fields:
            for tf in target_text_fields:
                if mode == "equals_any":
                    conditions.append(
                        f"(coalesce(trim(toString(s.{sf})), '') <> '' AND "
                        f"coalesce(trim(toString(p.{tf})), '') <> '' AND "
                        f"coalesce(trim(toString(p.{tf})), '') = coalesce(trim(toString(s.{sf})), ''))"
                    )
                else:
                    conditions.append(
                        f"(coalesce(trim(toString(s.{sf})), '') <> '' AND "
                        f"coalesce(trim(toString(p.{tf})), '') <> '' AND "
                        f"coalesce(toString(p.{tf}), '') CONTAINS coalesce(toString(s.{sf}), ''))"
                    )

        if not conditions:
            return {
                "status": "success",
                "preset": preset_name,
                "source_id": str(source_id),
                "edges_created": 0,
                "targets_sample": [],
            }

        where_clause = " OR ".join(conditions)
        sample_limit = max(0, min(int(sample_limit), 50))

        cypher = f"""
        MATCH (s:{source_label} {{ {source_id_field}: $sid }})
        MATCH (p:{target_label})
        WHERE ({where_clause})
        MERGE (s)-[:{rel_type}]->(p)
        RETURN count(p) as edges_created
        """

        sample_cypher = f"""
        MATCH (s:{source_label} {{ {source_id_field}: $sid }})-[:{rel_type}]->(p:{target_label})
        RETURN p.{target_id_field} as target_id
        LIMIT $limit
        """

        async with graphiti_app.driver.session() as session:
            res = await session.run(cypher, sid=source_id)
            rec = await res.single()
            edges_val = rec.get("edges_created") if rec else None
            edges_created = int(edges_val) if edges_val is not None else 0
            await res.consume()

            sample_res = await session.run(sample_cypher, sid=source_id, limit=sample_limit)
            sample_rows = await sample_res.data()
            await sample_res.consume()

        targets_sample = [r.get("target_id") for r in sample_rows if isinstance(r, dict)]
        return {
            "status": "success",
            "preset": preset_name,
            "source_id": str(source_id),
            "edges_created": edges_created,
            "targets_sample": targets_sample,
        }

    @staticmethod
    async def link_rule_preset_for_sources(preset_name: str, source_ids: List[Any], sample_limit: int = 0) -> Dict[str, Any]:
        total = 0
        per_source: List[Dict[str, Any]] = []
        for sid in source_ids:
            one = await GenericEntityService.link_rule_preset_for_source(
                preset_name=preset_name,
                source_id=sid,
                sample_limit=sample_limit,
            )
            if one.get("status") == "success":
                try:
                    total += int(one.get("edges_created") or 0)
                except Exception:
                    pass
            per_source.append(one)
        return {"status": "success", "preset": preset_name, "edges_created": total, "details": per_source}

    @staticmethod
    async def repair_rule_preset_for_source(preset_name: str, source_id: Any, sample_limit: int = 10) -> Dict[str, Any]:
        graphiti_app, _, _, _ = _get_graph_deps()
        preset = RULE_LINK_PRESETS.get(preset_name)
        if not preset:
            return {"status": "error", "message": "Preset not found", "preset": preset_name}

        source_label = GenericEntityService._validate_cypher_identifier(preset["source_label"])
        source_id_field = GenericEntityService._validate_cypher_identifier(preset["source_id_field"])
        target_label = GenericEntityService._validate_cypher_identifier(preset["target_label"])
        rel_type = GenericEntityService._validate_cypher_identifier(preset.get("rel_type") or "RELATED_TO")

        delete_cypher = f"""
        MATCH (s:{source_label} {{ {source_id_field}: $sid }})-[r:{rel_type}]->(p:{target_label})
        DELETE r
        RETURN count(r) as deleted
        """

        async with graphiti_app.driver.session() as session:
            del_res = await session.run(delete_cypher, sid=source_id)
            del_rec = await del_res.single()
            deleted_val = del_rec.get("deleted") if del_rec else None
            deleted = int(deleted_val) if deleted_val is not None else 0
            await del_res.consume()

        linked = await GenericEntityService.link_rule_preset_for_source(
            preset_name=preset_name,
            source_id=source_id,
            sample_limit=sample_limit,
        )
        linked["deleted"] = deleted
        return linked

    @staticmethod
    async def run_rule_preset(preset_name: str) -> Dict[str, Any]:
        preset = RULE_LINK_PRESETS.get(preset_name)
        if not preset:
            return {"status": "error", "message": "Preset not found"}

        edges = await GenericEntityService.link_entities_by_rules(**preset)
        return {"status": "success", "preset": preset_name, "edges_created": edges}

    @staticmethod
    async def count_rule_matches_for_item(preset_name: str, source_item: Dict[str, Any]) -> int:
        graphiti_app, _, _, _ = _get_graph_deps()
        preset = RULE_LINK_PRESETS.get(preset_name)
        if not preset:
            raise ValueError("Preset not found")

        target_label = GenericEntityService._validate_cypher_identifier(preset["target_label"])
        source_list_fields: List[str] = preset.get("source_list_fields") or []
        source_text_fields: List[str] = preset.get("source_text_fields") or []
        target_text_fields: List[str] = preset.get("target_text_fields") or []
        mode: str = preset.get("mode") or "contains_any"

        for f in source_list_fields + source_text_fields + target_text_fields:
            GenericEntityService._validate_cypher_identifier(f)

        if mode not in {"contains_any", "equals_any"}:
            raise ValueError("Invalid mode")

        if preset_name == "standard_to_product":
            keyword_candidates = GenericEntityService._extract_standard_keywords(source_item)
            keyword_candidates = [
                str(k).strip()
                for k in (keyword_candidates or [])
                if isinstance(k, str) and str(k).strip()
            ]
            if not keyword_candidates:
                print(
                    "[RULE_MATCH] preset=", preset_name,
                    "target_label=", target_label,
                    "mode=", mode,
                    "keywords=", [],
                    "matched_count=", 0,
                )
                return 0

            keyword = keyword_candidates[0]
            keyword_codepoints = [hex(ord(c)) for c in keyword]
            print(
                "[RULE_MATCH_PARAM] preset=", preset_name,
                "build=", GenericEntityService.BUILD_MARKER,
                "keyword_raw=", keyword,
                "keyword_repr=", repr(keyword),
                "keyword_codepoints=", keyword_codepoints,
            )

            or_clauses = []
            for tf in target_text_fields:
                or_clauses.append(f"coalesce(toString(p.{tf}), '') CONTAINS $keyword")
            where_clause = " OR ".join(or_clauses) if or_clauses else "false"

            cypher = f"""
            MATCH (p:{target_label})
            WHERE {where_clause}
            RETURN count(p) as matched_count
            """

            print(
                "[RULE_MATCH_CYPHER] preset=", preset_name,
                "target_label=", target_label,
                "mode=", mode,
                "using_keyword=", keyword,
                "cypher=", " ".join(line.strip() for line in cypher.splitlines() if line.strip()),
            )

            async with graphiti_app.driver.session() as session:
                result = await session.run(cypher, keyword=keyword)
                record = await result.single()
                matched_param = int(record["matched_count"]) if record and "matched_count" in record else 0

                literal_keyword = keyword.replace("\\", "\\\\").replace("'", "\\'")
                literal_or_clauses = []
                for tf in target_text_fields:
                    literal_or_clauses.append(f"coalesce(toString(p.{tf}), '') CONTAINS '{literal_keyword}'")
                literal_where = " OR ".join(literal_or_clauses) if literal_or_clauses else "false"
                literal_cypher = f"""
                MATCH (p:{target_label})
                WHERE {literal_where}
                RETURN count(p) as matched_count
                """

                print(
                    "[RULE_MATCH_CYPHER_LITERAL] preset=", preset_name,
                    "target_label=", target_label,
                    "mode=", mode,
                    "using_keyword=", keyword,
                    "cypher=", " ".join(line.strip() for line in literal_cypher.splitlines() if line.strip()),
                )

                result_literal = await session.run(literal_cypher)
                record_literal = await result_literal.single()
                matched_literal = int(record_literal["matched_count"]) if record_literal and "matched_count" in record_literal else 0

                final_matched = matched_literal or matched_param

                print(
                    "[RULE_MATCH] preset=", preset_name,
                    "target_label=", target_label,
                    "mode=", mode,
                    "keywords=", json.dumps(keyword_candidates, ensure_ascii=False),
                    "using_keyword=", keyword,
                    "matched_count_param=", matched_param,
                    "matched_count_literal=", matched_literal,
                    "matched_count_final=", final_matched,
                )

                return final_matched

        conditions: List[str] = []
        params: Dict[str, Any] = {}

        for sf in source_list_fields:
            raw = source_item.get(sf)
            if raw is None:
                continue
            raw_list = raw if isinstance(raw, list) else [raw]
            cleaned = [str(v).strip() for v in raw_list if str(v).strip()]
            if not cleaned:
                continue

            param_name = f"sf_list_{sf}"
            params[param_name] = cleaned

            per_target = []
            for tf in target_text_fields:
                if mode == "equals_any":
                    per_target.append(
                        f"trim(coalesce(toString(p.{tf}), '')) = trim(coalesce(toString(k), ''))"
                    )
                else:
                    per_target.append(
                        f"coalesce(toString(p.{tf}), '') CONTAINS coalesce(toString(k), '')"
                    )
            if per_target:
                conditions.append(
                    f"ANY(k IN ${param_name} WHERE coalesce(trim(toString(k)), '') <> '' AND ({' OR '.join(per_target)}))"
                )

        for sf in source_text_fields:
            raw = source_item.get(sf)
            if raw is None:
                continue
            text_val = str(raw).strip()
            if not text_val:
                continue

            param_name = f"sf_text_{sf}"
            params[param_name] = text_val

            per_target = []
            for tf in target_text_fields:
                if mode == "equals_any":
                    per_target.append(
                        f"trim(coalesce(toString(p.{tf}), '')) = trim(${param_name})"
                    )
                else:
                    per_target.append(
                        f"coalesce(toString(p.{tf}), '') CONTAINS ${param_name}"
                    )
            if per_target:
                conditions.append(
                    f"(trim(${param_name}) <> '' AND ({' OR '.join(per_target)}))"
                )

        if not conditions:
            print(
                "[RULE_MATCH] no conditions built",
                "preset=", preset_name,
                "target_label=", target_label,
                "mode=", mode,
                "source_list_fields=", source_list_fields,
                "source_text_fields=", source_text_fields,
                "target_text_fields=", target_text_fields,
                "source_item=", json.dumps(source_item, ensure_ascii=False)[:500],
            )
            return 0

        cypher_conditions = " OR ".join(conditions)
        cypher = f"""
        MATCH (p:{target_label})
        WHERE ({cypher_conditions})
        RETURN count(p) as matched_count
        """

        async with graphiti_app.driver.session() as session:
            result = await session.run(cypher, **params)
            record = await result.single()
            matched = int(record["matched_count"]) if record and "matched_count" in record else 0

            print(
                "[RULE_MATCH] preset=", preset_name,
                "target_label=", target_label,
                "mode=", mode,
                "source_list_fields=", source_list_fields,
                "source_text_fields=", source_text_fields,
                "target_text_fields=", target_text_fields,
                "params=", json.dumps(params, ensure_ascii=False),
                "matched_count=", matched,
            )

            return matched

    # ==========================================
    # 1. 通用写入 (Generic Write / Ingest)
    # ==========================================
    @staticmethod
    async def ingest_entities(
            data_list: List[Dict[str, Any]],
            label: str,
            id_field: str,
            vector_template: str,
            graph_exclude_fields: Optional[List[str]] = None,
            group_id: str = "default",
            concurrency: int = 5
    ):
        graphiti_app, vector_store, _, _ = _get_graph_deps()
        if graph_exclude_fields is None:
            graph_exclude_fields = []

        print(f"🚀 [Generic] 开始导入 {label}，共 {len(data_list)} 条，并发度: {concurrency}")

        semaphore = asyncio.Semaphore(concurrency)
        lock = asyncio.Lock()
        summary: Dict[str, Any] = {
            "label": label,
            "id_field": id_field,
            "received": len(data_list),
            "skipped_missing_id": 0,
            "neo4j_written": 0,
            "neo4j_failed": 0,
            "vector_written": 0,
            "vector_failed": 0,
            "errors": [],
        }

        def _push_error(entity_id: Any, stage: str, err: Exception) -> None:
            if len(summary["errors"]) >= 20:
                return
            summary["errors"].append(
                {
                    "entity_id": str(entity_id) if entity_id is not None else None,
                    "stage": stage,
                    "error": str(err),
                }
            )

        async def _process_single(item: Dict[str, Any]):
            async with semaphore:
                unique_id = item.get(id_field)
                if not unique_id:
                    print(f"⚠️ 跳过无主键数据: {str(item)[:50]}...")
                    async with lock:
                        summary["skipped_missing_id"] += 1
                    return

                graph_props = item.copy()
                for field in graph_exclude_fields:
                    if field in graph_props:
                        del graph_props[field]

                try:
                    await GenericEntityService._write_node_to_neo4j(label, id_field, unique_id, graph_props)
                    async with lock:
                        summary["neo4j_written"] += 1
                except Exception as e:
                    print(f"❌ Neo4j 写入失败 {label}:{unique_id} -> {e}")
                    async with lock:
                        summary["neo4j_failed"] += 1
                        _push_error(unique_id, "neo4j_write", e)
                    return

                try:
                    try:
                        safe_data = {k: v if v is not None else "" for k, v in item.items()}
                        text_content = vector_template.format(**safe_data)
                    except KeyError as e:
                        print(f"⚠️ 向量模版匹配失败 ({e})，降级为 JSON 文本")
                        text_content = json.dumps(item, ensure_ascii=False)
                    except Exception as e:
                        print(f"⚠️ 向量生成未知错误: {e}")
                        text_content = str(item)

                    doc = Document(
                        page_content=text_content,
                        metadata={
                            "entity_id": str(unique_id),
                            "entity_label": label,
                            "original_id_field": id_field,
                            "group_id": group_id
                        }
                    )

                    await vector_store.aadd_documents([doc])
                    async with lock:
                        summary["vector_written"] += 1
                except Exception as e:
                    print(f"⚠️ Qdrant 写入失败 {label}:{unique_id} -> {e}")
                    async with lock:
                        summary["vector_failed"] += 1
                        _push_error(unique_id, "vector_write", e)

        tasks = [_process_single(d) for d in data_list]
        if tasks:
            await asyncio.gather(*tasks)

        print(f"✅ {label} 导入流程结束")
        return summary

    @staticmethod
    async def _write_node_to_neo4j(label: str, id_field: str, unique_id: Any, properties: Dict):
        graphiti_app, _, _, _ = _get_graph_deps()
        driver = getattr(graphiti_app, "driver", None)
        if driver is None:
            raise RuntimeError("Neo4j driver not initialized on graphiti_app")
        clean_props = {}

        for k, v in properties.items():
            if v is None:
                continue

            if isinstance(v, dict):
                try:
                    clean_props[k] = json.dumps(v, ensure_ascii=False)
                except Exception:
                    clean_props[k] = str(v)

            elif isinstance(v, list):
                if len(v) > 0:
                    if isinstance(v[0], (dict, list)):
                        try:
                            clean_props[k] = json.dumps(v, ensure_ascii=False)
                        except Exception:
                            clean_props[k] = str(v)
                    else:
                        clean_props[k] = v
                else:
                    clean_props[k] = v

            else:
                clean_props[k] = v

        query = f"""
            MERGE (n:{label} {{ {id_field}: $uid }})
            SET n += $props, n.last_updated = datetime()
            """

        last_err: Exception | None = None
        for attempt in range(1, 4):
            try:
                async with driver.session() as session:
                    result = await session.run(query, uid=unique_id, props=clean_props)
                    await result.consume()
                return
            except Exception as e:
                last_err = e
                wait_s = min(8, 2 ** (attempt - 1))
                print(
                    "[NEO4J_WRITE_RETRY] label=", label,
                    "id_field=", id_field,
                    "uid=", str(unique_id),
                    "attempt=", attempt,
                    "wait_s=", wait_s,
                    "error=", str(e),
                    flush=True,
                )
                if attempt < 3:
                    await asyncio.sleep(wait_s)
        if last_err is not None:
            raise last_err

    # ==========================================
    # 2. 通用查询 (Generic Query / Search)
    # ==========================================
    @staticmethod
    async def generic_search(
            query: str,
            target_label: str,
            limit: int = 5
    ) -> Dict[str, Any]:
        graphiti_app, _, QDRANT_URL, langchain_embeddings = _get_graph_deps()
        print(f"🔎 [GenericSearch] 查 {target_label}: {query}")

        try:
            embed = langchain_embeddings.embed_query(query)
            resp = requests.post(
                f"{QDRANT_URL}/collections/multimodal_knowledge/points/search",
                json={
                    "vector": embed,
                    "limit": limit * 3,
                    "with_payload": True,
                },
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            points = data.get("result", [])
            vector_results = []
            for p in points:
                payload = p.get("payload") or {}
                page_content = payload.get("page_content", "")
                metadata = payload.get("metadata", {})
                doc = Document(page_content=page_content, metadata=metadata)
                score = p.get("score", 0.0)
                vector_results.append((doc, score))
        except Exception as e:
            print(f"❌ 向量搜索失败: {e}")
            return {"results": [], "error": str(e)}

        candidates = []
        candidate_ids = []
        original_id_field = "id"

        for doc, score in vector_results:
            meta = doc.metadata
            if meta.get("entity_label") == target_label:
                uid = meta.get("entity_id")
                original_id_field = meta.get("original_id_field", "id")

                if uid and uid not in candidate_ids:
                    candidates.append({
                        "id": uid,
                        "score": score,
                        "semantic_text": doc.page_content,
                        "metadata": meta,
                        "graph_data": {}
                    })
                    candidate_ids.append(uid)

            if len(candidate_ids) >= limit:
                break

        if not candidate_ids:
            return {"results": [], "message": "No matching entities found"}

        cypher = f"""
        MATCH (n:{target_label})
        WHERE n.{original_id_field} IN $ids
        RETURN n
        """

        try:
            async with graphiti_app.driver.session() as session:
                result = await session.run(cypher, ids=candidate_ids)
                records = await result.data()

                graph_map = {}
                for r in records:
                    node = r['n']
                    node_id = node.get(original_id_field)
                    if node_id:
                        graph_map[str(node_id)] = node

                for cand in candidates:
                    cand_id = cand['id']
                    if cand_id in graph_map:
                        cand['graph_data'] = graph_map[cand_id]
                    else:
                        cand['graph_data'] = {"_status": "Not found in Graph (Sync delay?)"}

        except Exception as e:
            print(f"⚠️ 图谱查询失败: {e}")

        return {"results": candidates}

    # ==========================================
    # 3. 通用修改 (Generic Modify / Update)
    # ==========================================
    @staticmethod
    async def update_entity(
            label: str,
            id_field: str,
            unique_id: str,
            update_data: Dict[str, Any]
    ) -> Optional[Dict]:
        graphiti_app, _, _, _ = _get_graph_deps()
        clean_props = {k: v for k, v in update_data.items() if v is not None}

        query = f"""
        MATCH (n:{label} {{ {id_field}: $uid }})
        SET n += $props, n.last_updated = datetime()
        RETURN n
        """

        try:
            async with graphiti_app.driver.session() as session:
                result = await session.run(query, uid=unique_id, props=clean_props)
                record = await result.single()
                if record:
                    return dict(record['n'])
                return None
        except Exception as e:
            print(f"❌ 更新失败: {e}")
            raise e

    @staticmethod
    async def delete_entity(
            label: str,
            id_field: str,
            unique_id: str
    ) -> int:
        graphiti_app, _, _, _ = _get_graph_deps()
        query = f"""
        MATCH (n:{label} {{ {id_field}: $uid }})
        DETACH DELETE n
        RETURN count(*) as deleted_count
        """

        try:
            async with graphiti_app.driver.session() as session:
                result = await session.run(query, uid=unique_id)
                record = await result.single()
                if record:
                    return record["deleted_count"]
                return 0
        except Exception as e:
            print(f"❌ 删除失败: {e}")
            raise e

    # ==========================================
    # 4. 语义自动关联 (Semantic Linking)
    # ==========================================
    @staticmethod
    async def link_entities_by_semantic(
            source_label: str,
            source_id_field: str,
            target_label: str,
            target_id_field: str,
            top_k: int = 10,
            score_threshold: float = 0.3
    ):
        graphiti_app, _, QDRANT_URL, langchain_embeddings = _get_graph_deps()
        print(f"🔗 [Linking] 开始建立关联: ({source_label}) -> ({target_label})")

        fetch_query = f"MATCH (n:{source_label}) RETURN n.{source_id_field} as uid, n.title as text_content"

        async with graphiti_app.driver.session() as session:
            result = await session.run(fetch_query)
            records = await result.data()

        print(f"   共找到 {len(records)} 个源实体待处理...")

        link_count = 0

        for rec in records:
            uid = rec["uid"]
            text = rec.get("text_content", "")

            if not uid or not text:
                continue

            try:
                vector = langchain_embeddings.embed_query(text)

                search_payload = {
                    "vector": vector,
                    "limit": top_k,
                    "with_payload": True,
                    "score_threshold": score_threshold,
                    "filter": {
                        "must": [
                            {
                                "key": "metadata.entity_label",
                                "match": {"value": target_label}
                            }
                        ]
                    }
                }

                response = requests.post(
                    f"{QDRANT_URL}/collections/multimodal_knowledge/points/search",
                    json=search_payload,
                    timeout=60
                )
                response.raise_for_status()
                search_res = response.json()
                points = search_res.get("result", [])

                targets_to_link = []
                resolved_target_id_field = target_id_field
                for point in points:
                    payload = point.get("payload", {})
                    metadata = payload.get("metadata", {})
                    resolved_target_id_field = metadata.get("original_id_field", resolved_target_id_field)
                    target_uid = metadata.get("entity_id")

                    if target_uid:
                        targets_to_link.append(target_uid)

                if targets_to_link:
                    created = await GenericEntityService._create_edges_batch(
                        source_label, source_id_field, uid,
                        target_label, resolved_target_id_field, targets_to_link,
                        rel_type="APPLIES_TO"
                    )
                    link_count += int(created)
                    print(f"   ✅ {uid} -> 关联了 {int(created)} 个产品")

            except Exception as e:
                print(f"❌ 处理 {uid} 关联失败: {e}")

        print(f"✅ 关联任务结束，共创建 {link_count} 条关系。")

    @staticmethod
    async def _create_edges_batch(src_lbl, src_key, src_val, tgt_lbl, tgt_key, tgt_val_list, rel_type) -> int:
        graphiti_app, _, _, _ = _get_graph_deps()
        cypher = f"""
        MATCH (s:{src_lbl} {{ {src_key}: $src_val }})
        MATCH (t:{tgt_lbl})
        WHERE t.{tgt_key} IN $tgt_vals
        MERGE (s)-[r:{rel_type}]->(t)
        RETURN count(r) as rel_count
        """
        async with graphiti_app.driver.session() as session:
            result = await session.run(cypher, src_val=src_val, tgt_vals=tgt_val_list)
            record = await result.single()
            if record and "rel_count" in record:
                return int(record["rel_count"])
            return 0

    @staticmethod
    async def run_rule_based_linking():
        graphiti_app, _, _, _ = _get_graph_deps()
        cypher = """
            MATCH (s:StandardDocument), (p:Product)
            WHERE 
                // 规则：标准文档的标签 出现在 产品的系列名中
                ANY(tag IN s.tags WHERE p.series CONTAINS tag)
                OR
                // 规则：或者标题包含产品的类别名称
                s.title CONTAINS p.className
            MERGE (s)-[:APPLIES_TO]->(p)
            RETURN count(*) as edges_created
            """
        async with graphiti_app.driver.session() as session:
            await session.run(cypher)
