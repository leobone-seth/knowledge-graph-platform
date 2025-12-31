import asyncio
import json
from typing import List, Dict, Any, Optional

import requests
from langchain_core.documents import Document

from backend.entity_registry import RULE_LINK_PRESETS


def _get_graph_deps():
    from backend.core.graph_client import graphiti_app, vector_store, QDRANT_URL, langchain_embeddings
    return graphiti_app, vector_store, QDRANT_URL, langchain_embeddings


class GenericEntityService:
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
            if record and "edges_created" in record:
                return int(record["edges_created"])
            return 0

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
            return 0

        cypher = f"""
        MATCH (p:{target_label})
        WHERE ({' OR '.join(conditions)})
        RETURN count(p) as matched_count
        """

        async with graphiti_app.driver.session() as session:
            result = await session.run(cypher, **params)
            record = await result.single()
            if record and "matched_count" in record:
                return int(record["matched_count"])
            return 0

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

        async def _process_single(item: Dict[str, Any]):
            async with semaphore:
                try:
                    unique_id = item.get(id_field)
                    if not unique_id:
                        print(f"⚠️ 跳过无主键数据: {str(item)[:50]}...")
                        return

                    graph_props = item.copy()
                    for field in graph_exclude_fields:
                        if field in graph_props:
                            del graph_props[field]

                    await GenericEntityService._write_node_to_neo4j(label, id_field, unique_id, graph_props)

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

                except Exception as e:
                    print(f"❌ 处理 {unique_id} 失败: {e}")

        tasks = [_process_single(d) for d in data_list]
        if tasks:
            await asyncio.gather(*tasks)

        print(f"✅ {label} 导入流程结束")

    @staticmethod
    async def _write_node_to_neo4j(label: str, id_field: str, unique_id: Any, properties: Dict):
        graphiti_app, _, _, _ = _get_graph_deps()
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

        async with graphiti_app.driver.session() as session:
            await session.run(query, uid=unique_id, props=clean_props)

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
                    await GenericEntityService._create_edges_batch(
                        source_label, source_id_field, uid,
                        target_label, resolved_target_id_field, targets_to_link,
                        rel_type="APPLIES_TO"
                    )
                    link_count += len(targets_to_link)
                    print(f"   ✅ {uid} -> 关联了 {len(targets_to_link)} 个产品")

            except Exception as e:
                print(f"❌ 处理 {uid} 关联失败: {e}")

        print(f"✅ 关联任务结束，共创建 {link_count} 条关系。")

    @staticmethod
    async def _create_edges_batch(src_lbl, src_key, src_val, tgt_lbl, tgt_key, tgt_val_list, rel_type):
        graphiti_app, _, _, _ = _get_graph_deps()
        cypher = f"""
        MATCH (s:{src_lbl} {{ {src_key}: $src_val }})
        MATCH (t:{tgt_lbl})
        WHERE t.{tgt_key} IN $tgt_vals
        MERGE (s)-[:{rel_type}]->(t)
        """
        async with graphiti_app.driver.session() as session:
            await session.run(cypher, src_val=src_val, tgt_vals=tgt_val_list)

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
