import asyncio
import json
from typing import List, Dict, Any, Optional

import requests
from langchain_core.documents import Document

from backend.core.graph_client import (
    graphiti_app,
    vector_store,
    QDRANT_URL,
    langchain_embeddings,
)


class GenericEntityService:
    """
    通用实体服务 (Generic Entity Service)

    职责：
    1. 提供对任意实体的 增(Ingest)、查(Search)、改(Update) 能力。
    2. 维护 Graph (Neo4j) 与 Vector (Qdrant) 的数据一致性。
    3. 支持读写分离策略：大字段只存向量库，不存图数据库。
    """

    RULE_LINK_PRESETS: Dict[str, Dict[str, Any]] = {
        "standard_to_product": {
            "source_label": "StandardDocument",
            "source_id_field": "standard_code",
            "target_label": "Product",
            "target_id_field": "code",
            "rel_type": "APPLIES_TO",
            "source_list_fields": ["tags"],
            "source_text_fields": ["title", "summary"],
            "target_text_fields": ["elem", "fun", "className", "series"],
            "mode": "contains_any",
        },
        "sensory_to_product": {
            "source_label": "Sensory",
            "source_id_field": "name",
            "target_label": "Product",
            "target_id_field": "code",
            "rel_type": "EVOKES",
            "source_list_fields": [],
            "source_text_fields": ["name"],
            "target_text_fields": ["fun"],
            "mode": "contains_any",
        },
        "season_to_product": {
            "source_label": "Season",
            "source_id_field": "name",
            "target_label": "Product",
            "target_id_field": "code",
            "rel_type": "SUITS_FOR",
            "source_list_fields": [],
            "source_text_fields": ["name"],
            "target_text_fields": ["season_marking"],
            "mode": "equals_any",
        },
    }

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
        preset = GenericEntityService.RULE_LINK_PRESETS.get(preset_name)
        if not preset:
            return {"status": "error", "message": "Preset not found"}

        edges = await GenericEntityService.link_entities_by_rules(**preset)
        return {"status": "success", "preset": preset_name, "edges_created": edges}

    # ==========================================
    # 1. 通用写入 (Generic Write / Ingest)
    # ==========================================
    @staticmethod
    async def ingest_entities(
            data_list: List[Dict[str, Any]],
            label: str,  # e.g., "Product", "User", "StandardDocument"
            id_field: str,  # e.g., "code", "user_id", "standard_code"
            vector_template: str,  # e.g., "标题: {title}, 摘要: {summary}"
            graph_exclude_fields: Optional[List[str]] = None,  # 不需要存入 Neo4j 的大字段列表
            group_id: str = "default",
            concurrency: int = 5
    ):
        """
        通用的批量入库方法

        Args:
            data_list: 待写入的数据字典列表
            label: Neo4j 中的节点标签 (Label)
            id_field: 数据中作为唯一主键的字段名
            vector_template: 用于生成向量文本的字符串模版
            graph_exclude_fields: 指定哪些字段不需要写入 Neo4j (例如超长的正文)
            group_id: 数据分组 ID
            concurrency: 并发写入的线程/任务数
        """
        if graph_exclude_fields is None:
            graph_exclude_fields = []

        print(f"🚀 [Generic] 开始导入 {label}，共 {len(data_list)} 条，并发度: {concurrency}")

        # 使用信号量控制并发
        semaphore = asyncio.Semaphore(concurrency)

        async def _process_single(item: Dict[str, Any]):
            async with semaphore:
                try:
                    # 1. 获取并校验主键
                    unique_id = item.get(id_field)
                    if not unique_id:
                        # 尝试转为字符串查找，或者跳过
                        print(f"⚠️ 跳过无主键数据: {str(item)[:50]}...")
                        return

                    # === Step A: 写入 Graph (Neo4j) ===
                    # 准备写入 Neo4j 的属性：过滤掉大字段
                    graph_props = item.copy()
                    for field in graph_exclude_fields:
                        if field in graph_props:
                            del graph_props[field]

                    # 动态写入节点
                    await GenericEntityService._write_node_to_neo4j(label, id_field, unique_id, graph_props)

                    # === Step B: 写入 Vector Store (Qdrant) ===
                    # 动态生成向量文本 (使用原始完整数据 item)
                    try:
                        # 处理 None 值，防止 format 报错
                        safe_data = {k: v if v is not None else "" for k, v in item.items()}
                        text_content = vector_template.format(**safe_data)
                    except KeyError as e:
                        print(f"⚠️ 向量模版匹配失败 ({e})，降级为 JSON 文本")
                        text_content = json.dumps(item, ensure_ascii=False)
                    except Exception as e:
                        print(f"⚠️ 向量生成未知错误: {e}")
                        text_content = str(item)

                    # 构建 Document 对象
                    doc = Document(
                        page_content=text_content,
                        metadata={
                            "entity_id": str(unique_id),  # 统一存储为字符串 ID
                            "entity_label": label,  # 用于过滤
                            "original_id_field": id_field,  # 用于回查
                            "group_id": group_id
                        }
                    )

                    # 异步写入向量库
                    await vector_store.aadd_documents([doc])

                except Exception as e:
                    print(f"❌ 处理 {unique_id} 失败: {e}")

        # 创建并执行任务
        tasks = [_process_single(d) for d in data_list]
        if tasks:
            await asyncio.gather(*tasks)

        print(f"✅ {label} 导入流程结束")

    @staticmethod
    async def _write_node_to_neo4j(label: str, id_field: str, unique_id: Any, properties: Dict):
        """
        内部方法：动态生成 MERGE 语句并写入 Neo4j
        [修改版] 自动将字典或复杂列表序列化为 JSON 字符串，防止 Neo4j 报错
        """
        clean_props = {}

        for k, v in properties.items():
            # 1. 跳过 None (Neo4j 不存 Null)
            if v is None:
                continue

            # 2. 处理字典 (Dict) -> 转 JSON 字符串
            # 解决 extra_metadata 报错的核心逻辑
            if isinstance(v, dict):
                try:
                    clean_props[k] = json.dumps(v, ensure_ascii=False)
                except Exception:
                    clean_props[k] = str(v)

            # 3. 处理列表 (List)
            elif isinstance(v, list):
                if len(v) > 0:
                    # 如果列表里包含字典 (e.g. List[Dict])，Neo4j 也不支持 -> 转 JSON 字符串
                    if isinstance(v[0], (dict, list)):
                        try:
                            clean_props[k] = json.dumps(v, ensure_ascii=False)
                        except Exception:
                            clean_props[k] = str(v)
                    else:
                        # 如果是 List[str] 或 List[int]，Neo4j 支持 -> 直接保留
                        clean_props[k] = v
                else:
                    # 空列表可以选择不存，或者存为空数组
                    clean_props[k] = v

            # 4. 其他基本类型 (int, str, float) -> 直接保留
            else:
                clean_props[k] = v

        # 动态构造 Cypher
        query = f"""
            MERGE (n:{label} {{ {id_field}: $uid }})
            SET n += $props, n.last_updated = datetime()
            """

        # 使用 graphiti_app 底层的 driver
        async with graphiti_app.driver.session() as session:
            await session.run(query, uid=unique_id, props=clean_props)
    # ==========================================
    # 2. 通用查询 (Generic Query / Search)
    # ==========================================
    @staticmethod
    async def generic_search(
            query: str,
            target_label: str,  # 限制搜索某种类型
            limit: int = 5
    ) -> Dict[str, Any]:
        """
        通用混合检索：向量召回 + 图谱属性补全
        """
        print(f"🔎 [GenericSearch] 查 {target_label}: {query}")

        # === Step A: 向量召回 ===
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
        original_id_field = "id"  # 默认值

        for doc, score in vector_results:
            meta = doc.metadata
            # 过滤：只保留目标 Label 的数据
            if meta.get("entity_label") == target_label:
                uid = meta.get("entity_id")
                # 记录该实体在图谱中的主键字段名 (e.g., "code" or "standard_code")
                original_id_field = meta.get("original_id_field", "id")

                # 去重
                if uid and uid not in candidate_ids:
                    candidates.append({
                        "id": uid,
                        "score": score,
                        "semantic_text": doc.page_content,  # 向量库里的文本（可能包含大字段的摘要）
                        "metadata": meta,
                        "graph_data": {}  # 稍后填充
                    })
                    candidate_ids.append(uid)

            if len(candidate_ids) >= limit:
                break

        if not candidate_ids:
            return {"results": [], "message": "No matching entities found"}

        # === Step B: 图谱补全 (Graph Lookup) ===
        # 使用 Cypher 批量查出这些实体的最新属性
        cypher = f"""
        MATCH (n:{target_label})
        WHERE n.{original_id_field} IN $ids
        RETURN n
        """

        try:
            async with graphiti_app.driver.session() as session:
                result = await session.run(cypher, ids=candidate_ids)
                records = await result.data()

                # 构建 ID -> Node Props 映射表
                graph_map = {}
                for r in records:
                    node = r['n']
                    # 获取该节点的主键值
                    # 注意：从 Neo4j 拿回来的 node 是 dict 结构
                    node_id = node.get(original_id_field)
                    if node_id:
                        graph_map[str(node_id)] = node

                # 将图谱数据回填到候选列表中
                for cand in candidates:
                    cand_id = cand['id']
                    if cand_id in graph_map:
                        cand['graph_data'] = graph_map[cand_id]
                    else:
                        cand['graph_data'] = {"_status": "Not found in Graph (Sync delay?)"}

        except Exception as e:
            print(f"⚠️ 图谱查询失败: {e}")
            # 即使图谱挂了，也返回向量结果

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
        """
        通用更新实体属性
        """
        # 移除 None
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
    # 4. 语义自动关联 (Semantic Linking) [新增]
    # ==========================================
    @staticmethod
    async def link_entities_by_semantic(
            source_label: str,  # 源节点 Label，如 "StandardDocument"
            source_id_field: str,  # 源节点主键字段，如 "standard_code"
            target_label: str,  # 目标节点 Label，如 "Product"
            target_id_field: str,  # 目标节点主键字段，如 "code"
            top_k: int = 10,  # 每个文档关联多少个最相似的产品
            score_threshold: float = 0.3  # 相似度阈值
    ):
        """
        [修复版] 语义关联：使用 HTTP 接口 + Server-side Filter，避免客户端版本兼容问题
        """
        print(f"🔗 [Linking] 开始建立关联: ({source_label}) -> ({target_label})")

        # 1. 从 Neo4j 获取所有源节点
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
                # 2. 生成向量 (使用 graph_client 中初始化的全局 embedding 模型)
                vector = langchain_embeddings.embed_query(text)

                # 3. 构造 Qdrant 搜索请求 (带 Filter)
                # LangChain 将 metadata 存在 payload.metadata 下，所以 key 是 "metadata.entity_label"
                search_payload = {
                    "vector": vector,
                    "limit": top_k,
                    "with_payload": True,
                    "score_threshold": score_threshold,  # Qdrant 支持直接传阈值，过滤掉低分结果
                    "filter": {
                        "must": [
                            {
                                "key": "metadata.entity_label",
                                "match": {"value": target_label}
                            }
                        ]
                    }
                }

                # 发送 HTTP 请求 (复用 generic_search 的逻辑)
                response = requests.post(
                    f"{QDRANT_URL}/collections/multimodal_knowledge/points/search",
                    json=search_payload,
                    timeout=60
                )
                response.raise_for_status()
                search_res = response.json()
                points = search_res.get("result", [])

                # 4. 提取目标 ID 并建立关联
                targets_to_link = []
                resolved_target_id_field = target_id_field
                for point in points:
                    payload = point.get("payload", {})
                    metadata = payload.get("metadata", {})
                    resolved_target_id_field = metadata.get("original_id_field", resolved_target_id_field)
                    target_uid = metadata.get("entity_id")

                    if target_uid:
                        targets_to_link.append(target_uid)

                # 5. 批量写入 Neo4j 边
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
        """
        Neo4j 批量建边 Cypher
        """
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
        """
        基于标签的精确匹配：
        如果 StandardDocument 的 tags 包含 '针织'，
        且 Product 的 attributes 或 series 包含 '针织'，则建立关联。
        """
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
