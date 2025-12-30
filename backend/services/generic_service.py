import asyncio
import json
from typing import List, Dict, Any, Optional

# 复用已有的核心组件
from backend.core.graph_client import graphiti_app, vector_store
from langchain_core.documents import Document


class GenericEntityService:
    """
    通用实体服务：实现对任意实体的 增(Write)、查(Query)、改(Modify)
    """

    # ==========================================
    # 1. 通用写入 (Generic Write / Ingest)
    # ==========================================
    @staticmethod
    async def ingest_entities(
            data_list: List[Dict[str, Any]],
            label: str,  # e.g., "Product", "User"
            id_field: str,  # e.g., "code", "user_id"
            vector_template: str,  # e.g., "姓名: {name}, 简介: {desc}"
            group_id: str = "default",
            concurrency: int = 5
    ):
        """
        通用的批量入库方法：同时写入 图数据库(Neo4j) 和 向量数据库(Qdrant)
        """
        print(f"🚀 [Generic] 开始导入 {label}，共 {len(data_list)} 条")
        semaphore = asyncio.Semaphore(concurrency)

        async def _process_single(item: Dict[str, Any]):
            async with semaphore:
                try:
                    # 1. 获取唯一主键值
                    unique_id = item.get(id_field)
                    if not unique_id:
                        print(f"⚠️ 跳过无主键数据: {item}")
                        return

                    # 2. 写入 Graph (Neo4j)
                    # 使用动态的 Label 和 ID 进行 MERGE 操作
                    await GenericEntityService._write_node_to_neo4j(label, id_field, unique_id, item)

                    # 3. 写入 Vector Store (Qdrant)
                    # 动态生成向量文本
                    try:
                        # 使用 format 填充模板，如 "{name} is {age}" -> "Bob is 20"
                        # {k: v or ""} 用于处理 None 值，防止 format 报错
                        safe_data = {k: v if v is not None else "" for k, v in item.items()}
                        text_content = vector_template.format(**safe_data)
                    except KeyError as e:
                        # 如果模板里的 key 在数据里找不到，降级为直接存 JSON 字符串
                        print(f"⚠️ 模版匹配失败 ({e})，使用 JSON 文本")
                        text_content = json.dumps(item, ensure_ascii=False)

                    doc = Document(
                        page_content=text_content,
                        metadata={
                            "entity_id": str(unique_id),  # 统一叫 entity_id
                            "entity_label": label,  # 存入 label 以便过滤
                            "group_id": group_id,
                            "original_id_field": id_field
                        }
                    )
                    # 异步写入
                    await vector_store.aadd_documents([doc])

                except Exception as e:
                    print(f"❌ 处理 {unique_id} 失败: {e}")

        # 并发执行所有任务
        tasks = [_process_single(d) for d in data_list]
        await asyncio.gather(*tasks)
        print(f"✅ {label} 导入完成")

    @staticmethod
    async def _write_node_to_neo4j(label: str, id_field: str, unique_id: str, properties: Dict):
        """
        内部方法：动态生成 MERGE 语句写入 Neo4j
        """
        # 移除 None 值，Neo4j 不支持 Null 属性
        clean_props = {k: v for k, v in properties.items() if v is not None}

        # 动态构造 Cypher: MERGE (n:User {user_id: $uid}) SET n += $props
        # 注意：这里使用 f-string 注入 label 和 id_field (作为Schema)，参数使用 $params 注入 (防止注入攻击)
        query = f"""
        MERGE (n:{label} {{ {id_field}: $uid }})
        SET n += $props, n.last_updated = datetime()
        """

        # 使用 graphiti_app.driver 直接获取 session
        async with graphiti_app.driver.session() as session:
            await session.run(query, uid=unique_id, props=clean_props)

    # ==========================================
    # 2. 通用查询 (Generic Query / Search)
    # ==========================================
    @staticmethod
    async def generic_search(
            query: str,
            target_label: str,  # 限制搜索某种类型，如 "Product"
            limit: int = 5
    ) -> Dict[str, Any]:
        """
        通用混合检索：向量召回 + 图谱属性补全
        """
        print(f"🔎 [GenericSearch] 查 {target_label}: {query}")

        # A. 向量召回 (Semantic Search)
        # 注意：我们需要过滤 metadata['entity_label'] == target_label
        # Qdrant/LangChain 的 filter 语法较复杂，这里为了通用性，先查出来再在内存过滤 (数据量大时建议用 filter 参数)
        try:
            vector_results = await vector_store.asimilarity_search_with_score(query, k=limit * 2)  # 多查一点供过滤
        except Exception as e:
            return {"error": str(e)}

        candidates = []
        candidate_ids = []
        original_id_field = "id"  # 默认值

        for doc, score in vector_results:
            meta = doc.metadata
            # 过滤：只保留目标类型的实体
            if meta.get("entity_label") == target_label:
                uid = meta.get("entity_id")
                original_id_field = meta.get("original_id_field", "id")

                if uid and uid not in candidate_ids:
                    candidates.append({
                        "id": uid,
                        "score": score,
                        "semantic_text": doc.page_content,
                        "metadata": meta,
                        "graph_data": {}  # 待填充
                    })
                    candidate_ids.append(uid)

            if len(candidate_ids) >= limit:
                break

        if not candidate_ids:
            return {"results": [], "message": "No matching entities found"}

        # B. 图谱补全 (Graph Lookup)
        # 动态生成 Cypher 查询详情
        cypher = f"""
        MATCH (n:{target_label})
        WHERE n.{original_id_field} IN $ids
        RETURN n
        """

        try:
            async with graphiti_app.driver.session() as session:
                result = await session.run(cypher, ids=candidate_ids)
                records = await result.data()

                # 建立 ID -> Node Data 的映射
                graph_map = {}
                for r in records:
                    node_data = r['n']
                    # 获取主键值
                    node_id = node_data.get(original_id_field)
                    graph_map[str(node_id)] = node_data

                # 回填数据
                for cand in candidates:
                    cand_id = cand['id']
                    if cand_id in graph_map:
                        cand['graph_data'] = graph_map[cand_id]

        except Exception as e:
            print(f"⚠️ 图谱补全失败: {e}")

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
    ):
        """
        通用更新实体属性
        """
        clean_props = {k: v for k, v in update_data.items() if v is not None}

        query = f"""
        MATCH (n:{label} {{ {id_field}: $uid }})
        SET n += $props, n.last_updated = datetime()
        RETURN n
        """
        async with graphiti_app.driver.session() as session:
            result = await session.run(query, uid=unique_id, props=clean_props)
            record = await result.single()
            return record['n'] if record else None