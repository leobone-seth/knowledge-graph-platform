import asyncio
import json
from typing import List, Dict, Any, Optional

# 引用核心组件
from langchain_core.documents import Document

from backend.core.graph_client import graphiti_app, vector_store


class GenericEntityService:
    """
    通用实体服务 (Generic Entity Service)

    职责：
    1. 提供对任意实体的 增(Ingest)、查(Search)、改(Update) 能力。
    2. 维护 Graph (Neo4j) 与 Vector (Qdrant) 的数据一致性。
    3. 支持读写分离策略：大字段只存向量库，不存图数据库。
    """

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
        """
        # 移除 None 值，Neo4j 不支持 Null 属性
        clean_props = {k: v for k, v in properties.items() if v is not None}

        # 动态构造 Cypher
        # 注意：Label 和 Key 无法参数化，必须拼接入字符串 (请确保 label/id_field 是可信的内部输入)
        # 属性使用 $props 参数化注入，安全
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
            # 召回多一点数据用于内存过滤 (LangChain 的 filter 构造较复杂，这里采用后过滤策略)
            vector_results = await vector_store.asimilarity_search_with_score(query, k=limit * 3)
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
