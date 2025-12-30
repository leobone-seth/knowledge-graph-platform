# 引用我们在 graph_client.py 中初始化的全局实例
from typing import Dict, Any, List

from backend.core.graph_client import graphiti_app, vector_store


class SearchService:
    """
    搜索服务：负责混合检索（向量召回 + 图谱补全）
    """

    @staticmethod
    async def hybrid_search(query: str, limit: int = 5) -> Dict[str, Any]:
        print(f"🔎 [Search] 用户查询: {query}")

        # ==========================================
        # 1. 向量召回 (Semantic Search) -> 获取 ID
        # ==========================================
        # 这一步用 LangChain 的 vector_store，因为它封装好了 Embedding 过程
        try:
            # 返回格式: List[(Document, score)]
            vector_results = await vector_store.asimilarity_search_with_score(query, k=limit)
        except Exception as e:
            print(f"❌ 向量搜索失败: {e}")
            return {"error": "Vector search failed"}

        if not vector_results:
            return {"results": [], "message": "No vector matches found"}

        # 提取关键信息：ID 和 分数
        candidate_ids = []
        candidates_map = {}  # 用 map 方便后续合并

        for doc, score in vector_results:
            # 这里的 metadata['product_code'] 是我们在 IngestionService 里存进去的锚点
            code = doc.metadata.get("product_code")
            if code:
                # 过滤重复 ID
                if code not in candidates_map:
                    candidate_ids.append(code)
                    candidates_map[code] = {
                        "code": code,
                        "score": score,
                        "semantic_text": doc.page_content,
                        "metadata": doc.metadata,
                        "graph_data": {}  # 占位
                    }

        print(f"✅ 向量召回 ID: {candidate_ids}")

        # ==========================================
        # 2. 图谱补全 (Graph Lookup) -> 获取结构化详情
        # ==========================================
        # 这里不使用 LLM，而是直接执行高效的 Cypher
        # 目的：查出这些 ID 对应的 Product 节点，以及它挂载的所有 Attribute

        cypher_query = """
        MATCH (p:Product)
        WHERE p.name IN $codes  // 使用参数化查询，安全且快

        // 1. 抓取产品自身属性 (从 EpisodeBody 解析出来的)
        // 假设 attributes 存在节点属性里，或者我们只查关联边

        // 2. 抓取关联的属性节点 (HAS_ATTR 边)
        OPTIONAL MATCH (p)-[:HAS_ATTR]->(attr:ProductAttr)

        RETURN p.name as code, 
               p.group_id as group_id,
               // 聚合该产品的所有属性为列表
               collect({field: attr.field, value: attr.value}) as attributes
        """

        try:
            # 使用 graphiti_app.driver 直接执行，绕过 LangChain Chain
            # async_session 是 Neo4j 官方驱动的异步会话
            async with graphiti_app.driver.session() as session:
                result = await session.run(cypher_query, codes=candidate_ids)
                records: List[Dict[str, Any]] = await result.data()

            # 将图谱查到的结果回填到 map 中
            for record in records:
                # 确保 record 是 dict[str, Any] 类型
                record_dict: Dict[str, Any] = dict(record) if isinstance(record, dict) else record
                code = record_dict["code"]
                if code in candidates_map:
                    candidates_map[code]["graph_data"] = {
                        "group_id": record_dict["group_id"],
                        "attributes": record_dict["attributes"]
                    }

            print(f"✅ 图谱补全完成，共 {len(records)} 条关联数据")

        except Exception as e:
            print(f"⚠️ 图谱查询失败 (降级为仅返回向量结果): {e}")

        # ==========================================
        # 3. 结果排序与返回
        # ==========================================
        # 将 map 转回 list，并按分数排序
        final_results = sorted(
            candidates_map.values(),
            key=lambda x: x["score"],
            reverse=True
        )

        return {
            "query": query,
            "total_candidates": len(final_results),
            "results": final_results
        }
