from typing import Dict, Any, List
import requests
from langchain_core.documents import Document

from backend.core.graph_client import graphiti_app, QDRANT_URL, langchain_embeddings


class SearchService:
    """
    搜索服务：负责混合检索（向量召回 + 图谱补全）
    """

    @staticmethod
    async def hybrid_search(query: str, limit: int = 5) -> Dict[str, Any]:
        print(f"🔎 [Search] 用户查询: {query}")

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
            vector_results: List[tuple[Document, float]] = []
            for p in points:
                payload = p.get("payload") or {}
                page_content = payload.get("page_content", "")
                metadata = payload.get("metadata", {})
                doc = Document(page_content=page_content, metadata=metadata)
                score = p.get("score", 0.0)
                vector_results.append((doc, score))
        except Exception as e:
            print(f"❌ 向量搜索失败: {e}")
            return {"error": "Vector search failed"}

        if not vector_results:
            return {"results": [], "message": "No vector matches found"}

        # 提取关键信息：ID 和 分数
        candidate_ids = []
        candidates_map = {}  # 用 map 方便后续合并

        for doc, score in vector_results:
            meta = doc.metadata
            if meta.get("entity_label") == "Product":
                code = meta.get("entity_id")
                if code and code not in candidates_map:
                    candidate_ids.append(code)
                    candidates_map[code] = {
                        "code": code,
                        "score": score,
                        "semantic_text": doc.page_content,
                        "metadata": meta,
                        "graph_data": {}
                    }

        print(f"✅ 向量召回 ID: {candidate_ids}")

        # ==========================================
        # 2. 图谱补全 (Graph Lookup) -> 获取结构化详情
        # ==========================================
        # 这里不使用 LLM，而是直接执行高效的 Cypher
        # 目的：查出这些 ID 对应的 Product 节点，以及它挂载的所有 Attribute

        cypher_query = """
        MATCH (p:Product)
        WHERE p.code IN $codes
        RETURN p
        """

        try:
            async with graphiti_app.driver.session() as session:
                result = await session.run(cypher_query, codes=candidate_ids)
                records: List[Dict[str, Any]] = await result.data()

            for record in records:
                record_dict: Dict[str, Any] = dict(record) if isinstance(record, dict) else record
                node = record_dict.get("p") or {}
                code = node.get("code")
                if code and code in candidates_map:
                    candidates_map[code]["graph_data"] = node

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
