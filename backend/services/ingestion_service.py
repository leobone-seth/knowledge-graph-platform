import json
import asyncio
from datetime import datetime, timezone
from typing import List, Dict, Any

from graphiti_core.nodes import EpisodeType, EntityNode
from graphiti_core.edges import EntityEdge

# 引用核心组件 (Singleton)
from backend.core.graph_client import graphiti_app
# 引用数据模型
from backend.models.product import Product
from langchain_core.documents import Document
from backend.core.graph_client import graphiti_app, vector_store
class IngestionService:
    """
    数据摄入服务：负责将业务数据转换为图谱节点和边
    """

    @staticmethod
    async def ingest_product_batch(
        products_data: List[Dict[str, Any]],
        group_id: str = "product_demo",
        concurrency: int = 5
    ) -> None:
        """
        并发摄入产品数据：自动构建 Product 节点及其属性关联

        Args:
            products_data: 包含原始产品数据的字典列表
            group_id: 图谱中的分组ID，用于隔离不同批次或业务域的数据
            concurrency: 并发任务数
        """
        total = len(products_data)
        print(f"🚀 开始并发导入产品数据，共 {total} 条，并发度: {concurrency}")

        # 使用信号量控制并发度，防止瞬间压垮数据库或 LLM 接口
        semaphore = asyncio.Semaphore(concurrency)

        async def _process_single_product(i: int, product_dict: Dict[str, Any]) -> None:
            async with semaphore:
                try:
                    product_model = Product(**product_dict)

                    # === 1. 确定唯一关联键 (Anchor Key) ===
                    # 这是连接 Neo4j 和 Qdrant 的纽带，必须保持一致！
                    unique_code = product_model.code or f"Product_{i}"

                    # === 2. 写入 Neo4j (Graphiti) ===
                    # (这部分逻辑保持不变，写入图谱)
                    name = unique_code
                    episode_body_json = json.dumps(product_model.model_dump(exclude_none=True), ensure_ascii=False)

                    await graphiti_app.add_episode(
                        name=name,
                        episode_body=episode_body_json,
                        # ...
                    )
                    # ... (构建 EntityNode 和 Edge 的逻辑保持不变) ...

                    # === 3. 写入 Qdrant (关联的关键) ===
                    # 我们把产品的"描述性文本"向量化，但把 unique_code 存入 metadata

                    # 构造一段利于语义搜索的文本
                    semantic_text = (
                        f"产品编码: {unique_code}。系列: {product_model.series}。"
                        f"功能特点: {product_model.fun}。材质: {product_model.elem}。"
                        f"描述: {product_model.className}"
                    )

                    # 封装为 Document，关键在于 metadata
                    doc = Document(
                        page_content=semantic_text,
                        metadata={
                            "product_code": unique_code,  # <--- 核心：这就是外键！
                            "group_id": group_id,
                            "series": product_model.series or "",  # 存入这些字段支持 Qdrant 的过滤(Filter)
                            "season": product_model.season_marking or ""
                        }
                    )

                    # 异步写入向量库 (LangChain 的 aadd_documents)
                    await vector_store.aadd_documents([doc])

                    print(f"[完成] {unique_code} (Graph + Vector Synced)")

                except Exception as e:
                    print(f"Error: {e}")
        # 创建并执行所有任务
        tasks = [
            _process_single_product(i, p)
            for i, p in enumerate(products_data)
        ]

        # 等待所有并发任务完成
        await asyncio.gather(*tasks)
        print("✅ 所有产品数据导入流程结束")

    @staticmethod
    async def clear_graph_data():
        """
        清空图谱数据的工具方法 (慎用)
        """
        from graphiti_core.utils.maintenance.graph_data_operations import clear_data
        print("⚠️ 正在清空图数据库...")
        await clear_data(graphiti_app.driver)
        print("✅ 图数据库已清空")