from typing import Any, Dict, Optional
from langchain_core.documents import Document
from backend.core.graph_client import vector_store


class VectorManager:
    """
    向量存储管理器 (Vector Store Manager)

    职责：
    封装 main.py 中所需的向量入库操作。
    底层复用 backend.core.graph_client 中初始化的 Qdrant vector_store 实例。
    """

    def __init__(self):
        # 直接使用 graph_client.py 中初始化的全局 vector_store
        # 这样可以共享连接池配置和 Embedding 模型
        self.store = vector_store

    def add_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        添加单条文本到向量数据库 (同步方法)

        Usage in main.py:
            vector_manager.add_text(
                text=data.summary_text,
                metadata={"image_id": data.image_id, "timestamp": data.timestamp}
            )
        """
        if not text:
            print("⚠️ [VectorManager] Empty text received, skipping ingestion.")
            return

        if metadata is None:
            metadata = {}

        # 1. 构造 LangChain Document 对象
        doc = Document(
            page_content=text,
            metadata=metadata
        )

        try:
            # 2. 调用同步的 add_documents 方法
            # 注意：main.py 中该调用在 try-except 块中，且未 await，所以这里必须用同步方法
            self.store.add_documents([doc])

            image_id = metadata.get("image_id", "unknown")
            print(f"✅ [VectorManager] Successfully indexed observation for Image ID: {image_id}")

        except Exception as e:
            print(f"❌ [VectorManager] Failed to index document: {e}")
            # 抛出异常以便 main.py 中的 try-except 捕获并打印 warning
            raise e