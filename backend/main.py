from dotenv import load_dotenv

from backend.models.product import Product

# 优先加载环境变量
load_dotenv()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from core.graphiti_adapter import GraphitiAdapter
from core.vector_store import VectorManager
from core.rag_engine import RAGEngine
# backend/main.py 修改示例
from backend.services.generic_service import GenericEntityService
app = FastAPI(title="Multimodal KG Platform")

# 初始化服务
graph_adapter = GraphitiAdapter()
# 注意：vector_manager 和 rag_engine 初始化依赖环境变量
vector_manager = VectorManager()
rag_engine = RAGEngine()


# --- 请求模型 ---
class IngestRequest(BaseModel):
    image_id: str
    timestamp: str
    summary_text: str
    objects: List[Dict[str, Any]]


class ChatRequest(BaseModel):
    question: str


class UpdateStatusRequest(BaseModel):
    new_status: str
    timestamp: str


class DeleteRequest(BaseModel):
    image_id: str





# ... 其他 imports

# ----------------------------------------------------
# 场景 1: 写入 User (用户)
# ----------------------------------------------------
@app.post("/api/users/ingest")
async def ingest_users(users: List[Dict[str, Any]]):
    """
    User 写入示例
    """
    await GenericEntityService.ingest_entities(
        data_list=users,
        label="User",  # 存入 Neo4j 的 Label: :User
        id_field="username",  # 主键字段名
        vector_template="用户姓名: {name}, 职业: {job}, 兴趣: {hobby}"  # 向量化模版
    )
    return {"status": "success", "count": len(users)}


# ----------------------------------------------------
# 场景 2: 写入 Product (产品) - 替换原来的 ingest_product_batch
# ----------------------------------------------------
@app.post("/api/products/ingest")
async def ingest_products(products: List[Product]):
    """
    接收符合 Product 模型结构的数据，并调用通用服务入库
    """
    try:
        # 1. 数据转换：将 Pydantic 对象列表转换为 Dict 列表
        # exclude_none=True 可以去除空值，避免覆盖旧数据的某些字段（视业务需求而定）
        data_list = [p.model_dump(exclude_none=True) for p in products]

        # 2. 调用通用服务
        await GenericEntityService.ingest_entities(
            data_list=data_list,
            label="Product",  # 指定在 Neo4j 中的标签
            id_field="code",  # 指定 Product 模型中的哪个字段是唯一主键
            # 3. 定义向量化模版 (根据 Product 模型字段动态填充)
            vector_template="产品编码: {code}, 系列: {series}, 功能: {fun}, 材质: {elem}, 描述: {className}"
        )

        return {
            "status": "success",
            "message": f"Successfully ingested {len(products)} products",
            "sample_id": products[0].code if products else None
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ----------------------------------------------------
# 场景 3: 通用查询 (Search)
# ----------------------------------------------------
@app.get("/api/search/{entity_type}")
async def search_entity(entity_type: str, q: str):
    """
    通用搜索接口: /api/search/Product?q=... 或 /api/search/User?q=...
    """
    # entity_type 可以是 "Product" 或 "User"
    # 注意首字母大写需与 ingest 时的 label 保持一致
    result = await GenericEntityService.generic_search(
        query=q,
        target_label=entity_type
    )
    return result


# --- API 路由 (纯 GET/POST) ---

# 1. CREATE -> POST (保持不变)
@app.post("/api/ingest")
async def ingest_data(data: IngestRequest):
    """
    [新增数据]
    """
    try:
        graph_adapter.add_observation(
            image_id=data.image_id,
            timestamp=data.timestamp,
            text_summary=data.summary_text,
            entities=data.objects
        )
        try:
            vector_manager.add_text(
                text=data.summary_text,
                metadata={"image_id": data.image_id, "timestamp": data.timestamp}
            )
        except Exception as v_e:
            print(f"Vector store warning: {v_e}")

        return {"status": "success", "message": f"Ingested {data.image_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 2. READ -> GET (保持不变)
@app.get("/api/entities/{name}")
async def get_entity(name: str):
    """
    [查询实体]
    """
    data = graph_adapter.get_entity_details(name)
    if not data:
        raise HTTPException(status_code=404, detail="Entity not found")
    return {"data": data}


# 3. UPDATE -> POST (原 PUT)
# URL 变更为 /update 结尾
@app.post("/api/entities/{name}/update")
async def update_entity(name: str, req: UpdateStatusRequest):
    try:
        result = graph_adapter.update_entity_status(name, req.new_status, req.timestamp)
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 4. DELETE -> POST (原 DELETE)
# URL 变更为 /delete 结尾，参数改为从 Body 或 Query 传
@app.post("/api/observations/delete")
async def delete_observation(req: DeleteRequest):
    try:
        graph_adapter.delete_observation_chain(req.image_id)
        return {"status": "success", "message": f"Deleted observation chain for {req.image_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 5. RAG Chat -> POST (保持不变)
@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    [智能问答]
    """
    response = rag_engine.query(request.question)
    return {"answer": response}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8077)
