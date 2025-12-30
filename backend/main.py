from dotenv import load_dotenv

# 优先加载环境变量
load_dotenv()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from core.graphiti_adapter import GraphitiAdapter
from core.vector_store import VectorManager
from core.rag_engine import RAGEngine

app = FastAPI(title="Multimodal KG Platform (GET/POST Only)")

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

    uvicorn.run(app, host="0.0.0.0", port=8000)
