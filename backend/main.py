import asyncio
import json

from dotenv import load_dotenv

from backend.models.product import Product

# 优先加载环境变量
load_dotenv()
from backend.models.standard_document import StandardDocument # 引入你已有的模型
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from core.graphiti_adapter import GraphitiAdapter
from core.vector_store import VectorManager
from core.rag_engine import RAGEngine
from backend.core.graph_client import QDRANT_URL, graphiti_app
from backend.services.entity_service import GenericEntityService
from backend.services.chat_service import ChatService
from backend.entity_registry import ENTITY_SPECS, RULE_LINK_PRESETS
import requests
app = FastAPI(title="Multimodal KG Platform")

graph_adapter = GraphitiAdapter()
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



class EntityDeleteRequest(BaseModel):
    label: str
    entity_id: str
    id_field: str = "code"


class CypherQueryRequest(BaseModel):
    cypher: str
    params: Dict[str, Any] | None = None


class RuleLinkRequest(BaseModel):
    source_label: str
    source_id_field: str
    target_label: str
    target_id_field: str
    rel_type: str = "RELATED_TO"
    source_list_fields: List[str] = []
    source_text_fields: List[str] = []
    target_text_fields: List[str] = []
    mode: str = "contains_any"


class GenericIngestBody(BaseModel):
    items: List[Dict[str, Any]]
    group_id: str = "default"
    concurrency: int = 5
    auto_link: bool = True
    score_threshold: float = 0.3


def _json_safe(data: Any) -> Any:
    return json.loads(json.dumps(data, ensure_ascii=False, default=str))


async def _ingest_by_entity_type(
        entity_type: str,
        items: List[Dict[str, Any]],
        group_id: str = "default",
        concurrency: int = 5,
        auto_link: bool = True,
        score_threshold: float = 0.3,
):
    spec = ENTITY_SPECS.get(entity_type)
    if not spec:
        raise HTTPException(status_code=400, detail=f"Unknown entity_type: {entity_type}")

    received_count = len(items)
    parsed_count = received_count
    skipped_parse_invalid = 0

    if entity_type == "StandardDocument":
        cleaned_data = []
        for item in items:
            try:
                doc_obj = StandardDocument.from_chinese_json(item)
                cleaned_data.append(doc_obj.model_dump(exclude_none=True))
            except Exception as e:
                print(f"Skipping invalid standard: {e}")
                skipped_parse_invalid += 1
        if not cleaned_data:
            return {"status": "error", "message": "No valid data parsed"}
        data_list = cleaned_data
        parsed_count = len(cleaned_data)
    else:
        data_list = items

    skipped_missing_id = 0
    validated_by_id: List[Dict[str, Any]] = []
    for item in data_list:
        uid = item.get(spec.id_field)
        if uid is None or (isinstance(uid, str) and not uid.strip()):
            skipped_missing_id += 1
            continue
        validated_by_id.append(item)

    skipped_unmatched = 0
    validated_data = validated_by_id

    if spec.ingest_match_preset:
        semaphore = asyncio.Semaphore(max(1, min(concurrency, 10)))
        kept: List[Dict[str, Any]] = []

        async def _check_and_keep(it: Dict[str, Any]):
            nonlocal skipped_unmatched
            async with semaphore:
                try:
                    matched_count = await GenericEntityService.count_rule_matches_for_item(
                        spec.ingest_match_preset,
                        it,
                    )
                    if matched_count > 0:
                        kept.append(it)
                    else:
                        skipped_unmatched += 1
                except Exception as e:
                    print(f"Skipping unmatched item due to match check error: {e}")
                    skipped_unmatched += 1

        await asyncio.gather(*[_check_and_keep(it) for it in validated_by_id])
        validated_data = kept

    if not validated_data:
        return {
            "status": "success",
            "entity_type": entity_type,
            "received": received_count,
            "parsed": parsed_count,
            "ingested": 0,
            "skipped": {
                "parse_invalid": skipped_parse_invalid,
                "missing_id": skipped_missing_id,
                "unmatched": skipped_unmatched,
            },
        }

    await GenericEntityService.ingest_entities(
        data_list=validated_data,
        label=spec.label,
        id_field=spec.id_field,
        vector_template=spec.vector_template,
        graph_exclude_fields=spec.graph_exclude_fields,
        group_id=group_id,
        concurrency=concurrency,
    )

    if auto_link and entity_type == "StandardDocument":
        asyncio.create_task(
            GenericEntityService.link_entities_by_semantic(
                source_label="StandardDocument",
                source_id_field=ENTITY_SPECS["StandardDocument"].id_field,
                target_label="Product",
                target_id_field=ENTITY_SPECS["Product"].id_field,
                score_threshold=score_threshold
            )
        )
        return {
            "status": "success",
            "entity_type": entity_type,
            "message": "Entities ingested, linking task started in background.",
            "received": received_count,
            "parsed": parsed_count,
            "ingested": len(validated_data),
            "skipped": {
                "parse_invalid": skipped_parse_invalid,
                "missing_id": skipped_missing_id,
                "unmatched": skipped_unmatched,
            },
        }

    return {
        "status": "success",
        "entity_type": entity_type,
        "received": received_count,
        "parsed": parsed_count,
        "ingested": len(validated_data),
        "skipped": {
            "parse_invalid": skipped_parse_invalid,
            "missing_id": skipped_missing_id,
            "unmatched": skipped_unmatched,
        },
    }



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

@app.get("/api/entities/types")
async def list_entity_types():
    return {"entity_types": sorted(list(ENTITY_SPECS.keys()))}


@app.post("/api/entities/{entity_type}/ingest")
async def ingest_entity_generic(entity_type: str, body: GenericIngestBody):
    return await _ingest_by_entity_type(
        entity_type=entity_type,
        items=body.items,
        group_id=body.group_id,
        concurrency=body.concurrency,
        auto_link=body.auto_link,
        score_threshold=body.score_threshold,
    )


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
        await _ingest_by_entity_type(
            entity_type="Product",
            items=data_list,
            group_id="default",
            concurrency=5,
            auto_link=False,
        )

        return {
            "status": "success",
            "message": f"Successfully ingested {len(products)} products",
            "sample_id": products[0].code if products else None
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
# 修改文件: backend/main.py



# ...

# ----------------------------------------------------
# 场景 4: 写入 StandardDocument (标准文档) 并自动关联
# ----------------------------------------------------
@app.post("/api/standards/ingest")
async def ingest_standards(standards: List[Dict[str, Any]]):
    """
    摄入标准文档，并自动与 Product 建立关联
    """
    return await _ingest_by_entity_type(
        entity_type="StandardDocument",
        items=standards,
        group_id="default",
        concurrency=5,
        auto_link=True,
        score_threshold=0.3,
    )
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


@app.post("/api/del/entities/delete")
async def delete_entity_by_id(req: EntityDeleteRequest):
    try:
        deleted = await GenericEntityService.delete_entity(
            label=req.label,
            id_field=req.id_field,
            unique_id=req.entity_id,
        )
        if deleted == 0:
            raise HTTPException(status_code=404, detail="Entity not found")
        return {"status": "success", "deleted": deleted}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 5. RAG Chat -> POST (保持不变)
@app.post("/api/chat")
async def chat(request: ChatRequest):
    result = await ChatService.run_deep_accuracy_chat(request.question)
    return result


@app.post("/api/chat/simple")
async def chat_simple(request: ChatRequest):
    answer = rag_engine.query(request.question)
    return {"answer": answer}


@app.post("/api/chat/deep")
async def chat_deep(request: ChatRequest):
    result = await ChatService.run_deep_accuracy_chat(request.question)
    return result


@app.post("/api/graph/query")
async def run_cypher_query(req: CypherQueryRequest):
    try:
        result = graph_adapter.run_cypher(req.cypher, req.params or {})
        return {"data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/linking/presets")
async def list_linking_presets():
    return {"presets": list(RULE_LINK_PRESETS.keys())}


@app.post("/api/linking/presets/{preset_name}/run")
async def run_linking_preset(preset_name: str):
    try:
        return await GenericEntityService.run_rule_preset(preset_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/linking/rules/run")
async def run_linking_rules(req: RuleLinkRequest):
    try:
        edges = await GenericEntityService.link_entities_by_rules(
            source_label=req.source_label,
            source_id_field=req.source_id_field,
            target_label=req.target_label,
            target_id_field=req.target_id_field,
            source_list_fields=req.source_list_fields,
            source_text_fields=req.source_text_fields,
            target_text_fields=req.target_text_fields,
            rel_type=req.rel_type,
            mode=req.mode,
        )
        return {"status": "success", "edges_created": edges}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/verify/neo4j/relationships")
async def verify_neo4j_relationships(
        source_label: str,
        target_label: str,
        rel_type: str,
        limit: int = 10,
):
    try:
        src_lbl = GenericEntityService._validate_cypher_identifier(source_label)
        tgt_lbl = GenericEntityService._validate_cypher_identifier(target_label)
        rel = GenericEntityService._validate_cypher_identifier(rel_type)
        limit = max(0, min(int(limit), 50))

        count_cypher = f"""
        MATCH (s:{src_lbl})-[r:{rel}]->(t:{tgt_lbl})
        RETURN count(r) as edge_count
        """
        sample_cypher = f"""
        MATCH (s:{src_lbl})-[r:{rel}]->(t:{tgt_lbl})
        RETURN properties(s) as source, properties(t) as target
        LIMIT $limit
        """

        async with graphiti_app.driver.session() as session:
            count_res = await session.run(count_cypher)
            count_rec = await count_res.single()
            edge_count = int(count_rec["edge_count"]) if count_rec else 0

            sample_res = await session.run(sample_cypher, limit=limit)
            samples = await sample_res.data()

        return {"edge_count": edge_count, "samples": _json_safe(samples)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/verify/qdrant/points/count")
async def verify_qdrant_points_count(entity_label: str, group_id: str | None = None):
    try:
        must = [{"key": "metadata.entity_label", "match": {"value": entity_label}}]
        if group_id:
            must.append({"key": "metadata.group_id", "match": {"value": group_id}})

        resp = requests.post(
            f"{QDRANT_URL}/collections/multimodal_knowledge/points/count",
            json={"filter": {"must": must}, "exact": True},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return {"count": data.get("result", {}).get("count", 0), "note": "Qdrant仅保存向量点，不保存Neo4j的边"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8077)
