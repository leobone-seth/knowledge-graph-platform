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
from core.rag_engine import RAGEngine
from backend.core.graph_client import QDRANT_URL, graphiti_app
from backend.services.entity_service import GenericEntityService
from backend.services.chat_service import ChatService
from backend.entity_registry import ENTITY_SPECS, RULE_LINK_PRESETS
import requests
import os
import logging
import sys
import time
app = FastAPI(title="Multimodal KG Platform")

logger = logging.getLogger("uvicorn.error")

graph_adapter = GraphitiAdapter()
rag_engine = RAGEngine()


# --- 请求模型 ---
class ChatRequest(BaseModel):
    question: str



class EntityDeleteRequest(BaseModel):
    label: str
    entity_id: str
    id_field: str = "code"


class CypherQueryRequest(BaseModel):
    cypher: str
    params: Dict[str, Any] | None = None


class ClearQdrantRequest(BaseModel):
    collection_name: str = "multimodal_knowledge"


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
    debug_match: bool = False
    allow_unmatched: bool = False


def _json_safe(data: Any) -> Any:
    return json.loads(json.dumps(data, ensure_ascii=False, default=str))


async def _ingest_by_entity_type(
        entity_type: str,
        items: List[Dict[str, Any]],
        group_id: str = "default",
        concurrency: int = 5,
        auto_link: bool = True,
        score_threshold: float = 0.3,
        debug_match: bool = False,
        allow_unmatched: bool = False,
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

    if spec.ingest_match_preset and not allow_unmatched:
        semaphore = asyncio.Semaphore(max(1, min(concurrency, 10)))
        kept: List[Dict[str, Any]] = []

        debug_match = debug_match or (
            os.getenv("DEBUG_INGEST_MATCH", "").strip().lower() in {"1", "true", "yes", "on"}
        )
        preset = RULE_LINK_PRESETS.get(spec.ingest_match_preset) or {}
        debug_fields = {
            "id_field": spec.id_field,
            "id": None,
            "preset": spec.ingest_match_preset,
            "source_list_fields": preset.get("source_list_fields") or [],
            "source_text_fields": preset.get("source_text_fields") or [],
            "target_label": preset.get("target_label"),
            "target_text_fields": preset.get("target_text_fields") or [],
            "mode": preset.get("mode"),
        }

        async def _check_and_keep(it: Dict[str, Any]):
            nonlocal skipped_unmatched
            async with semaphore:
                try:
                    matched_count = await GenericEntityService.count_rule_matches_for_item(
                        spec.ingest_match_preset,
                        it,
                    )
                    uid = it.get(spec.id_field)
                    source_list_payload = {k: it.get(k) for k in (debug_fields["source_list_fields"] or [])}
                    source_text_payload = {k: it.get(k) for k in (debug_fields["source_text_fields"] or [])}

                    if debug_match:
                        payload = {
                            "id_field": debug_fields["id_field"],
                            "id": uid,
                            "preset": debug_fields["preset"],
                            "mode": debug_fields["mode"],
                            "source_list_fields": source_list_payload,
                            "source_text_fields": source_text_payload,
                            "target": {
                                "label": debug_fields["target_label"],
                                "text_fields": debug_fields["target_text_fields"],
                            },
                            "matched_count": matched_count,
                        }
                        logger.info("[DEBUG_INGEST_MATCH] %s", json.dumps(_json_safe(payload), ensure_ascii=False))

                    if matched_count > 0:
                        kept.append(it)
                    else:
                        skipped_unmatched += 1
                        payload = {
                            "entity_type": entity_type,
                            "id_field": debug_fields["id_field"],
                            "id": uid,
                            "preset": debug_fields["preset"],
                            "mode": debug_fields["mode"],
                            "source_list_fields": source_list_payload,
                            "source_text_fields": source_text_payload,
                            "matched_count": matched_count,
                        }
                        logger.warning("[INGEST_UNMATCHED] %s", json.dumps(_json_safe(payload), ensure_ascii=False))
                except Exception as e:
                    logger.exception("[INGEST_MATCH_ERROR] %s", str(e))
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

    ingest_summary = await GenericEntityService.ingest_entities(
        data_list=validated_data,
        label=spec.label,
        id_field=spec.id_field,
        vector_template=spec.vector_template,
        graph_exclude_fields=spec.graph_exclude_fields,
        group_id=group_id,
        concurrency=concurrency,
    )

    if auto_link and entity_type == "StandardDocument":
        try:
            neo4j_written = int((ingest_summary or {}).get("neo4j_written") or 0)
        except Exception:
            neo4j_written = 0
        if neo4j_written <= 0:
            print(
                "[STD_LINK_SKIP] reason=neo4j_write_failed_or_zero",
                "ingested=", len(validated_data),
                "neo4j_written=", neo4j_written,
                flush=True,
            )
            return {
                "status": "success",
                "entity_type": entity_type,
                "message": "Entities ingested, linking skipped because Neo4j write failed.",
                "received": received_count,
                "parsed": parsed_count,
                "ingested": len(validated_data),
                "ingest_summary": _json_safe(ingest_summary),
                "skipped": {
                    "parse_invalid": skipped_parse_invalid,
                    "missing_id": skipped_missing_id,
                    "unmatched": skipped_unmatched,
                },
            }
        preset_name = ENTITY_SPECS["StandardDocument"].ingest_match_preset
        if preset_name:
            ids = [it.get(ENTITY_SPECS["StandardDocument"].id_field) for it in validated_data]
            ids = [i for i in ids if i]
            if ids:
                asyncio.create_task(
                    GenericEntityService.link_rule_preset_for_sources(
                        preset_name=preset_name,
                        source_ids=ids,
                        sample_limit=0,
                    )
                )
        return {
            "status": "success",
            "entity_type": entity_type,
            "message": "Entities ingested, linking task started in background.",
            "received": received_count,
            "parsed": parsed_count,
            "ingested": len(validated_data),
            "ingest_summary": _json_safe(ingest_summary),
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
        "ingest_summary": _json_safe(ingest_summary),
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
    """
    返回当前服务中注册的实体类型列表，便于前端做下拉选择
    """
    return {"entity_types": sorted(list(ENTITY_SPECS.keys()))}


@app.post("/api/entities/{entity_type}/ingest")
async def ingest_entity_generic(entity_type: str, body: GenericIngestBody):
    """
    通用实体写入入口，根据 entity_type 自动路由到对应的配置
    """
    force_standard_write = entity_type == "StandardDocument" and (
        os.getenv("FORCE_STANDARD_WRITE", "1").strip().lower() in {"1", "true", "yes", "on"}
    )
    return await _ingest_by_entity_type(
        entity_type=entity_type,
        items=body.items,
        group_id=body.group_id,
        concurrency=body.concurrency,
        auto_link=body.auto_link,
        score_threshold=body.score_threshold,
        debug_match=body.debug_match,
        allow_unmatched=body.allow_unmatched or force_standard_write,
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
        allow_unmatched=True,
    )


@app.get("/api/debug/standards/count")
async def debug_standards_count():
    """
    调试用：统计标准文档节点总数，验证写入是否成功
    """
    cypher = """
    MATCH (s:StandardDocument)
    RETURN count(s) as cnt
    """
    async with graphiti_app.driver.session() as session:
        result = await session.run(cypher)
        record = await result.single()
        cnt_val = record.get("cnt") if record else None
        cnt = int(cnt_val) if cnt_val is not None else 0
    return {"label": "StandardDocument", "count": cnt}


@app.get("/api/debug/standards/links")
async def debug_standard_links(standard_code: str | None = None, limit: int = 20, recent: int = 20):
    """
    调试用：查看某个标准与 Product 的关联情况，或最近更新的若干标准
    """
    sc = (standard_code or "").strip()
    limit = max(0, min(int(limit), 100))
    recent = max(0, min(int(recent), 200))

    if not sc:
        list_cypher = """
        MATCH (s:StandardDocument)
        OPTIONAL MATCH (s)-[:APPLIES_TO]->(p:Product)
        WITH s, count(p) as edge_count
        ORDER BY s.last_updated DESC
        RETURN s.standard_code as standard_code, edge_count as edge_count
        LIMIT $recent
        """
        async with graphiti_app.driver.session() as session:
            res = await session.run(list_cypher, recent=recent)
            rows = await res.data()
            await res.consume()
        return {
            "status": "missing_standard_code",
            "usage": "/api/debug/standards/links?standard_code=FZ/T%2073018-2021&limit=50",
            "recent": _json_safe(rows),
        }

    count_cypher = """
    MATCH (s:StandardDocument {standard_code: $standard_code})-[:APPLIES_TO]->(p:Product)
    RETURN count(p) as cnt
    """
    sample_cypher = """
    MATCH (s:StandardDocument {standard_code: $standard_code})-[:APPLIES_TO]->(p:Product)
    RETURN p.code as code, p.series as series, p.className as className, p.elem as elem, p.fun as fun
    LIMIT $limit
    """
    async with graphiti_app.driver.session() as session:
        count_res = await session.run(count_cypher, standard_code=sc)
        count_rec = await count_res.single()
        cnt_val = count_rec.get("cnt") if count_rec else None
        cnt = int(cnt_val) if cnt_val is not None else 0
        await count_res.consume()

        sample_res = await session.run(sample_cypher, standard_code=sc, limit=limit)
        rows = await sample_res.data()
        await sample_res.consume()

    return {"standard_code": sc, "edge_count": cnt, "products": _json_safe(rows)}


@app.get("/api/debug/products/match_count")
async def debug_products_match_count(keyword: str = "羊毛"):
    """
    调试用：根据关键字统计命中的 Product 数量，用于观察匹配范围
    """
    kw = (keyword or "").strip()
    cypher = """
    MATCH (p:Product)
    WHERE
        coalesce(toString(p.elem), '') CONTAINS $keyword
        OR coalesce(toString(p.fun), '') CONTAINS $keyword
        OR coalesce(toString(p.className), '') CONTAINS $keyword
        OR coalesce(toString(p.series), '') CONTAINS $keyword
    RETURN count(p) as matched_count
    """
    async with graphiti_app.driver.session() as session:
        result = await session.run(cypher, keyword=kw)
        record = await result.single()
        matched_val = record.get("matched_count") if record else None
        matched_count = int(matched_val) if matched_val is not None else 0
    logger.info("[DEBUG_PRODUCTS_MATCH_COUNT] keyword=%s matched_count=%s", kw, matched_count)
    return {"keyword": kw, "matched_count": matched_count, "cypher": cypher.strip()}


@app.get("/api/debug/neo4j/env")
async def debug_neo4j_env():
    """
    调试用：检查 Neo4j 连接状态、当前数据库与用户等基础信息
    """
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME")
    connectivity: Dict[str, Any] = {"ok": False}
    server: Dict[str, Any] | None = None
    current_user: Dict[str, Any] | None = None
    current_db: Dict[str, Any] | None = None

    try:
        async with graphiti_app.driver.session() as session:
            ping_res = await session.run("RETURN 1 as ok")
            await ping_res.consume()
            connectivity = {"ok": True}

            try:
                db_res = await session.run("SHOW CURRENT DATABASE YIELD name RETURN name")
                db_rec = await db_res.single()
                await db_res.consume()
                if db_rec and "name" in db_rec:
                    current_db = {"name": db_rec["name"]}
            except Exception:
                current_db = None

            comp_res = await session.run(
                "CALL dbms.components() YIELD name, versions, edition RETURN name, versions[0] as version, edition"
            )
            comp = await comp_res.single()
            if comp:
                server = {
                    "name": comp.get("name"),
                    "version": comp.get("version"),
                    "edition": comp.get("edition"),
                }

            try:
                user_res = await session.run("SHOW CURRENT USER YIELD user RETURN user")
                user_rec = await user_res.single()
                if user_rec and "user" in user_rec:
                    current_user = {"user": user_rec["user"]}
            except Exception:
                current_user = None
    except Exception as e:
        connectivity = {"ok": False, "error": str(e)}

    return {
        "neo4j_uri": uri,
        "neo4j_username": user,
        "connectivity": connectivity,
        "current_database": current_db,
        "server": server,
        "current_user": current_user,
    }


@app.get("/api/debug/runtime")
async def debug_runtime():
    """
    调试用：查看后端运行环境、关键文件的更新时间等信息
    """
    now = time.time()
    def _stat(path: str) -> Dict[str, Any] | None:
        try:
            st = os.stat(path)
            return {"mtime": st.st_mtime, "age_sec": round(now - st.st_mtime, 3)}
        except Exception:
            return None

    root = os.getcwd()
    main_path = os.path.abspath(__file__)
    svc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "services", "entity_service.py"))

    return {
        "pid": os.getpid(),
        "python": sys.version,
        "cwd": root,
        "files": {
            "main.py": {"path": main_path, "stat": _stat(main_path)},
            "entity_service.py": {"path": svc_path, "stat": _stat(svc_path)},
        },
        "env": {
            "NEO4J_URI": os.getenv("NEO4J_URI"),
            "NEO4J_USERNAME": os.getenv("NEO4J_USERNAME"),
            "FORCE_STANDARD_WRITE": os.getenv("FORCE_STANDARD_WRITE", "1"),
        },
    }
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


@app.post("/api/del/entities/delete")
async def delete_entity_by_id(req: EntityDeleteRequest):
    """
    根据 label + 主键字段 + 主键值 删除单个实体节点
    """
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
    """
    默认高精度聊天模式，内部走混合检索 + 复杂推理链路
    """
    result = await ChatService.run_deep_accuracy_chat(request.question)
    return result


@app.post("/api/chat/simple")
async def chat_simple(request: ChatRequest):
    """
    简单聊天模式，直接调用向量检索 + 轻量回答
    """
    answer = rag_engine.query(request.question)
    return {"answer": answer}


@app.post("/api/chat/deep")
async def chat_deep(request: ChatRequest):
    """
    显式调用深度聊天模式，与 /api/chat 行为一致
    """
    result = await ChatService.run_deep_accuracy_chat(request.question)
    return result


@app.post("/api/graph/query")
async def run_cypher_query(req: CypherQueryRequest):
    """
    运行原始 Cypher 查询（谨慎开放，通常只给内部调试使用）
    """
    try:
        result = graph_adapter.run_cypher(req.cypher, req.params or {})
        return {"data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/linking/presets")
async def list_linking_presets():
    """
    列出可用的规则联动预设名称，前端可以用来做下拉选择
    """
    return {"presets": list(RULE_LINK_PRESETS.keys())}


@app.post("/api/linking/presets/{preset_name}/run")
async def run_linking_preset(preset_name: str):
    """
    按预设名称运行整库规则联动任务（通常是批量离线任务）
    """
    try:
        return await GenericEntityService.run_rule_preset(preset_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/linking/presets/{preset_name}/run_one")
async def run_linking_preset_for_one(
        preset_name: str,
        source_id: str,
        sample_limit: int = 10,
        keyword: str | None = None,
):
    """
    对单个 source 节点运行规则联动，可选传入 keyword 做强化匹配
    """
    try:
        return await GenericEntityService.link_rule_preset_for_source(
            preset_name=preset_name,
            source_id=source_id,
            sample_limit=sample_limit,
            keyword=keyword,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/linking/presets/{preset_name}/repair_one")
async def repair_linking_preset_for_one(preset_name: str, source_id: str, sample_limit: int = 10):
    """
    修复单个 source 的联动结果，常用于某条数据修正后的重跑
    """
    try:
        return await GenericEntityService.repair_rule_preset_for_source(
            preset_name=preset_name,
            source_id=source_id,
            sample_limit=sample_limit,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/linking/rules/run")
async def run_linking_rules(req: RuleLinkRequest):
    """
    自定义规则联动入口，前端可以动态指定源/目标/字段/关系类型
    """
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
    """
    校验 Neo4j 中某类关系的数量，并抽样返回若干边的两端节点
    """
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
            edge_val = count_rec.get("edge_count") if count_rec else None
            edge_count = int(edge_val) if edge_val is not None else 0

            sample_res = await session.run(sample_cypher, limit=limit)
            samples = await sample_res.data()

        return {"edge_count": edge_count, "samples": _json_safe(samples)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/verify/qdrant/points/count")
async def verify_qdrant_points_count(entity_label: str, group_id: str | None = None):
    """
    校验 Qdrant 中某实体类型（及可选 group_id）的向量点数量
    """
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


@app.post("/api/admin/qdrant/collections/clear")
async def clear_qdrant_collection(req: ClearQdrantRequest):
    """
    管理接口：清空指定 collection 中的所有向量点，慎用
    """
    try:
        resp = requests.post(
            f"{QDRANT_URL}/collections/{req.collection_name}/points/delete",
            json={"filter": {}},
            params={"wait": "true"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return {"status": "success", "result": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8077)
