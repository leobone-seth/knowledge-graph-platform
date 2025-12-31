import asyncio
import json
import os
from typing import Dict, Any

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain

from backend.core.graph_client import langchain_llm
from backend.services.search_service import SearchService


class ChatService:
    """
    高精度问答服务：并行执行向量检索和图谱统计，由 LLM 交叉验证
    """

    # 初始化一个独立的 Neo4jGraph 用于 LangChain Chain
    _graph = Neo4jGraph(
        url=os.getenv("NEO4J_URI"),
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD")
    )

    # 定义 Cypher 生成规则
    _cypher_prompt = PromptTemplate(
        template="""
        Task: Generate Cypher query for Neo4j.
        Schema: {schema}
        Instructions:
        1. Nodes: `Product`, `ProductAttr`, `StandardDocument`.
        2. Edge: 
           - (Product)-[:HAS_ATTR]->(ProductAttr)
           - (StandardDocument)-[:APPLIES_TO]->(Product) 
        3. Do NOT query `Observation` or `Episode`.
        4. For aggregation/counting, use count().
        5. For string matching, use CONTAINS.

        Question: {question}
        """,
        input_variables=["schema", "question"]
    )

    _cypher_chain = GraphCypherQAChain.from_llm(
        llm=langchain_llm,
        graph=_graph,
        cypher_prompt=_cypher_prompt,
        verbose=True,
        return_direct=True,  # 直接返回数据结果，不让 Chain 自动回答
        allow_dangerous_requests=True
    )

    @staticmethod
    async def run_deep_accuracy_chat(question: str) -> Dict[str, Any]:
        print(f"🧠 [DeepMode] 启动高精度双路检索: {question}")

        # --- 并行执行：路A (向量) + 路B (统计) ---
        task_vector = SearchService.hybrid_search(question, limit=5)
        task_graph = ChatService._safe_cypher_run(question)

        # 等待结果
        vector_res, graph_res = await asyncio.gather(task_vector, task_graph)

        # --- 结果融合 ---
        context_str = "【来源1：语义检索 (Qdrant)】\n"
        if vector_res.get("results"):
            for item in vector_res["results"]:
                info = {
                    "产品编码": item["code"],
                    "属性": item["graph_data"],
                    "描述": item["semantic_text"][:100]
                }
                context_str += f"- {json.dumps(info, ensure_ascii=False, default=str)}\n"
        else:
            context_str += "(无相关结果)\n"

        context_str += "\n【来源2：全库统计 (Neo4j Cypher)】\n"
        if graph_res:
            context_str += f"{json.dumps(graph_res, ensure_ascii=False, default=str)}\n"
        else:
            context_str += "(无统计数据)\n"

        # --- 最终推理 ---
        final_answer = await ChatService._synthesize_answer(question, context_str)

        return {
            "answer": final_answer,
            "sources": {
                "vector": [x["code"] for x in vector_res.get("results", [])],
                "graph_query": str(graph_res)[:200]
            }
        }

    @staticmethod
    async def _safe_cypher_run(question: str):
        """安全运行 Cypher，失败则返回 None"""
        try:
            return await ChatService._cypher_chain.ainvoke({"query": question})
        except Exception as e:
            print(f"⚠️ Cypher 失败 (非统计类问题可忽略): {e}")
            return None

    @staticmethod
    async def _synthesize_answer(question: str, context: str) -> str:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个严谨的数据分析师。请根据提供的两份数据来源回答用户问题。\n"
                       "规则：\n"
                       "1. 如果是统计问题(多少个/总共)，优先信任【来源2】。\n"
                       "2. 如果是详情或推荐问题，优先信任【来源1】。\n"
                       "3. 如果都没有数据，请直接说不知道。"),
            ("human", "证据数据:\n{context}\n\n问题: {question}")
        ])
        chain = prompt | langchain_llm | StrOutputParser()
        return await chain.ainvoke({"context": context, "question": question})
