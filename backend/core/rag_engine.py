import requests
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from backend.core.graph_client import (
    langchain_llm,
    NEO4J_URI,
    NEO4J_USER,
    NEO4J_PWD,
    QDRANT_URL,
    langchain_embeddings,
)


class RAGEngine:
    def __init__(self, llm=None, embeddings=None):
        self.llm = llm or langchain_llm

        self.graph = Neo4jGraph(
            url=NEO4J_URI,
            username=NEO4J_USER,
            password=NEO4J_PWD,
        )

        self.qa_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful assistant. Answer the question using only the given context.",
                ),
                ("human", "Context:\n{context}\n\nQuestion: {question}"),
            ]
        )
        self.output_parser = StrOutputParser()

    def _search_docs(self, query: str, k: int = 5):
        embed = langchain_embeddings.embed_query(query)
        resp = requests.post(
            f"{QDRANT_URL}/collections/multimodal_knowledge/points/search",
            json={
                "vector": embed,
                "limit": k,
                "with_payload": True,
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        points = data.get("result", [])
        docs = []
        for p in points:
            payload = p.get("payload") or {}
            page_content = payload.get("page_content", "")
            metadata = payload.get("metadata", {})
            docs.append(Document(page_content=page_content, metadata=metadata))
        return docs

    def query(self, user_input: str):
        docs = self._search_docs(user_input, k=15)
        context = "\n\n".join(d.page_content for d in docs) if docs else ""
        chain = self.qa_prompt | self.llm | self.output_parser
        return chain.invoke({"context": context, "question": user_input})
