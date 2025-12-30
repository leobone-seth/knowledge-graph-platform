from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from backend.core.graph_client import (
    langchain_llm,
    vector_store,
    NEO4J_URI,
    NEO4J_USER,
    NEO4J_PWD,
)


class RAGEngine:
    def __init__(self, llm=None, embeddings=None):
        self.llm = llm or langchain_llm

        self.graph = Neo4jGraph(
            url=NEO4J_URI,
            username=NEO4J_USER,
            password=NEO4J_PWD
        )

        self.vector_store = vector_store
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})

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

    def query(self, user_input: str):
        docs = self.retriever.invoke(user_input)
        context = "\n\n".join(d.page_content for d in docs) if docs else ""
        chain = self.qa_prompt | self.llm | self.output_parser
        return chain.invoke({"context": context, "question": user_input})
