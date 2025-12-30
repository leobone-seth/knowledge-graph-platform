import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_community.vectorstores import Qdrant
from langchain.agents import initialize_agent, Tool
from langchain.chains import RetrievalQA


class RAGEngine:
    def __init__(self):
        # --- 关键修改：适配你的 .env 配置 ---
        self.llm = ChatOpenAI(
            model=os.getenv("OPENAI_API_MODEL", "gpt-4"),  # 读取 Qwen 模型名
            openai_api_base=os.getenv("OPENAI_API_BASE"),  # 读取 SiliconFlow 地址
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0
        )

        # 这里的 Embedding 如果也需要用 SiliconFlow，可能需要类似配置
        # 假设仍使用 OpenAI 官方或兼容接口
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_api_base=os.getenv("OPENAI_API_BASE")
        )

        self.graph = Neo4jGraph(
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD")
        )

        self.vector_store = Qdrant.from_existing_collection(
            embedding=self.embeddings,
            collection_name="multimodal_knowledge",
            url=os.getenv("QDRANT_URL")
        )

    def get_agent(self):
        # 保持原有逻辑不变
        cypher_chain = GraphCypherQAChain.from_llm(
            llm=self.llm,
            graph=self.graph,
            verbose=True,
            allow_dangerous_requests=True
        )

        vector_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 3})
        )

        tools = [
            Tool(
                name="GraphDatabase",
                func=cypher_chain.run,
                description="Use this for questions about relationships, counts, status logic."
            ),
            Tool(
                name="VisualDescriptions",
                func=vector_chain.run,
                description="Use this for questions about appearance, colors, visual summary."
            )
        ]

        return initialize_agent(tools, self.llm, agent="zero-shot-react-description", verbose=True)

    def query(self, user_input: str):
        agent = self.get_agent()
        return agent.run(user_input)