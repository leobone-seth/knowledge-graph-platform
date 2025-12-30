import os
from graphiti_core import Graphiti
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient

# LangChain 相关引用
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from qdrant_client import QdrantClient

# 引用上面的 LLM Client
from backend.core.llm_client import SiliconFlowGenericClient, llm_config

# 1. 环境变量读取
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USERNAME")
NEO4J_PWD = os.getenv("NEO4J_PASSWORD")
QDRANT_URL = os.getenv("QDRANT_URL")
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-8B")

# ==========================================
# A. 初始化 Graphiti (用于入库写图)
# ==========================================
embed_config = OpenAIEmbedderConfig(
    api_key=os.getenv("OPENAI_API_KEY"),
    embedding_model=EMBEDDING_MODEL,
    embedding_dim=4096,
    base_url=os.getenv("OPENAI_API_BASE"),
)
graphiti_embedder = OpenAIEmbedder(embed_config)
graphiti_llm_client = SiliconFlowGenericClient(config=llm_config)
cross_encoder = OpenAIRerankerClient(client=graphiti_llm_client.client, config=llm_config)

graphiti_app = Graphiti(
    uri=NEO4J_URI,
    user=NEO4J_USER,
    password=NEO4J_PWD,
    llm_client=graphiti_llm_client,
    embedder=graphiti_embedder,
    cross_encoder=cross_encoder,
)

# ==========================================
# B. 初始化 LangChain 组件 (用于查询)
# ==========================================

# 1. 标准 Chat LLM (用于 ChatService)
langchain_llm = ChatOpenAI(
    model=os.getenv("OPENAI_API_MODEL"),
    openai_api_base=os.getenv("OPENAI_API_BASE"),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0
)

# 2. Vector Store (Qdrant)
_qdrant_client = QdrantClient(url=QDRANT_URL)
langchain_embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base=os.getenv("OPENAI_API_BASE"),
    check_embedding_ctx_length=False
)

vector_store = Qdrant(
    client=_qdrant_client,
    collection_name="multimodal_knowledge", # 集合名称
    embeddings=langchain_embeddings,
)