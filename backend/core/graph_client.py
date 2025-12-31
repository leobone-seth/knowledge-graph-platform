import os
import builtins

if not hasattr(builtins, "AsyncOpenSearch"):
    class AsyncOpenSearch:
        pass

    builtins.AsyncOpenSearch = AsyncOpenSearch

from graphiti_core import Graphiti
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from backend.core.llm_client import SiliconFlowGenericClient, llm_config

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USERNAME")
NEO4J_PWD = os.getenv("NEO4J_PASSWORD")

_raw_qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
if _raw_qdrant_url and not _raw_qdrant_url.startswith("http"):
    _raw_qdrant_url = "http://" + _raw_qdrant_url
QDRANT_URL = _raw_qdrant_url
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-8B")

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

langchain_llm = ChatOpenAI(
    model=os.getenv("OPENAI_API_MODEL"),
    openai_api_base=os.getenv("OPENAI_API_BASE"),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0
)

_qdrant_client = QdrantClient(url=QDRANT_URL)

try:
    _qdrant_client.get_collection("multimodal_knowledge")
except Exception:
    try:
        _qdrant_client.create_collection(
            collection_name="multimodal_knowledge",
            vectors_config=VectorParams(size=4096, distance=Distance.COSINE),
        )
    except Exception as e:
        print(f"Qdrant init error: {e}")

langchain_embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base=os.getenv("OPENAI_API_BASE"),
    check_embedding_ctx_length=False
)

vector_store = Qdrant(
    client=_qdrant_client,
    collection_name="multimodal_knowledge",
    embeddings=langchain_embeddings,
)
