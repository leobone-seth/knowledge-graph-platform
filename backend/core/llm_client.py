import json
import builtins

if not hasattr(builtins, "AsyncOpenSearch"):
    class AsyncOpenSearch:  # 占位类型，避免旧版 graphiti_core 中的类型注解报错
        pass

    builtins.AsyncOpenSearch = AsyncOpenSearch

from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from graphiti_core.nodes import EpisodeType, EntityNode
from graphiti_core.edges import EntityEdge
from graphiti_core.utils.maintenance.graph_data_operations import clear_data

import os

# 读取环境变量 (建议使用 config.py 或 os.getenv)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.siliconflow.cn/v1")
OPENAI_API_MODEL = os.getenv("OPENAI_API_MODEL", "zai-org/GLM-4.7")

class SiliconFlowGenericClient(OpenAIGenericClient):
    """
    自定义适配器：处理 SiliconFlow 特定的返回格式或结构化输出
    """
    async def _create_structured_completion(
        self, model, messages, temperature, max_tokens, response_model,
        reasoning=None, verbosity=None
    ):
        # ... (保留你原代码中的 _create_structured_completion 逻辑) ...
        # 代码略，直接复制你提供的实现即可
        pass

# 初始化配置对象
llm_config = LLMConfig(
    api_key=OPENAI_API_KEY,
    model=OPENAI_API_MODEL,
    small_model=OPENAI_API_MODEL, # 假设 small model 一样
    base_url=OPENAI_API_BASE,
)
