# 多模态知识图谱平台实施方案

## 1. 项目概述 (Project Overview)
本项目旨在构建一个智能知识图谱基座，对接上游的**多模态识别系统**。系统能够接收识别出的结构化对象与非结构化描述，自动构建动态图谱，并支持基于 LLM 的混合检索问答（GraphRAG）。

## 2. 技术栈 (Tech Stack)

| 组件 | 技术选型 | 用途 |
| :--- | :--- | :--- |
| **语言** | Python 3.10+ | 核心开发语言 |
| **Web 框架** | FastAPI | 高性能 API 接口服务 |
| **图数据库** | **Neo4j** (v5.x) | 存储实体关系、拓扑结构 |
| **向量数据库** | **Qdrant** | 存储语义向量、支持模糊检索 |
| **动态图引擎** | **Graphiti** (Custom Adapter) | 处理时序数据、动态图构建 |
| **大模型框架** | **LangChain** | 编排 RAG 流程、Agent 路由 |
| **LLM** | GPT-4o / DeepSeek | 语义理解、Cypher 生成、推理 |

## 3. 系统架构 (Architecture)

### 3.1 数据流向 (Pipeline)
1.  **入库 (Write Path):** * 多模态模型 -> JSON 输出 -> Python Adapter
    * 分支 A: 结构化数据 (实体/状态) -> Graphiti -> Neo4j
    * 分支 B: 描述性文本 -> Embeddings -> Qdrant (带 Meta 关联 Neo4j ID)
2.  **查询 (Read Path):**
    * 用户提问 -> LangChain Agent
    * 意图识别 -> 路由至 (VectorSearch) 或 (GraphCypherQuery)
    * 结果融合 -> LLM 生成回答

## 4. 详细实施计划

### 阶段一：环境搭建与数据打通 (Week 1)
- [ ] **Docker 部署:** 启动 Neo4j 和 Qdrant 容器 (参考 `docker-compose.yml`)。
- [ ] **Adapter 开发:** 编写 `graphiti_adapter.py`，确保能解析上游 JSON 并写入 Neo4j。
- [ ] **双写验证:** 运行测试脚本，确认同一条数据在 Neo4j 中有节点，在 Qdrant 中有向量，且 ID 对应。

### 阶段二：查询引擎开发 (Week 2)
- [ ] **LangChain 集成:** 配置 `GraphCypherQAChain`，让 LLM 能自动写 Cypher。
- [ ] **Schema 调优:** 针对 Graphiti 生成的复杂图结构，优化 Prompt Template，告诉 LLM 如何查询“最新状态”。
- [ ] **混合检索:** 实现 Agent 逻辑，自动判断是查图还是查向量。

### 阶段三：API 与 平台化 (Week 3)
- [ ] **API 封装:** 完成 `/ingest` 和 `/chat` 接口。
- [ ] **前端对接:** (可选) 开发简单的 React/Streamlit 界面，展示实时入库效果。

## 5. 核心数据模型 (Schema Design)

### Neo4j 节点设计
- **`Image`**: 代表一次视觉采集。
    - 属性: `id`, `timestamp`, `location`
- **`Entity`**: 代表识别出的物理对象。
    - 属性: `name`, `type`, `model`
- **`Observation`** (核心): 连接 Image 和 Entity 的中间节点，记录当时的特定状态。
    - 关系: `(Image)-[:HAS_OBSERVATION]->(Observation)`
    - 关系: `(Observation)-[:DETECTED {status: 'leaking'}]->(Entity)`

### Qdrant Payload 设计
```json
{
  "page_content": "2024-12-29 10:00，检测到A区阀门出现严重锈蚀...",
  "metadata": {
    "image_id": "img_001",
    "timestamp": "2024-12-29T10:00:00",
    "entities": ["Valve_A"]
  }
}
```

## 6. 常见问题与解决方案

1.  **LLM 生成 Cypher 错误**: 
    * *解法:* 使用 `graph.refresh_schema()` 确保 schema 最新；在 Prompt 中加入 Few-shot 示例。
2.  **实体对齐问题**: 
    * *解法:* 在入库前增加简单的规则匹配（如根据设备 ID），确保“昨天的阀门A”和“今天的阀门A”在图里是同一个节点。

## 7. 启动指南

1. 启动数据库:
   ```bash
   docker-compose up -d
   ```
2. 安装依赖:
   ```bash
   pip install -r requirements.txt
   ```
3. 运行服务:
   ```bash
   python backend/main.py
   ```