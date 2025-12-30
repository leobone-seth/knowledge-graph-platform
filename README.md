# 多模态知识图谱平台 (Multimodal KG Platform)

这是一个基于 **Python + Neo4j + Qdrant + LangChain** 构建的智能知识图谱基座。该平台旨在解决多模态数据（如识别后的图像对象、状态、描述）的结构化存储与智能问答问题。

核心能力包括：
1.  **动态图谱构建**: 利用 Graphiti 逻辑处理时序数据（Time-variant data），记录实体的状态变化。
2.  **双路混合检索 (GraphRAG)**: 结合 Neo4j 的逻辑推理能力与 Qdrant 的语义搜索能力。
3.  **大模型增强**: 使用 LangChain 编排 LLM 进行 Text-to-Cypher 生成和最终答案合成。

---

## 🛠️ 技术栈 (Tech Stack)

* **后端框架**: FastAPI
* **图数据库**: Neo4j (Community Edition 5.x)
* **向量数据库**: Qdrant (Binary Mode)
* **LLM 编排**: LangChain, LangChain-Neo4j
* **大模型**: OpenAI GPT-4o (或兼容 API)
* **运行环境**: Python 3.10+, Java JDK 17 (Neo4j 依赖)

---

## 📋 前置准备 (Prerequisites)

由于本项目采用**原生部署 (Native Deployment)**，请确保本地机器已安装以下基础环境：

1.  **Python 3.10+**: [下载 Python](https://www.python.org/)
2.  **Java JDK 17**: **(必须)** Neo4j 5.x 强制要求 Java 17 环境。
    * 验证命令: `java -version`
3.  **Neo4j Community Server**: [下载链接](https://neo4j.com/download-center/#community)
    * 下载对应的压缩包（Windows/Linux/Mac）并解压。
4.  **Qdrant Binary**: [下载链接](https://github.com/qdrant/qdrant/releases)
    * 下载对应系统的二进制文件并解压。

---

## 🚀 安装与配置 (Installation)

### 1. 克隆项目与环境配置
```bash
# 进入项目目录
cd multimodal-kg-platform

# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 安装 Python 依赖
pip install -r requirements.txt