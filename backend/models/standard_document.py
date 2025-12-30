import re
from typing import List, Optional, Dict, Any
from pydantic import Field
from .base import BaseEntityModel


class StandardDocument(BaseEntityModel):
    """
    国家标准/行业标准文档实体模型
    覆盖：标题、摘要、正文、标签、提取的表格数据、发布年份等
    """

    # === 1. 核心标识 (Identity & Meta) ===
    standard_code: str = Field(
        ...,
        description="标准编号 (唯一主键)，如: GB/T 18830-2009"
    )
    title: str = Field(..., description="文档标题")
    publish_year: Optional[str] = Field(None, description="发布年份 (从编号提取)")

    # === 2. 核心内容 (Content) ===
    summary: Optional[str] = Field(None, description="全局摘要")

    # 注意：main_content 通常很大，建议在 GenericService 入库时配置 graph_exclude_fields 排除它
    main_content: Optional[str] = Field(None, description="正文内容")

    # === 3. 结构化特征 (Structured Features) ===
    tags: List[str] = Field(default_factory=list, description="文档标签")

    # === 4. 特殊数据: 表格 (Extracted Tables) ===
    # 存储从 OCR/图片中提取的表格文本，这是回答数值类问题的关键
    # Neo4j 支持存储 List[String] 类型
    extracted_tables: List[str] = Field(
        default_factory=list,
        description="从图片中提取的表格内容(Markdown/Text)"
    )

    # === 5. 扩展字段 (Extra) ===
    # 用于存储原始 JSON 中可能存在的其他未映射字段，保证数据不丢失
    extra_metadata: Dict[str, Any] = Field(default_factory=dict, description="额外元数据")

    # =========================================================
    # 核心逻辑：工厂方法 (Factory Method)
    # =========================================================
    @classmethod
    def from_chinese_json(cls, data: Dict[str, Any]) -> "StandardDocument":
        """
        将上游的中文 Key JSON 转换为标准实体对象
        """
        # 1. 基础映射
        raw_title = data.get("文档标题", "")
        raw_tags = data.get("文档标签", []) or []
        raw_summary = data.get("全局摘要", "")
        raw_content = data.get("正文内容", "")

        # 2. 表格数据扁平化提取
        # 原始结构: [{"图片序号": "0", "表格内容": "..."}, ...] -> ["表格内容..."]
        tables = []
        raw_imgs = data.get("图片内容", [])
        if isinstance(raw_imgs, list):
            for img in raw_imgs:
                # 过滤掉无效内容
                content = img.get("表格内容", "")
                if content and content != "无":
                    # 可以在这里把序号拼进去，保留上下文
                    idx = img.get("图片序号", "")
                    table_str = f"[图/表 {idx}] {content}" if idx else content
                    tables.append(table_str)

        # 3. 智能提取标准编号 (Standard Code)
        # 很多时候标题是空的，或者标题里没编号，标签里才有
        code = cls._extract_code(raw_title, raw_tags, raw_content)

        # 兜底逻辑：如果实在提取不到，用标题，还没有就用 Unknown
        final_code = code if code else (raw_title if raw_title else "UNKNOWN_CODE")
        final_title = raw_title if raw_title else final_code

        # 4. 提取年份
        year = None
        if final_code:
            match_year = re.search(r"(19|20)\d{2}", final_code)
            if match_year:
                year = match_year.group(0)

        # 5. 收集剩余的元数据
        known_keys = {"文档标题", "文档标签", "全局摘要", "正文内容", "图片内容"}
        extras = {k: v for k, v in data.items() if k not in known_keys}

        return cls(
            standard_code=final_code,
            title=final_title,
            summary=raw_summary,
            main_content=raw_content,
            tags=raw_tags,
            extracted_tables=tables,
            publish_year=year,
            extra_metadata=extras
        )

    @staticmethod
    def _extract_code(title: str, tags: List[str], content: str) -> Optional[str]:
        """
        正则提取标准号 (如 GB/T 18830-2009, ISO 139:2005)
        """
        # 匹配逻辑：字母(GB/ISO/EN) + 间隔 + 数字 + 可选年份
        pattern = r"([A-Z]+(?:/[A-Z]+)*\s?[\d\.:]+(?:-\d{4})?)"

        # 优先级 1: 标签 (通常最规范)
        for tag in tags:
            # 简单过滤：必须包含数字，且匹配正则
            if any(c.isdigit() for c in tag) and re.match(pattern, tag.strip()):
                return tag.strip()

        # 优先级 2: 标题
        match = re.search(pattern, title)
        if match:
            return match.group(1).strip()

        # 优先级 3: 正文开头
        if content:
            match = re.search(pattern, content[:300])
            if match:
                return match.group(1).strip()

        return None