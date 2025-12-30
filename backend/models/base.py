from pydantic import BaseModel, Field
from typing import Optional

class BaseEntityModel(BaseModel):
    """所有业务实体模型的基类"""
    # 可以在这里定义所有实体都共有的字段，比如 id, created_at 等
    pass