from typing import Optional
from pydantic import Field
from .base import BaseEntityModel

class Product(BaseEntityModel):
    code: Optional[str] = Field(None, description="Product code")
    weight: Optional[float] = Field(None, description="Product weight")
    elem: Optional[str] = Field(None, description="Fiber composition")
    inelem: Optional[str] = Field(None, description="Internal composition")
    mprice: Optional[float] = Field(None, description="Market price")
    season_marking: Optional[str] = Field(None, description="Season marking")
    series: Optional[str] = Field(None, description="Series name")
    dressing_category: Optional[str] = Field(None, description="Dressing category")
    fun: Optional[str] = Field(None, description="Functional features")
    fabric_erp: Optional[str] = Field(None, description="Fabric ERP name")
    className: Optional[str] = Field(None, description="Fabric class name")