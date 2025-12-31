from dataclasses import dataclass
from typing import Dict, Any, List, Optional


@dataclass(frozen=True)
class EntitySpec:
    label: str
    id_field: str
    vector_template: str
    graph_exclude_fields: Optional[List[str]] = None
    ingest_match_preset: Optional[str] = None


ENTITY_SPECS: Dict[str, EntitySpec] = {
    "Product": EntitySpec(
        label="Product",
        id_field="code",
        vector_template="产品编码: {code}, 系列: {series}, 功能: {fun}, 材质: {elem}, 描述: {className}",
        graph_exclude_fields=None,
    ),
    "StandardDocument": EntitySpec(
        label="StandardDocument",
        id_field="standard_code",
        vector_template="标题: {title}, 摘要: {summary}, 标签: {tags}, 内容: {main_content}",
        graph_exclude_fields=["main_content"],
        ingest_match_preset="standard_to_product",
    ),
}


RULE_LINK_PRESETS: Dict[str, Dict[str, Any]] = {
    "standard_to_product": {
        "source_label": "StandardDocument",
        "source_id_field": "standard_code",
        "target_label": "Product",
        "target_id_field": "code",
        "rel_type": "APPLIES_TO",
        "source_list_fields": ["tags"],
        "source_text_fields": ["title", "summary"],
        "target_text_fields": ["elem"],
        "mode": "contains_any",
    },
    "sensory_to_product": {
        "source_label": "Sensory",
        "source_id_field": "name",
        "target_label": "Product",
        "target_id_field": "code",
        "rel_type": "EVOKES",
        "source_list_fields": [],
        "source_text_fields": ["name"],
        "target_text_fields": ["fun"],
        "mode": "contains_any",
    },
    "season_to_product": {
        "source_label": "Season",
        "source_id_field": "name",
        "target_label": "Product",
        "target_id_field": "code",
        "rel_type": "SUITS_FOR",
        "source_list_fields": [],
        "source_text_fields": ["name"],
        "target_text_fields": ["season_marking"],
        "mode": "equals_any",
    },
}
