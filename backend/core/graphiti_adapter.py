import os
from langchain_community.graphs import Neo4jGraph

class GraphitiAdapter:
    def __init__(self):
        # 1. 加载配置
        self.neo4j_url = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_auth = (
            os.getenv("NEO4J_USERNAME", "neo4j"),
            os.getenv("NEO4J_PASSWORD", "password")
        )

        # 2. 初始化连接
        self.graph = Neo4jGraph(
            url=self.neo4j_url,
            username=self.neo4j_auth[0],
            password=self.neo4j_auth[1]
        )
        print("Graph Engine Initialized")

    # --- Create (增) ---
    def add_observation(self, image_id: str, timestamp: str, text_summary: str, entities: list):
        """写入观测数据"""
        print(f"Creating/Merging Observation for: {image_id}")
        cypher = """
        MERGE (img:Image {id: $image_id})
        SET img.timestamp = $timestamp

        MERGE (evt:Observation {id: $image_id + '_obs'})
        SET evt.summary = $summary, evt.at = $timestamp
        MERGE (img)-[:HAS_OBSERVATION]->(evt)

        FOREACH (entity IN $entities |
            MERGE (e:Entity {name: entity.name})
            SET e.type = entity.type
            MERGE (evt)-[:DETECTED]->(e)
            MERGE (e)-[s:STATUS_AT]->(evt)
            SET s.status = entity.status
        )
        """
        self.graph.query(
            cypher,
            params={
                "image_id": image_id,
                "timestamp": timestamp,
                "summary": text_summary,
                "entities": entities,
            },
        )

    # --- Read (查) ---
    def get_entity_details(self, name: str):
        """查询实体及其最近的一次观测状态"""
        cypher = """
        MATCH (e:Entity {name: $name})
        OPTIONAL MATCH (e)-[s:STATUS_AT]->(obs:Observation)
        RETURN e.name as name, e.type as type, s.status as latest_status, obs.at as timestamp
        ORDER BY obs.at DESC LIMIT 1
        """
        result = self.graph.query(cypher, params={"name": name})
        return result if result else None

    # --- Update (改) ---
    def update_entity_status(self, name: str, new_status: str, timestamp: str):
        """更新实体的状态"""
        cypher = """
        MATCH (e:Entity {name: $name})
        SET e.current_status = $new_status, e.last_updated = $timestamp
        RETURN e
        """
        self.graph.query(
            cypher,
            params={"name": name, "new_status": new_status, "timestamp": timestamp},
        )
        return {"name": name, "status": new_status, "updated_at": timestamp}

    # --- Delete (删) ---
    def delete_observation_chain(self, image_id: str):
        """级联删除：删除 Image 及其关联的 Observation 节点"""
        cypher = """
        MATCH (img:Image {id: $image_id})
        OPTIONAL MATCH (img)-[:HAS_OBSERVATION]->(obs:Observation)
        DETACH DELETE img, obs
        """
        self.graph.query(cypher, params={"image_id": image_id})
        return True
