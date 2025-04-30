from neo4j import GraphDatabase
import json

# Connect to Neo4j (update credentials if necessary)
URI = "bolt://localhost:7687"
USERNAME = "neo4j"
PASSWORD = "novanova"

# Nodes and relationships
with open("graph_4.json", "r") as f:
    data = json.load(f)

class Neo4jGraph:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def purge_old_data(self):
        with self.driver.session() as session:
            # Purge all existing data (nodes and relationships)
            session.run("MATCH (n) DETACH DELETE n")

    def insert_nodes(self, nodes):
        with self.driver.session() as session:
            for node in nodes:
                session.run("""
                    MERGE (n:Concept {name: $name})
                    SET n.usefulness = $usefulness, n.impact = $impact, n.efficiency = $efficiency, n.score = $score
                """, 
                name=node["name"], 
                usefulness=node["usefulness"], 
                impact=node["impact"], 
                efficiency=node["efficiency"], 
                score=node["score"])

    def insert_relationships(self, relationships):
        with self.driver.session() as session:
            for rel in relationships:
                # Use Python string formatting to inject the relationship type and ensure it's valid
                relationship_type = rel["type"].replace(" ", "_")  # Replace spaces with underscores
                query = f"""
                    MATCH (a:Concept {{name: $source}}), (b:Concept {{name: $target}})
                    MERGE (a)-[r:{relationship_type}]->(b)
                    SET r.score = $score
                """
                session.run(query, source=rel["from"], target=rel["to"], score=rel["score"])

# Initialize Neo4j connection
graph = Neo4jGraph(URI, USERNAME, PASSWORD)

# Purge old data
graph.purge_old_data()

# Insert nodes and relationships
graph.insert_nodes(data["nodes"])
graph.insert_relationships(data["relationships"])

# Close connection
graph.close()

print("Data inserted into Neo4j. Open Neo4j Browser and run:")
print("MATCH (n)-[r]->(m) RETURN n,r,m")
