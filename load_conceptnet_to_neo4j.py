import json
from tqdm import tqdm  
from neo4j import GraphDatabase
import os

# Neo4j Connection Config
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "novanova"

# Connect to Neo4j
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# File path to ConceptNet edges CSV
CONCEPTNET_FILE = "/Users/christian/Desktop/Nova POC/conceptnet/conceptnet-assertions-5.7.0.csv"
CHECKPOINT_FILE = "checkpoint.txt"

# Batch settings
BATCH_SIZE = 10_000  # Process 10,000 lines per batch
CHECKPOINT_INTERVAL = 10  # Save checkpoint every 10 batches

def purge_database():
    # """Deletes all nodes and relationships from Neo4j."""
    # with driver.session() as session:
    #     session.run("MATCH (n) DETACH DELETE n")
    # print("Database purged.")

def setup_indexes():
    """Ensures Neo4j indexes are created for faster querying."""
    with driver.session() as session:
        session.run("CREATE INDEX IF NOT EXISTS FOR (c:Concept) ON (c.uri);")
    print("Indexes created.")

def process_line(line):
    """Parses a ConceptNet line and extracts nodes, relation, and metadata."""
    parts = line.strip().split("\t")
    if len(parts) < 5:
        return None  

    uri, relation, start, end, metadata_json = parts
    try:
        json.loads(metadata_json)  # Validate JSON
    except json.JSONDecodeError:
        metadata_json = "{}"  # Use empty JSON if invalid

    if "/c/en/" in start and "/c/en/" in end:
        return {"start": start, "relation": relation, "end": end, "metadata": metadata_json}  
    return None

def insert_data(tx, batch):
    """Inserts nodes and relationships into Neo4j efficiently using UNWIND."""
    query = """
    UNWIND $batch AS row
    MERGE (a:Concept {uri: row.start})
    MERGE (b:Concept {uri: row.end})
    MERGE (a)-[r:RELATION {type: row.relation, metadata: row.metadata}]->(b)
    MERGE (b)-[r_reverse:RELATION {type: row.relation, metadata: row.metadata}]->(a)
    """
    tx.run(query, batch=batch)

def load_conceptnet():
    """Purges the database, sets up indexes, then loads ConceptNet into Neo4j."""
    # purge_database()
    setup_indexes()

    last_processed_line = 0
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            last_processed_line = int(f.read().strip())

    with driver.session() as session, open(CONCEPTNET_FILE, "r", encoding="utf-8") as file:
        batch = []
        batch_count = 0

        for i, line in tqdm(enumerate(file), desc="Processing ConceptNet", unit="lines"):
            if i <= last_processed_line:
                continue  

            data = process_line(line)
            if data:
                batch.append(data)

            if len(batch) >= BATCH_SIZE:
                session.write_transaction(insert_data, batch)
                batch = []  
                batch_count += 1

                if batch_count % CHECKPOINT_INTERVAL == 0:
                    with open(CHECKPOINT_FILE, "w") as f:
                        f.write(str(i))

        if batch:
            session.write_transaction(insert_data, batch)
            with open(CHECKPOINT_FILE, "w") as f:
                f.write(str(i))

    print(f"Import completed. Processed up to line {i}.")

# Run the loader
load_conceptnet()

# Close Neo4j connection
driver.close()



