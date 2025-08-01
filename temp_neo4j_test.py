
import sys
import os
import asyncio

sys.path.append(os.path.join(os.path.dirname(__file__), 'ClaudeCode', 'architecture_intelligence', 'core'))
from neo4j_knowledge_graph import Neo4jKnowledgeGraph

async def test_neo4j_connection():
    try:
        kg = Neo4jKnowledgeGraph()
        print("Successfully connected to Neo4j.")
        kg.close()
    except Exception as e:
        print(f"Failed to connect to Neo4j: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(test_neo4j_connection())
