import sys
import os
import asyncio
import json
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), 'ClaudeCode', 'architecture_intelligence', 'core'))
from neo4j_knowledge_graph import Neo4jKnowledgeGraph, ArchitecturalPattern, KnowledgeScope

async def add_adr_to_neo4j(adr_file_path):
    try:
        with open(adr_file_path, 'r') as f:
            adr_data = json.load(f)

        kg = Neo4jKnowledgeGraph()

        # Extract data from ADR
        topic = adr_data.get("topic", "Untitled ADR")
        decision = adr_data.get("decision", "No decision provided.")
        rationale = adr_data.get("rationale", "No rationale provided.")
        implementation_details = adr_data.get("implementation_details", "No implementation details.")
        timestamp_str = adr_data.get("timestamp", datetime.now().isoformat())

        # Create an ArchitecturalPattern from the ADR
        pattern = ArchitecturalPattern(
            name=topic,
            category="Architectural Decision Record",
            description=f"Decision: {decision}\nRationale: {rationale}\nImplementation: {implementation_details}",
            benefits=["Improved clarity", "Documented rationale"],
            drawbacks=["Overhead of documentation"],
            implementation_guidance=implementation_details,
            source="memory-bank/decisionLog.md",
            author="Gemini CLI",
            confidence_score=1.0,
            extraction_date=datetime.fromisoformat(timestamp_str)
        )

        pattern_id = await kg.add_pattern(pattern, scope=KnowledgeScope.PROJECT, project_id="current_project")
        print(f"ADR '{topic}' added to Neo4j with ID: {pattern_id}")
        kg.close()
    except Exception as e:
        print(f"Error adding ADR to Neo4j: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        asyncio.run(add_adr_to_neo4j(sys.argv[1]))
    else:
        print("Usage: python add_adr_to_neo4j.py <adr_file_path>")
        sys.exit(1)