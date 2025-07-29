#!/usr/bin/env python3
"""
Query the Neo4j Knowledge Graph to see what was stored
"""

from neo4j import GraphDatabase
import json


def query_knowledge():
    """Query what patterns and knowledge were stored"""
    
    print("Architecture Knowledge Graph - Query Results")
    print("=" * 50)
    
    try:
        driver = GraphDatabase.driver(
            "bolt://localhost:7687",
            auth=("neo4j", "architecture123")
        )
        
        with driver.session() as session:
            # Count all nodes
            result = session.run("MATCH (n) RETURN labels(n) as labels, count(n) as count")
            print("Node counts by type:")
            for record in result:
                labels = record["labels"]
                count = record["count"]
                print(f"  {', '.join(labels)}: {count}")
            
            print()
            
            # Get all patterns
            result = session.run("""
                MATCH (p:Pattern) 
                RETURN p.name as name, p.category as category, 
                       p.benefits as benefits, p.source as source
                LIMIT 10
            """)
            
            patterns = list(result)
            print(f"Patterns found: {len(patterns)}")
            
            for pattern in patterns:
                print(f"\n  Pattern: {pattern['name']}")
                print(f"  Category: {pattern['category']}")
                if pattern['benefits']:
                    print(f"  Benefits: {', '.join(pattern['benefits'][:3])}")
                if pattern['source']:
                    print(f"  Source: {pattern['source'][:50]}...")
            
            # Get relationships
            result = session.run("""
                MATCH (a:Author)-[r:RECOMMENDS]->(p:Pattern)
                RETURN a.name as author, p.name as pattern
                LIMIT 5
            """)
            
            relationships = list(result)
            if relationships:
                print(f"\nAuthor -> Pattern relationships: {len(relationships)}")
                for rel in relationships:
                    print(f"  {rel['author']} -> {rel['pattern']}")
            
            # Get conflicts if any
            result = session.run("MATCH (c:Conflict) RETURN count(c) as count")
            conflict_count = result.single()["count"]
            print(f"\nConflicts detected: {conflict_count}")
            
        driver.close()
        
    except Exception as e:
        print(f"Query failed: {e}")
        return
    
    print("\n" + "=" * 50)
    print("SUCCESS: Knowledge graph is working!")
    print("\nTo explore more:")
    print("1. Open Neo4j Browser: http://localhost:7474")
    print("2. Login: neo4j / architecture123")
    print("3. Try queries:")
    print("   MATCH (p:Pattern) RETURN p")
    print("   MATCH (p:Pattern)-[:IMPLEMENTED_IN]->(f:Framework) RETURN p, f")


if __name__ == "__main__":
    query_knowledge()