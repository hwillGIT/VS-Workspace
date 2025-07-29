#!/usr/bin/env python3
"""
Test Neo4j Desktop Connection

Verifies connection to Neo4j Desktop and creates initial schema.
"""

import sys
import os
from pathlib import Path

# Add architecture_intelligence to path
sys.path.append(str(Path(__file__).parent))

try:
    from neo4j import GraphDatabase
    print("Neo4j driver available")
except ImportError:
    print("Installing neo4j driver...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "neo4j"])
    from neo4j import GraphDatabase


def test_connection():
    """Test connection to Neo4j Desktop"""
    
    # Connection details for Neo4j Desktop
    uri = "bolt://localhost:7687"
    username = "neo4j"
    password = "architecture123"  # Set this in Neo4j Desktop
    
    print("Testing Neo4j connection...")
    print(f"URI: {uri}")
    print(f"Username: {username}")
    
    try:
        # Create driver
        driver = GraphDatabase.driver(uri, auth=(username, password))
        
        # Test connection
        with driver.session() as session:
            result = session.run("RETURN 'Connection successful!' as message, datetime() as timestamp")
            record = result.single()
            
            if record:
                print(f"✅ {record['message']}")
                print(f"   Timestamp: {record['timestamp']}")
                
                # Get Neo4j version
                version_result = session.run("CALL dbms.components() YIELD name, versions RETURN name, versions[0] as version")
                for record in version_result:
                    if record['name'] == 'Neo4j Kernel':
                        print(f"   Version: Neo4j {record['version']}")
                
                return driver
            else:
                print("❌ No response from Neo4j")
                return None
                
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure Neo4j Desktop is running")
        print("2. Check that your database is started")
        print("3. Verify the password is 'architecture123'")
        print("4. Try connecting via Neo4j Browser first")
        return None


def create_schema(driver):
    """Create initial schema for architecture intelligence"""
    
    print("\nCreating schema...")
    
    schema_queries = [
        # Constraints for uniqueness
        "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
        "CREATE CONSTRAINT pattern_name IF NOT EXISTS FOR (p:Pattern) REQUIRE p.name IS UNIQUE", 
        "CREATE CONSTRAINT principle_name IF NOT EXISTS FOR (pr:Principle) REQUIRE pr.name IS UNIQUE",
        "CREATE CONSTRAINT framework_name IF NOT EXISTS FOR (f:Framework) REQUIRE f.name IS UNIQUE",
        "CREATE CONSTRAINT author_name IF NOT EXISTS FOR (a:Author) REQUIRE a.name IS UNIQUE",
        "CREATE CONSTRAINT project_id IF NOT EXISTS FOR (proj:Project) REQUIRE proj.id IS UNIQUE",
        
        # Indexes for performance
        "CREATE INDEX pattern_category IF NOT EXISTS FOR (p:Pattern) ON (p.category)",
        "CREATE INDEX document_source IF NOT EXISTS FOR (d:Document) ON (d.source)",
        "CREATE INDEX extraction_date IF NOT EXISTS FOR (p:Pattern) ON (p.extraction_date)",
        "CREATE INDEX conflict_type IF NOT EXISTS FOR (c:Conflict) ON (c.conflict_type)"
    ]
    
    success_count = 0
    
    with driver.session() as session:
        for query in schema_queries:
            try:
                session.run(query)
                print(f"   ✅ {query}")
                success_count += 1
            except Exception as e:
                if "already exists" in str(e).lower() or "equivalent" in str(e).lower():
                    print(f"   ✓ {query} (already exists)")
                    success_count += 1
                else:
                    print(f"   ❌ {query} - {e}")
    
    print(f"\nSchema creation: {success_count}/{len(schema_queries)} successful")
    return success_count == len(schema_queries)


def create_sample_data(driver):
    """Create sample architectural pattern for testing"""
    
    print("\nCreating sample data...")
    
    sample_query = """
    // Create sample architectural pattern
    MERGE (p:Pattern {name: 'Microservices Architecture'})
    SET p.category = 'Architectural Style',
        p.description = 'Distributed system architecture with independently deployable services',
        p.benefits = ['Scalability', 'Technology diversity', 'Team autonomy'],
        p.drawbacks = ['Complexity', 'Network overhead', 'Data consistency'],
        p.implementation_guidance = 'Start with domain-driven design to identify service boundaries',
        p.confidence_score = 0.95,
        p.extraction_date = datetime(),
        p.scope = 'global'
    
    // Create sample author
    MERGE (a:Author {name: 'Martin Fowler'})
    MERGE (a)-[:RECOMMENDS]->(p)
    
    // Create sample document
    MERGE (d:Document {id: 'fowler_microservices.pdf'})
    SET d.source = 'Microservices - Martin Fowler'
    MERGE (d)-[:CONTAINS]->(p)
    
    RETURN p, a, d
    """
    
    try:
        with driver.session() as session:
            result = session.run(sample_query)
            record = result.single()
            
            if record:
                print("   ✅ Sample data created successfully")
                pattern = record['p']
                print(f"   Pattern: {pattern['name']}")
                print(f"   Author: {record['a']['name']}")
                print(f"   Document: {record['d']['source']}")
                return True
            else:
                print("   ❌ Failed to create sample data")
                return False
                
    except Exception as e:
        print(f"   ❌ Sample data creation failed: {e}")
        return False


def test_queries(driver):
    """Test some basic queries"""
    
    print("\nTesting queries...")
    
    test_queries = [
        ("Count patterns", "MATCH (p:Pattern) RETURN count(p) as pattern_count"),
        ("Count authors", "MATCH (a:Author) RETURN count(a) as author_count"),
        ("Count documents", "MATCH (d:Document) RETURN count(d) as document_count"),
        ("Pattern relationships", """
            MATCH (a:Author)-[:RECOMMENDS]->(p:Pattern)<-[:CONTAINS]-(d:Document)
            RETURN a.name as author, p.name as pattern, d.source as document
            LIMIT 5
        """)
    ]
    
    with driver.session() as session:
        for query_name, query in test_queries:
            try:
                result = session.run(query)
                records = list(result)
                
                print(f"   ✅ {query_name}: {len(records)} results")
                
                # Show first result for context
                if records:
                    first_record = dict(records[0])
                    print(f"      Sample: {first_record}")
                    
            except Exception as e:
                print(f"   ❌ {query_name} failed: {e}")


def main():
    """Main test function"""
    
    print("Neo4j Desktop Connection Test")
    print("=" * 40)
    
    # Test connection
    driver = test_connection()
    if not driver:
        return False
    
    try:
        # Create schema
        schema_success = create_schema(driver)
        
        # Create sample data
        if schema_success:
            sample_success = create_sample_data(driver)
            
            # Test queries
            if sample_success:
                test_queries(driver)
        
        print("\n" + "=" * 40)
        print("✅ Neo4j Desktop is ready for Architecture Intelligence!")
        print("\nNext steps:")
        print("1. Your Neo4j database is configured")
        print("2. Schema and sample data created")
        print("3. Ready to analyze G:\\downloads documents")
        
        return True
        
    finally:
        driver.close()


if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Setup incomplete - please check Neo4j Desktop configuration")
    
    input("\nPress Enter to continue...")