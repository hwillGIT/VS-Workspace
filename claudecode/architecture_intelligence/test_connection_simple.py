#!/usr/bin/env python3
"""
Simple Neo4j Connection Test - Windows Compatible
"""

import sys
from pathlib import Path

try:
    from neo4j import GraphDatabase
    print("Neo4j driver available")
except ImportError:
    print("Installing neo4j driver...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "neo4j"])
    from neo4j import GraphDatabase


def test_connection():
    """Test Neo4j Desktop connection"""
    
    print("Testing Neo4j Desktop connection...")
    print("URI: bolt://localhost:7687")
    print("Username: neo4j")
    
    try:
        driver = GraphDatabase.driver(
            "bolt://localhost:7687",
            auth=("neo4j", "architecture123")
        )
        
        with driver.session() as session:
            result = session.run("RETURN 'Connected!' as message")
            record = result.single()
            
            if record:
                print(f"SUCCESS: {record['message']}")
                
                # Test creating a simple node
                session.run("CREATE (test:TestNode {created: datetime()}) RETURN test")
                print("SUCCESS: Can create nodes")
                
                # Clean up test node
                session.run("MATCH (test:TestNode) DELETE test")
                print("SUCCESS: Can delete nodes")
                
                driver.close()
                return True
                
    except Exception as e:
        print(f"CONNECTION FAILED: {e}")
        print("\nTroubleshooting Steps:")
        print("1. Open Neo4j Desktop")
        print("2. Make sure your database is STARTED (green play button)")
        print("3. Check that you set password to 'architecture123'")
        print("4. Try opening Neo4j Browser and connecting there first")
        print("5. If using different password, update the .env file")
        return False


def main():
    print("Neo4j Desktop Connection Test")
    print("=" * 30)
    
    if test_connection():
        print("\nSUCCESS: Neo4j Desktop is ready!")
        print("You can now run document analysis on G:\\downloads")
    else:
        print("\nFAILED: Please check Neo4j Desktop setup")
    
    input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()