#!/usr/bin/env python3
"""
Neo4j Setup Script for Architecture Intelligence

Automatically sets up Neo4j using Docker for the architecture knowledge graph.
"""

import subprocess
import time
import sys
import os
from pathlib import Path


def check_docker_available():
    """Check if Docker is available and running"""
    try:
        result = subprocess.run(
            ["docker", "--version"], 
            capture_output=True, 
            text=True, 
            timeout=10
        )
        if result.returncode == 0:
            print(f"Docker found: {result.stdout.strip()}")
            return True
        else:
            print("Docker command failed")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("Docker not found or not responding")
        return False


def check_neo4j_running():
    """Check if Neo4j container is already running"""
    try:
        result = subprocess.run(
            ["docker", "ps", "--filter", "name=neo4j-architecture", "--format", "{{.Names}}"],
            capture_output=True,
            text=True
        )
        return "neo4j-architecture" in result.stdout
    except:
        return False


def check_neo4j_exists():
    """Check if Neo4j container exists (but may be stopped)"""
    try:
        result = subprocess.run(
            ["docker", "ps", "-a", "--filter", "name=neo4j-architecture", "--format", "{{.Names}}"],
            capture_output=True,
            text=True
        )
        return "neo4j-architecture" in result.stdout
    except:
        return False


def start_existing_neo4j():
    """Start existing Neo4j container"""
    print("📦 Starting existing Neo4j container...")
    try:
        result = subprocess.run(
            ["docker", "start", "neo4j-architecture"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("✅ Neo4j container started successfully")
            return True
        else:
            print(f"❌ Failed to start Neo4j: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error starting Neo4j: {e}")
        return False


def create_neo4j_container():
    """Create and start new Neo4j container"""
    print("🚀 Creating new Neo4j container...")
    
    # Create volumes first
    volumes = ["neo4j_architecture_data", "neo4j_architecture_logs", "neo4j_architecture_import"]
    
    for volume in volumes:
        try:
            subprocess.run(
                ["docker", "volume", "create", volume],
                capture_output=True,
                text=True
            )
        except:
            pass  # Volume might already exist
    
    # Create and start container
    docker_cmd = [
        "docker", "run",
        "--name", "neo4j-architecture",
        "-p", "7474:7474",  # HTTP port
        "-p", "7687:7687",  # Bolt port
        "-d",               # Detached mode
        "-v", "neo4j_architecture_data:/data",
        "-v", "neo4j_architecture_logs:/logs",
        "-v", "neo4j_architecture_import:/var/lib/neo4j/import",
        "-e", "NEO4J_AUTH=neo4j/architecture123",
        "-e", "NEO4J_PLUGINS=[\"apoc\"]",  # Enable APOC procedures
        "-e", "NEO4J_apoc_export_file_enabled=true",
        "-e", "NEO4J_apoc_import_file_enabled=true",
        "neo4j:5.15"
    ]
    
    try:
        result = subprocess.run(docker_cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Neo4j container created and started successfully")
            return True
        else:
            print(f"❌ Failed to create Neo4j container: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error creating Neo4j container: {e}")
        return False


def wait_for_neo4j_ready(max_wait=60):
    """Wait for Neo4j to be ready to accept connections"""
    print("⏳ Waiting for Neo4j to be ready...")
    
    for i in range(max_wait):
        try:
            # Test if Neo4j is responding
            result = subprocess.run(
                ["docker", "exec", "neo4j-architecture", "cypher-shell", "-u", "neo4j", "-p", "architecture123", "RETURN 1;"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                print("✅ Neo4j is ready!")
                return True
                
        except subprocess.TimeoutExpired:
            pass
        except Exception:
            pass
        
        print(f"   Still waiting... ({i+1}/{max_wait})")
        time.sleep(1)
    
    print("⚠️  Neo4j may not be fully ready, but continuing...")
    return False


def install_python_dependencies():
    """Install required Python packages"""
    print("📦 Installing Python dependencies...")
    
    packages = ["neo4j", "chromadb"]
    
    for package in packages:
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", package],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                print(f"✅ Installed {package}")
            else:
                print(f"⚠️  Warning: Failed to install {package}: {result.stderr}")
        except Exception as e:
            print(f"⚠️  Warning: Error installing {package}: {e}")


def test_connection():
    """Test connection to Neo4j"""
    print("🔍 Testing Neo4j connection...")
    
    try:
        from neo4j import GraphDatabase
        
        driver = GraphDatabase.driver(
            "bolt://localhost:7687",
            auth=("neo4j", "architecture123")
        )
        
        with driver.session() as session:
            result = session.run("RETURN 'Connection successful!' as message")
            record = result.single()
            if record:
                print(f"✅ {record['message']}")
                return True
                
    except ImportError:
        print("⚠️  Neo4j Python driver not available")
        return False
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        return False
    
    return False


def create_initial_schema():
    """Create initial Neo4j schema for architecture intelligence"""
    print("🏗️  Creating initial schema...")
    
    try:
        from neo4j import GraphDatabase
        
        driver = GraphDatabase.driver(
            "bolt://localhost:7687",
            auth=("neo4j", "architecture123")
        )
        
        schema_queries = [
            "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT pattern_name IF NOT EXISTS FOR (p:Pattern) REQUIRE p.name IS UNIQUE",
            "CREATE CONSTRAINT author_name IF NOT EXISTS FOR (a:Author) REQUIRE a.name IS UNIQUE",
            "CREATE INDEX pattern_category IF NOT EXISTS FOR (p:Pattern) ON (p.category)"
        ]
        
        with driver.session() as session:
            for query in schema_queries:
                try:
                    session.run(query)
                    print(f"   ✅ {query}")
                except Exception as e:
                    print(f"   ⚠️  {query} - {e}")
        
        driver.close()
        print("✅ Schema creation completed")
        return True
        
    except Exception as e:
        print(f"❌ Schema creation failed: {e}")
        return False


def main():
    """Main setup process"""
    print("Architecture Intelligence Neo4j Setup")
    print("=" * 50)
    
    # Step 1: Check Docker
    if not check_docker_available():
        print("\n❌ Setup failed: Docker is required but not available.")
        print("Please install Docker and try again.")
        return False
    
    # Step 2: Check Neo4j status
    if check_neo4j_running():
        print("✅ Neo4j is already running!")
    elif check_neo4j_exists():
        if not start_existing_neo4j():
            return False
    else:
        if not create_neo4j_container():
            return False
    
    # Step 3: Wait for Neo4j to be ready
    wait_for_neo4j_ready()
    
    # Step 4: Install Python dependencies
    install_python_dependencies()
    
    # Step 5: Test connection
    if not test_connection():
        print("⚠️  Connection test failed, but continuing...")
    
    # Step 6: Create schema
    create_initial_schema()
    
    # Final status
    print("\n" + "=" * 50)
    print("✅ Neo4j setup completed!")
    print("\n📊 Connection Details:")
    print("   Browser:  http://localhost:7474")
    print("   Bolt:     bolt://localhost:7687")
    print("   Username: neo4j")
    print("   Password: architecture123")
    
    print("\n🎯 Next Steps:")
    print("   1. Open http://localhost:7474 in your browser")
    print("   2. Login with neo4j/architecture123")
    print("   3. Run: python -c \"from architecture_intelligence.core.integrated_knowledge_manager import create_integrated_knowledge_manager; import asyncio; asyncio.run(create_integrated_knowledge_manager())\"")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)