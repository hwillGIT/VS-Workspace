#!/usr/bin/env python3
"""
Simple Neo4j Setup for Windows

Sets up Neo4j using Docker without Unicode characters for Windows compatibility.
"""

import subprocess
import time
import sys
import os


def check_docker():
    """Check if Docker is available"""
    try:
        result = subprocess.run(["docker", "--version"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"Docker found: {result.stdout.strip()}")
            return True
        else:
            print("Docker command failed")
            return False
    except:
        print("Docker not found")
        return False


def setup_neo4j():
    """Setup Neo4j container"""
    print("Setting up Neo4j...")
    
    # Check if container exists
    try:
        result = subprocess.run(
            ["docker", "ps", "-a", "--filter", "name=neo4j-architecture"],
            capture_output=True, text=True
        )
        
        if "neo4j-architecture" in result.stdout:
            print("Container exists, starting...")
            subprocess.run(["docker", "start", "neo4j-architecture"])
        else:
            print("Creating new container...")
            subprocess.run([
                "docker", "run", "--name", "neo4j-architecture",
                "-p", "7474:7474", "-p", "7687:7687", "-d",
                "-e", "NEO4J_AUTH=neo4j/architecture123",
                "neo4j:5.15"
            ])
            
        print("Neo4j started successfully!")
        print("Browser: http://localhost:7474")
        print("Username: neo4j | Password: architecture123")
        
        return True
        
    except Exception as e:
        print(f"Failed to setup Neo4j: {e}")
        return False


def install_deps():
    """Install Python dependencies"""
    print("Installing dependencies...")
    packages = ["neo4j", "chromadb"]
    
    for package in packages:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", package], 
                         capture_output=True, text=True)
            print(f"Installed {package}")
        except:
            print(f"Warning: Could not install {package}")


def main():
    print("Architecture Intelligence Neo4j Setup")
    print("=" * 40)
    
    if not check_docker():
        print("ERROR: Docker is required. Please install Docker Desktop.")
        return False
    
    if not setup_neo4j():
        return False
        
    install_deps()
    
    print("\nSetup complete!")
    print("Next: Open http://localhost:7474 and login with neo4j/architecture123")
    
    return True


if __name__ == "__main__":
    main()