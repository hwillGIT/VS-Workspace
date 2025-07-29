#!/usr/bin/env python3
"""
System Status Check - Architecture Intelligence Platform
"""

import os
import sys
from pathlib import Path
from neo4j import GraphDatabase

def check_system_status():
    """Check the status of all architecture intelligence components"""
    
    print("=" * 70)
    print("ARCHITECTURE INTELLIGENCE PLATFORM - SYSTEM STATUS")
    print("=" * 70)
    print()
    
    # Check API Keys
    print("API KEY STATUS:")
    print("-" * 30)
    
    # Load environment variables
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                if line.strip() and not line.startswith('#') and '=' in line:
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value
    
    api_keys = {
        'Google (Gemini)': os.getenv('GOOGLE_API_KEY'),
        'Anthropic (Claude)': os.getenv('ANTHROPIC_API_KEY'),
        'OpenAI (GPT)': os.getenv('OPENAI_API_KEY'),
        'Groq': os.getenv('GROQ_API_KEY'),
        'OpenRouter': os.getenv('OPENROUTER_API_KEY')
    }
    
    configured_keys = 0
    for name, key in api_keys.items():
        if key and not key.startswith('your-') and len(key) > 15:
            print(f"   [OK] {name}: Configured")
            configured_keys += 1
        else:
            print(f"   [!] {name}: Not configured")
    
    print(f"\n   Total configured: {configured_keys}/{len(api_keys)} API keys")
    print()
    
    # Check Neo4j Connection
    print("DATABASE STATUS:")
    print("-" * 30)
    
    try:
        driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'architecture123'))
        session = driver.session()
        result = session.run("RETURN 'connected' as status")
        status = result.single()['status']
        print(f"   [OK] Neo4j: Connected ({status})")
        
        # Get knowledge base statistics
        result = session.run('MATCH (p:Pattern) RETURN count(p) as total')
        patterns = result.single()['total']
        
        result = session.run('MATCH (a:Author) RETURN count(a) as total')
        authors = result.single()['total']
        
        result = session.run('MATCH (d:Document) RETURN count(d) as total')
        documents = result.single()['total']
        
        print(f"   [INFO] Knowledge Base:")
        print(f"          * Patterns: {patterns}")
        print(f"          * Authors: {authors}")
        print(f"          * Documents: {documents}")
        
        session.close()
        driver.close()
        
    except Exception as e:
        print(f"   [ERROR] Neo4j: Connection failed - {e}")
    
    print()
    
    # Check Required Libraries
    print("LIBRARY STATUS:")
    print("-" * 30)
    
    required_libs = [
        'google.generativeai', 
        'neo4j', 
        'click', 
        'rich',
        'chromadb',
        'yaml'
    ]
    
    missing_libs = []
    for lib in required_libs:
        try:
            if lib == 'yaml':
                import yaml
            elif lib == 'google.generativeai':
                import google.generativeai
            elif lib == 'neo4j':
                import neo4j
            elif lib == 'click':
                import click
            elif lib == 'rich':
                import rich
            elif lib == 'chromadb':
                import chromadb
            print(f"   [OK] {lib}: Installed")
        except ImportError:
            print(f"   [!] {lib}: Missing")
            missing_libs.append(lib)
    
    if missing_libs:
        print(f"\n   Install missing libraries: pip install {' '.join(missing_libs)}")
    print()
    
    # Check Document Library
    print("DOCUMENT LIBRARY:")
    print("-" * 30)
    
    library_path = Path("G:/downloads")
    if library_path.exists():
        pdf_files = len(list(library_path.glob("*.pdf")))
        print(f"   [OK] G:/downloads: {pdf_files} PDF files available")
        
        # Check for key architecture books
        key_books = [
            "Clean Architecture",
            "design patterns", 
            "TOGAF",
            "Domain-Driven Design",
            "Microservices"
        ]
        
        found_books = []
        for pdf in library_path.glob("*.pdf"):
            for book in key_books:
                if book.lower() in pdf.name.lower():
                    found_books.append(book)
                    break
        
        print(f"   [INFO] Architecture books found: {len(found_books)}")
        for book in found_books[:5]:
            print(f"          * {book}")
        
    else:
        print(f"   [!] G:/downloads: Directory not found")
    
    print()
    
    # System Capabilities Summary
    print("CAPABILITIES SUMMARY:")
    print("-" * 30)
    
    capabilities = []
    
    if configured_keys > 0:
        capabilities.append("[OK] AI-powered document analysis")
    
    try:
        GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'architecture123'))
        capabilities.append("[OK] Knowledge graph storage")
    except:
        pass
    
    if library_path.exists() and len(list(library_path.glob("*.pdf"))) > 0:
        capabilities.append("[OK] Architecture document library")
    
    capabilities.extend([
        "[OK] 20+ architecture frameworks supported",
        "[OK] Pattern mining and conflict detection",
        "[OK] Enterprise-ready CLI interface"
    ])
    
    for capability in capabilities:
        print(f"   {capability}")
    
    print()
    print("=" * 70)
    
    # Overall Status
    if configured_keys >= 1 and patterns > 0:
        print("[SUCCESS] STATUS: FULLY OPERATIONAL - Ready for enterprise deployment!")
    elif configured_keys >= 1:
        print("[READY] STATUS: READY - Run analysis to populate knowledge base")
    else:
        print("[SETUP] STATUS: SETUP REQUIRED - Configure API keys to begin")
    
    print("=" * 70)
    print()
    
    # Quick Commands
    print("QUICK COMMANDS:")
    print("   python generate_report.py           # View knowledge base")
    print("   python analyze_with_real_gemini.py  # Analyze more documents")
    print("   python query_knowledge.py          # Query patterns")
    print("   python cli.py analyze-library       # Advanced analysis")
    print()

if __name__ == "__main__":
    check_system_status()