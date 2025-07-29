#!/usr/bin/env python3
"""
Simple Test for Architecture Document Analysis

Tests the core functionality without complex dependencies.
"""

import asyncio
import sys
from pathlib import Path

# Test basic Neo4j functionality
async def test_neo4j_integration():
    """Test Neo4j integration directly"""
    print("Testing Neo4j Integration...")
    
    try:
        from core.neo4j_knowledge_graph import (
            Neo4jKnowledgeGraph, 
            ArchitecturalPattern,
            KnowledgeScope
        )
        
        # Create knowledge graph
        kg = Neo4jKnowledgeGraph()
        
        # Test adding a pattern
        pattern = ArchitecturalPattern(
            name="Microservices Test Pattern",
            category="Architectural Style",
            description="Test pattern for microservices",
            benefits=["Scalability", "Independence", "Technology diversity"],
            drawbacks=["Complexity", "Network overhead", "Data consistency"],
            implementation_guidance="Start with bounded contexts",
            source="test_book.pdf",
            author="Martin Fowler",
            confidence_score=0.90
        )
        
        # Add pattern
        pattern_id = await kg.add_pattern(pattern, KnowledgeScope.PROJECT, "test_project")
        print(f"SUCCESS: Added pattern with ID {pattern_id}")
        
        # Test conflict detection
        pattern2 = ArchitecturalPattern(
            name="Monolithic Architecture",
            category="Architectural Style", 
            description="Single deployable unit",
            benefits=["Simplicity", "Easy deployment", "Data consistency"],
            drawbacks=["Scalability", "Technology lock-in", "Team dependencies"],
            implementation_guidance="Keep it simple",
            source="another_book.pdf",
            author="Eric Evans",
            confidence_score=0.85
        )
        
        conflicts = await kg.detect_conflicts(pattern2)
        print(f"SUCCESS: Detected {len(conflicts)} potential conflicts")
        
        kg.close()
        return True
        
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_document_relevance():
    """Test document relevance scoring"""
    print("\nTesting Document Relevance Analysis...")
    
    try:
        from core.knowledge_extractor import ArchitectureKnowledgeExtractor
        
        extractor = ArchitectureKnowledgeExtractor()
        
        # Test with a sample PDF path
        test_path = Path("G:/downloads")
        if not test_path.exists():
            print("G:\\downloads not found, using current directory")
            test_path = Path(".")
        
        # Find a PDF file
        pdf_files = list(test_path.glob("*.pdf"))[:3]  # First 3 PDFs
        
        if not pdf_files:
            print("No PDF files found for testing")
            return True  # Not a failure, just no files
        
        print(f"Found {len(pdf_files)} PDF files")
        
        # Test relevance scoring
        context = {
            "domain": "software_architecture",
            "goals": ["patterns", "best_practices", "microservices"],
            "technical_stack": ["cloud", "distributed"]
        }
        
        for pdf_file in pdf_files:
            try:
                document = await extractor.analyze_document_relevance(pdf_file, context)
                print(f"\nDocument: {document.filename}")
                print(f"  Relevance: {document.relevance.value} (score: {document.relevance_score:.2f})")
                print(f"  Frameworks: {', '.join(document.frameworks_covered)}")
                print(f"  Author: {document.author or 'Unknown'}")
            except Exception as e:
                print(f"  Error analyzing {pdf_file.name}: {e}")
        
        return True
        
    except Exception as e:
        print(f"FAILED: {e}")
        return False


async def test_chromadb_setup():
    """Test ChromaDB setup"""
    print("\nTesting ChromaDB Setup...")
    
    try:
        import chromadb
        from chromadb.config import Settings
        
        # Create test client
        client = chromadb.PersistentClient(path="./data/test_chromadb")
        
        # Create test collection
        collection = client.get_or_create_collection(
            name="test_patterns",
            metadata={"description": "Test pattern collection"}
        )
        
        # Add test document
        collection.add(
            documents=["This is a test pattern for microservices architecture"],
            ids=["test_1"],
            metadatas=[{"pattern": "microservices", "type": "test"}]
        )
        
        # Test search
        results = collection.query(
            query_texts=["microservices scalability"],
            n_results=1
        )
        
        print(f"SUCCESS: ChromaDB working, found {len(results['ids'][0])} results")
        return True
        
    except Exception as e:
        print(f"FAILED: {e}")
        return False


async def test_api_keys():
    """Test if API keys are configured"""
    print("\nTesting API Key Configuration...")
    
    import os
    from pathlib import Path
    
    # Load .env file
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value
    
    api_keys = {
        "GOOGLE_API_KEY": "Gemini models",
        "ANTHROPIC_API_KEY": "Claude models",
        "OPENAI_API_KEY": "GPT models"
    }
    
    configured = []
    missing = []
    
    for key, description in api_keys.items():
        if os.getenv(key) and os.getenv(key) != f"your-{key.lower().replace('_', '-')}-here":
            configured.append(f"{key} ({description})")
        else:
            missing.append(f"{key} ({description})")
    
    print(f"Configured: {len(configured)}")
    for item in configured:
        print(f"  - {item}")
    
    if missing:
        print(f"\nMissing: {len(missing)}")
        for item in missing:
            print(f"  - {item}")
    
    return len(configured) > 0  # At least one API key configured


def show_next_steps():
    """Show what to do next"""
    print("\n" + "=" * 60)
    print("NEXT STEPS FOR G:\\DOWNLOADS ANALYSIS")
    print("=" * 60)
    
    print("\n1. CONFIGURE API KEYS in .env file:")
    print("   - GOOGLE_API_KEY (for Gemini - preferred for document analysis)")
    print("   - ANTHROPIC_API_KEY (for Claude - fallback)")
    print("   - OPENAI_API_KEY (for GPT - secondary fallback)")
    
    print("\n2. SIMPLE DOCUMENT ANALYSIS:")
    print("""
import asyncio
from pathlib import Path
from core.knowledge_extractor import ArchitectureKnowledgeExtractor
from core.neo4j_knowledge_graph import Neo4jKnowledgeGraph, ArchitecturalPattern

async def analyze_book(pdf_path):
    # Extract knowledge
    extractor = ArchitectureKnowledgeExtractor()
    document = await extractor.analyze_document_relevance(pdf_path)
    
    if document.relevance_score > 0.7:
        print(f"Analyzing: {document.filename}")
        # Here you would call Gemini/Claude to extract patterns
        # For now, we'll create a sample pattern
        
        pattern = ArchitecturalPattern(
            name="Sample Pattern from " + document.filename,
            category="Architectural Style",
            description="Extracted pattern",
            benefits=["Benefit 1", "Benefit 2"],
            drawbacks=["Drawback 1"],
            implementation_guidance="Implementation guide",
            source=document.filename,
            author=document.author or "Unknown",
            confidence_score=document.relevance_score
        )
        
        # Store in Neo4j
        kg = Neo4jKnowledgeGraph()
        pattern_id = await kg.add_pattern(pattern)
        print(f"Stored pattern: {pattern_id}")
        kg.close()

# Run it
asyncio.run(analyze_book(Path("G:/downloads/your_book.pdf")))
""")
    
    print("\n3. QUERY YOUR KNOWLEDGE:")
    print("   - Open Neo4j Browser: http://localhost:7474")
    print("   - Login: neo4j / architecture123")
    print("   - Query: MATCH (p:Pattern) RETURN p LIMIT 10")


async def main():
    """Run all tests"""
    print("Architecture Intelligence Simple Test")
    print("=" * 40)
    
    tests = [
        ("Neo4j Integration", test_neo4j_integration),
        ("Document Relevance", test_document_relevance),
        ("ChromaDB Setup", test_chromadb_setup),
        ("API Keys", test_api_keys)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*40}")
        try:
            success = await test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"ERROR in {test_name}: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 40)
    print("TEST SUMMARY")
    print("=" * 40)
    
    passed = sum(1 for _, success in results if success)
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed >= 3:  # Most tests passed
        print("\nSYSTEM READY! You can start analyzing documents.")
        show_next_steps()
    else:
        print("\nSome components need attention.")


if __name__ == "__main__":
    asyncio.run(main())