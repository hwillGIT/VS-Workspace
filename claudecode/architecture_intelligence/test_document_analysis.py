#!/usr/bin/env python3
"""
Test Document Analysis Pipeline

Tests the complete pipeline from PDF documents to knowledge graph
using the integrated architecture intelligence system.
"""

import asyncio
import sys
import os
from pathlib import Path
import json

# Add paths for imports
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

# Test if we can import our modules
try:
    from core.integrated_knowledge_manager import create_integrated_knowledge_manager
    from core.neo4j_knowledge_graph import ArchitecturalPattern, KnowledgeScope
    from agents.document_analysis_agent import create_document_analysis_agent
    print("All modules imported successfully")
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're in the right directory and Neo4j is running")
    sys.exit(1)


async def test_knowledge_manager():
    """Test the integrated knowledge manager"""
    
    print("Testing Integrated Knowledge Manager...")
    
    try:
        # Create knowledge manager for testing
        km = await create_integrated_knowledge_manager(
            project_id="document_analysis_test",
            chroma_path="./data/test_knowledge"
        )
        
        print("SUCCESS: Knowledge manager created")
        
        # Test adding a sample pattern
        sample_pattern = ArchitecturalPattern(
            name="Test Microservices Pattern",
            category="Architectural Style", 
            description="A test pattern for microservices architecture",
            benefits=["Scalability", "Independence"],
            drawbacks=["Complexity", "Network overhead"],
            implementation_guidance="Start with domain boundaries",
            source="test_document.pdf",
            author="Test Author",
            confidence_score=0.85
        )
        
        # Add pattern with conflict detection
        result = await km.add_pattern(sample_pattern, KnowledgeScope.PROJECT)
        
        print(f"SUCCESS: Pattern added with ID {result['pattern_id']}")
        print(f"Conflicts detected: {len(result['conflicts_detected'])}")
        
        # Test semantic search
        search_results = await km.semantic_search_patterns("microservices scalability")
        print(f"SUCCESS: Found {len(search_results)} patterns via semantic search")
        
        # Get knowledge statistics
        stats = await km.get_knowledge_statistics()
        print(f"SUCCESS: Knowledge base stats: {stats['chromadb_stats']}")
        
        await km.shutdown()
        return True
        
    except Exception as e:
        print(f"FAILED: Knowledge manager test failed: {e}")
        return False


async def test_document_agent():
    """Test the document analysis agent"""
    
    print("\nTesting Document Analysis Agent...")
    
    try:
        # Create document analysis agent  
        agent = await create_document_analysis_agent(
            prefer_models=["gemini-2.0-flash-exp", "claude-3-5-sonnet-20241022"]
        )
        
        print("SUCCESS: Document analysis agent created")
        
        # Test with a sample document path (you can change this)
        test_library_path = Path("G:/downloads")
        
        if not test_library_path.exists():
            print(f"WARNING: {test_library_path} not found, using current directory for test")
            test_library_path = Path(".")
        
        # Analyze document library (limit to 2 for testing)
        context = {
            "domain": "software_architecture",
            "goals": ["scalability", "maintainability"],
            "technical_stack": ["microservices", "cloud_native"]
        }
        
        print(f"Analyzing documents in: {test_library_path}")
        
        results = await agent.analyze_document_library(
            library_path=test_library_path,
            context=context,
            max_documents=2,  # Limit for testing
            prefer_gemini=True
        )
        
        print(f"SUCCESS: Analyzed {results['documents_analyzed']} documents")
        print(f"Model usage: {results['model_usage']}")
        print(f"Knowledge summary: {results['knowledge_summary']}")
        
        await agent.shutdown()
        return True
        
    except Exception as e:
        print(f"FAILED: Document agent test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_full_pipeline():
    """Test the complete document analysis pipeline"""
    
    print("\nTesting Complete Pipeline...")
    
    # This would test:
    # 1. PDF extraction from G:\downloads
    # 2. Gemini analysis with model routing
    # 3. Pattern extraction and conflict detection
    # 4. Knowledge graph storage
    # 5. Semantic search capabilities
    
    print("Full pipeline test - checking components...")
    
    # Check environment variables
    required_vars = [
        "GOOGLE_API_KEY", 
        "ANTHROPIC_API_KEY", 
        "NEO4J_URI", 
        "NEO4J_USERNAME", 
        "NEO4J_PASSWORD"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"WARNING: Missing environment variables: {missing_vars}")
        print("Set these in your .env file for full functionality")
    else:
        print("SUCCESS: All environment variables configured")
    
    return len(missing_vars) == 0


def show_next_steps():
    """Show user what to do next"""
    
    print("\n" + "=" * 50)
    print("NEXT STEPS FOR G:\\DOWNLOADS ANALYSIS")
    print("=" * 50)
    
    print("\n1. VERIFY SETUP:")
    print("   - Neo4j Desktop is running")
    print("   - API keys are set in .env file")
    print("   - G:\\downloads folder exists with PDF files")
    
    print("\n2. ANALYZE YOUR LIBRARY:")
    print("   python -c \"")
    print("   import asyncio")
    print("   from agents.document_analysis_agent import create_document_analysis_agent")
    print("   from pathlib import Path")
    print("   ")
    print("   async def analyze():")
    print("       agent = await create_document_analysis_agent()")
    print("       results = await agent.analyze_document_library(")
    print("           library_path=Path('G:/downloads'),")
    print("           max_documents=5")
    print("       )")
    print("       print(f'Analyzed: {results[\\\"documents_analyzed\\\"]} documents')")
    print("       await agent.shutdown()")
    print("   ")
    print("   asyncio.run(analyze())")
    print("   \"")
    
    print("\n3. EXPLORE RESULTS:")
    print("   - Open Neo4j Browser: http://localhost:7474")
    print("   - Query: MATCH (p:Pattern) RETURN p LIMIT 10")
    print("   - Search for conflicts: MATCH (c:Conflict) RETURN c")


async def main():
    """Main test function"""
    
    print("Architecture Intelligence Pipeline Test")
    print("=" * 40)
    
    # Test components sequentially
    tests = [
        ("Knowledge Manager", test_knowledge_manager),
        ("Document Agent", test_document_agent), 
        ("Full Pipeline Check", test_full_pipeline)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n=== {test_name} ===")
        try:
            success = await test_func()
            results.append((test_name, success))
            if success:
                print(f"SUCCESS: {test_name} passed")
            else:
                print(f"FAILED: {test_name} failed")
        except Exception as e:
            print(f"ERROR: {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 40)
    print("TEST RESULTS SUMMARY")
    print("=" * 40)
    
    passed = 0
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{status}: {test_name}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\nSUCCESS: System ready for document analysis!")
        show_next_steps()
    else:
        print("\nWARNING: Some components need attention before proceeding")


if __name__ == "__main__":
    asyncio.run(main())