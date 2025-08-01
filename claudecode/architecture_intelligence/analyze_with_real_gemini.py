#!/usr/bin/env python3
"""
Architecture Analysis with Real Gemini API

This version will use the actual Gemini API once you provide a real API key.
For now, it shows what would happen with real analysis.
"""

import asyncio
import os
import sys
from pathlib import Path
import json

sys.path.append(str(Path(__file__).parent))

from core.neo4j_knowledge_graph import (
    Neo4jKnowledgeGraph,
    ArchitecturalPattern,
    KnowledgeScope
)


def check_api_keys():
    """Check if API keys are properly configured"""
    
    # Load .env file
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                if line.strip() and not line.startswith('#') and '=' in line:
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value
    
    # Use global API key manager for multi-key failover support
    try:
        from global_api_keys import get_api_key_sync
        google_key = get_api_key_sync('GOOGLE_API_KEY')
        anthropic_key = get_api_key_sync('ANTHROPIC_API_KEY') 
        openai_key = get_api_key_sync('OPENAI_API_KEY')
    except ImportError:
        # Fallback to direct access if global manager not available
        google_key = os.getenv('GOOGLE_API_KEY')
        anthropic_key = os.getenv('ANTHROPIC_API_KEY') 
        openai_key = os.getenv('OPENAI_API_KEY')
    
    print("API Key Status:")
    print("=" * 30)
    
    # Check each key
    keys = {
        "Google (Gemini)": google_key,
        "Anthropic (Claude)": anthropic_key,
        "OpenAI (GPT)": openai_key
    }
    
    configured_count = 0
    for name, key in keys.items():
        if key and not key.startswith('your-') and len(key) > 20:
            print(f"[OK] {name}: Configured")
            configured_count += 1
        else:
            print(f"[!] {name}: Not configured (placeholder)")
    
    print(f"\nConfigured: {configured_count}/3 API keys")
    
    return configured_count > 0, google_key


async def analyze_with_real_gemini(book_path: Path):
    """Analyze a book with real Gemini API"""
    
    try:
        import google.generativeai as genai
        
        # Configure Gemini
        api_key = os.getenv('GOOGLE_API_KEY')
        genai.configure(api_key=api_key)
        
        # Use the 2.0 Flash model for document analysis
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # Create a comprehensive analysis prompt
        prompt = f"""
        Analyze this architecture book: {book_path.name}
        
        Based on the title and your knowledge of architecture books, extract:
        
        1. **Key Architectural Patterns** (3-5 patterns):
           - Pattern name
           - Category (Architectural Style, Design Pattern, etc.)
           - Brief description
           - Main benefits (2-3)
           - Main drawbacks (1-2)
           - Implementation guidance
        
        2. **Design Principles** discussed
        
        3. **Author's main architectural philosophy**
        
        Please provide response in JSON format:
        {{
            "book_analysis": {{
                "title": "extracted title",
                "author": "author name",
                "main_themes": ["theme1", "theme2"],
                "architectural_philosophy": "brief description"
            }},
            "patterns": [
                {{
                    "name": "Pattern Name",
                    "category": "Pattern Category",
                    "description": "What this pattern does",
                    "benefits": ["benefit1", "benefit2", "benefit3"],
                    "drawbacks": ["drawback1", "drawback2"],
                    "implementation_guidance": "How to implement this pattern",
                    "context": "When to use this pattern"
                }}
            ],
            "principles": [
                {{
                    "name": "Principle Name",
                    "statement": "Brief statement of the principle",
                    "rationale": "Why this principle matters"
                }}
            ]
        }}
        
        Focus on extracting real, valuable architectural knowledge that would be useful in a knowledge base.
        """
        
        print(f"Analyzing with Gemini 2.0 Flash: {book_path.name[:60]}...")
        
        # Generate response
        response = model.generate_content(prompt)
        
        # Parse JSON response
        text = response.text
        
        # Find JSON in the response
        start = text.find('{')
        end = text.rfind('}') + 1
        
        if start >= 0 and end > start:
            json_text = text[start:end]
            analysis = json.loads(json_text)
            
            print(f"[SUCCESS] Successfully analyzed: {analysis['book_analysis']['title']}")
            print(f"   Author: {analysis['book_analysis']['author']}")
            print(f"   Patterns found: {len(analysis.get('patterns', []))}")
            print(f"   Principles found: {len(analysis.get('principles', []))}")
            
            return analysis
        else:
            print(f"[ERROR] Could not parse JSON from response")
            return None
            
    except ImportError:
        print("[ERROR] Google AI library not installed. Run: pip install google-generativeai")
        return None
    except Exception as e:
        print(f"[ERROR] Gemini analysis failed: {e}")
        return None


async def demonstrate_real_analysis():
    """Demonstrate what real Gemini analysis would do"""
    
    print("Architecture Intelligence - Real Gemini Analysis Demo")
    print("=" * 60)
    
    # Check API keys
    has_keys, google_key = check_api_keys()
    
    if not has_keys:
        print("\n[ERROR] No API keys configured!")
        print("\nTo use real Gemini analysis:")
        print("1. Get a Google AI API key from: https://makersuite.google.com/app/apikey")
        print("2. Update .env file: GOOGLE_API_KEY=your_actual_key_here")
        print("3. Run this script again")
        return
    
    # Check if Google key is real
    if not google_key or google_key.startswith('your-') or len(google_key) < 20:
        print("\n[ERROR] Google API key is still a placeholder!")
        print("\nTo get a real Gemini API key:")
        print("1. Go to: https://makersuite.google.com/app/apikey")
        print("2. Click 'Create API Key'")
        print("3. Copy the key")
        print("4. Update .env: GOOGLE_API_KEY=your_actual_key")
        print("\nFor now, showing simulated analysis...")
        await simulate_gemini_analysis()
        return
    
    # Try real analysis
    print(f"\n[GEMINI] Using real Gemini API key: {google_key[:10]}...")
    
    # Find a good architecture book
    library_path = Path("G:/downloads")
    pdf_files = list(library_path.glob("*.pdf"))
    
    # Look for Clean Architecture or Design Patterns book
    target_books = [
        "Clean Architecture",
        "design patterns",
        "TOGAF",
        "Domain-Driven Design"
    ]
    
    selected_book = None
    for pdf in pdf_files:
        for target in target_books:
            if target.lower() in pdf.name.lower():
                selected_book = pdf
                break
        if selected_book:
            break
    
    if not selected_book:
        selected_book = pdf_files[0]  # Use first available
    
    # Analyze with real Gemini
    analysis = await analyze_with_real_gemini(selected_book)
    
    if analysis:
        # Store in Neo4j
        kg = Neo4jKnowledgeGraph()
        
        patterns_stored = 0
        for pattern_data in analysis.get('patterns', []):
            pattern = ArchitecturalPattern(
                name=pattern_data['name'],
                category=pattern_data['category'],
                description=pattern_data['description'],
                benefits=pattern_data['benefits'],
                drawbacks=pattern_data['drawbacks'],
                implementation_guidance=pattern_data['implementation_guidance'],
                source=selected_book.name,
                author=analysis['book_analysis']['author'],
                confidence_score=0.95  # High confidence from Gemini
            )
            
            try:
                pattern_id = await kg.add_pattern(pattern, KnowledgeScope.GLOBAL)
                print(f"   Stored pattern: {pattern.name} (ID: {pattern_id})")
                patterns_stored += 1
            except Exception as e:
                print(f"   Storage error: {e}")
        
        kg.close()
        print(f"\n[SUCCESS] Analysis complete! Stored {patterns_stored} patterns.")
        print("\nQuery in Neo4j Browser:")
        print("MATCH (p:Pattern) WHERE p.source CONTAINS 'Clean' RETURN p")


async def simulate_gemini_analysis():
    """Simulate what Gemini analysis would produce"""
    
    print("\n[SIMULATION] Simulated Gemini Analysis (what you'd get with real API key):")
    print("-" * 50)
    
    simulated_analysis = {
        "book_analysis": {
            "title": "Clean Architecture: A Craftsman's Guide to Software Structure and Design",
            "author": "Robert C. Martin",
            "main_themes": ["dependency_inversion", "layered_architecture", "testing"],
            "architectural_philosophy": "Architecture should maximize the number of decisions not made"
        },
        "patterns": [
            {
                "name": "Clean Architecture",
                "category": "Architectural Style",
                "description": "Concentric circles of dependencies pointing inward",
                "benefits": ["Testable", "Framework independent", "UI independent", "Database independent"],
                "drawbacks": ["Initial complexity", "Learning curve"],
                "implementation_guidance": "Start with entities and use cases, add interfaces last",
                "context": "Large applications requiring long-term maintainability"
            },
            {
                "name": "Dependency Inversion Principle",
                "category": "Design Principle", 
                "description": "High-level modules should not depend on low-level modules",
                "benefits": ["Flexibility", "Testability", "Reusability"],
                "drawbacks": ["Indirection", "Initial abstraction overhead"],
                "implementation_guidance": "Use interfaces and dependency injection",
                "context": "When you need to swap implementations"
            }
        ]
    }
    
    print(f"[BOOK] {simulated_analysis['book_analysis']['title']}")
    print(f"[AUTHOR] {simulated_analysis['book_analysis']['author']}")
    print(f"[PHILOSOPHY] {simulated_analysis['book_analysis']['architectural_philosophy']}")
    print(f"[PATTERNS] {len(simulated_analysis['patterns'])}")
    
    for pattern in simulated_analysis['patterns']:
        print(f"\n  * {pattern['name']} ({pattern['category']})")
        print(f"     {pattern['description']}")
        print(f"     Benefits: {', '.join(pattern['benefits'][:2])}")


if __name__ == "__main__":
    asyncio.run(demonstrate_real_analysis())