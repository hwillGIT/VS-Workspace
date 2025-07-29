#!/usr/bin/env python3
"""
Analyze Top Architecture Books from G:\downloads

Focuses on the most relevant architecture books found.
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
from datetime import datetime

sys.path.append(str(Path(__file__).parent))

from core.neo4j_knowledge_graph import (
    Neo4jKnowledgeGraph,
    ArchitecturalPattern,
    KnowledgeScope
)
from core.knowledge_extractor import ArchitectureKnowledgeExtractor


# Top architecture books identified
TOP_BOOKS = [
    "Modeling Enterprise Architecture with TOGAF",
    "Java design patterns_ a tour of 23 gang of four design patterns",
    "Clean Architecture_ A Craftsman",
    "Solution Architecture Patterns for Enterprise",
    "System Design Interview",
    "Building Microservices_ Designing Fine-Grained Systems",
    "Domain-Driven Design_ Tackling Complexity",
    "Implementing Domain-Driven Design",
    "Patterns of Enterprise Application Architecture"
]


async def analyze_top_books():
    """Analyze the top architecture books"""
    
    print("Architecture Intelligence - Top Books Analysis")
    print("=" * 60)
    
    # Load API keys
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                if line.strip() and not line.startswith('#') and '=' in line:
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value
    
    # Initialize components
    kg = Neo4jKnowledgeGraph()
    extractor = ArchitectureKnowledgeExtractor()
    
    # Find matching books
    library_path = Path("G:/downloads")
    pdf_files = list(library_path.glob("*.pdf"))
    
    matched_books = []
    for pdf in pdf_files:
        for book_title in TOP_BOOKS:
            if book_title.lower() in pdf.name.lower():
                matched_books.append(pdf)
                break
    
    print(f"Found {len(matched_books)} top architecture books:")
    for i, book in enumerate(matched_books[:5], 1):
        print(f"  {i}. {book.name[:80]}...")
    
    # Analyze each book
    context = {
        "domain": "software_architecture",
        "goals": ["patterns", "principles", "best_practices", "microservices"],
        "technical_stack": ["distributed", "cloud", "enterprise"]
    }
    
    for book_path in matched_books[:3]:  # Start with top 3
        print(f"\n{'='*60}")
        print(f"Analyzing: {book_path.name[:80]}...")
        
        try:
            # Get document metadata
            doc = await extractor.analyze_document_relevance(book_path, context)
            print(f"Relevance: {doc.relevance_score:.2f}")
            print(f"Frameworks: {', '.join(doc.frameworks_covered)}")
            
            # Extract patterns based on book type
            patterns = await extract_patterns_from_book(book_path, doc)
            
            if patterns:
                print(f"\nExtracted {len(patterns)} patterns:")
                
                for pattern in patterns[:3]:  # Show first 3
                    print(f"\n  Pattern: {pattern.name}")
                    print(f"  Category: {pattern.category}")
                    print(f"  Benefits: {', '.join(pattern.benefits[:2])}")
                    
                    # Check for conflicts
                    conflicts = await kg.detect_conflicts(pattern)
                    if conflicts:
                        print(f"  Conflicts: {len(conflicts)} detected")
                    
                    # Store in Neo4j
                    try:
                        pattern_id = await kg.add_pattern(
                            pattern,
                            KnowledgeScope.GLOBAL,  # These are well-known patterns
                            "architecture_books"
                        )
                        print(f"  Stored: ID {pattern_id}")
                    except Exception as e:
                        print(f"  Storage error: {e}")
                
        except Exception as e:
            print(f"Error analyzing book: {e}")
    
    kg.close()
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print("\nYou can now query Neo4j:")
    print("  MATCH (p:Pattern) RETURN p.name, p.category, p.benefits")
    print("  MATCH (a:Author)-[:RECOMMENDS]->(p:Pattern) RETURN a.name, p.name")


async def extract_patterns_from_book(book_path: Path, doc) -> List[ArchitecturalPattern]:
    """Extract patterns based on book content (simulated)"""
    
    patterns = []
    
    # Pattern extraction based on book title
    if "clean architecture" in book_path.name.lower():
        patterns.extend([
            ArchitecturalPattern(
                name="Clean Architecture",
                category="Architectural Style",
                description="Architecture that separates concerns into layers with dependency rules",
                benefits=["Testability", "Independence of frameworks", "Independence of UI", "Independence of database"],
                drawbacks=["Initial complexity", "More boilerplate code"],
                implementation_guidance="Start with entities, then use cases, then interface adapters",
                source=doc.filename,
                author="Robert C. Martin",
                confidence_score=0.95
            ),
            ArchitecturalPattern(
                name="Dependency Rule",
                category="Design Principle",
                description="Dependencies should point inwards toward higher-level policies",
                benefits=["Loose coupling", "High cohesion", "Testability"],
                drawbacks=["Requires discipline", "Can lead to indirection"],
                implementation_guidance="Use dependency injection and interfaces",
                source=doc.filename,
                author="Robert C. Martin",
                confidence_score=0.95
            )
        ])
    
    elif "design patterns" in book_path.name.lower():
        patterns.extend([
            ArchitecturalPattern(
                name="Factory Pattern",
                category="Creational Pattern",
                description="Creates objects without specifying exact classes",
                benefits=["Flexibility", "Decoupling", "Code reuse"],
                drawbacks=["Added complexity", "Indirection"],
                implementation_guidance="Use when object creation logic is complex",
                source=doc.filename,
                author="Gang of Four",
                confidence_score=0.90
            ),
            ArchitecturalPattern(
                name="Observer Pattern",
                category="Behavioral Pattern",
                description="Defines one-to-many dependency between objects",
                benefits=["Loose coupling", "Dynamic relationships", "Open/closed principle"],
                drawbacks=["Memory leaks if not careful", "Unexpected updates"],
                implementation_guidance="Use for event-driven systems",
                source=doc.filename,
                author="Gang of Four",
                confidence_score=0.90
            )
        ])
    
    elif "togaf" in book_path.name.lower():
        patterns.extend([
            ArchitecturalPattern(
                name="Architecture Development Method (ADM)",
                category="Enterprise Architecture",
                description="TOGAF's core process for developing enterprise architecture",
                benefits=["Comprehensive approach", "Industry standard", "Proven methodology"],
                drawbacks=["Can be heavyweight", "Requires tailoring"],
                implementation_guidance="Start with preliminary phase and tailor to organization",
                source=doc.filename,
                author="The Open Group",
                confidence_score=0.85
            )
        ])
    
    elif "microservices" in book_path.name.lower():
        patterns.extend([
            ArchitecturalPattern(
                name="Database per Service",
                category="Microservices Pattern",
                description="Each microservice has its own database",
                benefits=["Service autonomy", "Independent scaling", "Technology diversity"],
                drawbacks=["Data consistency challenges", "Complex queries", "Data duplication"],
                implementation_guidance="Use event-driven architecture for data synchronization",
                source=doc.filename,
                author="Sam Newman",
                confidence_score=0.88
            )
        ])
    
    return patterns


if __name__ == "__main__":
    asyncio.run(analyze_top_books())