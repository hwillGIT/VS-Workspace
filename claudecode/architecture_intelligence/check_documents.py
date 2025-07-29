#!/usr/bin/env python3
"""
Check what architecture documents are available in G:\downloads
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from core.knowledge_extractor import ArchitectureKnowledgeExtractor
import asyncio


async def check_library():
    """Check what's in the library"""
    library_path = Path("G:/downloads")
    
    if not library_path.exists():
        print(f"Path {library_path} not found")
        return
    
    print(f"Checking documents in: {library_path}")
    
    # Find PDFs
    pdf_files = list(library_path.glob("*.pdf"))
    print(f"Total PDF files: {len(pdf_files)}")
    
    # Architecture keywords to look for
    keywords = [
        "architect", "pattern", "design", "domain", "microservice", 
        "fowler", "evans", "vernon", "richardson", "newman",
        "ddd", "clean", "hexagonal", "event", "cqrs", "saga",
        "distributed", "system", "enterprise", "software"
    ]
    
    # Find potentially relevant files by name
    relevant_files = []
    for pdf in pdf_files:
        name_lower = pdf.name.lower()
        if any(keyword in name_lower for keyword in keywords):
            relevant_files.append(pdf)
    
    print(f"\nPotentially relevant files (by name): {len(relevant_files)}")
    
    # Show top 20
    for i, pdf in enumerate(relevant_files[:20], 1):
        print(f"{i:2d}. {pdf.name}")
    
    if len(relevant_files) > 20:
        print(f"... and {len(relevant_files) - 20} more")
    
    # Now analyze with our extractor
    print("\n" + "="*60)
    print("Analyzing relevance scores...")
    
    extractor = ArchitectureKnowledgeExtractor()
    context = {
        "domain": "software_architecture",
        "goals": ["patterns", "microservices", "domain_driven_design", "best_practices"],
        "technical_stack": ["distributed", "cloud", "event_driven"]
    }
    
    scored_docs = []
    for pdf in relevant_files[:10]:  # Analyze first 10
        try:
            doc = await extractor.analyze_document_relevance(pdf, context)
            scored_docs.append((doc.relevance_score, doc))
        except Exception as e:
            print(f"Error analyzing {pdf.name}: {e}")
    
    # Sort by score
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    
    print("\nTop architecture documents by relevance:")
    for score, doc in scored_docs:
        if score > 0:
            print(f"  [{score:.2f}] {doc.filename}")
            if doc.frameworks_covered:
                print(f"         Frameworks: {', '.join(doc.frameworks_covered)}")


if __name__ == "__main__":
    asyncio.run(check_library())