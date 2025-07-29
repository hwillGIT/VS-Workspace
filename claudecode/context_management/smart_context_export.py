"""
Smart Context Export - Uses ChromaDB's semantic search to create intelligent context files

This maintains the power of embeddings while creating a file you can share with Claude.
"""

import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from chroma_context_manager import ChromaContextManager, ContextLevel


class SmartContextExporter:
    """
    Intelligently exports context using semantic search based on:
    1. Recent activity
    2. Current work description
    3. Specific queries
    """
    
    def __init__(self, project: str = "default"):
        self.context_manager = ChromaContextManager()
        self.context_manager.set_project(project)
        self.project = project
        
    def export_for_work_session(self, 
                               work_description: str,
                               output_file: str = "CONTEXT.md",
                               max_items_per_level: int = 5) -> str:
        """
        Export context relevant to current work session.
        
        Args:
            work_description: What you're working on
            output_file: Where to save context
            max_items_per_level: Max items from each context level
            
        Returns:
            Path to exported file
        """
        output_path = Path(output_file)
        
        # Gather relevant context using semantic search
        context_sections = []
        
        # Header
        context_sections.append(f"# Context for: {self.project}")
        context_sections.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        context_sections.append(f"**Work Focus:** {work_description}\n")
        
        # Search for relevant context at each level
        levels = [
            (ContextLevel.GLOBAL, "Universal Best Practices & Patterns"),
            (ContextLevel.PROJECT, "Project-Specific Knowledge"),
            (ContextLevel.SESSION, "Recent Decisions & Changes"),
            (ContextLevel.IMMEDIATE, "Current Conversation Context")
        ]
        
        for level, description in levels:
            # Search for relevant items
            results = self.context_manager.search_context(
                work_description,
                level=level,
                n_results=max_items_per_level
            )
            
            if results:
                context_sections.append(f"## {description}")
                for i, result in enumerate(results, 1):
                    relevance = max(0, 1 - result['distance']) * 100
                    
                    # Only include if reasonably relevant
                    if relevance > 20:  # 20% relevance threshold
                        context_sections.append(f"\n### {i}. [{relevance:.0f}% relevant]")
                        
                        # Add metadata if meaningful
                        metadata = result['metadata']
                        if 'type' in metadata:
                            context_sections.append(f"**Type:** {metadata['type']}")
                        if 'timestamp' in metadata:
                            context_sections.append(f"**When:** {metadata['timestamp']}")
                        
                        # Add content
                        context_sections.append(f"\n{result['content']}\n")
                
                context_sections.append("")  # Empty line between sections
        
        # Add quick queries section
        context_sections.append("## Quick Context Queries")
        context_sections.append("*Use these to get more specific context:*")
        context_sections.append("- Recent technical decisions")
        context_sections.append("- Code patterns in this project")
        context_sections.append("- Security considerations")
        context_sections.append("- Performance optimizations")
        
        # Write to file
        content = "\n".join(context_sections)
        output_path.write_text(content, encoding='utf-8')
        
        print(f"[OK] Exported smart context to: {output_path}")
        print(f"  - Used semantic search for: '{work_description}'")
        print(f"  - Found relevant context from {len([s for s in context_sections if s.startswith('###')])} items")
        
        return str(output_path)
    
    def export_by_queries(self,
                         queries: List[str],
                         output_file: str = "CONTEXT.md") -> str:
        """
        Export context based on multiple semantic queries.
        
        Args:
            queries: List of queries to search for
            output_file: Where to save
            
        Returns:
            Path to exported file
        """
        output_path = Path(output_file)
        context_sections = []
        
        # Header
        context_sections.append(f"# Targeted Context Export: {self.project}")
        context_sections.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Search for each query
        for query in queries:
            results = self.context_manager.search_context(query, n_results=5)
            
            if results:
                context_sections.append(f"## Query: {query}")
                
                for result in results:
                    relevance = max(0, 1 - result['distance']) * 100
                    if relevance > 25:  # Higher threshold for targeted queries
                        context_sections.append(f"\n### [{relevance:.0f}% relevant] From {result['level']}")
                        context_sections.append(result['content'])
                        
                        # Add useful metadata
                        if result['metadata'].get('type'):
                            context_sections.append(f"\n*Type: {result['metadata']['type']}*")
                
                context_sections.append("")  # Empty line
        
        # Write to file
        content = "\n".join(context_sections)
        output_path.write_text(content, encoding='utf-8')
        
        print(f"[OK] Exported targeted context to: {output_path}")
        
        return str(output_path)
    
    def create_daily_context(self, output_file: Optional[str] = None) -> str:
        """
        Create a daily context file with most relevant information.
        
        Args:
            output_file: Override default daily file name
            
        Returns:
            Path to exported file
        """
        if not output_file:
            date_str = datetime.now().strftime('%Y%m%d')
            output_file = f"CONTEXT_{self.project}_{date_str}.md"
        
        # Common daily queries
        daily_queries = [
            "current implementation",
            "technical decisions",
            "code patterns",
            "todo tasks remaining", 
            "bugs issues problems",
            "architecture design"
        ]
        
        return self.export_by_queries(daily_queries, output_file)


def main():
    """Command line interface for smart context export."""
    parser = argparse.ArgumentParser(description="Smart context export using semantic search")
    parser.add_argument("project", help="Project name")
    
    subparsers = parser.add_subparsers(dest="command", help="Export commands")
    
    # Work session export
    work_parser = subparsers.add_parser("work", help="Export for current work")
    work_parser.add_argument("description", help="What you're working on")
    work_parser.add_argument("-o", "--output", default="CONTEXT.md", help="Output file")
    work_parser.add_argument("-n", "--max-items", type=int, default=5, help="Max items per level")
    
    # Query-based export
    query_parser = subparsers.add_parser("query", help="Export by queries")
    query_parser.add_argument("queries", nargs="+", help="Queries to search for")
    query_parser.add_argument("-o", "--output", default="CONTEXT.md", help="Output file")
    
    # Daily context
    daily_parser = subparsers.add_parser("daily", help="Create daily context file")
    daily_parser.add_argument("-o", "--output", help="Output file (auto-named if not specified)")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize exporter
    exporter = SmartContextExporter(args.project)
    
    # Execute command
    if args.command == "work":
        exporter.export_for_work_session(args.description, args.output, args.max_items)
    elif args.command == "query":
        exporter.export_by_queries(args.queries, args.output)
    elif args.command == "daily":
        exporter.create_daily_context(args.output)


if __name__ == "__main__":
    main()


# Example usage:
# python smart_context_export.py trading_system work "implementing order execution engine"
# python smart_context_export.py trading_system query "database design" "API endpoints" "error handling"
# python smart_context_export.py trading_system daily