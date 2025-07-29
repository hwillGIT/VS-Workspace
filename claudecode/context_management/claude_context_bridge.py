"""
Claude Context Bridge

This script demonstrates how to use ChromaDB to maintain context
and provide relevant information to Claude in your conversations.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from chroma_context_manager import ChromaContextManager, ContextLevel


class ClaudeContextBridge:
    """
    Bridges ChromaDB context management with Claude conversations.
    
    This allows you to:
    1. Store conversation history persistently
    2. Retrieve relevant context for new queries
    3. Build up project knowledge over time
    """
    
    def __init__(self, project_name: str = "claude_assistant"):
        """Initialize the context bridge."""
        self.context_manager = ChromaContextManager()
        self.context_manager.set_project(project_name)
        self.conversation_log = []
        
    def add_conversation_turn(self, user_input: str, assistant_response: str):
        """
        Add a conversation turn to the context.
        
        Args:
            user_input: What the user asked
            assistant_response: Claude's response
        """
        # Add user input to immediate context
        self.context_manager.add_context(
            user_input,
            ContextLevel.IMMEDIATE,
            metadata={
                "type": "user_query",
                "turn": len(self.conversation_log)
            }
        )
        
        # Add assistant response to immediate context
        self.context_manager.add_context(
            assistant_response,
            ContextLevel.IMMEDIATE,
            metadata={
                "type": "assistant_response",
                "turn": len(self.conversation_log)
            }
        )
        
        # Log the conversation
        self.conversation_log.append({
            "timestamp": datetime.now().isoformat(),
            "user": user_input,
            "assistant": assistant_response
        })
        
    def save_important_decision(self, decision: str, reason: str):
        """
        Save an important technical decision to session context.
        
        Args:
            decision: The decision made
            reason: Why this decision was made
        """
        content = f"Decision: {decision}\nReason: {reason}"
        self.context_manager.add_context(
            content,
            ContextLevel.SESSION,
            metadata={
                "type": "technical_decision",
                "decision": decision,
                "reason": reason
            }
        )
        
    def save_code_pattern(self, pattern_name: str, code: str, description: str):
        """
        Save a useful code pattern to project context.
        
        Args:
            pattern_name: Name of the pattern
            code: The code pattern
            description: What it does
        """
        content = f"Pattern: {pattern_name}\n\n{code}\n\nDescription: {description}"
        self.context_manager.add_context(
            content,
            ContextLevel.PROJECT,
            metadata={
                "type": "code_pattern",
                "pattern_name": pattern_name,
                "language": "python"  # Could be dynamic
            }
        )
        
    def save_best_practice(self, practice: str, category: str):
        """
        Save a best practice to global context.
        
        Args:
            practice: The best practice
            category: Category (security, performance, etc.)
        """
        self.context_manager.add_context(
            practice,
            ContextLevel.GLOBAL,
            metadata={
                "type": "best_practice",
                "category": category
            }
        )
        
    def get_relevant_context(self, query: str, max_results: int = 5) -> str:
        """
        Get relevant context for a query, formatted for Claude.
        
        Args:
            query: The user's query
            max_results: Maximum number of context items
            
        Returns:
            Formatted context string to include in Claude prompt
        """
        # Search across all levels
        results = self.context_manager.search_context(
            query,
            n_results=max_results
        )
        
        if not results:
            return "No relevant context found."
        
        # Format context for Claude
        context_parts = []
        
        # Group by level
        by_level = {}
        for result in results:
            level = result['level']
            if level not in by_level:
                by_level[level] = []
            by_level[level].append(result)
        
        # Format each level
        level_order = ['global', 'project', 'session', 'immediate']
        for level in level_order:
            if level in by_level:
                context_parts.append(f"\n### {level.upper()} Context:")
                for item in by_level[level]:
                    relevance = 1 - item['distance']  # Convert distance to relevance
                    context_parts.append(
                        f"- [{relevance:.2%} relevant] {item['content'][:200]}..."
                    )
                    if 'type' in item['metadata']:
                        context_parts.append(f"  Type: {item['metadata']['type']}")
        
        return "\n".join(context_parts)
    
    def prepare_claude_prompt(self, user_query: str, include_context: bool = True) -> str:
        """
        Prepare a prompt for Claude with relevant context.
        
        Args:
            user_query: The user's question
            include_context: Whether to include retrieved context
            
        Returns:
            Complete prompt with context
        """
        prompt_parts = []
        
        if include_context:
            context = self.get_relevant_context(user_query)
            prompt_parts.append("## Relevant Context from Previous Conversations:")
            prompt_parts.append(context)
            prompt_parts.append("\n## Current Question:")
        
        prompt_parts.append(user_query)
        
        return "\n".join(prompt_parts)
    
    def export_session_summary(self) -> Dict[str, Any]:
        """Export a summary of the current session."""
        session_context = self.context_manager.get_session_context()
        
        summary = {
            "session_id": self.context_manager.current_session_id,
            "project": self.context_manager.current_project,
            "conversation_turns": len(self.conversation_log),
            "decisions_made": sum(1 for ctx in session_context 
                                if ctx['metadata'].get('type') == 'technical_decision'),
            "patterns_saved": sum(1 for ctx in session_context 
                                if ctx['metadata'].get('type') == 'code_pattern'),
            "start_time": self.conversation_log[0]['timestamp'] if self.conversation_log else None,
            "last_activity": datetime.now().isoformat()
        }
        
        return summary


# Example usage script
def example_workflow():
    """Demonstrate how to use the context bridge."""
    
    print("=== Claude Context Bridge Demo ===\n")
    
    # Initialize bridge for a specific project
    bridge = ClaudeContextBridge("trading_system_project")
    
    # Simulate a conversation
    print("1. Simulating conversation...")
    bridge.add_conversation_turn(
        "How should I structure the database for a trading system?",
        "For a trading system database, I recommend: 1) Separate tables for orders, trades, positions..."
    )
    
    # Save an important decision
    print("2. Saving technical decision...")
    bridge.save_important_decision(
        "Use PostgreSQL with TimescaleDB extension",
        "Need time-series optimization for market data and ACID compliance for transactions"
    )
    
    # Save a useful code pattern
    print("3. Saving code pattern...")
    bridge.save_code_pattern(
        "async_db_connection_pool",
        """async def create_db_pool():
    return await asyncpg.create_pool(
        host='localhost',
        database='trading',
        min_size=10,
        max_size=20
    )""",
        "Creates an async connection pool for high-performance database access"
    )
    
    # Save a best practice
    print("4. Saving best practice...")
    bridge.save_best_practice(
        "Always use Decimal type for monetary values, never float",
        "financial_accuracy"
    )
    
    # Now simulate a new query that should retrieve relevant context
    print("\n5. Testing context retrieval for new query...")
    new_query = "What database should I use for financial data?"
    
    context = bridge.get_relevant_context(new_query)
    print("\nRetrieved Context:")
    print(context)
    
    # Show how to prepare a full prompt
    print("\n6. Full Claude Prompt with Context:")
    print("-" * 50)
    full_prompt = bridge.prepare_claude_prompt(new_query)
    print(full_prompt)
    print("-" * 50)
    
    # Export session summary
    print("\n7. Session Summary:")
    summary = bridge.export_session_summary()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    example_workflow()