"""
Save current workspace state to ChromaDB
"""

from chroma_context_manager import ChromaContextManager, ContextLevel
from datetime import datetime
import json

def save_workspace_state():
    # Initialize context manager
    cm = ChromaContextManager(persist_directory="./chroma_context_db")
    
    # Set project context
    cm.set_project("claude_code_workspace")
    
    # Save today's work at different context levels
    
    # 1. Immediate context - Current conversation
    cm.add_context(
        "Integrated parallel agents analysis guidelines into workspace configuration",
        ContextLevel.IMMEDIATE,
        metadata={
            "type": "task_completed",
            "date": "2025-07-31",
            "files_modified": "/CLAUDE.md, /roles/claude-code.md, /DEVELOPMENT_GUIDELINES.md"
        }
    )
    
    cm.add_context(
        "Created UI Designer role with emphasis on standardized modules only",
        ContextLevel.IMMEDIATE,
        metadata={
            "type": "task_completed",
            "date": "2025-07-31",
            "files_created": "/roles/claude-ui-designer.md, /UI_DESIGN_PRINCIPLES.md"
        }
    )
    
    # 2. Session context - Today's work session
    cm.add_context(
        """Session Summary: Enhanced workspace with AI collaboration guidelines and UI design principles.
        Key concepts: Plan→Code→Review loop, standardized modules only, Gestalt principles, 
        theme/effect customization allowed but core behavior immutable.""",
        ContextLevel.SESSION,
        metadata={
            "type": "session_summary",
            "date": "2025-07-31",
            "major_changes": "AI collaboration directives, UI design system"
        }
    )
    
    # 3. Project context - Workspace configuration knowledge
    cm.add_context(
        """Core workspace philosophy: All components must use standardized modules.
        No custom components allowed. Modules can be themed but not structurally modified.
        Every UI element extends from the standardized module library.""",
        ContextLevel.PROJECT,
        metadata={
            "type": "design_principle",
            "category": "ui_architecture",
            "importance": "critical"
        }
    )
    
    cm.add_context(
        """AI Agent collaboration principles: Always plan before coding, run tests automatically,
        respect permission settings, update knowledge before proposing solutions.""",
        ContextLevel.PROJECT,
        metadata={
            "type": "collaboration_principle",
            "category": "ai_workflow",
            "importance": "high"
        }
    )
    
    # 4. Global context - Universal principles
    cm.add_context(
        """Software development best practices: Plan→Code→Review loop, automated testing,
        security by default, explicit communication, continuous improvement.""",
        ContextLevel.GLOBAL,
        metadata={
            "type": "best_practice",
            "domain": "software_engineering",
            "source": "parallel_agents_analysis"
        }
    )
    
    cm.add_context(
        """UI Design principles: Gestalt principles (proximity, similarity, continuation, closure,
        figure/ground, common fate) form the foundation of all layout decisions.""",
        ContextLevel.GLOBAL,
        metadata={
            "type": "design_principle",
            "domain": "ui_ux",
            "framework": "gestalt"
        }
    )
    
    # Print statistics
    stats = cm.get_statistics()
    print(f"Context saved successfully!")
    print(f"Database statistics: {json.dumps(stats, indent=2)}")
    
    # Search test
    print("\nTesting search for 'standardized modules':")
    results = cm.search_context("standardized modules", n_results=3)
    for result in results:
        print(f"- [{result['level']}] {result['content'][:100]}...")

if __name__ == "__main__":
    save_workspace_state()