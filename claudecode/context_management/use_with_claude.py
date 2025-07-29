"""
Practical Workflow: How to Use ChromaDB Context with Claude

This script shows how to integrate ChromaDB context management
into your actual workflow with Claude.
"""

from claude_context_bridge import ClaudeContextBridge
import sys
import json
from datetime import datetime


def interactive_session():
    """
    Interactive session that demonstrates the workflow.
    """
    print("=== ChromaDB + Claude Integration ===")
    print("This tool helps you maintain context across Claude conversations.\n")
    
    # Get project name
    project = input("Enter project name (or press Enter for 'default'): ").strip()
    if not project:
        project = "default"
    
    # Initialize bridge
    bridge = ClaudeContextBridge(project)
    print(f"\n✓ Initialized context for project: {project}")
    
    while True:
        print("\n" + "="*50)
        print("Choose an action:")
        print("1. Ask a question (with context)")
        print("2. Save the last Q&A exchange")
        print("3. Record a technical decision")
        print("4. Save a code pattern")
        print("5. Add a best practice")
        print("6. View session summary")
        print("7. Export context for Claude")
        print("8. Start new session")
        print("9. Switch project")
        print("0. Exit")
        
        choice = input("\nYour choice: ").strip()
        
        if choice == "1":
            # Ask a question with context
            question = input("\nYour question: ").strip()
            if question:
                print("\n--- COPY THIS TO CLAUDE ---")
                print(bridge.prepare_claude_prompt(question))
                print("--- END ---")
                
                # Store the question temporarily
                bridge._last_question = question
                
        elif choice == "2":
            # Save Q&A exchange
            if hasattr(bridge, '_last_question'):
                response = input("\nPaste Claude's response (or press Enter to skip): ").strip()
                if response:
                    bridge.add_conversation_turn(bridge._last_question, response)
                    print("✓ Saved conversation turn")
                    delattr(bridge, '_last_question')
            else:
                print("⚠ No question to save. Ask a question first.")
                
        elif choice == "3":
            # Record decision
            decision = input("\nWhat decision was made? ").strip()
            reason = input("Why? ").strip()
            if decision and reason:
                bridge.save_important_decision(decision, reason)
                print("✓ Saved technical decision")
                
        elif choice == "4":
            # Save code pattern
            pattern_name = input("\nPattern name: ").strip()
            print("Enter code (type 'END' on a new line when done):")
            code_lines = []
            while True:
                line = input()
                if line == "END":
                    break
                code_lines.append(line)
            code = "\n".join(code_lines)
            description = input("Description: ").strip()
            
            if pattern_name and code and description:
                bridge.save_code_pattern(pattern_name, code, description)
                print("✓ Saved code pattern")
                
        elif choice == "5":
            # Add best practice
            practice = input("\nBest practice: ").strip()
            category = input("Category (security/performance/design/other): ").strip()
            if practice and category:
                bridge.save_best_practice(practice, category)
                print("✓ Saved best practice")
                
        elif choice == "6":
            # View summary
            summary = bridge.export_session_summary()
            print("\nSession Summary:")
            print(json.dumps(summary, indent=2))
            
        elif choice == "7":
            # Export context
            query = input("\nWhat topic to export context for? ").strip()
            if query:
                context = bridge.get_relevant_context(query, max_results=10)
                print("\n--- CONTEXT FOR CLAUDE ---")
                print(context)
                print("--- END ---")
                
        elif choice == "8":
            # New session
            bridge.context_manager.new_session()
            bridge.conversation_log = []
            print("✓ Started new session")
            
        elif choice == "9":
            # Switch project
            new_project = input("\nNew project name: ").strip()
            if new_project:
                bridge.context_manager.set_project(new_project)
                print(f"✓ Switched to project: {new_project}")
                
        elif choice == "0":
            print("\nGoodbye!")
            break
            
        else:
            print("Invalid choice. Try again.")


def batch_export(project: str, output_file: str):
    """
    Export all context for a project to a file.
    
    Args:
        project: Project name
        output_file: Where to save the context
    """
    bridge = ClaudeContextBridge(project)
    
    # Get all context
    stats = bridge.context_manager.get_statistics()
    
    output = {
        "project": project,
        "exported_at": datetime.now().isoformat(),
        "statistics": stats,
        "contexts": {
            "immediate": [],
            "session": [],
            "project": [],
            "global": []
        }
    }
    
    # Export each level
    for level in ["immediate", "session", "project", "global"]:
        results = bridge.context_manager.search_context("", n_results=100)
        for result in results:
            if result['level'] == level:
                output['contexts'][level].append({
                    "content": result['content'],
                    "metadata": result['metadata']
                })
    
    # Save to file
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"✓ Exported context to {output_file}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "export":
        # Batch export mode
        if len(sys.argv) != 4:
            print("Usage: python use_with_claude.py export <project> <output_file>")
            sys.exit(1)
        batch_export(sys.argv[2], sys.argv[3])
    else:
        # Interactive mode
        interactive_session()