"""
Enhanced Workflow Manager

Combines the original context workflow with the new command suite capabilities.
Provides a unified interface for both context export and command execution.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

from context_command_suite import ContextAwareCommandSuite, CommandNamespace
from smart_context_export import SmartContextExporter


class EnhancedWorkflowManager:
    """
    Unified workflow manager that combines:
    - Context-aware command execution
    - Smart context export
    - Project memory management
    """
    
    def __init__(self, project: str):
        self.project = project
        self.command_suite = ContextAwareCommandSuite(project)
        self.context_exporter = SmartContextExporter(project)
    
    def execute_workflow(self, workflow_type: str, **kwargs) -> str:
        """Execute a workflow with appropriate context and commands"""
        
        if workflow_type == "code-review":
            return self._code_review_workflow(**kwargs)
        elif workflow_type == "security-audit":
            return self._security_audit_workflow(**kwargs)
        elif workflow_type == "performance-audit":
            return self._performance_audit_workflow(**kwargs)
        elif workflow_type == "debug-session":
            return self._debug_workflow(**kwargs)
        elif workflow_type == "architecture-review":
            return self._architecture_review_workflow(**kwargs)
        elif workflow_type == "test-generation":
            return self._test_generation_workflow(**kwargs)
        else:
            # Fallback to context export
            return self.context_exporter.export_for_work_session(
                workflow_type, 
                f"CONTEXT_{workflow_type}.md"
            )
    
    def _code_review_workflow(self, focus: Optional[str] = None, **kwargs) -> str:
        """Context-aware code review workflow"""
        query = focus or "code review quality patterns"
        
        # Execute command with project context
        result = self.command_suite.execute_command(
            "dev", 
            "code-review",
            context_query=query,
            max_context_items=15
        )
        
        # Save to file
        output_file = "CODE_REVIEW_CONTEXT.md"
        Path(output_file).write_text(result, encoding='utf-8')
        
        print(f"[OK] Generated context-enhanced code review guide: {output_file}")
        print("   - Includes project-specific code patterns")
        print("   - References previous review feedback")
        print("   - Applies established coding standards")
        
        return output_file
    
    def _security_audit_workflow(self, focus: Optional[str] = None, **kwargs) -> str:
        """Context-aware security audit workflow"""
        query = focus or "security vulnerabilities authentication authorization"
        
        result = self.command_suite.execute_command(
            "security",
            "security-audit", 
            context_query=query,
            max_context_items=12
        )
        
        output_file = "SECURITY_AUDIT_CONTEXT.md"
        Path(output_file).write_text(result, encoding='utf-8')
        
        print(f"[OK] Generated context-enhanced security audit: {output_file}")
        print("   - Includes previous security decisions")
        print("   - References project threat model")
        print("   - Applies security best practices")
        
        return output_file
    
    def _performance_audit_workflow(self, focus: Optional[str] = None, **kwargs) -> str:
        """Context-aware performance audit workflow"""
        query = focus or "performance optimization bottlenecks scaling"
        
        result = self.command_suite.execute_command(
            "performance",
            "performance-audit",
            context_query=query,
            max_context_items=12
        )
        
        output_file = "PERFORMANCE_AUDIT_CONTEXT.md"
        Path(output_file).write_text(result, encoding='utf-8')
        
        print(f"[OK] Generated context-enhanced performance audit: {output_file}")
        print("   - Includes previous optimization decisions")
        print("   - References performance benchmarks")
        print("   - Applies scaling patterns")
        
        return output_file
    
    def _debug_workflow(self, issue: Optional[str] = None, **kwargs) -> str:
        """Context-aware debugging workflow"""
        query = issue or "debugging troubleshooting error resolution"
        
        result = self.command_suite.execute_command(
            "debug",
            "troubleshoot",
            context_query=query,
            max_context_items=10
        )
        
        output_file = "DEBUG_SESSION_CONTEXT.md"
        Path(output_file).write_text(result, encoding='utf-8')
        
        print(f"[OK] Generated context-enhanced debug guide: {output_file}")
        print("   - Includes similar past issues")
        print("   - References debugging patterns")
        print("   - Applies project-specific solutions")
        
        return output_file
    
    def _architecture_review_workflow(self, focus: Optional[str] = None, **kwargs) -> str:
        """Context-aware architecture review workflow"""
        query = focus or "architecture design patterns scalability"
        
        result = self.command_suite.execute_command(
            "project",
            "architecture-review",
            context_query=query,
            max_context_items=15
        )
        
        output_file = "ARCHITECTURE_REVIEW_CONTEXT.md"
        Path(output_file).write_text(result, encoding='utf-8')
        
        print(f"[OK] Generated context-enhanced architecture review: {output_file}")
        print("   - Includes architectural decisions")
        print("   - References design patterns")
        print("   - Applies project constraints")
        
        return output_file
    
    def _test_generation_workflow(self, focus: Optional[str] = None, **kwargs) -> str:
        """Context-aware test generation workflow"""
        query = focus or "testing patterns coverage quality"
        
        result = self.command_suite.execute_command(
            "test",
            "generate-tests",
            context_query=query,
            max_context_items=10
        )
        
        output_file = "TEST_GENERATION_CONTEXT.md"
        Path(output_file).write_text(result, encoding='utf-8')
        
        print(f"[OK] Generated context-enhanced test guide: {output_file}")
        print("   - Includes testing patterns")
        print("   - References past bug reports")
        print("   - Applies quality standards")
        
        return output_file
    
    def add_workflow_memory(self, workflow: str, content: str, context_type: str):
        """Add memory from a completed workflow"""
        self.command_suite.add_context_to_workflow(workflow, content, context_type)
        print(f"[OK] Added {context_type} memory to {workflow} workflow")
    
    def list_workflows(self) -> Dict[str, List[str]]:
        """List available workflows"""
        built_in_workflows = {
            "development": ["code-review", "debug-session", "architecture-review"],
            "quality": ["security-audit", "performance-audit", "test-generation"],
            "context": ["work-session", "research", "daily-export"]
        }
        
        command_workflows = self.command_suite.list_commands()
        
        # Combine both
        all_workflows = built_in_workflows.copy()
        all_workflows.update(command_workflows)
        
        return all_workflows


def main():
    """Enhanced workflow CLI"""
    parser = argparse.ArgumentParser(description="Enhanced Context-Aware Workflow Manager")
    parser.add_argument("project", help="Project name")
    parser.add_argument("workflow", nargs="?", help="Workflow type or command")
    parser.add_argument("-f", "--focus", help="Specific focus area or query")
    parser.add_argument("-l", "--list", action="store_true", help="List available workflows")
    parser.add_argument("--add-memory", help="Add memory to workflow", nargs=3, 
                       metavar=("WORKFLOW", "CONTENT", "TYPE"))
    
    args = parser.parse_args()
    
    manager = EnhancedWorkflowManager(args.project)
    
    if args.list:
        workflows = manager.list_workflows()
        print("Available Workflows:")
        for category, wf_list in workflows.items():
            print(f"\n{category.upper()}:")
            for wf in wf_list:
                print(f"  - {wf}")
        return
    
    if args.add_memory:
        workflow, content, context_type = args.add_memory
        manager.add_workflow_memory(workflow, content, context_type)
        return
    
    if not args.workflow:
        parser.error("workflow argument is required unless using --list or --add-memory")
    
    # Execute workflow
    try:
        output_file = manager.execute_workflow(args.workflow, focus=args.focus)
        print(f"\n[SUCCESS] Workflow completed!")
        print(f"Output saved to: {output_file}")
        print("\nNext steps:")
        print("1. Review the generated context")
        print("2. Copy relevant sections to your Claude conversation")
        print("3. Use the enhanced prompts for better results")
        
    except Exception as e:
        print(f"[ERROR] Workflow failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()


# Quick usage examples:
"""
# Code review with context
python enhanced_workflow.py trading_system code-review --focus "authentication module"

# Security audit
python enhanced_workflow.py trading_system security-audit --focus "API endpoints"

# Performance analysis
python enhanced_workflow.py trading_system performance-audit --focus "database queries"

# Debug session
python enhanced_workflow.py trading_system debug-session --focus "websocket connection issues"

# Architecture review
python enhanced_workflow.py trading_system architecture-review --focus "microservices design"

# List all workflows
python enhanced_workflow.py trading_system --list

# Add memory from completed work
python enhanced_workflow.py trading_system --add-memory security "OAuth2 implementation completed" "implementation_note"
"""