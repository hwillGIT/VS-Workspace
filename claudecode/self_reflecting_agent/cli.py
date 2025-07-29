"""
Command Line Interface for Self-Reflecting Agent System

Provides global access to the agent system from any directory,
with automatic project context detection and CLAUDE.md integration.
"""

import asyncio
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import yaml

from .main import SelfReflectingAgent
from .global_manager import GlobalAgentManager


def find_project_root() -> Path:
    """Find the project root by looking for common project indicators."""
    
    current_dir = Path.cwd()
    indicators = [
        '.git',
        'package.json',
        'requirements.txt',
        'pyproject.toml',
        'Cargo.toml',
        'go.mod',
        'pom.xml',
        'build.gradle',
        'CLAUDE.md'
    ]
    
    # Walk up the directory tree
    for path in [current_dir] + list(current_dir.parents):
        for indicator in indicators:
            if (path / indicator).exists():
                return path
    
    # If no project root found, use current directory
    return current_dir


def detect_project_type(project_root: Path) -> str:
    """Detect the type of project based on files present."""
    
    if (project_root / 'package.json').exists():
        return 'javascript'
    elif (project_root / 'requirements.txt').exists() or (project_root / 'pyproject.toml').exists():
        return 'python'
    elif (project_root / 'Cargo.toml').exists():
        return 'rust'
    elif (project_root / 'go.mod').exists():
        return 'golang'
    elif (project_root / 'pom.xml').exists() or (project_root / 'build.gradle').exists():
        return 'java'
    elif (project_root / '.git').exists():
        return 'git_repository'
    else:
        return 'general'


def load_claude_md(project_root: Path) -> Optional[Dict[str, Any]]:
    """Load and parse CLAUDE.md if it exists."""
    
    claude_md_path = project_root / 'CLAUDE.md'
    if not claude_md_path.exists():
        return None
    
    try:
        content = claude_md_path.read_text(encoding='utf-8')
        
        # Parse CLAUDE.md for structured information
        # This is a simplified parser - could be enhanced
        claude_config = {
            'content': content,
            'project_description': '',
            'technologies': [],
            'requirements': [],
            'constraints': []
        }
        
        lines = content.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if line.startswith('#'):
                current_section = line.lower()
            elif current_section and line:
                if 'project' in current_section or 'description' in current_section:
                    claude_config['project_description'] += line + ' '
                elif 'tech' in current_section or 'stack' in current_section:
                    claude_config['technologies'].append(line)
                elif 'requirement' in current_section:
                    claude_config['requirements'].append(line)
                elif 'constraint' in current_section:
                    claude_config['constraints'].append(line)
        
        return claude_config
        
    except Exception as e:
        print(f"Warning: Could not parse CLAUDE.md: {e}")
        return None


async def handle_task_command(args) -> None:
    """Handle the 'task' command to execute a development task."""
    
    project_root = find_project_root()
    project_type = detect_project_type(project_root)
    claude_config = load_claude_md(project_root)
    
    print(f"🚀 Self-Reflecting Agent - Executing Task")
    print(f"📁 Project Root: {project_root}")
    print(f"🏷️  Project Type: {project_type}")
    
    try:
        # Initialize global agent manager
        manager = GlobalAgentManager()
        
        # Get or create agent for this project
        agent = await manager.get_agent_for_project(
            project_path=str(project_root),
            project_type=project_type,
            claude_config=claude_config
        )
        
        # Prepare task context
        task_context = {
            'project_root': str(project_root),
            'project_type': project_type,
            'working_directory': str(Path.cwd())
        }
        
        if claude_config:
            task_context.update({
                'project_description': claude_config.get('project_description', ''),
                'technologies': claude_config.get('technologies', []),
                'requirements': claude_config.get('requirements', []),
                'constraints': claude_config.get('constraints', [])
            })
        
        if args.requirements:
            task_context['additional_requirements'] = args.requirements
        
        if args.constraints:
            task_context['additional_constraints'] = args.constraints
        
        # Execute the task
        result = await agent.execute_task(
            task_description=args.description,
            requirements=task_context,
            constraints={'max_execution_time': args.timeout} if args.timeout else {}
        )
        
        # Display results
        print("\n" + "="*60)
        print("📊 TASK EXECUTION RESULT")
        print("="*60)
        
        if result.get('status') == 'completed':
            print("✅ Status: Completed Successfully")
            if 'files_created' in result:
                print(f"📝 Files Created: {len(result['files_created'])}")
            if 'files_modified' in result:
                print(f"✏️  Files Modified: {len(result['files_modified'])}")
            if 'summary' in result:
                print(f"📋 Summary: {result['summary']}")
        else:
            print("❌ Status: Failed")
            if 'error' in result:
                print(f"🚨 Error: {result['error']}")
        
        # Save results to project
        if args.save_results:
            results_file = project_root / '.sra_results.json'
            with open(results_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            print(f"💾 Results saved to: {results_file}")
        
    except Exception as e:
        print(f"❌ Task execution failed: {e}")
        sys.exit(1)


async def handle_workflow_command(args) -> None:
    """Handle the 'workflow' command to execute domain workflows."""
    
    project_root = find_project_root()
    project_type = detect_project_type(project_root)
    claude_config = load_claude_md(project_root)
    
    print(f"🔄 Self-Reflecting Agent - Executing Workflow")
    print(f"📁 Project Root: {project_root}")
    print(f"🏷️  Project Type: {project_type}")
    print(f"🎯 Domain: {args.domain}")
    print(f"⚙️  Workflow: {args.workflow}")
    
    try:
        # Initialize global agent manager
        manager = GlobalAgentManager()
        
        # Get or create agent for this project
        agent = await manager.get_agent_for_project(
            project_path=str(project_root),
            project_type=project_type,
            claude_config=claude_config
        )
        
        # Prepare workflow context
        workflow_context = {
            'project_root': str(project_root),
            'project_type': project_type,
            'working_directory': str(Path.cwd()),
            'task': args.description
        }
        
        if claude_config:
            workflow_context.update({
                'project_description': claude_config.get('project_description', ''),
                'technologies': claude_config.get('technologies', []),
                'requirements': claude_config.get('requirements', []),
                'constraints': claude_config.get('constraints', [])
            })
        
        if args.context:
            # Parse JSON context if provided
            try:
                additional_context = json.loads(args.context)
                workflow_context.update(additional_context)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse context JSON: {args.context}")
        
        # Execute the workflow
        result = await agent.execute_domain_workflow(
            domain_name=args.domain,
            workflow_name=args.workflow,
            task_description=args.description,
            task_context=workflow_context
        )
        
        # Display results
        print("\n" + "="*60)
        print("📊 WORKFLOW EXECUTION RESULT")
        print("="*60)
        
        if 'error' not in result:
            print("✅ Status: Completed Successfully")
            print(f"🎯 Workflow: {result.get('workflow', 'Unknown')}")
            print(f"🏗️  Domain: {result.get('domain', 'Unknown')}")
            
            if 'type' in result and result['type'] == 'multi_perspective_planning':
                print("🧠 Type: Multi-Perspective Planning")
                execution_summary = result.get('execution_summary', {})
                print(f"👁️  Perspectives: {', '.join(execution_summary.get('perspectives_used', []))}")
                print(f"📈 Validation Score: {execution_summary.get('validation_score', 0.0):.2f}")
                
                if 'synthesized_plan' in result:
                    plan = result['synthesized_plan']
                    print(f"\n📋 Executive Summary:")
                    print(plan.get('executive_summary', 'No summary available'))
            else:
                execution_summary = result.get('execution_summary', {})
                print(f"🤖 Agents Used: {', '.join(execution_summary.get('agents_used', []))}")
                print(f"⚡ Sequence: {execution_summary.get('sequence', 'Unknown')}")
        else:
            print("❌ Status: Failed")
            print(f"🚨 Error: {result['error']}")
        
        # Save results to project
        if args.save_results:
            results_file = project_root / f'.sra_workflow_{args.domain}_{args.workflow}.json'
            with open(results_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            print(f"💾 Results saved to: {results_file}")
        
    except Exception as e:
        print(f"❌ Workflow execution failed: {e}")
        sys.exit(1)


async def handle_info_command(args) -> None:
    """Handle the 'info' command to show system information."""
    
    project_root = find_project_root()
    project_type = detect_project_type(project_root)
    claude_config = load_claude_md(project_root)
    
    print(f"ℹ️  Self-Reflecting Agent - System Information")
    print("="*60)
    print(f"📁 Project Root: {project_root}")
    print(f"🏷️  Project Type: {project_type}")
    print(f"📍 Current Directory: {Path.cwd()}")
    print(f"📄 CLAUDE.md: {'✅ Found' if claude_config else '❌ Not found'}")
    
    try:
        # Initialize global agent manager
        manager = GlobalAgentManager()
        
        # Get available domains and agents
        agent = await manager.get_agent_for_project(
            project_path=str(project_root),
            project_type=project_type,
            claude_config=claude_config
        )
        
        print(f"\n🌐 Available Domains:")
        domains = agent.list_available_domains()
        for domain in domains:
            print(f"  • {domain}")
            agents = agent.list_domain_agents(domain)
            for agent_name in agents:
                print(f"    - {agent_name}")
        
        print(f"\n📊 Domain Statistics:")
        stats = agent.get_domain_statistics()
        for domain, domain_stats in stats.get('domains', {}).items():
            print(f"  • {domain}: {domain_stats['agent_count']} agents, {domain_stats['workflows']} workflows")
        
        if claude_config:
            print(f"\n📄 CLAUDE.md Configuration:")
            print(f"  • Technologies: {', '.join(claude_config.get('technologies', []))}")
            print(f"  • Requirements: {len(claude_config.get('requirements', []))} items")
            print(f"  • Constraints: {len(claude_config.get('constraints', []))} items")
        
    except Exception as e:
        print(f"❌ Could not retrieve system information: {e}")


def main():
    """Main CLI entry point."""
    
    parser = argparse.ArgumentParser(
        description="Self-Reflecting Claude Code Agent - Global AI Development Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Execute a development task
  sra task "Create a REST API for user management"
  
  # Execute a domain workflow
  sra workflow software_development architecture_review "Review my microservices design"
  
  # Multi-perspective planning
  sra workflow software_development comprehensive_project_planning "Plan an e-commerce platform"
  
  # Show system information
  sra info
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Task command
    task_parser = subparsers.add_parser('task', help='Execute a development task')
    task_parser.add_argument('description', help='Task description')
    task_parser.add_argument('--requirements', help='Additional requirements (JSON string)')
    task_parser.add_argument('--constraints', help='Additional constraints (JSON string)')
    task_parser.add_argument('--timeout', type=int, help='Task timeout in seconds')
    task_parser.add_argument('--save-results', action='store_true', help='Save results to project directory')
    
    # Workflow command
    workflow_parser = subparsers.add_parser('workflow', help='Execute a domain workflow')
    workflow_parser.add_argument('domain', help='Domain name (e.g., software_development)')
    workflow_parser.add_argument('workflow', help='Workflow name (e.g., architecture_review)')
    workflow_parser.add_argument('description', help='Task description for the workflow')
    workflow_parser.add_argument('--context', help='Additional context (JSON string)')
    workflow_parser.add_argument('--save-results', action='store_true', help='Save results to project directory')
    
    # Info command  
    info_parser = subparsers.add_parser('info', help='Show system information')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Run the appropriate command
    try:
        if args.command == 'task':
            asyncio.run(handle_task_command(args))
        elif args.command == 'workflow':
            asyncio.run(handle_workflow_command(args))
        elif args.command == 'info':
            asyncio.run(handle_info_command(args))
    except KeyboardInterrupt:
        print("\n👋 Operation cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()