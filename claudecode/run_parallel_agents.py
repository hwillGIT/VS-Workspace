#!/usr/bin/env python3
"""
Run Parallel Agents Command
Easy-to-use command for launching multiple Claude Code subagents in parallel.
"""

import asyncio
import argparse
import sys
from pathlib import Path
from claude_code_integration import (
    ClaudeCodeAgentLauncher, 
    TaskTemplates,
    run_code_review_workflow,
    run_research_workflow,
    run_optimization_workflow
)
from parallel_agent_launcher import AgentTask
import json


def create_quick_tasks():
    """Create some quick example tasks for demonstration."""
    return [
        AgentTask(
            task_id="trading_system_overview",
            agent_type="general-purpose",
            description="Analyze the trading system architecture and components",
            prompt="Analyze the overall trading system architecture. Provide an overview of the multi-agent design, identify key components, and suggest areas for improvement.",
            inputs={"system_type": "multi-agent trading system"},
            priority=1
        ),
        AgentTask(
            task_id="risk_management_review",
            agent_type="code-reviewer",
            description="Review risk management implementation",
            prompt="Review the risk management components of the trading system. Focus on risk calculation algorithms, position sizing, and risk limit enforcement.",
            inputs={"component": "risk_management", "focus": "algorithms and limits"},
            priority=1
        ),
        AgentTask(
            task_id="performance_optimization",
            agent_type="code-architect",
            description="Analyze performance optimization opportunities",
            prompt="Identify performance optimization opportunities in the trading system. Consider latency, throughput, memory usage, and scalability improvements.",
            inputs={"optimization_type": "performance", "metrics": ["latency", "throughput", "memory"]},
            priority=2,
            dependencies=["trading_system_overview"]
        )
    ]


async def run_custom_workflow(config_file: str, max_agents: int = 3):
    """Run a custom workflow from a configuration file."""
    launcher = ClaudeCodeAgentLauncher(max_concurrent_agents=max_agents)
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        tasks = []
        for task_data in config.get('tasks', []):
            task = AgentTask(**task_data)
            tasks.append(task)
        
        if not tasks:
            print("No tasks found in configuration file.")
            return None
        
        launcher.add_batch_tasks(tasks)
        
        print(f"Loaded {len(tasks)} tasks from {config_file}")
        print("Starting parallel execution...")
        
        def progress_callback(result, completed, total):
            print(f"[{completed}/{total}] {result.task_id} - {result.status} ({result.execution_time:.1f}s)")
        
        results = await launcher.execute_parallel(progress_callback=progress_callback)
        
        return {
            "summary": launcher.get_execution_summary(),
            "detailed_results": results,
            "config_file": config_file
        }
        
    except FileNotFoundError:
        print(f"Configuration file not found: {config_file}")
        return None
    except json.JSONDecodeError as e:
        print(f"Invalid JSON in configuration file: {e}")
        return None
    except Exception as e:
        print(f"Error running custom workflow: {e}")
        return None


async def run_quick_demo():
    """Run a quick demonstration with predefined tasks."""
    launcher = ClaudeCodeAgentLauncher(max_concurrent_agents=3)
    
    tasks = create_quick_tasks()
    launcher.add_batch_tasks(tasks)
    
    print("Running quick demo with 3 predefined tasks...")
    print("Tasks:")
    for task in tasks:
        deps_str = f" (depends on: {', '.join(task.dependencies)})" if task.dependencies else ""
        print(f"  - {task.task_id}: {task.description}{deps_str}")
    
    print("\nStarting execution...")
    
    def progress_callback(result, completed, total):
        status_emoji = "✅" if result.status == "success" else "❌"
        print(f"{status_emoji} [{completed}/{total}] {result.task_id} completed in {result.execution_time:.1f}s")
    
    results = await launcher.execute_parallel(progress_callback=progress_callback)
    
    return {
        "summary": launcher.get_execution_summary(),
        "detailed_results": results
    }


def print_results_summary(results):
    """Print a formatted summary of results."""
    if not results:
        return
    
    summary = results["summary"]
    
    print("\n" + "="*60)
    print("EXECUTION SUMMARY")
    print("="*60)
    
    print(f"📊 Total tasks: {summary['total_tasks']}")
    print(f"✅ Successful: {summary['successful']}")
    print(f"❌ Failed: {summary['failed']}")
    print(f"⏰ Timed out: {summary['timed_out']}")
    print(f"📈 Success rate: {summary['success_rate']}")
    print(f"⏱️  Total execution time: {summary['total_execution_time']}")
    print(f"⚡ Average execution time: {summary['average_execution_time']}")
    
    if "results_by_agent_type" in summary and summary["results_by_agent_type"]:
        print(f"\n📋 Results by agent type:")
        for agent_type, stats in summary["results_by_agent_type"].items():
            print(f"   {agent_type}: {stats['success']} success, {stats['failed']} failed, {stats['timeout']} timeout")


def save_results(results, output_file: str):
    """Save results to a file."""
    if not results:
        return
    
    try:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Results saved to: {output_file}")
    except Exception as e:
        print(f"❌ Error saving results: {e}")


async def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Run multiple Claude Code subagents in parallel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick demo with predefined tasks
  python run_parallel_agents.py demo

  # Code review workflow
  python run_parallel_agents.py code-review file1.py file2.py file3.py

  # Research workflow
  python run_parallel_agents.py research "AI in trading" "Risk management" "Market microstructure"

  # System optimization workflow
  python run_parallel_agents.py optimize "Data Pipeline" "ML Models" "Risk Engine"

  # Custom workflow from config file
  python run_parallel_agents.py custom agent_config_template.json --max-agents 5

  # Generate a template config file
  python run_parallel_agents.py generate-config custom_tasks.json
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run quick demo with predefined tasks')
    
    # Code review command
    review_parser = subparsers.add_parser('code-review', help='Comprehensive code review workflow')
    review_parser.add_argument('files', nargs='+', help='Files to review')
    
    # Research command
    research_parser = subparsers.add_parser('research', help='Research and analysis workflow')
    research_parser.add_argument('topics', nargs='+', help='Topics to research')
    
    # Optimization command
    optimize_parser = subparsers.add_parser('optimize', help='System optimization workflow')
    optimize_parser.add_argument('components', nargs='+', help='Components to analyze and optimize')
    
    # Custom workflow command
    custom_parser = subparsers.add_parser('custom', help='Run custom workflow from config file')
    custom_parser.add_argument('config_file', help='JSON configuration file')
    custom_parser.add_argument('--max-agents', type=int, default=3, help='Maximum concurrent agents')
    
    # Generate config command
    generate_parser = subparsers.add_parser('generate-config', help='Generate a template configuration file')
    generate_parser.add_argument('output_file', help='Output file name for the template')
    
    # Global options
    parser.add_argument('--output', '-o', default=None, help='Output file for results (auto-generated if not specified)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Handle generate-config command
    if args.command == 'generate-config':
        template_config = {
            "description": "Custom agent tasks configuration",
            "max_concurrent_agents": 3,
            "tasks": [
                {
                    "task_id": "example_task_1",
                    "agent_type": "general-purpose",
                    "description": "Example research task",
                    "prompt": "Research and analyze the specified topic. Provide comprehensive insights and recommendations.",
                    "inputs": {"topic": "your_topic_here"},
                    "priority": 1,
                    "timeout": 300,
                    "retry_count": 2,
                    "dependencies": []
                },
                {
                    "task_id": "example_task_2",
                    "agent_type": "code-reviewer",
                    "description": "Example code review task",
                    "prompt": "Review the specified code for quality, security, and performance. Provide actionable recommendations.",
                    "inputs": {"file_path": "path/to/your/file.py"},
                    "priority": 1,
                    "timeout": 180,
                    "retry_count": 1,
                    "dependencies": []
                }
            ]
        }
        
        try:
            with open(args.output_file, 'w') as f:
                json.dump(template_config, f, indent=2)
            print(f"✅ Template configuration saved to: {args.output_file}")
            print("Edit the file to customize your tasks and then run:")
            print(f"python run_parallel_agents.py custom {args.output_file}")
        except Exception as e:
            print(f"❌ Error saving template: {e}")
        return
    
    # Execute the appropriate workflow
    print("🚀 Claude Code Parallel Agent Launcher")
    print("="*50)
    
    results = None
    
    try:
        if args.command == 'demo':
            print("Running demonstration workflow...")
            results = await run_quick_demo()
            
        elif args.command == 'code-review':
            print(f"Starting code review workflow for {len(args.files)} files...")
            results = await run_code_review_workflow(args.files)
            
        elif args.command == 'research':
            print(f"Starting research workflow for {len(args.topics)} topics...")
            results = await run_research_workflow(args.topics)
            
        elif args.command == 'optimize':
            print(f"Starting optimization workflow for {len(args.components)} components...")
            results = await run_optimization_workflow(args.components)
            
        elif args.command == 'custom':
            print(f"Running custom workflow from {args.config_file}...")
            results = await run_custom_workflow(args.config_file, args.max_agents)
        
        # Print summary
        print_results_summary(results)
        
        # Save results
        if results:
            if args.output:
                output_file = args.output
            else:
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = f"agent_results_{args.command}_{timestamp}.json"
            
            save_results(results, output_file)
        
        # Print detailed results if verbose
        if args.verbose and results and "detailed_results" in results:
            print("\n" + "="*60)
            print("DETAILED RESULTS")
            print("="*60)
            
            for task_id, result in results["detailed_results"].items():
                print(f"\n📋 Task: {task_id}")
                print(f"   Status: {result.status}")
                print(f"   Agent: {result.agent_type}")
                print(f"   Time: {result.execution_time:.2f}s")
                
                if result.status == "success" and result.result:
                    print("   Result summary:")
                    if isinstance(result.result, dict):
                        for key, value in result.result.items():
                            if isinstance(value, (str, int, float)):
                                print(f"     {key}: {value}")
                            elif isinstance(value, list) and len(value) <= 3:
                                print(f"     {key}: {value}")
                            else:
                                print(f"     {key}: [complex data structure]")
                
                if result.error:
                    print(f"   Error: {result.error}")
        
        print(f"\n🎉 Workflow completed!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Workflow interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error running workflow: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())