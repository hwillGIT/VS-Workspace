#!/usr/bin/env python3
"""
Parallel Agent Launcher for Claude Code
A reusable tool for launching multiple subagents concurrently to perform various tasks.
"""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class AgentTask:
    """Represents a task to be executed by a subagent."""
    task_id: str
    agent_type: str
    description: str
    prompt: str
    inputs: Dict[str, Any]
    priority: int = 1  # 1 = highest, 5 = lowest
    timeout: int = 300  # seconds
    retry_count: int = 2
    dependencies: List[str] = None  # Task IDs this task depends on
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


@dataclass
class AgentResult:
    """Represents the result from a subagent execution."""
    task_id: str
    agent_type: str
    status: str  # 'success', 'failed', 'timeout', 'cancelled'
    result: Any
    error: Optional[str]
    execution_time: float
    timestamp: datetime
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ParallelAgentLauncher:
    """Main class for launching and managing parallel subagents."""
    
    def __init__(self, max_concurrent_agents: int = 5):
        self.max_concurrent_agents = max_concurrent_agents
        self.active_tasks: Dict[str, AgentTask] = {}
        self.completed_tasks: Dict[str, AgentResult] = {}
        self.task_queue: List[AgentTask] = []
        self.dependency_graph: Dict[str, List[str]] = {}
        
        # Agent type configurations
        self.agent_configs = {
            'code-reviewer': {
                'description': 'Code review and quality analysis',
                'default_timeout': 180
            },
            'general-purpose': {
                'description': 'General research and analysis tasks',
                'default_timeout': 300
            },
            'code-architect': {
                'description': 'Software architecture and design',
                'default_timeout': 240
            }
        }
    
    def add_task(self, task: AgentTask) -> None:
        """Add a task to the execution queue."""
        self.task_queue.append(task)
        self.active_tasks[task.task_id] = task
        
        # Build dependency graph
        if task.dependencies:
            self.dependency_graph[task.task_id] = task.dependencies
        
        logger.info(f"Added task {task.task_id} ({task.agent_type}): {task.description}")
    
    def add_batch_tasks(self, tasks: List[AgentTask]) -> None:
        """Add multiple tasks at once."""
        for task in tasks:
            self.add_task(task)
    
    async def execute_task(self, task: AgentTask) -> AgentResult:
        """Execute a single task with a subagent."""
        start_time = time.time()
        
        try:
            logger.info(f"Starting task {task.task_id} with {task.agent_type} agent")
            
            # Simulate subagent execution (replace with actual Claude Code Task tool call)
            result = await self._call_subagent(task)
            
            execution_time = time.time() - start_time
            
            agent_result = AgentResult(
                task_id=task.task_id,
                agent_type=task.agent_type,
                status='success',
                result=result,
                error=None,
                execution_time=execution_time,
                timestamp=datetime.now(),
                metadata={'retries': 0}
            )
            
            logger.info(f"Task {task.task_id} completed successfully in {execution_time:.2f}s")
            return agent_result
            
        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            logger.error(f"Task {task.task_id} timed out after {execution_time:.2f}s")
            
            return AgentResult(
                task_id=task.task_id,
                agent_type=task.agent_type,
                status='timeout',
                result=None,
                error=f"Task timed out after {task.timeout} seconds",
                execution_time=execution_time,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Task {task.task_id} failed: {str(e)}")
            
            return AgentResult(
                task_id=task.task_id,
                agent_type=task.agent_type,
                status='failed',
                result=None,
                error=str(e),
                execution_time=execution_time,
                timestamp=datetime.now()
            )
    
    async def _call_subagent(self, task: AgentTask) -> Any:
        """Call the actual Claude Code subagent (placeholder for Task tool integration)."""
        # This is where you would integrate with Claude Code's Task tool
        # For now, simulate the call
        
        # Example of how this would work with actual Task tool:
        # from claude_code_tools import Task
        # 
        # result = Task(
        #     description=task.description,
        #     prompt=task.prompt,
        #     subagent_type=task.agent_type
        # )
        # return result
        
        # Simulation for demonstration
        await asyncio.sleep(1)  # Simulate processing time
        return {
            'analysis': f"Completed {task.description}",
            'recommendations': ['Recommendation 1', 'Recommendation 2'],
            'metadata': task.inputs
        }
    
    def get_ready_tasks(self) -> List[AgentTask]:
        """Get tasks that are ready to execute (dependencies satisfied)."""
        ready_tasks = []
        
        for task in self.task_queue:
            if task.task_id in self.completed_tasks:
                continue  # Already completed
            
            # Check if all dependencies are satisfied
            dependencies_satisfied = all(
                dep_id in self.completed_tasks and 
                self.completed_tasks[dep_id].status == 'success'
                for dep_id in task.dependencies
            )
            
            if dependencies_satisfied:
                ready_tasks.append(task)
        
        # Sort by priority (lower number = higher priority)
        ready_tasks.sort(key=lambda t: t.priority)
        
        return ready_tasks
    
    async def execute_parallel(self, progress_callback: Optional[Callable] = None) -> Dict[str, AgentResult]:
        """Execute all tasks in parallel, respecting dependencies and concurrency limits."""
        logger.info(f"Starting parallel execution of {len(self.task_queue)} tasks")
        
        semaphore = asyncio.Semaphore(self.max_concurrent_agents)
        running_tasks = {}
        
        while len(self.completed_tasks) < len(self.task_queue):
            # Get tasks ready for execution
            ready_tasks = self.get_ready_tasks()
            
            # Start new tasks up to concurrency limit
            for task in ready_tasks:
                if (task.task_id not in running_tasks and 
                    task.task_id not in self.completed_tasks and
                    len(running_tasks) < self.max_concurrent_agents):
                    
                    # Wrap task execution with semaphore
                    async def execute_with_semaphore(t):
                        async with semaphore:
                            return await self.execute_task(t)
                    
                    # Start the task
                    future = asyncio.create_task(execute_with_semaphore(task))
                    running_tasks[task.task_id] = future
            
            # Wait for at least one task to complete
            if running_tasks:
                done, pending = await asyncio.wait(
                    running_tasks.values(),
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # Process completed tasks
                for future in done:
                    result = await future
                    self.completed_tasks[result.task_id] = result
                    
                    # Remove from running tasks
                    task_id_to_remove = None
                    for tid, fut in running_tasks.items():
                        if fut == future:
                            task_id_to_remove = tid
                            break
                    
                    if task_id_to_remove:
                        del running_tasks[task_id_to_remove]
                    
                    # Call progress callback if provided
                    if progress_callback:
                        progress_callback(result, len(self.completed_tasks), len(self.task_queue))
            
            # Small delay to prevent busy waiting
            await asyncio.sleep(0.1)
        
        logger.info(f"All tasks completed. {len(self.completed_tasks)} results available.")
        return self.completed_tasks
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get a summary of the execution results."""
        if not self.completed_tasks:
            return {"message": "No tasks have been executed yet"}
        
        total_tasks = len(self.completed_tasks)
        successful = sum(1 for r in self.completed_tasks.values() if r.status == 'success')
        failed = sum(1 for r in self.completed_tasks.values() if r.status == 'failed')
        timed_out = sum(1 for r in self.completed_tasks.values() if r.status == 'timeout')
        
        total_time = sum(r.execution_time for r in self.completed_tasks.values())
        avg_time = total_time / total_tasks if total_tasks > 0 else 0
        
        return {
            "total_tasks": total_tasks,
            "successful": successful,
            "failed": failed,
            "timed_out": timed_out,
            "success_rate": f"{(successful / total_tasks * 100):.1f}%" if total_tasks > 0 else "0%",
            "total_execution_time": f"{total_time:.2f}s",
            "average_execution_time": f"{avg_time:.2f}s",
            "results_by_agent_type": self._group_results_by_agent_type()
        }
    
    def _group_results_by_agent_type(self) -> Dict[str, Dict[str, int]]:
        """Group results by agent type."""
        grouped = {}
        
        for result in self.completed_tasks.values():
            agent_type = result.agent_type
            if agent_type not in grouped:
                grouped[agent_type] = {'success': 0, 'failed': 0, 'timeout': 0}
            
            grouped[agent_type][result.status] += 1
        
        return grouped
    
    def save_results(self, output_file: Path) -> None:
        """Save execution results to a JSON file."""
        results_data = {
            "execution_summary": self.get_execution_summary(),
            "detailed_results": {
                task_id: asdict(result) 
                for task_id, result in self.completed_tasks.items()
            },
            "execution_timestamp": datetime.now().isoformat()
        }
        
        with open(output_file, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        logger.info(f"Results saved to {output_file}")


# Utility functions for common task patterns

def create_code_review_tasks(file_paths: List[str], review_criteria: List[str] = None) -> List[AgentTask]:
    """Create code review tasks for multiple files."""
    if review_criteria is None:
        review_criteria = ["code quality", "security", "performance", "maintainability"]
    
    tasks = []
    for i, file_path in enumerate(file_paths):
        task = AgentTask(
            task_id=f"code_review_{i+1}",
            agent_type="code-reviewer",
            description=f"Review code quality and suggest improvements for {file_path}",
            prompt=f"Please review the code in {file_path} focusing on {', '.join(review_criteria)}. Provide specific recommendations for improvements.",
            inputs={"file_path": file_path, "criteria": review_criteria},
            priority=1
        )
        tasks.append(task)
    
    return tasks


def create_research_tasks(topics: List[str], research_depth: str = "comprehensive") -> List[AgentTask]:
    """Create research tasks for multiple topics."""
    tasks = []
    for i, topic in enumerate(topics):
        task = AgentTask(
            task_id=f"research_{i+1}",
            agent_type="general-purpose",
            description=f"Research and analyze {topic}",
            prompt=f"Conduct a {research_depth} research on {topic}. Provide key findings, insights, and actionable recommendations.",
            inputs={"topic": topic, "depth": research_depth},
            priority=2
        )
        tasks.append(task)
    
    return tasks


def create_architecture_analysis_tasks(components: List[str]) -> List[AgentTask]:
    """Create architecture analysis tasks for system components."""
    tasks = []
    
    # First, analyze individual components
    for i, component in enumerate(components):
        task = AgentTask(
            task_id=f"analyze_component_{i+1}",
            agent_type="code-architect",
            description=f"Analyze architecture of {component}",
            prompt=f"Analyze the architecture and design patterns used in {component}. Identify strengths, weaknesses, and improvement opportunities.",
            inputs={"component": component},
            priority=1
        )
        tasks.append(task)
    
    # Then, create an integration analysis task that depends on component analyses
    integration_task = AgentTask(
        task_id="integration_analysis",
        agent_type="code-architect",
        description="Analyze system integration and overall architecture",
        prompt="Based on the individual component analyses, evaluate the overall system architecture, integration patterns, and provide recommendations for improvements.",
        inputs={"components": components},
        priority=2,
        dependencies=[f"analyze_component_{i+1}" for i in range(len(components))]
    )
    tasks.append(integration_task)
    
    return tasks


# CLI interface
async def main():
    """Main CLI interface for the parallel agent launcher."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Parallel Agent Launcher for Claude Code")
    parser.add_argument("--config", type=str, help="JSON configuration file with tasks")
    parser.add_argument("--max-agents", type=int, default=3, help="Maximum concurrent agents")
    parser.add_argument("--output", type=str, default="results.json", help="Output file for results")
    
    # Predefined task patterns
    parser.add_argument("--code-review", nargs="+", help="Files to review")
    parser.add_argument("--research", nargs="+", help="Topics to research")
    parser.add_argument("--architecture", nargs="+", help="Components to analyze")
    
    args = parser.parse_args()
    
    launcher = ParallelAgentLauncher(max_concurrent_agents=args.max_agents)
    
    # Load tasks based on arguments
    if args.config:
        # Load tasks from configuration file
        with open(args.config, 'r') as f:
            config = json.load(f)
        
        tasks = [AgentTask(**task_data) for task_data in config.get('tasks', [])]
        launcher.add_batch_tasks(tasks)
    
    elif args.code_review:
        tasks = create_code_review_tasks(args.code_review)
        launcher.add_batch_tasks(tasks)
    
    elif args.research:
        tasks = create_research_tasks(args.research)
        launcher.add_batch_tasks(tasks)
    
    elif args.architecture:
        tasks = create_architecture_analysis_tasks(args.architecture)
        launcher.add_batch_tasks(tasks)
    
    else:
        print("No tasks specified. Use --config, --code-review, --research, or --architecture")
        return
    
    # Progress callback
    def progress_callback(result: AgentResult, completed: int, total: int):
        print(f"Progress: {completed}/{total} - {result.task_id} completed ({result.status})")
    
    # Execute tasks
    print(f"Executing {len(launcher.task_queue)} tasks with up to {args.max_agents} concurrent agents...")
    results = await launcher.execute_parallel(progress_callback=progress_callback)
    
    # Print summary
    summary = launcher.get_execution_summary()
    print("\n" + "="*60)
    print("EXECUTION SUMMARY")
    print("="*60)
    print(f"Total tasks: {summary['total_tasks']}")
    print(f"Successful: {summary['successful']}")
    print(f"Failed: {summary['failed']}")
    print(f"Timed out: {summary['timed_out']}")
    print(f"Success rate: {summary['success_rate']}")
    print(f"Total time: {summary['total_execution_time']}")
    print(f"Average time: {summary['average_execution_time']}")
    
    # Save results
    launcher.save_results(Path(args.output))
    print(f"\nDetailed results saved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())