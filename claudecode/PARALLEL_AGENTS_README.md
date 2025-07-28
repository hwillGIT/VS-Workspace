# Claude Code Parallel Agent Launcher

A powerful, reusable tool for launching multiple Claude Code subagents in parallel to perform various tasks efficiently. This system enables concurrent execution of code reviews, research tasks, architecture analysis, and custom workflows.

## 🚀 Features

- **Parallel Execution**: Run multiple subagents concurrently with configurable limits
- **Dependency Management**: Support for task dependencies and execution ordering
- **Task Templates**: Predefined workflows for common scenarios
- **Progress Tracking**: Real-time progress monitoring and reporting
- **Flexible Configuration**: JSON-based configuration for custom workflows
- **Error Handling**: Robust error handling with retry mechanisms
- **Results Management**: Comprehensive result collection and reporting

## 📦 Components

### Core Files

- `parallel_agent_launcher.py` - Core parallel execution engine
- `claude_code_integration.py` - Claude Code specific integration and workflows
- `run_parallel_agents.py` - Command-line interface for easy usage
- `agent_config_template.json` - Template for custom task configurations

### Key Classes

- `ParallelAgentLauncher` - Main orchestration engine
- `ClaudeCodeAgentLauncher` - Claude Code specific implementation
- `AgentTask` - Task definition and configuration
- `AgentResult` - Execution result container
- `TaskTemplates` - Predefined workflow templates

## 🛠️ Installation

No additional installation required - works with your existing Claude Code setup.

```bash
# Make sure you're in your Claude Code project directory
cd /path/to/your/claude-code-project

# Copy the parallel agent files to your project
# (files should be in the same directory as this README)
```

## 🎯 Quick Start

### 1. Run a Quick Demo

```bash
python run_parallel_agents.py demo
```

This runs a demonstration with 3 predefined tasks to show how the system works.

### 2. Code Review Workflow

```bash
python run_parallel_agents.py code-review file1.py file2.py file3.py
```

Performs comprehensive code review of multiple files in parallel.

### 3. Research Workflow

```bash
python run_parallel_agents.py research "AI in Trading" "Risk Management" "Market Analysis"
```

Conducts parallel research on multiple topics.

### 4. System Optimization Workflow

```bash
python run_parallel_agents.py optimize "Data Pipeline" "ML Models" "Risk Engine"
```

Analyzes system components for optimization opportunities.

## 📋 Available Commands

### Demo Command
```bash
python run_parallel_agents.py demo
```
Runs a quick demonstration with predefined tasks.

### Code Review Command
```bash
python run_parallel_agents.py code-review <files...>
```
- Performs comprehensive code review
- Analyzes quality, security, performance
- Checks SOLID principles and design patterns
- Provides architectural recommendations

### Research Command
```bash
python run_parallel_agents.py research <topics...>
```
- Conducts comprehensive research on topics
- Provides market analysis and trends
- Offers strategic recommendations
- Synthesizes findings across topics

### Optimization Command
```bash
python run_parallel_agents.py optimize <components...>
```
- Analyzes system components for performance
- Identifies bottlenecks and optimization opportunities
- Provides architectural improvement suggestions
- Creates implementation plans

### Custom Workflow Command
```bash
python run_parallel_agents.py custom config.json --max-agents 5
```
- Runs custom tasks from configuration file
- Supports complex dependency chains
- Configurable concurrency limits

### Generate Config Template
```bash
python run_parallel_agents.py generate-config my_config.json
```
Creates a template configuration file for custom workflows.

## 🔧 Configuration

### Agent Types

The system supports three main agent types:

#### `code-reviewer`
- **Purpose**: Code quality analysis, security review, performance optimization
- **Capabilities**: SOLID principles, design patterns, complexity analysis
- **Default Timeout**: 180 seconds

#### `general-purpose`
- **Purpose**: Research, analysis, documentation, planning
- **Capabilities**: Multi-step tasks, complex analysis, strategic planning
- **Default Timeout**: 300 seconds

#### `code-architect`
- **Purpose**: Architecture design, system planning, integration analysis
- **Capabilities**: System design, architectural patterns, scalability analysis
- **Default Timeout**: 240 seconds

### Task Configuration

Tasks are defined using the `AgentTask` class:

```python
AgentTask(
    task_id="unique_task_identifier",
    agent_type="code-reviewer",  # or "general-purpose", "code-architect"
    description="Brief description of the task",
    prompt="Detailed prompt for the agent",
    inputs={"key": "value"},  # Additional inputs
    priority=1,  # 1 = highest, 5 = lowest
    timeout=300,  # seconds
    retry_count=2,
    dependencies=["other_task_id"]  # Tasks this depends on
)
```

### JSON Configuration Format

```json
{
  "description": "Custom workflow description",
  "max_concurrent_agents": 3,
  "tasks": [
    {
      "task_id": "task_1",
      "agent_type": "code-reviewer",
      "description": "Review main application code",
      "prompt": "Perform comprehensive code review focusing on...",
      "inputs": {
        "file_path": "src/main.py",
        "focus_areas": ["security", "performance"]
      },
      "priority": 1,
      "timeout": 180,
      "retry_count": 2,
      "dependencies": []
    },
    {
      "task_id": "task_2",
      "agent_type": "general-purpose",
      "description": "Research optimization techniques",
      "prompt": "Research latest optimization techniques for...",
      "inputs": {
        "domain": "trading systems",
        "focus": "latency optimization"
      },
      "priority": 2,
      "timeout": 300,
      "retry_count": 1,
      "dependencies": ["task_1"]
    }
  ]
}
```

## 🔄 Workflow Examples

### Example 1: Comprehensive Code Review

```python
from claude_code_integration import run_code_review_workflow

files = [
    "trading_system/main.py",
    "trading_system/risk_management.py",
    "trading_system/data_pipeline.py"
]

results = await run_code_review_workflow(files)
```

This creates:
1. Individual file reviews (parallel execution)
2. Overall architecture analysis (depends on file reviews)
3. Integration pattern analysis
4. Consolidated recommendations

### Example 2: Market Research Analysis

```python
from claude_code_integration import run_research_workflow

topics = [
    "AI in Algorithmic Trading",
    "Real-time Risk Management",
    "Market Microstructure Evolution"
]

results = await run_research_workflow(topics)
```

This creates:
1. Individual topic research (parallel execution)
2. Cross-topic synthesis and strategic analysis
3. Integrated recommendations

### Example 3: System Optimization

```python
from claude_code_integration import run_optimization_workflow

components = [
    "Data Processing Pipeline",
    "ML Model Inference",
    "Risk Calculation Engine",
    "Portfolio Optimization"
]

results = await run_optimization_workflow(components)
```

This creates:
1. Performance analysis per component (parallel execution)
2. Architecture optimization strategy
3. Implementation planning with priorities

## 📊 Results and Reporting

### Execution Summary

Each workflow provides a comprehensive summary:

```python
{
    "total_tasks": 5,
    "successful": 4,
    "failed": 1,
    "timed_out": 0,
    "success_rate": "80.0%",
    "total_execution_time": "45.3s",
    "average_execution_time": "9.1s",
    "results_by_agent_type": {
        "code-reviewer": {"success": 2, "failed": 0, "timeout": 0},
        "general-purpose": {"success": 1, "failed": 1, "timeout": 0},
        "code-architect": {"success": 1, "failed": 0, "timeout": 0}
    }
}
```

### Detailed Results

Each task result includes:

```python
{
    "task_id": "code_review_main",
    "agent_type": "code-reviewer",
    "status": "success",
    "result": {
        "review_summary": "...",
        "findings": {...},
        "recommendations": [...]
    },
    "error": null,
    "execution_time": 12.5,
    "timestamp": "2023-12-01T10:30:45",
    "metadata": {...}
}
```

## ⚡ Performance Features

### Concurrency Control
- Configurable maximum concurrent agents
- Semaphore-based resource management
- Optimal task scheduling

### Dependency Management
- Automatic dependency resolution
- Topological task ordering
- Efficient dependency tracking

### Error Handling
- Configurable retry mechanisms
- Timeout management
- Graceful failure handling
- Partial result recovery

### Progress Monitoring
- Real-time progress callbacks
- Execution time tracking
- Status reporting
- Completion notifications

## 🎛️ Advanced Usage

### Custom Task Creation

```python
from parallel_agent_launcher import AgentTask, ParallelAgentLauncher

# Create custom launcher
launcher = ParallelAgentLauncher(max_concurrent_agents=5)

# Define custom task
custom_task = AgentTask(
    task_id="custom_analysis",
    agent_type="general-purpose",
    description="Custom analysis task",
    prompt="Perform custom analysis of...",
    inputs={"data": "custom_data"},
    priority=1
)

# Add and execute
launcher.add_task(custom_task)
results = await launcher.execute_parallel()
```

### Progress Monitoring

```python
def progress_callback(result, completed, total):
    print(f"Progress: {completed}/{total}")
    print(f"Task {result.task_id} completed: {result.status}")
    if result.status == "success":
        print(f"Execution time: {result.execution_time:.2f}s")

results = await launcher.execute_parallel(progress_callback=progress_callback)
```

### Result Processing

```python
# Process results by status
successful_results = [
    result for result in results.values() 
    if result.status == "success"
]

# Group by agent type
from collections import defaultdict
by_agent_type = defaultdict(list)
for result in results.values():
    by_agent_type[result.agent_type].append(result)

# Calculate statistics
total_time = sum(result.execution_time for result in results.values())
avg_time = total_time / len(results)
```

## 🔍 Troubleshooting

### Common Issues

#### Tasks Taking Too Long
```bash
# Increase timeout in configuration
"timeout": 600  # 10 minutes
```

#### Too Many Concurrent Tasks
```bash
# Reduce concurrent agents
python run_parallel_agents.py custom config.json --max-agents 2
```

#### Dependency Cycles
```bash
# Check task dependencies for circular references
# Dependencies should form a directed acyclic graph (DAG)
```

### Debug Mode

```bash
# Run with verbose output for debugging
python run_parallel_agents.py demo --verbose
```

### Error Recovery

The system automatically handles:
- Task timeouts with configurable retry
- Agent failures with graceful degradation
- Partial completion with available results
- Resource cleanup on interruption

## 📈 Best Practices

### Task Design
1. **Clear Descriptions**: Use descriptive task IDs and descriptions
2. **Appropriate Timeouts**: Set realistic timeouts based on task complexity
3. **Logical Dependencies**: Structure dependencies to maximize parallelism
4. **Focused Prompts**: Write specific, actionable prompts for agents

### Performance Optimization
1. **Concurrency Tuning**: Adjust max_concurrent_agents based on system resources
2. **Task Prioritization**: Use priority levels to ensure important tasks run first
3. **Dependency Minimization**: Minimize dependencies to maximize parallel execution
4. **Resource Management**: Monitor memory and CPU usage during execution

### Error Handling
1. **Retry Configuration**: Set appropriate retry counts for different task types
2. **Timeout Management**: Use realistic timeouts to avoid unnecessary delays
3. **Graceful Degradation**: Design workflows to handle partial failures
4. **Result Validation**: Validate results before using them as dependencies

## 🤝 Integration with Claude Code

This system is designed to integrate seamlessly with Claude Code's Task tool. The actual integration point is in the `_call_subagent` method:

```python
# In actual implementation, replace simulation with:
from claude_code_tools import Task

result = await Task(
    description=task.description,
    prompt=task.prompt,
    subagent_type=task.agent_type
)
```

## 📝 Examples and Templates

### Trading System Analysis Template

```json
{
  "description": "Comprehensive trading system analysis",
  "max_concurrent_agents": 4,
  "tasks": [
    {
      "task_id": "risk_analysis",
      "agent_type": "code-reviewer",
      "description": "Analyze risk management components",
      "prompt": "Review risk management implementation focusing on calculation accuracy, limit enforcement, and error handling.",
      "inputs": {"component": "risk_management"},
      "priority": 1
    },
    {
      "task_id": "performance_analysis",
      "agent_type": "code-reviewer",
      "description": "Analyze system performance characteristics",
      "prompt": "Evaluate system performance including latency, throughput, and resource utilization.",
      "inputs": {"focus": "performance_metrics"},
      "priority": 1
    },
    {
      "task_id": "architecture_review",
      "agent_type": "code-architect",
      "description": "Review overall system architecture",
      "prompt": "Analyze the multi-agent architecture and suggest improvements for scalability and maintainability.",
      "inputs": {"system_type": "multi_agent_trading"},
      "priority": 2,
      "dependencies": ["risk_analysis", "performance_analysis"]
    }
  ]
}
```

This parallel agent launcher provides a powerful foundation for scaling your Claude Code workflows and efficiently managing complex, multi-step analysis tasks.