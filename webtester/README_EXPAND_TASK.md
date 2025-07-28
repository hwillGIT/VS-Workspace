# Task Expansion Tool

## Overview
The `expand_task` tool uses AI to automatically break down complex tasks into smaller, more manageable subtasks.

## Installation
1. Install required dependencies:
```bash
pip install -r requirements-expand-task.txt
```

2. Set up your Google API key in a `.env` file:
```
GOOGLE_API_KEY=your_api_key_here
```

## Usage

### Basic Usage
```python
from expand_task import expand_task

# Basic task expansion
task = "Develop a new web application"
result = expand_task(task)
print(result['subtasks'])
```

### Advanced Options
```python
# Customize number of subtasks
result = expand_task(task, num_subtasks=4)

# Add a focus prompt
result = expand_task(
    task, 
    focus_prompt="Prioritize user experience and security"
)
```

## Parameters
- `task_description` (str): The main task to be expanded
- `num_subtasks` (int, optional): Number of subtasks to generate (default: 3)
- `focus_prompt` (str, optional): Additional context for task expansion
- `use_research` (bool, optional): Whether to use research-backed generation

## Return Value
Returns a dictionary with:
- `original_task`: The original task description
- `subtasks`: A list of generated subtask descriptions

## Logging
The tool automatically logs task expansion operations to `logs/task_expansion_*.log`

## Testing
Run tests using:
```bash
python -m unittest test_expand_task.py
```

## Dependencies
- google-generativeai
- python-dotenv

## Limitations
- Requires a valid Google API key
- Task expansion quality depends on the AI model's capabilities