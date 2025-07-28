import json
from expand_task import expand_task

def main():
    # Demonstrate basic task expansion
    print("=== Basic Task Expansion ===")
    task = "Develop a new web application for task management"
    result = expand_task(task)
    print(json.dumps(result, indent=2))

    # Demonstrate custom subtask count
    print("\n=== Custom Subtask Count ===")
    task = "Create a machine learning model for predictive analytics"
    result = expand_task(task, num_subtasks=4)
    print(json.dumps(result, indent=2))

    # Demonstrate focus prompt
    print("\n=== Focus Prompt ===")
    task = "Design an e-commerce platform"
    focus = "Prioritize user experience and security"
    result = expand_task(task, focus_prompt=focus)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()