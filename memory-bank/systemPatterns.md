# System Patterns: [Project Name]

## 1. System Architecture

Describe the high-level architecture of the project. What architectural style is being used (e.g., microservices, monolithic, serverless)? What are the main components or modules?

- Choose and describe the overall system architecture.
- Identify the key components and their responsibilities.
- Explain how the components interact with each other.
- Diagram the system architecture if applicable (using Mermaid or similar).

## 2. Design Patterns

Document the key design patterns being used in the project. Why were these patterns chosen? How do they contribute to the project's goals (e.g., maintainability, scalability, flexibility)?

- List the major design patterns applied in the project.
- For each pattern, explain its purpose and benefits in this context.
- Provide examples of where these patterns are used in the codebase.

## 3. Component Relationships

If applicable, describe the relationships between different components or modules in more detail. How do they depend on each other? What are the communication pathways?

- Elaborate on the dependencies and interactions between components.
- Describe data flow and communication mechanisms.
- Diagram component relationships if helpful (using Mermaid or similar).


## 4. Workflow Patterns and Tools

[2025-04-08 19:25:44] - Added Automated Task Expansion capability.

*   **Tool: `expand_task`**
    *   **Description:** Takes a high-level task description and uses an LLM to break it down into a specified number of smaller, actionable subtasks. Can optionally focus the expansion with a prompt or use research-backed generation. Intended primarily for use by Orchestrator mode.
    *   **Parameters:**
        *   `task_description`: (Required, String) The description of the complex task to expand.
        *   `num_subtasks`: (Optional, Integer) The desired number of subtasks.
        *   `focus_prompt`: (Optional, String) Additional context or focus for the expansion.
        *   `use_research`: (Optional, Boolean) Whether to use research-backed generation.
    *   **Output:** JSON array of subtask description strings.
*   **`progress.md` Structure:**
    *   To support task expansion, `progress.md` will use a hierarchical structure for tracking tasks and their subtasks.
    *   Subtasks will be indented under their parent task and use a dot notation for IDs (e.g., Parent Task ID: `T1`, Subtask IDs: `T1.1`, `T1.2`).

[2025-05-13 20:33:43] - Documentation Standard: Extensive Code Comments
## Standard
* All code will be extensively documented using comments.
* Comments should explain the purpose, logic, and functionality of code sections, functions, classes, etc.
* Comments should be written in a clear and simple manner, assuming the reader is unfamiliar with the specific programming language or codebase.
* Focus on explaining *why* the code does something, not just *what* it does (unless the 'what' is complex).
## Rationale
* Ensures maintainability and makes the codebase accessible to new contributors, regardless of their language proficiency.
* Facilitates understanding of complex logic and design decisions.
* Supports the project's goal of being easily reproducible and understandable.
