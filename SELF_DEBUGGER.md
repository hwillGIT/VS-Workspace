# SELF_DEBUGGER.md Directives for Anthropic Debugging

This document outlines the directives and conditions for interacting with Anthropic to debug issues encountered by the Gemini CLI agent.

## Purpose

To leverage Anthropic's advanced debugging capabilities to diagnose and propose fixes for errors encountered during Gemini CLI operations, particularly when internal self-correction mechanisms are insufficient.

## Conditions for Invocation

The Gemini CLI agent will invoke Anthropic for debugging under the following conditions:

1.  **Persistent Errors:** When a task or operation consistently fails after multiple internal retry attempts or self-correction cycles.
2.  **Unclear Error Messages:** When the error message is ambiguous, generic, or does not provide sufficient information for internal diagnosis.
3.  **Complex Logical Failures:** When the failure appears to stem from a deep logical flaw in the agent's plan or execution, rather than a simple syntax or environmental issue.
4.  **Tool Interaction Failures:** Specifically, when interactions with external tools (like `run_shell_command`, `write_file`, `read_file`, `glob`, `search_file_content`, `web_fetch`, `replace`, `list_directory`) produce unexpected errors or behavior that cannot be resolved internally.
5.  **Integration Challenges:** When attempting to integrate with existing project components (e.g., Neo4j, ChromaDB, parallel agents) results in unforeseen issues.

## Debugging Protocol

When invoking Anthropic for debugging, the Gemini CLI agent will provide the following information:

1.  **Problem Description:** A clear and concise statement of the issue, including what was attempted, what was expected, and what actually happened.
2.  **Error Message & Stack Trace:** The full error message and any available stack trace.
3.  **Relevant Code Snippets:** The code that was executed, particularly the part directly related to the error.
4.  **Contextual Information:**
    *   Current working directory.
    *   Operating system.
    *   Relevant project files or configurations (e.g., `.env`, `package.json`, `requirements.txt`, `Dockerfile`).
    *   Any previous attempts at self-correction and their outcomes.
    *   The specific goal or task the agent was trying to achieve when the error occurred.
5.  **Specific Questions for Anthropic:** These questions will be posed via an API call using the `call_anthropic_api.py` script.
    *   "Based on the provided information, what is the root cause of this error?"
    *   "Please propose a detailed fix, including any necessary code modifications or environmental adjustments."
    *   "Are there any alternative approaches to achieve the original goal that might avoid this type of error?"

## Post-Debugging Actions

Upon receiving a response from Anthropic, the Gemini CLI agent will:

1.  **Analyze Proposed Fixes:** Evaluate the suggested solutions for feasibility, adherence to project standards, and potential side effects.
2.  **Implement Fixes:** Apply the most appropriate fix.
3.  **Verify Resolution:** Re-run the problematic operation to confirm the issue is resolved.
4.  **Document Learning:** Update internal knowledge base or relevant project documentation (e.g., `memory-bank/decisionLog.md`) with the problem, solution, and lessons learned.

---

## Specific Query for Anthropic (Initial Problem Context)

**Context:** I am the Gemini CLI agent, currently working on integrating my internal memory with the project's Neo4j and ChromaDB databases. I have encountered persistent issues with executing a Python script (`temp_chroma_add.py`) designed to add data to ChromaDB, specifically related to argument parsing and path handling when invoked via `run_shell_command`.

**Question for Anthropic:**

"You, as Anthropic, have successfully used this framework prior to me. I am trying to use the `temp_chroma_add.py` script to write to ChromaDB, but I'm facing persistent `KeyError: 'MEMORY'` and argument parsing issues when invoking it via `run_shell_command`. The `sys.argv` output shows incorrect splitting of arguments despite quoting. How did you successfully handle executing Python scripts with multiple arguments (especially those containing spaces or special characters) using `run_shell_command` in this environment, particularly when writing to databases like Neo4j or ChromaDB? What was your established mechanism for passing complex data to scripts, and how did you ensure reliable execution?"
