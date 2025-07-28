# Product Context

This file provides a high-level overview of the project and the expected product that will be created. Initially it is based upon projectBrief.md (if provided) and all other available project-related information in the working directory. This file is intended to be updated as the project evolves, and should be used to inform all other modes of the project's goals and context.
2025-04-08 01:41:38 - Log of updates made will be appended as footnotes to the end of this file.

*   A self-contained web application with the following components:

    ## Technology Stack
    ### Frontend UI
    - React + TypeScript
    - Monaco Editor (VS Code in browser)
    - Jest/Vitest for UI tests
    - WebAssembly for client-side validation
    
    ### LLM Integration Layer
    - Python FastAPI backend
    - LangChain framework for LLM abstraction (Supports OpenAI, Anthropic, Gemini)
    - Redis for rate limiting
    - Prometheus/Grafana for monitoring
    
    ### Code Validation Engine
    - Tree-sitter for syntax analysis
    - Semgrep for security scanning
    - Custom rule engine in Python
    - Language Server Protocol integration
    
    ### Test Execution Environment
    - Docker-in-Docker for isolation
    - Pyodide for browser-based execution
    - pytest/unittest/jest support
    - Custom result parser
    
    ### Reporting System
    - React-based dashboard
    - Diff2Html for comparisons
    - IndexedDB for browser storage
    - PDFMake for PDF exports

    ## Implementation Status
    
    ### Completed Phases
    1. Core LLM Integration (v1.2 - Gemini support pending)
    - Supports OpenAI GPT-4/4o and Anthropic Claude 3 (via existing adapters)
    - Google Gemini support planned (requires new adapter)
    - Dynamic rate limiting with Redis
    - Basic payload validation
    
    2. Code Validation Engine (v0.9)
    - Tree-sitter based syntax validation
    - Semgrep security rules integration
    - Test compatibility checking
    
    ### Current Focus
    3. Execution Environment (v0.5)
    - Docker-in-Docker isolation
    - Python/JavaScript execution support
    - Basic metrics collection
    
    ### Future Roadmap
    4. Enhanced Reporting System
    - Real-time WebSocket updates
    - Comparative analysis dashboard
    - Multi-format export (PDF/HTML/JSON)
    
    5. Language Expansion
    - Go and Rust support
    - WASM-based execution environments

    ## Key Decisions
    - Use browser-based execution where possible for security
    - Leverage WebAssembly for heavy computations
    - Standardize on JSON API for all components
    - Implement plugin architecture for LLM providers (OpenAI, Anthropic, Gemini)

## Project Goal

*   A tool to verify the correctness and accuracy of test-driven generated code, and the ability of AI to generate code from tests, and tests from code.

## Key Features

*   Generate code based on provided test cases.
*   Generate test cases based on provided code.
*   Execute generated tests against the generated code.
*   Provide a report comparing expected vs. actual results.

## Overall Architecture

*