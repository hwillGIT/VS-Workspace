# Webtester Application Redesign - Architecture Document

## 1. Overview

This document outlines the proposed architecture for the redesigned `webtester` application. The goal is to create a self-contained web application for exploring the effectiveness of Test-Driven Development (TDD) by generating code from tests and tests from code, allowing comparison, and displaying metrics.

## 2. High-Level Architecture

A **Client-Server architecture** will be used:

-   **Frontend (Client):** Responsible for the user interface, interactions, and communication with the backend.
-   **Backend (Server):** Responsible for core logic, code/test generation, diffing, metrics calculation, and data management.

## 3. Technology Stack

-   **Frontend:** React (JavaScript framework)
-   **Backend:** Python (using Flask or Django web framework - *Flask chosen for simplicity initially*)
-   **Code/Test Generation:** Existing Python LLM integration components (`llm_integration_api`, `expand_task.py`, etc.) accessed via the backend.
-   **Diffing Engine:** Python `difflib` module (or similar library) integrated into the backend.
-   **Metrics Calculation:** Python logic within the backend.
-   **Data Storage:** File system managed by the backend server (storing test specs, generated code/tests per session/project).

## 4. Component Diagram

```mermaid
graph TD
    subgraph Frontend (Browser - React)
        UI[User Interface Components - Canvases, Tabs, Metrics Display]
        StateMgmt[State Management (React Context API / Redux)]
        APIClient[API Client (fetch/axios)]
    end

    subgraph Backend (Python - Flask)
        API[RESTful API Endpoints (/generate_code, /generate_tests, /diff, /metrics)]
        Controller[Request Handling & Orchestration]
        GenService[Code/Test Generation Service]
        DiffService[Diffing Service]
        MetricService[Metrics Calculation Service]
        StorageMgr[Iteration Storage Manager (File System)]
    end

    subgraph External Services
        LLM_API[LLM Integration API (Python)]
    end

    UI -- Interacts --> StateMgmt
    UI -- Uses --> APIClient
    APIClient -- HTTP Requests --> API
    API -- Routes to --> Controller
    Controller -- Calls --> GenService
    Controller -- Calls --> DiffService
    Controller -- Calls --> MetricService
    Controller -- Uses --> StorageMgr
    GenService -- Calls --> LLM_API
    GenService -- Uses --> StorageMgr
    DiffService -- Uses --> StorageMgr
    MetricService -- Uses --> StorageMgr
```

## 5. Interfaces

-   **Frontend <-> Backend (REST API):**
    -   `POST /generate_code`: Input test specs, returns generated code.
    -   `POST /generate_tests`: Input code, returns generated tests.
    -   `POST /diff`: Input two iteration IDs (code vs code or test vs test), returns diff results.
    -   `GET /iterations`: Returns list of available code/test iterations.
    -   `GET /metrics`: Returns calculated TDD metrics for the current session/project.
    -   *(Specific request/response formats TBD)*
-   **Backend <-> LLM Service:** Direct Python function calls.
-   **Backend <-> Storage:** File system operations (read/write files for iterations).

## 6. Data Storage Structure (Initial Proposal)

```
webtester_data/
└── session_id_1/
    ├── code/
    │   ├── iteration_1.py
    │   └── iteration_2.py
    ├── tests/
    │   ├── iteration_1_tests.py
    │   └── iteration_2_tests.py
    └── metadata.json  # (Optional: session info, metrics)
└── session_id_2/
    └── ...
```

## 7. Next Steps

-   Refine API endpoint details (request/response formats).
-   Set up the basic project structure for both Frontend (React) and Backend (Flask).
-   Begin implementation of core components.

*[2025-04-08 21:55:39] - Initial architecture documented.*