"""
Basic usage example for the Self-Reflecting Claude Code Agent.

This example demonstrates how to set up and use the agent system
for simple development tasks.
"""

import asyncio
import logging
from pathlib import Path

from ..main import SelfReflectingAgent


async def basic_usage_example():
    """
    Basic usage example showing how to:
    1. Initialize the agent system
    2. Execute a simple development task
    3. Get system status and metrics
    """
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("=== Self-Reflecting Agent Basic Usage Example ===")
    
    try:
        # 1. Initialize the agent system
        logger.info("Step 1: Initializing agent system...")
        
        agent = SelfReflectingAgent(
            project_path="./example_project",
            enable_memory=True,
            enable_self_improvement=True
        )
        
        # Initialize all components
        initialized = await agent.initialize()
        if not initialized:
            logger.error("Failed to initialize agent system")
            return
        
        logger.info("✅ Agent system initialized successfully")
        
        # 2. Execute a simple development task
        logger.info("Step 2: Executing development task...")
        
        task_description = """
        Create a simple Python library for calculating mathematical operations.
        The library should include:
        - Basic arithmetic operations (add, subtract, multiply, divide)
        - Advanced operations (power, square root, factorial)
        - Input validation and error handling
        - Unit tests for all functions
        - Documentation with usage examples
        """
        
        requirements = {
            "language": "python",
            "testing_framework": "pytest",
            "documentation_format": "docstrings",
            "code_style": "PEP 8"
        }
        
        constraints = {
            "max_files": 10,
            "max_lines_per_file": 200,
            "no_external_dependencies": True
        }
        
        result = await agent.execute_task(
            task_description=task_description,
            requirements=requirements,
            constraints=constraints
        )
        
        logger.info(f"✅ Task execution completed with status: {result.get('status')}")
        logger.info(f"📊 Completion percentage: {result.get('project_summary', {}).get('completion_percentage', 0):.1f}%")
        
        # 3. Display execution results
        if result.get("status") == "completed":
            logger.info("📁 Generated deliverables:")
            deliverables = result.get("project_summary", {}).get("deliverables", [])
            for deliverable in deliverables:
                logger.info(f"   - {deliverable}")
            
            # Show quality gates
            quality_gates = result.get("project_summary", {}).get("quality_gates", {})
            logger.info("🔍 Quality gates:")
            for gate, passed in quality_gates.items():
                status = "✅" if passed else "❌"
                logger.info(f"   {status} {gate}")
        
        # 4. Get system status and metrics
        logger.info("Step 3: Getting system status...")
        
        status = agent.get_system_status()
        logger.info(f"🏗️  System initialized: {status['initialized']}")
        logger.info(f"📁 Project path: {status['project_path']}")
        
        # Agent metrics
        logger.info("🤖 Agent metrics:")
        for agent_id, metrics in status.get("agents", {}).items():
            tasks_completed = metrics.get("tasks_completed", 0)
            success_rate = metrics.get("success_rate", 0.0)
            logger.info(f"   {agent_id}: {tasks_completed} tasks, {success_rate:.1%} success rate")
        
        # 5. Add some knowledge to the system
        logger.info("Step 4: Adding knowledge to the system...")
        
        knowledge_content = """
        Python Best Practices for Mathematical Libraries:
        
        1. Use descriptive function names that clearly indicate their purpose
        2. Implement comprehensive input validation to handle edge cases
        3. Use type hints for better code documentation and IDE support
        4. Include detailed docstrings with examples for all public functions
        5. Handle special cases like division by zero, negative numbers for square root
        6. Use appropriate data types (int, float, Decimal) based on precision needs
        7. Implement unit tests covering normal cases, edge cases, and error conditions
        8. Consider performance implications for computationally intensive operations
        """
        
        await agent.add_knowledge(
            content=knowledge_content,
            source="best_practices",
            metadata={"topic": "python_math_libraries", "type": "guidelines"}
        )
        
        logger.info("✅ Knowledge added to system")
        
        # 6. Search for relevant knowledge
        logger.info("Step 5: Searching knowledge base...")
        
        search_results = await agent.search_knowledge(
            query="Python input validation for mathematical functions",
            max_results=3
        )
        
        logger.info(f"🔍 Found {len(search_results)} relevant knowledge items:")
        for i, result in enumerate(search_results, 1):
            logger.info(f"   {i}. Score: {result.get('score', 0):.3f}")
            logger.info(f"      Content: {result.get('content', '')[:100]}...")
        
        # 7. Export system state
        logger.info("Step 6: Exporting system state...")
        
        export_path = "./agent_system_export.json"
        exported = await agent.export_system_state(export_path)
        
        if exported:
            logger.info(f"✅ System state exported to: {export_path}")
        
        logger.info("=== Basic Usage Example Completed Successfully ===")
        
        # 8. Shutdown gracefully
        await agent.shutdown()
        
    except Exception as e:
        logger.error(f"❌ Example failed: {e}")
        raise


async def demonstrate_agent_interaction():
    """
    Demonstrate direct interaction with individual agents.
    """
    
    logger = logging.getLogger(__name__)
    logger.info("=== Agent Interaction Demonstration ===")
    
    try:
        # Initialize system
        agent = SelfReflectingAgent()
        await agent.initialize()
        
        # Direct interaction with researcher agent
        logger.info("🔬 Researcher Agent - Technology Analysis:")
        
        research_task = {
            "type": "technology_analysis",
            "title": "Analyze FastAPI for REST API Development",
            "technology": "FastAPI",
            "use_case": "Building high-performance REST APIs",
            "alternatives": ["Flask", "Django REST Framework", "Tornado"]
        }
        
        research_result = await agent.agents["researcher"].process_task(research_task)
        
        if research_result.get("status") == "completed":
            logger.info("✅ Research completed:")
            logger.info(f"   Technology: {research_result.get('technology', 'N/A')}")
            logger.info(f"   Analysis depth: {research_result.get('analysis_depth', 'N/A')}")
        
        # Direct interaction with coder agent
        logger.info("💻 Coder Agent - Simple Implementation:")
        
        coding_task = {
            "type": "implementation",
            "title": "Create Utility Function",
            "specification": "Create a Python function that validates email addresses using regex",
            "requirements": {
                "language": "python",
                "include_tests": True,
                "validation_strict": True  
            },
            "context": {"style": "functional", "error_handling": "exceptions"}
        }
        
        coding_result = await agent.agents["coder"].process_task(coding_task)
        
        if coding_result.get("status") == "completed":
            logger.info("✅ Implementation completed:")
            code_files = coding_result.get("code_files", [])
            logger.info(f"   Generated {len(code_files)} code files")
            for file_info in code_files:
                logger.info(f"   - {file_info.get('path', 'unknown')}: {file_info.get('description', 'N/A')}")
        
        # Direct interaction with reviewer agent
        logger.info("🔍 Reviewer Agent - Code Review:")
        
        sample_code = '''
def validate_email(email):
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if re.match(pattern, email):
        return True
    else:
        return False
'''
        
        review_task = {
            "type": "code_review",
            "title": "Review Email Validation Function",
            "code": sample_code,
            "file_path": "email_validator.py"
        }
        
        review_result = await agent.agents["reviewer"].process_task(review_task)
        
        if review_result.get("status") == "completed":
            logger.info("✅ Code review completed:")
            logger.info(f"   Overall score: {review_result.get('overall_score', 'N/A')}/10")
            logger.info(f"   Total findings: {review_result.get('total_findings', 0)}")
            quality_gate = review_result.get('quality_gate_passed', False)
            logger.info(f"   Quality gate: {'✅ Passed' if quality_gate else '❌ Failed'}")
        
        await agent.shutdown()
        logger.info("=== Agent Interaction Demonstration Completed ===")
        
    except Exception as e:
        logger.error(f"❌ Demonstration failed: {e}")
        raise


if __name__ == "__main__":
    # Run the basic usage example
    asyncio.run(basic_usage_example())
    
    # Run the agent interaction demonstration
    asyncio.run(demonstrate_agent_interaction())