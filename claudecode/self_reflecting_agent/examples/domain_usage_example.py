"""
Domain-Specific Agent Usage Examples

Demonstrates how to use domain-specific agents for specialized tasks
like software architecture, security auditing, and performance analysis.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from main import SelfReflectingAgent


async def software_development_examples():
    """Examples using software development domain agents."""
    
    print("=== Software Development Domain Examples ===\n")
    
    # Initialize the agent system
    agent = SelfReflectingAgent(
        config_path="config.yaml",
        project_path="./example_project",
        enable_memory=True,
        enable_self_improvement=True
    )
    
    # Initialize the system
    await agent.initialize()
    
    # Check available domains
    domains = agent.list_available_domains()
    print(f"Available domains: {domains}")
    
    if "software_development" in domains:
        # List agents in software development domain
        sw_agents = agent.list_domain_agents("software_development")
        print(f"Software development agents: {sw_agents}")
        
        # Example 1: Multi-Perspective Comprehensive Planning
        print("\n--- Multi-Perspective Comprehensive Planning ---")
        comprehensive_result = await agent.execute_domain_workflow(
            domain_name="software_development",
            workflow_name="comprehensive_project_planning",
            task_description="Plan a comprehensive e-commerce platform",
            task_context={
                "project_type": "e_commerce_platform",
                "requirements": {
                    "functional": {
                        "user_management": "registration, authentication, profiles",
                        "product_catalog": "browse, search, categories, inventory",
                        "shopping_cart": "add, remove, modify quantities",
                        "payment_processing": "multiple payment methods, secure transactions",
                        "order_management": "tracking, history, status updates"
                    },
                    "non_functional": {
                        "performance": "handle 10k concurrent users",
                        "scalability": "horizontal scaling capability",
                        "availability": "99.9% uptime requirement",
                        "security": "PCI DSS compliance, data protection"
                    }
                },
                "constraints": {
                    "timeline": "6 months to MVP",
                    "budget": "moderate budget constraints",
                    "team_size": "8 developers, 2 DevOps, 1 designer"
                },
                "stakeholders": ["product_owner", "development_team", "business_stakeholders"]
            }
        )
        print(f"Comprehensive planning result: {comprehensive_result}")
        
        # Example 2: Web Application Planning
        print("\n--- Web Application Planning ---")
        web_app_result = await agent.execute_domain_workflow(
            domain_name="software_development",
            workflow_name="web_application_planning",
            task_description="Plan a social media platform with real-time features",
            task_context={
                "project_type": "social_media_platform",
                "requirements": {
                    "real_time_chat": "instant messaging between users",
                    "content_sharing": "posts, images, videos",
                    "user_connections": "follow/unfollow, friend requests",
                    "notifications": "real-time push notifications",
                    "content_moderation": "automated and manual content review"
                },
                "constraints": {
                    "scalability": "expect rapid user growth",
                    "real_time_performance": "sub-second message delivery",
                    "content_safety": "COPPA compliance for younger users"
                }
            }
        )
        print(f"Web application planning result: {web_app_result}")
        
        # Example 3: Traditional Architecture Review Workflow
        print("\n--- Traditional Architecture Review Workflow ---")
        architecture_result = await agent.execute_domain_workflow(
            domain_name="software_development",
            workflow_name="architecture_review",
            task_description="Review the architecture of a microservices e-commerce system",
            task_context={
                "system_type": "microservices",
                "domain": "e-commerce",
                "scale": "medium",
                "technologies": ["Python", "FastAPI", "PostgreSQL", "Redis", "Docker"]
            }
        )
        print(f"Architecture review result: {architecture_result}")
        
        # Example 2: Code Quality Audit Workflow  
        print("\n--- Code Quality Audit Workflow ---")
        quality_result = await agent.execute_domain_workflow(
            domain_name="software_development", 
            workflow_name="code_quality_audit",
            task_description="Audit code quality for a Python web application",
            task_context={
                "language": "Python",
                "framework": "FastAPI",
                "codebase_size": "medium",
                "focus_areas": ["SOLID principles", "design patterns", "complexity"]
            }
        )
        print(f"Code quality audit result: {quality_result}")
        
        # Example 3: System Analysis Workflow
        print("\n--- System Analysis Workflow ---")
        analysis_result = await agent.execute_domain_workflow(
            domain_name="software_development",
            workflow_name="system_analysis", 
            task_description="Analyze system dependencies and security posture",
            task_context={
                "analysis_type": "comprehensive",
                "include_security": True,
                "include_dependencies": True,
                "include_performance": True
            }
        )
        print(f"System analysis result: {analysis_result}")
        
        # Example 4: Direct Agent Usage
        print("\n--- Direct Agent Usage ---")
        
        # Get specific domain agent
        architect = agent.get_domain_agent("software_development", "architect")
        if architect:
            # Use architect directly for system design
            design_result = await architect.design_system_architecture(
                requirements={
                    "type": "web_application",
                    "users": "10000_concurrent",
                    "features": ["user_auth", "real_time_chat", "file_upload"],
                    "constraints": {"budget": "medium", "timeline": "3_months"}
                },
                constraints={
                    "technology_stack": "Python/JavaScript",
                    "deployment": "cloud_native",
                    "scalability": "horizontal"
                }
            )
            print(f"System design result: {design_result}")
        
        # Get security auditor
        security_auditor = agent.get_domain_agent("software_development", "security_auditor")
        if security_auditor:
            print("\nSecurity auditor capabilities:")
            capabilities = await security_auditor.get_agent_capabilities()
            print(f"Capabilities: {capabilities}")
    
    # Get domain statistics
    stats = agent.get_domain_statistics()
    print(f"\nDomain statistics: {stats}")


async def mixed_domain_example():
    """Example combining multiple domains (when available)."""
    
    print("\n=== Mixed Domain Example ===\n")
    
    agent = SelfReflectingAgent(config_path="config.yaml")
    await agent.initialize()
    
    # Check what domains are available
    domains = agent.list_available_domains()
    print(f"Available domains: {domains}")
    
    # Use core agents for general tasks
    general_result = await agent.execute_task(
        task_description="Create a secure authentication system",
        requirements={
            "security_level": "high",
            "authentication_methods": ["password", "2fa", "oauth"],
            "compliance": ["GDPR", "SOC2"]
        }
    )
    print(f"General task result: {general_result}")
    
    # If software development domain is available, get specialized review
    if "software_development" in domains:
        security_review = await agent.execute_domain_workflow(
            domain_name="software_development",
            workflow_name="system_analysis",
            task_description="Security analysis of the authentication system",
            task_context={
                "focus": "security",
                "system_component": "authentication",
                "compliance_requirements": ["GDPR", "SOC2"]
            }
        )
        print(f"Security review result: {security_review}")


async def agent_specialization_demo():
    """Demonstrate different agent specializations."""
    
    print("\n=== Agent Specialization Demo ===\n")
    
    agent = SelfReflectingAgent(config_path="config.yaml")
    await agent.initialize()
    
    domains = agent.list_available_domains()
    
    if "software_development" in domains:
        # Test different specialized agents
        test_cases = [
            {
                "agent": "architect",
                "task": "Design a microservices architecture for a social media platform",
                "context": {"scale": "large", "users": "1M+", "regions": "global"}
            },
            {
                "agent": "security_auditor", 
                "task": "Audit security vulnerabilities in a REST API",
                "context": {"api_type": "REST", "authentication": "JWT", "data_sensitivity": "high"}
            },
            {
                "agent": "performance_auditor",
                "task": "Analyze performance bottlenecks in a database-heavy application", 
                "context": {"database": "PostgreSQL", "queries_per_second": "1000+", "data_size": "100GB+"}
            },
            {
                "agent": "design_patterns_expert",
                "task": "Recommend design patterns for a complex business rules engine",
                "context": {"complexity": "high", "rules_count": "500+", "changeability": "frequent"}
            }
        ]
        
        for test_case in test_cases:
            print(f"\n--- Testing {test_case['agent']} ---")
            
            domain_agent = agent.get_domain_agent("software_development", test_case["agent"])
            if domain_agent:
                # Each agent would have specialized methods
                if hasattr(domain_agent, 'process_task'):
                    result = await domain_agent.process_task(
                        test_case["task"],
                        test_case["context"]
                    )
                    print(f"Result: {result}")
                else:
                    print(f"Agent {test_case['agent']} loaded but no process_task method")
            else:
                print(f"Agent {test_case['agent']} not available")


async def main():
    """Run all examples."""
    
    print("Domain-Specific Agent System Examples")
    print("=" * 50)
    
    try:
        # Software development examples
        await software_development_examples()
        
        # Mixed domain examples
        await mixed_domain_example()
        
        # Agent specialization demo
        await agent_specialization_demo()
        
    except Exception as e:
        print(f"Example execution failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())