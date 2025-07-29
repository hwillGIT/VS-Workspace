"""
Intelligent Model Routing Example

Demonstrates the complete intelligent model routing system with:
- Task-based model selection
- Context preservation across model switches
- RAG and semantic search integration
- Performance monitoring and fallback handling
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from routed_agent import RoutedSelfReflectingAgent, create_routed_agent
from routing.model_router import TaskType


async def demonstrate_basic_routing():
    """Demonstrate basic intelligent routing capabilities."""
    
    print("🤖 Intelligent Model Routing Demo")
    print("=" * 60)
    
    # Create routed agent
    agent = await create_routed_agent(
        project_path="./demo_project",
        enable_rag=True,
        enable_semantic_search=True
    )
    
    # Check router status
    status = agent.get_routing_status()
    print(f"📊 Router Status: {len(status['router_status']['models'])} models available")
    
    # Available models
    for model_name, model_info in status['router_status']['models'].items():
        enabled = "✅" if model_info['enabled'] else "❌"
        has_key = "🔑" if model_info['has_api_key'] else "🚫"
        print(f"  {enabled} {has_key} {model_name} ({model_info['provider']})")
    
    print("\n" + "=" * 60)
    
    # Test different task types with intelligent routing
    test_cases = [
        {
            "name": "💬 Conversational Task",
            "message": "Hello! Can you help me understand how this routing system works?",
            "task_type": TaskType.CONVERSATION,
            "expected_models": ["claude-3-haiku-20240307", "gpt-4o-mini"]
        },
        {
            "name": "🏗️ Architecture Planning",
            "message": "Design a scalable microservices architecture for an e-commerce platform with 1M+ users",
            "task_type": TaskType.ARCHITECTURE,
            "expected_models": ["claude-3-5-sonnet-20241022", "gpt-4o"]
        },
        {
            "name": "💻 Code Generation",
            "message": "Create a Python function to implement a LRU cache with thread safety",
            "task_type": TaskType.CODE_GENERATION,
            "expected_models": ["claude-3-5-sonnet-20241022", "gpt-4o"]
        },
        {
            "name": "🐛 Debugging Task",
            "message": "I'm getting a 'KeyError: user_id' when trying to access session data. Help me debug this issue.",
            "task_type": TaskType.DEBUGGING,
            "expected_models": ["gemini-2.0-flash-exp", "claude-3-5-sonnet-20241022"]
        },
        {
            "name": "📝 Documentation",
            "message": "Write API documentation for a REST endpoint that creates user profiles",
            "task_type": TaskType.DOCUMENTATION,
            "expected_models": ["claude-3-haiku-20240307", "gpt-4o-mini"]
        }
    ]
    
    session_id = "routing_demo_session"
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['name']}")
        print("-" * 40)
        print(f"📋 Task: {test_case['message'][:60]}...")
        
        try:
            # Execute task with intelligent routing
            result = await agent.execute_task(
                task_description=test_case["message"],
                session_id=session_id,
                task_type=test_case["task_type"]
            )
            
            # Display routing decision
            print(f"🎯 Selected Model: {result['model_used']} ({result['provider']})")
            print(f"🧠 Reasoning: {result['routing_reasoning']}")
            print(f"⚡ Execution Time: {result['execution_time_ms']}ms")
            print(f"💰 Estimated Cost: ${result.get('estimated_cost', 0):.4f}")
            
            if result['fallback_models']:
                print(f"🔄 Fallback Models: {', '.join(result['fallback_models'][:3])}")
            
            # Show first part of response
            response_preview = result['content'][:100] + "..." if len(result['content']) > 100 else result['content']
            print(f"💭 Response Preview: {response_preview}")
            
            # Context and enhancement info
            if result.get('rag_enhanced'):
                print("🔍 Enhanced with RAG context")
            if result.get('memory_enhanced'):
                print("🧠 Enhanced with memory context")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print()
    
    # Show session statistics
    session_info = agent.get_session_info(session_id)
    if session_info:
        print("📈 Session Statistics:")
        print(f"  • Total Tasks: {session_info.get('task_count', 0)}")
        print(f"  • Models Used: {', '.join(session_info.get('models_used', []))}")
        print(f"  • Total Tokens: {session_info.get('total_tokens', 0)}")
    
    await agent.shutdown()


async def demonstrate_context_preservation():
    """Demonstrate context preservation across model switches."""
    
    print("\n🔄 Context Preservation Demo")
    print("=" * 60)
    
    agent = await create_routed_agent()
    session_id = "context_demo_session"
    
    # Start with a conversation that builds context
    conversation_steps = [
        {
            "message": "I'm working on a Python web application using FastAPI. I need to implement user authentication.",
            "task_type": TaskType.CONVERSATION
        },
        {
            "message": "Create the database models for users with email, password hash, and role fields.",
            "task_type": TaskType.CODE_GENERATION
        },
        {
            "message": "Now create JWT token generation and validation functions.",
            "task_type": TaskType.CODE_GENERATION
        },
        {
            "message": "I'm getting an error: 'jose.exceptions.JWKError: Unable to find a signing key'. Help me debug this.",
            "task_type": TaskType.DEBUGGING  # This should route to Gemini 2.5 Pro
        },
        {
            "message": "Write comprehensive tests for the authentication system we just built.",
            "task_type": TaskType.TESTING
        }
    ]
    
    for i, step in enumerate(conversation_steps, 1):
        print(f"\n{i}. Task Type: {step['task_type'].value}")
        print(f"📝 Message: {step['message'][:60]}...")
        
        result = await agent.execute_task(
            task_description=step["message"],
            session_id=session_id,
            task_type=step["task_type"]
        )
        
        print(f"🎯 Model: {result['model_used']}")
        print(f"📊 Context Stats: {result['context_stats']['total_turns']} turns, {result['context_stats']['current_context_length']} chars")
        
        # Show if context was compressed
        if "compressed" in result.get('routing_reasoning', '').lower():
            print("🗜️ Context was compressed for smaller model")
    
    print(f"\n📋 Final Context Summary: {agent.router_agent.router.get_context_stats(session_id)['conversation_summary']}")
    
    await agent.shutdown()


async def demonstrate_semantic_search():
    """Demonstrate semantic search capabilities."""
    
    print("\n🔍 Semantic Search Demo")
    print("=" * 60)
    
    agent = await create_routed_agent(
        enable_semantic_search=True,
        project_path="."  # Current directory
    )
    
    # Add some sample documents for demonstration
    if agent.semantic_search:
        sample_docs = [
            {
                "content": "The model router intelligently selects the best AI model for each task based on task type, model capabilities, availability, and performance history. It supports fallback mechanisms and context preservation.",
                "metadata": {"source": "routing_docs", "type": "documentation"},
                "source": "internal://routing_system",
                "id": "routing_overview"
            },
            {
                "content": "Context preservation is achieved through intelligent compression and summarization when switching between models with different context windows. The system maintains conversation history and important context.",
                "metadata": {"source": "context_docs", "type": "documentation"},
                "source": "internal://context_system",
                "id": "context_preservation"
            },
            {
                "content": "RAG (Retrieval-Augmented Generation) enhances responses by finding relevant context from indexed documents. It uses hybrid search combining semantic similarity and keyword matching.",
                "metadata": {"source": "rag_docs", "type": "documentation"},
                "source": "internal://rag_system",
                "id": "rag_overview"
            }
        ]
        
        await agent.semantic_search.add_documents(sample_docs)
        
        # Perform searches
        search_queries = [
            "How does model selection work?",
            "What happens when switching between models?",
            "How does RAG improve responses?",
            "Semantic search hybrid approach"
        ]
        
        for query in search_queries:
            print(f"\n🔍 Query: {query}")
            results = await agent.semantic_search(query, k=2, search_type="hybrid")
            
            for i, result in enumerate(results, 1):
                print(f"  {i}. Score: {result['score']:.3f}")
                print(f"     Content: {result['content'][:80]}...")
                print(f"     Source: {result['source']}")
        
        # Demonstrate search-and-answer
        print(f"\n🤔 Search and Answer Demo:")
        question = "How does the system handle model failures and fallbacks?"
        
        result = await agent.search_and_answer(
            question=question,
            session_id="search_demo_session"
        )
        
        print(f"❓ Question: {question}")
        print(f"🤖 Model Used: {result['model_used']}")
        print(f"🔍 Search Results: {len(result['search_results'])} found")
        print(f"💭 Answer: {result['content'][:200]}...")
    
    await agent.shutdown()


async def demonstrate_performance_monitoring():
    """Demonstrate performance monitoring and optimization."""
    
    print("\n📊 Performance Monitoring Demo")
    print("=" * 60)
    
    agent = await create_routed_agent()
    
    # Execute multiple tasks to generate performance data
    print("🏃 Executing tasks to generate performance data...")
    
    tasks = [
        "Write a simple hello world function",
        "Explain how dictionaries work in Python",
        "Debug this error: NameError: name 'x' is not defined",
        "Design a simple REST API structure",
        "Create unit tests for a calculator function"
    ]
    
    session_id = "performance_demo_session"
    
    for task in tasks:
        try:
            await agent.execute_task(
                task_description=task,
                session_id=session_id
            )
        except:
            pass  # Continue even if some tasks fail
    
    # Display performance statistics
    router_status = agent.get_routing_status()
    
    print("\n📈 Model Performance Statistics:")
    for model_name, model_info in router_status['router_status']['models'].items():
        if 'performance' in model_info:
            perf = model_info['performance']
            print(f"\n🤖 {model_name}:")
            print(f"  • Success Rate: {perf['success_rate']:.1%}")
            print(f"  • Avg Latency: {perf['avg_latency_ms']:.0f}ms")
            print(f"  • Avg Cost: ${perf['avg_cost']:.4f}")
            print(f"  • Total Requests: {perf['total_requests']}")
            
            if perf['last_success']:
                print(f"  • Last Success: {perf['last_success']}")
            if perf['last_failure']:
                print(f"  • Last Failure: {perf['last_failure']}")
    
    # Show semantic search stats if available
    if agent.semantic_search:
        search_stats = agent.semantic_search.get_search_stats()
        print(f"\n🔍 Semantic Search Statistics:")
        print(f"  • Total Documents: {search_stats['total_documents']}")
        print(f"  • Cache Size: {search_stats['cache_size']}")
        print(f"  • Total Searches: {search_stats['search_stats']['total_searches']}")
        print(f"  • Cache Hits: {search_stats['search_stats']['cache_hits']}")
        print(f"  • Avg Search Time: {search_stats['search_stats']['avg_search_time']:.3f}s")
    
    await agent.shutdown()


async def demonstrate_advanced_features():
    """Demonstrate advanced routing features."""
    
    print("\n🚀 Advanced Features Demo")
    print("=" * 60)
    
    agent = await create_routed_agent()
    
    # Test cost-sensitive routing
    print("💰 Cost-Sensitive Routing:")
    result = await agent.execute_task(
        task_description="Write simple documentation for a function",
        cost_sensitive=True
    )
    print(f"  Selected: {result['model_used']} (cost-optimized)")
    
    # Test latency-sensitive routing
    print("\n⚡ Latency-Sensitive Routing:")
    result = await agent.execute_task(
        task_description="Quick debugging help needed",
        task_type=TaskType.DEBUGGING,
        latency_sensitive=True
    )
    print(f"  Selected: {result['model_used']} (speed-optimized)")
    
    # Test specific model preferences
    print("\n🎯 Task-Specific Optimization:")
    
    # Architecture task (should prefer Claude for system thinking)
    result = await agent.plan_architecture(
        requirements="Design a distributed messaging system for real-time chat",
        session_id="advanced_demo"
    )
    print(f"  Architecture: {result['model_used']} - {result['routing_reasoning'][:60]}...")
    
    # Debugging task (should prefer Gemini 2.5 Pro)
    result = await agent.debug_code(
        code_or_error="Getting 'Connection refused' error when connecting to database",
        session_id="advanced_demo"
    )
    print(f"  Debugging: {result['model_used']} - {result['routing_reasoning'][:60]}...")
    
    # Code generation (should prefer Claude or GPT-4)
    result = await agent.generate_code(
        request="Create a thread-safe singleton pattern in Python",
        language="python",
        session_id="advanced_demo"
    )
    print(f"  Code Gen: {result['model_used']} - {result['routing_reasoning'][:60]}...")
    
    await agent.shutdown()


async def main():
    """Run all demonstrations."""
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🎯 Intelligent Model Routing System Demo")
    print("🚀 This demo showcases intelligent AI model selection based on task characteristics")
    print("=" * 80)
    
    try:
        # Run demonstrations
        await demonstrate_basic_routing()
        await demonstrate_context_preservation()
        await demonstrate_semantic_search()
        await demonstrate_performance_monitoring()
        await demonstrate_advanced_features()
        
        print("\n" + "=" * 80)
        print("✅ Demo completed successfully!")
        print("\n🔑 Key Features Demonstrated:")
        print("  • Intelligent model selection based on task type")
        print("  • Context preservation across model switches")
        print("  • Fallback handling for unavailable models")
        print("  • Performance tracking and optimization")
        print("  • RAG and semantic search integration")
        print("  • Cost and latency optimization")
        print("  • Advanced routing strategies")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())