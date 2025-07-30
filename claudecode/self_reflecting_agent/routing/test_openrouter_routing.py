#!/usr/bin/env python3
"""
Test OpenRouter Integration with Model Router

This script tests the complete OpenRouter integration including:
- Model availability checking
- Route selection with OpenRouter models
- Actual API calls through OpenRouter
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add the parent directory to the path to import the modules
sys.path.append(str(Path(__file__).parent.parent))

from routing.model_router import ModelRouter, TaskContext, TaskType
from routing.openrouter_client import OpenRouterClient, create_openrouter_client


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_openrouter_availability():
    """Test OpenRouter API key and basic connectivity."""
    logger.info("Testing OpenRouter availability...")
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        logger.error("OPENROUTER_API_KEY environment variable not set")
        return False
    
    try:
        client = await create_openrouter_client()
        
        # Test getting models
        models = await client.get_models()
        logger.info(f"OpenRouter has {len(models)} available models")
        
        # Show some popular models
        popular_models = [
            "anthropic/claude-3-5-sonnet",
            "openai/gpt-4o",
            "google/gemini-pro-1.5",
            "meta-llama/llama-3.1-405b-instruct"
        ]
        
        available_popular = []
        for model_id in popular_models:
            if any(m.get("id") == model_id for m in models):
                available_popular.append(model_id)
        
        logger.info(f"Popular models available: {available_popular}")
        
        # Test credits
        try:
            credits = await client.get_credits()
            logger.info(f"OpenRouter credits info: {credits}")
        except Exception as e:
            logger.warning(f"Could not get credits info: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"OpenRouter availability test failed: {e}")
        return False


async def test_model_router_with_openrouter():
    """Test model router's ability to select and use OpenRouter models."""
    logger.info("Testing model router with OpenRouter integration...")
    
    try:
        # Create model router
        router = ModelRouter()
        
        # Test different task types to see OpenRouter model selection
        test_tasks = [
            TaskContext(
                task_type=TaskType.CODE_GENERATION,
                complexity="medium",
                estimated_tokens=4000,
                requires_code=True,
                requires_reasoning=True
            ),
            TaskContext(
                task_type=TaskType.ARCHITECTURE,
                complexity="high",
                estimated_tokens=8000,
                requires_reasoning=True
            ),
            TaskContext(
                task_type=TaskType.DEBUGGING,
                complexity="high",
                estimated_tokens=12000,
                requires_code=True,
                requires_reasoning=True,
                latency_sensitive=True
            ),
            TaskContext(
                task_type=TaskType.DOCUMENTATION,
                complexity="low",
                estimated_tokens=2000,
                cost_sensitive=True
            )
        ]
        
        for i, task in enumerate(test_tasks):
            logger.info(f"\nTest {i+1}: {task.task_type.value} task")
            logger.info(f"  Complexity: {task.complexity}")
            logger.info(f"  Estimated tokens: {task.estimated_tokens}")
            
            try:
                # Get routing decision
                decision = await router.route_task(task)
                
                logger.info(f"  Selected model: {decision.selected_model}")
                logger.info(f"  Provider: {decision.provider.value}")
                logger.info(f"  Reasoning: {decision.reasoning}")
                logger.info(f"  Fallback models: {decision.fallback_models[:3]}")
                
                if decision.estimated_cost:
                    logger.info(f"  Estimated cost: ${decision.estimated_cost:.4f}")
                
                # Check if OpenRouter model was selected
                if decision.provider.value == "openrouter":
                    logger.info("  ✅ OpenRouter model selected!")
                else:
                    logger.info(f"  ℹ️  Non-OpenRouter model selected: {decision.provider.value}")
                
            except Exception as e:
                logger.error(f"  ❌ Routing failed: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"Model router test failed: {e}")
        return False


async def test_openrouter_model_execution():
    """Test actual execution of OpenRouter models through the router."""
    logger.info("Testing OpenRouter model execution...")
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        logger.warning("Skipping execution test - no OPENROUTER_API_KEY")
        return True
    
    try:
        client = await create_openrouter_client()
        
        # Test a simple completion
        test_model = "anthropic/claude-3-haiku"  # Affordable model for testing
        test_messages = [
            {"role": "user", "content": "Write a simple 'Hello World' program in Python."}
        ]
        
        logger.info(f"Testing completion with {test_model}...")
        
        response = await client.complete(
            model=test_model,
            messages=test_messages,
            max_tokens=200,
            temperature=0.3
        )
        
        logger.info(f"Response content: {response.content[:200]}...")
        logger.info(f"Model used: {response.model}")
        logger.info(f"Tokens used: {response.usage}")
        logger.info("✅ OpenRouter execution test successful!")
        
        return True
        
    except Exception as e:
        logger.error(f"OpenRouter execution test failed: {e}")
        return False


async def test_model_status_reporting():
    """Test that OpenRouter models appear in status reporting."""
    logger.info("Testing model status reporting...")
    
    try:
        router = ModelRouter()
        status = router.get_model_status()
        
        logger.info(f"Total models configured: {status['total_models']}")
        logger.info(f"Enabled models: {status['enabled_models']}")
        
        # Check for OpenRouter models
        openrouter_models = [
            name for name, config in status['models'].items()
            if config['provider'] == 'openrouter'
        ]
        
        logger.info(f"OpenRouter models configured: {len(openrouter_models)}")
        for model_name in openrouter_models:
            model_status = status['models'][model_name]
            has_key = model_status['has_api_key']
            logger.info(f"  - {model_name}: {'✅' if has_key else '❌'} API key")
        
        return True
        
    except Exception as e:
        logger.error(f"Status reporting test failed: {e}")
        return False


async def run_all_tests():
    """Run all OpenRouter integration tests."""
    logger.info("🚀 Starting OpenRouter Integration Tests")
    logger.info("=" * 60)
    
    tests = [
        ("OpenRouter Availability", test_openrouter_availability),
        ("Model Router Integration", test_model_router_with_openrouter),
        ("Model Status Reporting", test_model_status_reporting),
        ("OpenRouter Execution", test_openrouter_model_execution),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running: {test_name}")
        logger.info("-" * 40)
        
        try:
            result = await test_func()
            results.append((test_name, result))
            
            if result:
                logger.info(f"✅ {test_name}: PASSED")
            else:
                logger.info(f"❌ {test_name}: FAILED")
                
        except Exception as e:
            logger.error(f"❌ {test_name}: ERROR - {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All OpenRouter integration tests passed!")
        return True
    else:
        logger.info("⚠️  Some tests failed. Check the logs above.")
        return False


def main():
    """Main entry point."""
    logger.info("OpenRouter Integration Test Suite")
    
    # Check for required environment variables
    if not os.getenv("OPENROUTER_API_KEY"):
        logger.warning("⚠️  OPENROUTER_API_KEY not set - some tests will be skipped")
        logger.info("Set OPENROUTER_API_KEY to run full tests")
    
    # Run tests
    success = asyncio.run(run_all_tests())
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()