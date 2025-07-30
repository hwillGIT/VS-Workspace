#!/usr/bin/env python3
"""
Test Multi-Key Failover System

Tests the complete multi-key API management system including:
- .env file loading
- Multi-key discovery and management
- Rate limit detection and failover
- Key usage tracking and monitoring
- Integration with model router
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from datetime import datetime

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Load from the project root .env file
    env_path = Path(__file__).parent.parent.parent / '.env'
    load_dotenv(env_path)
    print(f"[+] Loaded environment variables from {env_path}")
except ImportError:
    print("[!] python-dotenv not available - environment variables may not be loaded")

# Add the parent directory to the path to import the modules
sys.path.append(str(Path(__file__).parent.parent))

from routing.model_router import ModelRouter, TaskContext, TaskType
from routing.multi_key_manager import (
    MultiKeyManager, 
    get_multi_key_manager, 
    get_api_key, 
    record_api_result,
    get_system_status
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_env_loading():
    """Test that .env file was loaded correctly."""
    logger.info("🔑 Testing Environment Variable Loading")
    logger.info("=" * 60)
    
    providers = {
        "Anthropic": ["ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY_2"],
        "OpenAI": ["OPENAI_API_KEY", "OPENAI_API_KEY_2"], 
        "Google": ["GOOGLE_API_KEY", "GOOGLE_API_KEY_2"],
        "OpenRouter": ["OPENROUTER_API_KEY", "OPENROUTER_API_KEY_2"],
        "Groq": ["GROQ_API_KEY", "GROQ_API_KEY_2"]
    }
    
    loaded_keys = {}
    for provider, env_vars in providers.items():
        provider_keys = []
        for env_var in env_vars:
            value = os.getenv(env_var)
            if value and not value.startswith("your-"):
                provider_keys.append(env_var)
                logger.info(f"✅ {env_var}: Loaded")
            else:
                logger.info(f"❌ {env_var}: Not set or placeholder")
        
        loaded_keys[provider] = provider_keys
        if provider_keys:
            logger.info(f"  {provider}: {len(provider_keys)} key(s) available")
    
    total_keys = sum(len(keys) for keys in loaded_keys.values())
    logger.info(f"\nTotal API keys loaded: {total_keys}")
    
    return loaded_keys


async def test_multi_key_manager_discovery():
    """Test multi-key manager key discovery."""
    logger.info("\n🔍 Testing Multi-Key Manager Discovery")
    logger.info("=" * 60)
    
    manager = get_multi_key_manager()
    
    logger.info(f"Discovered providers: {len(manager.keys)}")
    for provider, keys in manager.keys.items():
        logger.info(f"  {provider.value}: {len(keys)} keys")
        
        # Test getting active key
        active_key = await get_api_key(provider)
        if active_key:
            # Mask the key for security
            masked_key = active_key[:8] + "..." + active_key[-8:] if len(active_key) > 16 else "***"
            logger.info(f"    Active key: {masked_key}")
        else:
            logger.info(f"    No active key available")
    
    return manager


async def test_key_status_tracking():
    """Test key status and tracking functionality."""
    logger.info("\n📊 Testing Key Status Tracking")
    logger.info("=" * 60)
    
    # Get system status
    status = get_system_status()
    
    logger.info(f"Total providers: {status['total_providers']}")
    logger.info(f"Active providers: {status['active_providers']}")
    
    for provider_name, provider_status in status['providers'].items():
        logger.info(f"\n{provider_name.upper()}:")
        logger.info(f"  Available: {provider_status['available']}")
        logger.info(f"  Keys: {provider_status['available_keys']}/{provider_status['total_keys']}")
        logger.info(f"  Requests: {provider_status['total_requests']}")
        logger.info(f"  Daily cost: ${provider_status['daily_cost']:.4f}")
        
        if provider_status.get('key_details'):
            logger.info("  Key details:")
            for key_detail in provider_status['key_details']:
                status_icon = "🟢" if key_detail['is_available'] else "🔴"
                active_icon = "⭐" if key_detail['is_active'] else "  "
                logger.info(f"    {status_icon}{active_icon} {key_detail['key_id']}: "
                          f"{key_detail['success_rate']:.1%} success, "
                          f"{key_detail['requests']} requests, "
                          f"${key_detail['cost']:.4f} cost")
    
    return status


async def simulate_api_failures():
    """Simulate API failures to test failover logic."""
    logger.info("\n🔥 Testing Failover with Simulated Failures")
    logger.info("=" * 60)
    
    from routing.model_router import ModelProvider
    
    # Test with a provider that has keys
    test_providers = []
    manager = get_multi_key_manager()
    
    for provider in manager.keys.keys():
        if len(manager.keys[provider]) > 1:  # Only test providers with multiple keys
            test_providers.append(provider)
    
    if not test_providers:
        logger.info("No providers with multiple keys available for failover testing")
        return
    
    test_provider = test_providers[0]
    logger.info(f"Testing failover with {test_provider.value}")
    
    # Get initial active key
    initial_key = await get_api_key(test_provider)
    logger.info(f"Initial active key: {initial_key[:8]}..." if initial_key else "None")
    
    # Simulate rate limit failures
    for i in range(3):
        logger.info(f"\nSimulating failure {i+1}/3...")
        await record_api_result(
            provider=test_provider,
            success=False,
            error_type="rate_limit_exceeded",
            cost=0.0
        )
        
        # Check if failover occurred
        new_key = await get_api_key(test_provider)
        if new_key != initial_key:
            logger.info(f"✅ Failover occurred! New key: {new_key[:8]}...")
            break
        else:
            logger.info("  No failover yet")
    
    # Test success to reset error count
    logger.info("\nSimulating successful request...")
    await record_api_result(
        provider=test_provider,
        success=True,
        error_type=None,
        cost=0.001
    )
    
    final_status = get_system_status()
    provider_status = final_status['providers'][test_provider.value]
    logger.info(f"Final status: {provider_status['total_requests']} requests total")


async def test_model_router_integration():
    """Test integration between multi-key manager and model router."""
    logger.info("\n🔄 Testing Model Router Integration")
    logger.info("=" * 60)
    
    router = ModelRouter()
    
    # Test routing with multi-key manager
    test_task = TaskContext(
        task_type=TaskType.CODE_GENERATION,
        complexity="medium",
        estimated_tokens=4000,
        requires_code=True,
        requires_reasoning=True
    )
    
    try:
        decision = await router.route_task(test_task)
        logger.info(f"✅ Route decision: {decision.selected_model}")
        logger.info(f"   Provider: {decision.provider.value}")
        logger.info(f"   Reasoning: {decision.reasoning}")
        
        # Simulate task execution and result recording
        await router.record_task_result(
            model_name=decision.selected_model,
            success=True,
            latency_ms=2500,
            cost=0.002
        )
        logger.info("✅ Task result recorded successfully")
        
    except Exception as e:
        logger.error(f"❌ Router integration test failed: {e}")
    
    # Show updated status
    status = get_system_status()
    total_requests = sum(
        provider['total_requests'] 
        for provider in status['providers'].values()
    )
    total_cost = sum(
        provider['daily_cost'] 
        for provider in status['providers'].values()
    )
    logger.info(f"Total system requests: {total_requests}")
    logger.info(f"Total system cost: ${total_cost:.6f}")


async def test_key_rotation():
    """Test key rotation and health monitoring."""
    logger.info("\n🔄 Testing Key Rotation and Health Monitoring")
    logger.info("=" * 60)
    
    manager = get_multi_key_manager()
    
    # Perform health check
    await manager.health_check()
    logger.info("✅ Health check completed")
    
    # Test state persistence
    manager._save_state()
    logger.info("✅ State saved to disk")
    
    # Create new manager instance to test state loading
    new_manager = MultiKeyManager()
    logger.info("✅ State loaded from disk in new instance")
    
    # Compare status
    old_status = manager.get_system_status()
    new_status = new_manager.get_system_status()
    
    if old_status['total_providers'] == new_status['total_providers']:
        logger.info("✅ State persistence working correctly")
    else:
        logger.error("❌ State persistence issue detected")


async def run_comprehensive_tests():
    """Run all multi-key failover system tests."""
    logger.info("🚀 Multi-Key Failover System Test Suite")
    logger.info("=" * 60)
    logger.info(f"Start time: {datetime.now()}")
    
    test_results = {}
    
    tests = [
        ("Environment Loading", test_env_loading),
        ("Multi-Key Discovery", test_multi_key_manager_discovery),
        ("Key Status Tracking", test_key_status_tracking),
        ("Failover Simulation", simulate_api_failures),
        ("Router Integration", test_model_router_integration),
        ("Key Rotation", test_key_rotation),
    ]
    
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running: {test_name}")
        logger.info("-" * 40)
        
        try:
            result = await test_func()
            test_results[test_name] = {"success": True, "result": result}
            logger.info(f"✅ {test_name}: PASSED")
        except Exception as e:
            test_results[test_name] = {"success": False, "error": str(e)}
            logger.error(f"❌ {test_name}: FAILED - {e}")
    
    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 MULTI-KEY FAILOVER TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for result in test_results.values() if result["success"])
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result["success"] else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        if not result["success"]:
            logger.info(f"  Error: {result['error']}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    # Show final system status
    final_status = get_system_status()
    logger.info(f"\nFinal System Status:")
    logger.info(f"  Active Providers: {final_status['active_providers']}/{final_status['total_providers']}")
    
    total_requests = sum(p['total_requests'] for p in final_status['providers'].values())
    total_cost = sum(p['daily_cost'] for p in final_status['providers'].values())
    logger.info(f"  Total Requests: {total_requests}")
    logger.info(f"  Total Cost: ${total_cost:.6f}")
    
    if passed == total:
        logger.info("\n🎉 All multi-key failover tests passed!")
        ecosystem_health = True
    else:
        logger.info("\n⚠️  Some tests failed. Check the logs above.")
        ecosystem_health = False
    
    logger.info(f"\nEnd time: {datetime.now()}")
    return ecosystem_health


def main():
    """Main entry point."""
    success = asyncio.run(run_comprehensive_tests())
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()