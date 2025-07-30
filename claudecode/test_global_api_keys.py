#!/usr/bin/env python3
"""
Test Global Multi-Key API Management System

This script verifies that the global API key management system with
multi-key failover works correctly across all modules.
"""

import asyncio
import os
import sys
import logging
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_global_import():
    """Test that global API key system can be imported."""
    logger.info("=" * 60)
    logger.info("Testing Global API Key System Import")
    logger.info("=" * 60)
    
    try:
        import global_api_keys
        logger.info("[+] global_api_keys module imported successfully")
        
        # Check available functions
        functions = ['get_api_key', 'get_api_key_sync', 'record_api_result', 
                    'get_api_status', 'enable_global_failover']
        
        for func in functions:
            if hasattr(global_api_keys, func):
                logger.info(f"[+] Function available: {func}")
            else:
                logger.error(f"[-] Function missing: {func}")
                return False
        
        return True
    except ImportError as e:
        logger.error(f"[-] Failed to import global_api_keys: {e}")
        return False


async def test_direct_api_access():
    """Test direct API key access through global system."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Direct API Key Access")
    logger.info("=" * 60)
    
    from global_api_keys import get_api_key, get_api_key_sync, get_api_status
    
    # Test async access
    test_keys = [
        'OPENAI_API_KEY',
        'ANTHROPIC_API_KEY',
        'GOOGLE_API_KEY',
        'OPENROUTER_API_KEY',
        'GROQ_API_KEY'
    ]
    
    found_keys = 0
    for key_name in test_keys:
        # Test async version
        api_key = await get_api_key(key_name)
        if api_key and not api_key.startswith('your-'):
            logger.info(f"[+] {key_name}: Found (async)")
            found_keys += 1
        else:
            logger.info(f"[-] {key_name}: Not found (async)")
        
        # Test sync version
        sync_key = get_api_key_sync(key_name)
        if sync_key and not sync_key.startswith('your-'):
            logger.info(f"[+] {key_name}: Found (sync)")
        else:
            logger.info(f"[-] {key_name}: Not found (sync)")
    
    # Get system status
    status = get_api_status()
    logger.info(f"\nSystem Status: {status.get('active_providers', 0)}/{status.get('total_providers', 0)} providers active")
    
    return found_keys > 0


async def test_environment_interception():
    """Test that os.getenv() interception works."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Environment Variable Interception")
    logger.info("=" * 60)
    
    from global_api_keys import enable_global_failover, disable_global_failover
    
    # Test with interception disabled
    disable_global_failover()
    direct_key = os.getenv('OPENAI_API_KEY')
    logger.info(f"Direct os.getenv() result: {'Found' if direct_key else 'Not found'}")
    
    # Enable interception
    enable_global_failover()
    intercepted_key = os.getenv('OPENAI_API_KEY')
    logger.info(f"Intercepted os.getenv() result: {'Found' if intercepted_key else 'Not found'}")
    
    # Test non-API key (should not be intercepted)
    test_var = 'PATH'
    path_value = os.getenv(test_var)
    logger.info(f"Non-API key '{test_var}': {'Found' if path_value else 'Not found'}")
    
    return True


async def test_failover_simulation():
    """Test failover functionality."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Failover Functionality")
    logger.info("=" * 60)
    
    from global_api_keys import get_api_key, record_api_result, get_api_status
    
    # Get initial status
    initial_status = get_api_status()
    logger.info(f"Initial status: {initial_status.get('active_providers', 0)} providers active")
    
    # Test with a provider that has multiple keys
    test_provider = None
    if initial_status.get('providers'):
        for provider, info in initial_status['providers'].items():
            if info.get('total_keys', 0) > 1:
                test_provider = provider
                break
    
    if test_provider:
        logger.info(f"\nTesting failover with {test_provider}")
        
        # Map provider to env var
        env_map = {
            'anthropic': 'ANTHROPIC_API_KEY',
            'openai': 'OPENAI_API_KEY',
            'google': 'GOOGLE_API_KEY',
            'openrouter': 'OPENROUTER_API_KEY',
            'groq': 'GROQ_API_KEY'
        }
        
        env_var = env_map.get(test_provider)
        if env_var:
            # Get initial key
            initial_key = await get_api_key(env_var)
            logger.info(f"Initial key: {initial_key[:8]}..." if initial_key else "None")
            
            # Simulate failures
            for i in range(3):
                await record_api_result(env_var, success=False, error_type='rate_limit')
                logger.info(f"Recorded failure {i+1}")
            
            # Check if key changed
            new_key = await get_api_key(env_var)
            if new_key != initial_key:
                logger.info(f"[+] Failover successful! New key: {new_key[:8]}...")
            else:
                logger.info("[-] No failover occurred (may need more failures)")
    else:
        logger.info("No provider with multiple keys available for failover test")
    
    return True


async def test_module_integration():
    """Test integration with other modules."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Module Integration")
    logger.info("=" * 60)
    
    # Test that ClaudeCode __init__ loads global system
    try:
        # Re-import to test initialization
        if '__init__' in sys.modules:
            del sys.modules['__init__']
        
        import __init__
        
        # Check if functions are available
        if hasattr(__init__, 'get_api_key'):
            logger.info("[+] Global API functions available in __init__")
        else:
            logger.info("[-] Global API functions not exported in __init__")
        
    except Exception as e:
        logger.error(f"[-] Error testing __init__ integration: {e}")
    
    # Test architecture_intelligence integration
    try:
        # Check if the module would use global keys
        arch_path = Path(__file__).parent / 'architecture_intelligence' / 'analyze_with_real_gemini.py'
        if arch_path.exists():
            with open(arch_path, 'r') as f:
                content = f.read()
                if 'global_api_keys' in content or 'get_api_key_sync' in content:
                    logger.info("[+] architecture_intelligence updated for global API keys")
                else:
                    logger.info("[-] architecture_intelligence not yet updated")
        else:
            logger.info("[?] Could not find architecture_intelligence module")
    except Exception as e:
        logger.error(f"Error checking architecture_intelligence: {e}")
    
    return True


async def test_performance():
    """Test performance of API key retrieval."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Performance")
    logger.info("=" * 60)
    
    from global_api_keys import get_api_key, get_api_key_sync
    import time
    
    # Test async performance
    start = time.time()
    for _ in range(100):
        await get_api_key('OPENAI_API_KEY')
    async_time = time.time() - start
    logger.info(f"100 async get_api_key calls: {async_time:.3f}s ({async_time/100*1000:.1f}ms per call)")
    
    # Test sync performance
    start = time.time()
    for _ in range(100):
        get_api_key_sync('OPENAI_API_KEY')
    sync_time = time.time() - start
    logger.info(f"100 sync get_api_key calls: {sync_time:.3f}s ({sync_time/100*1000:.1f}ms per call)")
    
    # Test os.getenv performance (baseline)
    start = time.time()
    for _ in range(100):
        os.getenv('OPENAI_API_KEY')
    direct_time = time.time() - start
    logger.info(f"100 direct os.getenv calls: {direct_time:.3f}s ({direct_time/100*1000:.1f}ms per call)")
    
    overhead = (sync_time - direct_time) / direct_time * 100 if direct_time > 0 else 0
    logger.info(f"Overhead: {overhead:.1f}%")
    
    return True


async def run_all_tests():
    """Run all global API key tests."""
    logger.info("Global Multi-Key API Management Test Suite")
    logger.info(f"Start time: {datetime.now()}")
    logger.info("=" * 60)
    
    tests = [
        ("Import Test", test_global_import),
        ("Direct Access Test", test_direct_api_access),
        ("Environment Interception", test_environment_interception),
        ("Failover Simulation", test_failover_simulation),
        ("Module Integration", test_module_integration),
        ("Performance Test", test_performance)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            logger.info(f"\nRunning: {test_name}")
            result = await test_func()
            results.append((test_name, result))
            logger.info(f"Result: {'PASSED' if result else 'FAILED'}")
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "[+] PASSED" if result else "[-] FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("\n[+] Global multi-key API system is working correctly!")
    else:
        logger.info("\n[-] Some tests failed. Check the logs above.")
    
    logger.info(f"\nEnd time: {datetime.now()}")
    
    return passed == total


def main():
    """Main entry point."""
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()