#!/usr/bin/env python3
"""
Test Complete Multi-Provider Ecosystem

This script tests the complete model routing system with all providers:
- Anthropic (Claude)
- OpenAI (GPT)
- Google (Gemini)
- OpenRouter (Multi-model access)
- Groq (Ultra-fast inference)
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from datetime import datetime

# Add the parent directory to the path to import the modules
sys.path.append(str(Path(__file__).parent.parent))

from routing.model_router import ModelRouter, TaskContext, TaskType
from routing.openrouter_client import create_openrouter_client
from routing.groq_client import create_groq_client


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def check_provider_status():
    """Check which providers have API keys configured."""
    logger.info("🔑 Checking Provider API Keys")
    logger.info("=" * 60)
    
    providers = {
        "Anthropic": "ANTHROPIC_API_KEY",
        "OpenAI": "OPENAI_API_KEY", 
        "Google": "GOOGLE_API_KEY",
        "OpenRouter": "OPENROUTER_API_KEY",
        "Groq": "GROQ_API_KEY"
    }
    
    available_providers = []
    for provider, env_var in providers.items():
        if os.getenv(env_var):
            logger.info(f"✅ {provider}: API key found")
            available_providers.append(provider)
        else:
            logger.info(f"❌ {provider}: No API key")
    
    logger.info(f"\nTotal providers available: {len(available_providers)}/5")
    return available_providers


async def test_model_availability():
    """Test model availability across all providers."""
    logger.info("\n📊 Testing Model Availability")
    logger.info("=" * 60)
    
    router = ModelRouter()
    status = router.get_model_status()
    
    logger.info(f"Total models configured: {status['total_models']}")
    logger.info(f"Enabled models: {status['enabled_models']}")
    
    # Group models by provider
    providers_models = {}
    for model_name, model_info in status['models'].items():
        provider = model_info['provider']
        if provider not in providers_models:
            providers_models[provider] = []
        providers_models[provider].append({
            'name': model_name,
            'has_key': model_info['has_api_key'],
            'enabled': model_info['enabled'],
            'priority': model_info['priority']
        })
    
    # Display models by provider
    for provider, models in sorted(providers_models.items()):
        logger.info(f"\n{provider.upper()} Models ({len(models)}):")
        for model in sorted(models, key=lambda x: x['priority'], reverse=True):
            status_icon = "✅" if model['has_key'] and model['enabled'] else "❌"
            logger.info(f"  {status_icon} {model['name']} (priority: {model['priority']})")
    
    return status


async def test_task_routing():
    """Test routing decisions for various task types."""
    logger.info("\n🚦 Testing Task Routing")
    logger.info("=" * 60)
    
    router = ModelRouter()
    
    # Test scenarios covering different requirements
    test_scenarios = [
        {
            'name': 'Quick Debugging',
            'context': TaskContext(
                task_type=TaskType.DEBUGGING,
                complexity="medium",
                estimated_tokens=4000,
                requires_code=True,
                latency_sensitive=True,  # Favors fast models
                cost_sensitive=False
            )
        },
        {
            'name': 'Large Context Analysis',
            'context': TaskContext(
                task_type=TaskType.ANALYSIS,
                complexity="high",
                estimated_tokens=100000,  # Large context
                requires_reasoning=True,
                latency_sensitive=False
            )
        },
        {
            'name': 'Cost-Sensitive Documentation',
            'context': TaskContext(
                task_type=TaskType.DOCUMENTATION,
                complexity="low",
                estimated_tokens=2000,
                cost_sensitive=True,  # Favors cheap models
                latency_sensitive=False
            )
        },
        {
            'name': 'High-Quality Code Generation',
            'context': TaskContext(
                task_type=TaskType.CODE_GENERATION,
                complexity="high",
                estimated_tokens=8000,
                requires_code=True,
                requires_reasoning=True,
                cost_sensitive=False
            )
        },
        {
            'name': 'Fast Conversation',
            'context': TaskContext(
                task_type=TaskType.CONVERSATION,
                complexity="low",
                estimated_tokens=1000,
                latency_sensitive=True,  # Favors ultra-fast models
                cost_sensitive=True
            )
        }
    ]
    
    routing_results = []
    
    for scenario in test_scenarios:
        logger.info(f"\n📋 Scenario: {scenario['name']}")
        
        try:
            decision = await router.route_task(scenario['context'])
            
            logger.info(f"  Selected: {decision.selected_model}")
            logger.info(f"  Provider: {decision.provider.value}")
            logger.info(f"  Reasoning: {decision.reasoning}")
            
            if decision.estimated_cost:
                logger.info(f"  Est. Cost: ${decision.estimated_cost:.4f}")
            if decision.expected_latency:
                logger.info(f"  Est. Latency: {decision.expected_latency}ms")
            
            logger.info(f"  Fallbacks: {decision.fallback_models[:3]}")
            
            routing_results.append({
                'scenario': scenario['name'],
                'success': True,
                'model': decision.selected_model,
                'provider': decision.provider.value
            })
            
        except Exception as e:
            logger.error(f"  ❌ Routing failed: {e}")
            routing_results.append({
                'scenario': scenario['name'],
                'success': False,
                'error': str(e)
            })
    
    return routing_results


async def test_provider_specific_features():
    """Test provider-specific features and capabilities."""
    logger.info("\n🔧 Testing Provider-Specific Features")
    logger.info("=" * 60)
    
    results = {}
    
    # Test Groq ultra-fast inference
    if os.getenv("GROQ_API_KEY"):
        logger.info("\n⚡ Testing Groq Ultra-Fast Inference...")
        try:
            groq_client = await create_groq_client()
            
            start_time = asyncio.get_event_loop().time()
            response = await groq_client.complete(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": "Say 'fast' three times."}],
                max_tokens=20,
                temperature=0.1
            )
            end_time = asyncio.get_event_loop().time()
            
            latency = (end_time - start_time) * 1000  # Convert to ms
            logger.info(f"  ✅ Groq response in {latency:.0f}ms: {response.content}")
            results['groq'] = {'success': True, 'latency_ms': latency}
            
        except Exception as e:
            logger.error(f"  ❌ Groq test failed: {e}")
            results['groq'] = {'success': False, 'error': str(e)}
    
    # Test OpenRouter model diversity
    if os.getenv("OPENROUTER_API_KEY"):
        logger.info("\n🌐 Testing OpenRouter Model Diversity...")
        try:
            or_client = await create_openrouter_client()
            
            # Get available models
            models = await or_client.get_models()
            unique_providers = set(m.get('id', '').split('/')[0] for m in models if '/' in m.get('id', ''))
            
            logger.info(f"  ✅ OpenRouter provides access to {len(models)} models")
            logger.info(f"  ✅ From {len(unique_providers)} different providers")
            results['openrouter'] = {'success': True, 'model_count': len(models)}
            
        except Exception as e:
            logger.error(f"  ❌ OpenRouter test failed: {e}")
            results['openrouter'] = {'success': False, 'error': str(e)}
    
    return results


async def test_performance_tracking():
    """Test performance tracking and model learning."""
    logger.info("\n📈 Testing Performance Tracking")
    logger.info("=" * 60)
    
    router = ModelRouter()
    
    # Simulate some model interactions
    test_model = "gpt-4o-mini"  # Common model likely to be available
    
    # Record success
    await router.record_task_result(
        model_name=test_model,
        success=True,
        latency_ms=1500,
        cost=0.001
    )
    
    # Record failure
    await router.record_task_result(
        model_name=test_model,
        success=False,
        latency_ms=0,
        error_reason="Test simulated failure"
    )
    
    # Check performance data
    if test_model in router.performance_tracker.performance_data:
        perf = router.performance_tracker.performance_data[test_model]
        logger.info(f"Performance data for {test_model}:")
        logger.info(f"  Success rate: {perf.success_rate:.2%}")
        logger.info(f"  Avg latency: {perf.avg_latency_ms:.0f}ms")
        logger.info(f"  Total requests: {perf.total_requests}")
    else:
        logger.info("No performance data recorded yet")
    
    return True


async def run_ecosystem_tests():
    """Run all ecosystem tests."""
    logger.info("🚀 Multi-Provider Ecosystem Test Suite")
    logger.info("=" * 60)
    logger.info(f"Start time: {datetime.now()}")
    
    all_results = {
        'providers': [],
        'routing': [],
        'features': {},
        'performance': False
    }
    
    # Check provider availability
    available_providers = await check_provider_status()
    all_results['providers'] = available_providers
    
    # Test model availability
    model_status = await test_model_availability()
    
    # Test task routing
    routing_results = await test_task_routing()
    all_results['routing'] = routing_results
    
    # Test provider-specific features
    feature_results = await test_provider_specific_features()
    all_results['features'] = feature_results
    
    # Test performance tracking
    perf_result = await test_performance_tracking()
    all_results['performance'] = perf_result
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 ECOSYSTEM TEST SUMMARY")
    logger.info("=" * 60)
    
    # Provider summary
    logger.info(f"\n✅ Active Providers: {len(all_results['providers'])}/5")
    for provider in all_results['providers']:
        logger.info(f"  - {provider}")
    
    # Routing summary
    successful_routes = sum(1 for r in all_results['routing'] if r['success'])
    logger.info(f"\n✅ Routing Tests: {successful_routes}/{len(all_results['routing'])} passed")
    
    # Feature summary
    if all_results['features']:
        logger.info(f"\n✅ Provider Features:")
        for provider, result in all_results['features'].items():
            if result['success']:
                logger.info(f"  - {provider}: ✅ Working")
            else:
                logger.info(f"  - {provider}: ❌ Failed")
    
    # Overall health
    ecosystem_health = (
        len(all_results['providers']) >= 3 and  # At least 3 providers
        successful_routes > len(all_results['routing']) / 2  # Most routes work
    )
    
    if ecosystem_health:
        logger.info("\n🎉 Multi-Provider Ecosystem is HEALTHY!")
    else:
        logger.info("\n⚠️  Multi-Provider Ecosystem needs attention")
    
    logger.info(f"\nEnd time: {datetime.now()}")
    
    return ecosystem_health


def main():
    """Main entry point."""
    success = asyncio.run(run_ecosystem_tests())
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()