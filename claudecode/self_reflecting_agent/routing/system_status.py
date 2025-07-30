#!/usr/bin/env python3
"""
Multi-Key System Status Monitor

Provides comprehensive monitoring and status reporting for the multi-key
API management system including provider health, key status, and usage statistics.
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent.parent / '.env'
    load_dotenv(env_path)
except ImportError:
    pass

# Add the parent directory to the path to import the modules
sys.path.append(str(Path(__file__).parent.parent))

from routing.multi_key_manager import get_multi_key_manager, get_system_status
from routing.model_router import ModelRouter


def format_key_for_display(key: str) -> str:
    """Safely format API key for display (mask sensitive parts)."""
    if not key or len(key) < 16:
        return "***"
    return f"{key[:8]}...{key[-8:]}"


def format_cost(cost: float) -> str:
    """Format cost for display."""
    if cost == 0:
        return "$0.00"
    elif cost < 0.001:
        return f"${cost:.6f}"
    elif cost < 0.01:
        return f"${cost:.4f}"
    else:
        return f"${cost:.2f}"


def format_duration(td: timedelta) -> str:
    """Format timedelta for display."""
    total_seconds = int(td.total_seconds())
    if total_seconds < 60:
        return f"{total_seconds}s"
    elif total_seconds < 3600:
        return f"{total_seconds // 60}m {total_seconds % 60}s"
    else:
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        return f"{hours}h {minutes}m"


class SystemStatusReporter:
    """Comprehensive system status reporter."""
    
    def __init__(self):
        self.manager = get_multi_key_manager()
        self.router = ModelRouter()
    
    def print_header(self, title: str, width: int = 80):
        """Print a formatted header."""
        print("\n" + "=" * width)
        print(f" {title}")
        print("=" * width)
    
    def print_section(self, title: str, width: int = 60):
        """Print a formatted section header."""
        print(f"\n{title}")
        print("-" * width)
    
    def report_environment_status(self):
        """Report on environment variable status."""
        self.print_header("ENVIRONMENT VARIABLES STATUS")
        
        providers = {
            "Anthropic": ["ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY_2"],
            "OpenAI": ["OPENAI_API_KEY", "OPENAI_API_KEY_2"],
            "Google": ["GOOGLE_API_KEY", "GOOGLE_API_KEY_2"],
            "OpenRouter": ["OPENROUTER_API_KEY", "OPENROUTER_API_KEY_2"],
            "Groq": ["GROQ_API_KEY", "GROQ_API_KEY_2"]
        }
        
        total_loaded = 0
        for provider, env_vars in providers.items():
            print(f"\n{provider}:")
            provider_keys = 0
            for env_var in env_vars:
                value = os.getenv(env_var)
                if value and not value.startswith("your-"):
                    print(f"  [+] {env_var}: {format_key_for_display(value)}")
                    provider_keys += 1
                    total_loaded += 1
                else:
                    print(f"  [-] {env_var}: Not set")
            print(f"  Total: {provider_keys}/2 keys")
        
        print(f"\nOverall: {total_loaded} API keys loaded")
    
    def report_provider_status(self):
        """Report detailed provider status."""
        self.print_header("PROVIDER STATUS OVERVIEW")
        
        status = get_system_status()
        
        print(f"Active Providers: {status['active_providers']}/{status['total_providers']}")
        print(f"Configuration Strategy: {status['config']['strategy']}")
        print(f"Rate Limit Backoff: {status['config']['rate_limit_backoff_minutes']} minutes")
        print(f"Max Daily Cost per Key: {format_cost(status['config']['max_daily_cost_per_key'])}")
        
        # Calculate totals
        total_requests = sum(p['total_requests'] for p in status['providers'].values())
        total_cost = sum(p['daily_cost'] for p in status['providers'].values())
        
        print(f"\nSystem Totals:")
        print(f"  Total Requests: {total_requests}")
        print(f"  Total Daily Cost: {format_cost(total_cost)}")
    
    def report_detailed_provider_status(self):
        """Report detailed status for each provider."""
        self.print_header("DETAILED PROVIDER STATUS")
        
        status = get_system_status()
        
        for provider_name, provider_status in status['providers'].items():
            self.print_section(f"{provider_name.upper()} PROVIDER")
            
            # Provider overview
            health_icon = "[+]" if provider_status['available'] else "[-]"
            print(f"{health_icon} Status: {'Available' if provider_status['available'] else 'Unavailable'}")
            print(f"   Keys: {provider_status['available_keys']}/{provider_status['total_keys']} available")
            print(f"   Requests: {provider_status['total_requests']}")
            print(f"   Daily Cost: {format_cost(provider_status['daily_cost'])}")
            
            if provider_status['active_key']:
                print(f"   Active Key: {provider_status['active_key']}")
            
            # Individual key details
            if provider_status.get('key_details'):
                print("\n   Key Details:")
                for key_detail in provider_status['key_details']:
                    status_icon = "[+]" if key_detail['is_available'] else "[-]"
                    active_icon = "*" if key_detail['is_active'] else " "
                    
                    print(f"   {status_icon}{active_icon} {key_detail['key_id']}:")
                    print(f"      Success Rate: {key_detail['success_rate']:.1%}")
                    print(f"      Requests: {key_detail['requests']}")
                    print(f"      Cost: {format_cost(key_detail['cost'])}")
                    
                    if key_detail['rate_limited_until']:
                        rate_limit_time = datetime.fromisoformat(key_detail['rate_limited_until'])
                        if rate_limit_time > datetime.now():
                            remaining = rate_limit_time - datetime.now()
                            print(f"      Rate Limited: {format_duration(remaining)} remaining")
                        else:
                            print(f"      Rate Limited: Expired")
    
    def report_model_router_status(self):
        """Report model router integration status."""
        self.print_header("MODEL ROUTER INTEGRATION")
        
        router_status = self.router.get_model_status()
        
        print(f"Total Models Configured: {router_status['total_models']}")
        print(f"Enabled Models: {router_status['enabled_models']}")
        
        # Group by provider
        models_by_provider = {}
        for model_name, model_info in router_status['models'].items():
            provider = model_info['provider']
            if provider not in models_by_provider:
                models_by_provider[provider] = []
            models_by_provider[provider].append({
                'name': model_name,
                'enabled': model_info['enabled'],
                'priority': model_info['priority'],
                'has_key': model_info['has_api_key']
            })
        
        for provider, models in models_by_provider.items():
            self.print_section(f"{provider.upper()} MODELS")
            
            enabled_models = [m for m in models if m['enabled']]
            available_models = [m for m in enabled_models if m['has_key']]
            
            print(f"Models: {len(available_models)}/{len(enabled_models)} available, {len(models)} total")
            
            for model in sorted(models, key=lambda x: x['priority'], reverse=True):
                if model['enabled']:
                    status_icon = "[+]" if model['has_key'] else "[-]"
                    print(f"  {status_icon} {model['name']} (priority: {model['priority']})")
    
    def report_performance_metrics(self):
        """Report performance metrics and trends."""
        self.print_header("PERFORMANCE METRICS")
        
        # Get performance data from router
        perf_data = self.router.performance_tracker.performance_data
        
        if not perf_data:
            print("No performance data available yet.")
            return
        
        print(f"Models with Performance Data: {len(perf_data)}")
        
        for model_name, perf in perf_data.items():
            print(f"\n{model_name}:")
            print(f"  Success Rate: {perf.success_rate:.1%}")
            print(f"  Avg Latency: {perf.avg_latency_ms:.0f}ms")
            print(f"  Total Requests: {perf.total_requests}")
            print(f"  Avg Cost per Request: {format_cost(perf.avg_cost_per_request)}")
            
            if perf.last_success:
                if isinstance(perf.last_success, str):
                    last_success = datetime.fromisoformat(perf.last_success)
                else:
                    last_success = perf.last_success
                time_since = datetime.now() - last_success
                print(f"  Last Success: {format_duration(time_since)} ago")
    
    def report_health_recommendations(self):
        """Provide health recommendations based on system status."""
        self.print_header("HEALTH RECOMMENDATIONS")
        
        status = get_system_status()
        recommendations = []
        
        # Check provider availability
        unavailable_providers = [
            name for name, info in status['providers'].items()
            if not info['available']
        ]
        
        if unavailable_providers:
            recommendations.append(
                f"[!] {len(unavailable_providers)} provider(s) unavailable: {', '.join(unavailable_providers)}"
            )
        
        # Check for single-key providers
        single_key_providers = [
            name for name, info in status['providers'].items()
            if info['total_keys'] == 1 and info['available']
        ]
        
        if single_key_providers:
            recommendations.append(
                f"[!] {len(single_key_providers)} provider(s) have only 1 key (no failover): {', '.join(single_key_providers)}"
            )
        
        # Check cost levels
        high_cost_providers = [
            name for name, info in status['providers'].items()
            if info['daily_cost'] > 10.0
        ]
        
        if high_cost_providers:
            recommendations.append(
                f"[$] High daily costs detected: {', '.join(high_cost_providers)}"
            )
        
        # Check for rate-limited keys
        rate_limited_count = 0
        for provider_info in status['providers'].values():
            if provider_info.get('key_details'):
                for key_detail in provider_info['key_details']:
                    if key_detail.get('rate_limited_until'):
                        rate_limit_time = datetime.fromisoformat(key_detail['rate_limited_until'])
                        if rate_limit_time > datetime.now():
                            rate_limited_count += 1
        
        if rate_limited_count > 0:
            recommendations.append(
                f"[T] {rate_limited_count} key(s) currently rate limited"
            )
        
        if not recommendations:
            print("[+] System is healthy! No issues detected.")
        else:
            print("Issues and recommendations:")
            for rec in recommendations:
                print(f"  {rec}")
        
        # Positive recommendations
        print(f"\n[+] {status['active_providers']}/{status['total_providers']} providers active")
        
        total_keys = sum(info['total_keys'] for info in status['providers'].values())
        available_keys = sum(info['available_keys'] for info in status['providers'].values())
        print(f"[+] {available_keys}/{total_keys} keys available")
    
    def generate_full_report(self):
        """Generate a comprehensive status report."""
        print(f"Multi-Key API Management System Status Report")
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        self.report_environment_status()
        self.report_provider_status()
        self.report_detailed_provider_status()
        self.report_model_router_status()
        self.report_performance_metrics()
        self.report_health_recommendations()
        
        print(f"\n{'='*80}")
        print("Report completed successfully!")


async def main():
    """Main entry point."""
    reporter = SystemStatusReporter()
    reporter.generate_full_report()


if __name__ == "__main__":
    asyncio.run(main())