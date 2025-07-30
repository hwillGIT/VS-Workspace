"""
Multi-Key API Manager

Handles multiple API keys per provider with automatic failover, rate limit detection,
and intelligent key rotation to maximize API quota utilization.
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Tuple
from pathlib import Path

from .model_router import ModelProvider


@dataclass
class KeyStatus:
    """Status information for an API key."""
    key_id: str
    provider: ModelProvider  
    is_active: bool = True
    rate_limited_until: Optional[datetime] = None
    total_requests: int = 0
    failed_requests: int = 0
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    daily_cost: float = 0.0
    error_count: int = 0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate for this key."""
        if self.total_requests == 0:
            return 1.0
        return (self.total_requests - self.failed_requests) / self.total_requests
    
    @property
    def is_rate_limited(self) -> bool:
        """Check if key is currently rate limited."""
        if not self.rate_limited_until:
            return False
        return datetime.now() < self.rate_limited_until
    
    @property
    def is_available(self) -> bool:
        """Check if key is available for use."""
        return self.is_active and not self.is_rate_limited


class RateLimitStrategy(Enum):
    """Strategies for handling rate limits."""
    FAILOVER = "failover"  # Switch to next available key
    BACKOFF = "backoff"    # Wait and retry with same key
    ROUND_ROBIN = "round_robin"  # Distribute load across keys


@dataclass 
class MultiKeyConfig:
    """Configuration for multi-key management."""
    strategy: RateLimitStrategy = RateLimitStrategy.FAILOVER
    rate_limit_backoff_minutes: int = 15
    max_daily_cost_per_key: float = 50.0
    key_rotation_interval_hours: int = 6
    health_check_interval_minutes: int = 5
    persistent_state_file: str = "multi_key_state.json"


class MultiKeyManager:
    """
    Manages multiple API keys per provider with intelligent failover and load balancing.
    
    Features:
    - Automatic failover on rate limits
    - Key health monitoring and rotation
    - Cost tracking per key
    - Persistent state across restarts
    - Intelligent key selection based on performance
    """
    
    def __init__(self, config: Optional[MultiKeyConfig] = None):
        self.config = config or MultiKeyConfig()
        self.logger = logging.getLogger(__name__)
        
        # Key registry: provider -> list of keys
        self.keys: Dict[ModelProvider, List[str]] = {}
        
        # Key status tracking
        self.key_status: Dict[str, KeyStatus] = {}
        
        # Current active key per provider
        self.active_keys: Dict[ModelProvider, str] = {}
        
        # Load persistent state
        self._load_state()
        
        # Register available keys from environment
        self._discover_keys()
    
    def _discover_keys(self):
        """Discover available API keys from environment variables."""
        
        # Anthropic keys
        anthropic_keys = []
        if os.getenv("ANTHROPIC_API_KEY"):
            anthropic_keys.append(os.getenv("ANTHROPIC_API_KEY"))
        if os.getenv("ANTHROPIC_API_KEY_2"):
            anthropic_keys.append(os.getenv("ANTHROPIC_API_KEY_2"))
        
        if anthropic_keys:
            self.keys[ModelProvider.ANTHROPIC] = anthropic_keys
            self._initialize_key_status(ModelProvider.ANTHROPIC, anthropic_keys)
        
        # OpenAI keys
        openai_keys = []
        if os.getenv("OPENAI_API_KEY"):
            openai_keys.append(os.getenv("OPENAI_API_KEY"))
        if os.getenv("OPENAI_API_KEY_2"):
            openai_keys.append(os.getenv("OPENAI_API_KEY_2"))
        
        if openai_keys:
            self.keys[ModelProvider.OPENAI] = openai_keys
            self._initialize_key_status(ModelProvider.OPENAI, openai_keys)
        
        # Google keys
        google_keys = []
        if os.getenv("GOOGLE_API_KEY"):
            google_keys.append(os.getenv("GOOGLE_API_KEY"))
        if os.getenv("GOOGLE_API_KEY_2"):
            google_keys.append(os.getenv("GOOGLE_API_KEY_2"))
        
        if google_keys:
            self.keys[ModelProvider.GOOGLE] = google_keys
            self._initialize_key_status(ModelProvider.GOOGLE, google_keys)
        
        # OpenRouter keys
        openrouter_keys = []
        if os.getenv("OPENROUTER_API_KEY"):
            openrouter_keys.append(os.getenv("OPENROUTER_API_KEY"))
        if os.getenv("OPENROUTER_API_KEY_2"):
            openrouter_keys.append(os.getenv("OPENROUTER_API_KEY_2"))
        
        if openrouter_keys:
            self.keys[ModelProvider.OPENROUTER] = openrouter_keys
            self._initialize_key_status(ModelProvider.OPENROUTER, openrouter_keys)
        
        # Groq keys
        groq_keys = []
        if os.getenv("GROQ_API_KEY"):
            groq_keys.append(os.getenv("GROQ_API_KEY"))
        if os.getenv("GROQ_API_KEY_2"):
            groq_keys.append(os.getenv("GROQ_API_KEY_2"))
        
        if groq_keys:
            self.keys[ModelProvider.GROQ] = groq_keys
            self._initialize_key_status(ModelProvider.GROQ, groq_keys)
        
        self.logger.info(f"Discovered keys for {len(self.keys)} providers")
        for provider, keys in self.keys.items():
            self.logger.info(f"  {provider.value}: {len(keys)} keys")
    
    def _initialize_key_status(self, provider: ModelProvider, keys: List[str]):
        """Initialize status tracking for provider keys."""
        for i, key in enumerate(keys):
            key_id = f"{provider.value}_key_{i+1}"
            
            if key_id not in self.key_status:
                self.key_status[key_id] = KeyStatus(
                    key_id=key_id,
                    provider=provider
                )
            
            # Set first key as active if no active key set
            if provider not in self.active_keys:
                self.active_keys[provider] = key_id
    
    def get_active_key(self, provider: ModelProvider) -> Optional[str]:
        """Get the currently active API key for a provider."""
        if provider not in self.keys:
            return None
        
        # Get current active key ID
        active_key_id = self.active_keys.get(provider)
        if not active_key_id:
            return None
        
        # Check if active key is available
        key_status = self.key_status.get(active_key_id)
        if not key_status or not key_status.is_available:
            # Try to find an available key
            available_key = self._find_available_key(provider)
            if available_key:
                self.active_keys[provider] = available_key
                active_key_id = available_key
            else:
                return None
        
        # Return the actual API key value
        key_index = int(active_key_id.split('_')[-1]) - 1
        return self.keys[provider][key_index]
    
    def _find_available_key(self, provider: ModelProvider) -> Optional[str]:
        """Find an available key for the provider."""
        if provider not in self.keys:
            return None
        
        # Get all key IDs for this provider
        provider_key_ids = [
            key_id for key_id, status in self.key_status.items()
            if status.provider == provider
        ]
        
        # Sort by success rate and availability
        available_keys = [
            key_id for key_id in provider_key_ids
            if self.key_status[key_id].is_available
        ]
        
        if not available_keys:
            return None
        
        # Return key with best success rate
        best_key = max(available_keys, key=lambda k: self.key_status[k].success_rate)
        return best_key
    
    async def record_request_result(
        self,
        provider: ModelProvider,
        success: bool,
        error_type: Optional[str] = None,
        cost: float = 0.0
    ):
        """Record the result of an API request for tracking and failover logic."""
        
        active_key_id = self.active_keys.get(provider)
        if not active_key_id:
            return
        
        status = self.key_status.get(active_key_id)
        if not status:
            return
        
        # Update request counters
        status.total_requests += 1
        status.daily_cost += cost
        
        if success:
            status.last_success = datetime.now()
        else:
            status.failed_requests += 1
            status.last_failure = datetime.now()
            status.error_count += 1
            
            # Handle rate limiting
            if error_type and "rate" in error_type.lower():
                self._handle_rate_limit(provider, active_key_id)
            
            # Failover if key is consistently failing
            elif status.error_count >= 3:
                self.logger.warning(f"Key {active_key_id} has {status.error_count} consecutive errors, attempting failover")
                self._attempt_failover(provider)
        
        # Reset error count on success
        if success:
            status.error_count = 0
        
        # Save state periodically
        if status.total_requests % 10 == 0:
            self._save_state()
    
    def _handle_rate_limit(self, provider: ModelProvider, key_id: str):
        """Handle rate limiting for a specific key."""
        status = self.key_status.get(key_id)
        if not status:
            return
        
        # Mark key as rate limited
        status.rate_limited_until = datetime.now() + timedelta(
            minutes=self.config.rate_limit_backoff_minutes
        )
        
        self.logger.warning(f"Key {key_id} rate limited until {status.rate_limited_until}")
        
        # Attempt failover
        self._attempt_failover(provider)
    
    def _attempt_failover(self, provider: ModelProvider):
        """Attempt to failover to another available key."""
        available_key = self._find_available_key(provider)
        
        if available_key and available_key != self.active_keys.get(provider):
            old_key = self.active_keys.get(provider, "none")
            self.active_keys[provider] = available_key
            self.logger.info(f"Failed over {provider.value} from {old_key} to {available_key}")
        else:
            self.logger.error(f"No available keys for failover: {provider.value}")
    
    def get_provider_status(self, provider: ModelProvider) -> Dict[str, Any]:
        """Get comprehensive status for a provider."""
        if provider not in self.keys:
            return {"available": False, "keys": 0}
        
        provider_keys = [
            (key_id, status) for key_id, status in self.key_status.items()
            if status.provider == provider
        ]
        
        total_keys = len(provider_keys)
        available_keys = sum(1 for _, status in provider_keys if status.is_available)
        active_key_id = self.active_keys.get(provider)
        
        total_requests = sum(status.total_requests for _, status in provider_keys)
        total_cost = sum(status.daily_cost for _, status in provider_keys)
        
        return {
            "available": available_keys > 0,
            "total_keys": total_keys,
            "available_keys": available_keys,
            "active_key": active_key_id,
            "total_requests": total_requests,
            "daily_cost": total_cost,
            "key_details": [
                {
                    "key_id": key_id,
                    "is_active": key_id == active_key_id,
                    "is_available": status.is_available,
                    "success_rate": status.success_rate,
                    "requests": status.total_requests,
                    "cost": status.daily_cost,
                    "rate_limited_until": status.rate_limited_until.isoformat() if status.rate_limited_until else None
                }
                for key_id, status in provider_keys
            ]
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status across all providers."""
        return {
            "providers": {
                provider.value: self.get_provider_status(provider)
                for provider in self.keys.keys()
            },
            "total_providers": len(self.keys),
            "active_providers": sum(
                1 for provider in self.keys.keys()
                if self.get_provider_status(provider)["available"]
            ),
            "config": {
                "strategy": self.config.strategy.value,
                "rate_limit_backoff_minutes": self.config.rate_limit_backoff_minutes,
                "max_daily_cost_per_key": self.config.max_daily_cost_per_key
            }
        }
    
    def _load_state(self):
        """Load persistent state from disk."""
        state_file = Path(self.config.persistent_state_file)
        if not state_file.exists():
            return
        
        try:
            with open(state_file, 'r') as f:
                data = json.load(f)
            
            # Restore key status
            for key_id, status_data in data.get("key_status", {}).items():
                status = KeyStatus(
                    key_id=key_id,
                    provider=ModelProvider(status_data["provider"]),
                    is_active=status_data.get("is_active", True),
                    total_requests=status_data.get("total_requests", 0),
                    failed_requests=status_data.get("failed_requests", 0),
                    daily_cost=status_data.get("daily_cost", 0.0),
                    error_count=status_data.get("error_count", 0)
                )
                
                # Parse datetime fields
                if status_data.get("rate_limited_until"):
                    status.rate_limited_until = datetime.fromisoformat(status_data["rate_limited_until"])
                if status_data.get("last_success"):
                    status.last_success = datetime.fromisoformat(status_data["last_success"])
                if status_data.get("last_failure"):
                    status.last_failure = datetime.fromisoformat(status_data["last_failure"])
                
                self.key_status[key_id] = status
            
            # Restore active keys
            for provider_str, key_id in data.get("active_keys", {}).items():
                provider = ModelProvider(provider_str)
                self.active_keys[provider] = key_id
            
            self.logger.info(f"Loaded state for {len(self.key_status)} keys")
            
        except Exception as e:
            self.logger.error(f"Failed to load state: {e}")
    
    def _save_state(self):
        """Save persistent state to disk."""
        try:
            data = {
                "key_status": {},
                "active_keys": {}
            }
            
            # Serialize key status
            for key_id, status in self.key_status.items():
                data["key_status"][key_id] = {
                    "provider": status.provider.value,
                    "is_active": status.is_active,
                    "total_requests": status.total_requests,
                    "failed_requests": status.failed_requests,
                    "daily_cost": status.daily_cost,
                    "error_count": status.error_count,
                    "rate_limited_until": status.rate_limited_until.isoformat() if status.rate_limited_until else None,
                    "last_success": status.last_success.isoformat() if status.last_success else None,
                    "last_failure": status.last_failure.isoformat() if status.last_failure else None
                }
            
            # Serialize active keys
            for provider, key_id in self.active_keys.items():
                data["active_keys"][provider.value] = key_id
            
            with open(self.config.persistent_state_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")
    
    def reset_daily_stats(self):
        """Reset daily statistics for all keys."""
        for status in self.key_status.values():
            status.daily_cost = 0.0
        self._save_state()
        self.logger.info("Reset daily statistics for all keys")
    
    async def health_check(self):
        """Perform health checks on all keys."""
        for provider in self.keys.keys():
            # Check if rate limited keys should be restored
            for key_id, status in self.key_status.items():
                if status.provider == provider and status.is_rate_limited:
                    if not status.is_rate_limited:  # Rate limit expired
                        self.logger.info(f"Rate limit expired for {key_id}, key restored")
        
        self._save_state()


# Global instance
_multi_key_manager = None

def get_multi_key_manager() -> MultiKeyManager:
    """Get the global multi-key manager instance."""
    global _multi_key_manager
    if _multi_key_manager is None:
        _multi_key_manager = MultiKeyManager()
    return _multi_key_manager


# Convenience functions for integration
async def get_api_key(provider: ModelProvider) -> Optional[str]:
    """Get an available API key for the specified provider."""
    manager = get_multi_key_manager()
    return manager.get_active_key(provider)


async def record_api_result(provider: ModelProvider, success: bool, error_type: Optional[str] = None, cost: float = 0.0):
    """Record the result of an API call for failover logic."""
    manager = get_multi_key_manager()
    await manager.record_request_result(provider, success, error_type, cost)


def get_provider_status(provider: ModelProvider) -> Dict[str, Any]:
    """Get status information for a provider."""
    manager = get_multi_key_manager()
    return manager.get_provider_status(provider)


def get_system_status() -> Dict[str, Any]:
    """Get comprehensive system status."""
    manager = get_multi_key_manager()
    return manager.get_system_status()