"""
Minimal Global API Key Management System

A lightweight version of the global API key manager that works everywhere
without heavy dependencies, while still providing multi-key failover support.
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)

# Try to load .env file - check multiple locations
try:
    from dotenv import load_dotenv
    
    # Check multiple .env file locations in order of preference
    possible_env_paths = [
        Path(__file__).parent.parent / '.env',  # VS Workspace level
        Path(__file__).parent / '.env',         # ClaudeCode level
        Path.cwd() / '.env',                    # Current directory
    ]
    
    for env_path in possible_env_paths:
        if env_path.exists():
            load_dotenv(env_path)
            logger.info(f"Loaded .env from: {env_path}")
            break
    else:
        logger.info("No .env file found in standard locations")
        
except ImportError:
    logger.info("python-dotenv not available - using environment variables as-is")


class Provider(Enum):
    """Supported providers."""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"
    OPENROUTER = "openrouter"
    GROQ = "groq"


class MinimalKeyStatus:
    """Minimal key status tracking."""
    def __init__(self, key_id: str, provider: str):
        self.key_id = key_id
        self.provider = provider
        self.is_active = True
        self.rate_limited_until = None
        self.total_requests = 0
        self.failed_requests = 0
        self.consecutive_errors = 0
        self.last_success = None
        self.last_failure = None
        self.daily_cost = 0.0
    
    @property
    def is_available(self) -> bool:
        """Check if key is available for use."""
        if not self.is_active:
            return False
        if self.rate_limited_until:
            if datetime.now() < self.rate_limited_until:
                return False
        return True
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 1.0
        return (self.total_requests - self.failed_requests) / self.total_requests


class MinimalMultiKeyManager:
    """
    Minimal multi-key manager with no external dependencies.
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.keys: Dict[str, List[str]] = {}
            self.key_status: Dict[str, MinimalKeyStatus] = {}
            self.active_keys: Dict[str, str] = {}
            self.state_file = Path.home() / '.claudecode' / 'minimal_api_keys_state.json'
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Load state
            self._load_state()
            
            # Discover keys
            self._discover_keys()
            
            self.__class__._initialized = True
    
    def _discover_keys(self):
        """Discover available API keys from environment."""
        provider_patterns = {
            Provider.ANTHROPIC: ['ANTHROPIC_API_KEY', 'ANTHROPIC_API_KEY_2'],
            Provider.OPENAI: ['OPENAI_API_KEY', 'OPENAI_API_KEY_2'],
            Provider.GOOGLE: ['GOOGLE_API_KEY', 'GOOGLE_API_KEY_2'],
            Provider.OPENROUTER: ['OPENROUTER_API_KEY', 'OPENROUTER_API_KEY_2'],
            Provider.GROQ: ['GROQ_API_KEY', 'GROQ_API_KEY_2']
        }
        
        for provider, env_vars in provider_patterns.items():
            provider_keys = []
            for env_var in env_vars:
                value = os.getenv(env_var)
                if value and not value.startswith("your-"):
                    provider_keys.append(value)
            
            if provider_keys:
                self.keys[provider.value] = provider_keys
                
                # Initialize status for each key
                for i, key in enumerate(provider_keys):
                    key_id = f"{provider.value}_key_{i+1}"
                    if key_id not in self.key_status:
                        self.key_status[key_id] = MinimalKeyStatus(key_id, provider.value)
                    
                    # Set first key as active if none set
                    if provider.value not in self.active_keys:
                        self.active_keys[provider.value] = key_id
    
    def get_active_key(self, provider: str) -> Optional[str]:
        """Get the currently active API key for a provider."""
        if provider not in self.keys:
            return None
        
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
    
    def _find_available_key(self, provider: str) -> Optional[str]:
        """Find an available key for the provider."""
        if provider not in self.keys:
            return None
        
        # Get all key IDs for this provider
        provider_key_ids = [
            key_id for key_id, status in self.key_status.items()
            if status.provider == provider
        ]
        
        # Find available keys
        available_keys = [
            key_id for key_id in provider_key_ids
            if self.key_status[key_id].is_available
        ]
        
        if not available_keys:
            return None
        
        # Return key with best success rate
        best_key = max(available_keys, key=lambda k: self.key_status[k].success_rate)
        return best_key
    
    def record_result(self, provider: str, success: bool, error_type: Optional[str] = None, cost: float = 0.0):
        """Record the result of an API request."""
        active_key_id = self.active_keys.get(provider)
        if not active_key_id or active_key_id not in self.key_status:
            return
        
        status = self.key_status[active_key_id]
        status.total_requests += 1
        status.daily_cost += cost
        
        if success:
            status.last_success = datetime.now()
            status.consecutive_errors = 0
        else:
            status.failed_requests += 1
            status.last_failure = datetime.now()
            status.consecutive_errors += 1
            
            # Handle rate limiting
            if error_type and "rate" in error_type.lower():
                status.rate_limited_until = datetime.now() + timedelta(minutes=15)
                logger.info(f"Key {active_key_id} rate limited until {status.rate_limited_until}")
                self._attempt_failover(provider)
            
            # Failover after consecutive errors
            elif status.consecutive_errors >= 3:
                logger.warning(f"Key {active_key_id} has {status.consecutive_errors} consecutive errors")
                self._attempt_failover(provider)
        
        # Save state periodically
        if status.total_requests % 10 == 0:
            self._save_state()
    
    def _attempt_failover(self, provider: str):
        """Attempt to failover to another key."""
        available_key = self._find_available_key(provider)
        
        if available_key and available_key != self.active_keys.get(provider):
            old_key = self.active_keys.get(provider, "none")
            self.active_keys[provider] = available_key
            logger.info(f"Failed over {provider} from {old_key} to {available_key}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get system status."""
        active_providers = len([p for p in self.keys if any(
            self.key_status[kid].is_available 
            for kid in self.key_status 
            if self.key_status[kid].provider == p
        )])
        
        return {
            "available": True,
            "total_providers": len(self.keys),
            "active_providers": active_providers,
            "total_keys": sum(len(keys) for keys in self.keys.values()),
            "providers": {
                provider: {
                    "available": any(
                        self.key_status[kid].is_available 
                        for kid in self.key_status 
                        if self.key_status[kid].provider == provider
                    ),
                    "total_keys": len(self.keys.get(provider, [])),
                    "active_key": self.active_keys.get(provider)
                }
                for provider in self.keys
            }
        }
    
    def _save_state(self):
        """Save state to disk."""
        try:
            state = {
                "active_keys": self.active_keys,
                "key_status": {}
            }
            
            for key_id, status in self.key_status.items():
                state["key_status"][key_id] = {
                    "provider": status.provider,
                    "is_active": status.is_active,
                    "total_requests": status.total_requests,
                    "failed_requests": status.failed_requests,
                    "consecutive_errors": status.consecutive_errors,
                    "daily_cost": status.daily_cost,
                    "rate_limited_until": status.rate_limited_until.isoformat() if status.rate_limited_until else None,
                    "last_success": status.last_success.isoformat() if status.last_success else None,
                    "last_failure": status.last_failure.isoformat() if status.last_failure else None
                }
            
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    def _load_state(self):
        """Load state from disk."""
        if not self.state_file.exists():
            return
        
        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            
            self.active_keys = state.get("active_keys", {})
            
            for key_id, status_data in state.get("key_status", {}).items():
                status = MinimalKeyStatus(key_id, status_data["provider"])
                status.is_active = status_data.get("is_active", True)
                status.total_requests = status_data.get("total_requests", 0)
                status.failed_requests = status_data.get("failed_requests", 0)
                status.consecutive_errors = status_data.get("consecutive_errors", 0)
                status.daily_cost = status_data.get("daily_cost", 0.0)
                
                if status_data.get("rate_limited_until"):
                    status.rate_limited_until = datetime.fromisoformat(status_data["rate_limited_until"])
                if status_data.get("last_success"):
                    status.last_success = datetime.fromisoformat(status_data["last_success"])
                if status_data.get("last_failure"):
                    status.last_failure = datetime.fromisoformat(status_data["last_failure"])
                
                self.key_status[key_id] = status
                
        except Exception as e:
            logger.error(f"Failed to load state: {e}")


# Global instance
_minimal_manager = MinimalMultiKeyManager()


# Provider mapping
PROVIDER_MAP = {
    'ANTHROPIC_API_KEY': Provider.ANTHROPIC.value,
    'OPENAI_API_KEY': Provider.OPENAI.value,
    'GOOGLE_API_KEY': Provider.GOOGLE.value,
    'OPENROUTER_API_KEY': Provider.OPENROUTER.value,
    'GROQ_API_KEY': Provider.GROQ.value
}


def get_api_key(env_var: str) -> Optional[str]:
    """Get API key with failover support."""
    # Map env var to provider
    provider = None
    for pattern, prov in PROVIDER_MAP.items():
        # Check exact match or numbered variant (e.g., ANTHROPIC_API_KEY_2)
        if env_var == pattern or env_var.startswith(pattern + '_'):
            provider = prov
            break
    
    if provider:
        return _minimal_manager.get_active_key(provider)
    
    # Fallback to direct access
    return _original_getenv(env_var)


def record_api_result(env_var: str, success: bool, error_type: Optional[str] = None, cost: float = 0.0):
    """Record API result for failover logic."""
    # Map env var to provider
    provider = None
    for pattern, prov in PROVIDER_MAP.items():
        # Check exact match or numbered variant (e.g., ANTHROPIC_API_KEY_2)
        if env_var == pattern or env_var.startswith(pattern + '_'):
            provider = prov
            break
    
    if provider:
        _minimal_manager.record_result(provider, success, error_type, cost)


def get_api_status() -> Dict[str, Any]:
    """Get system status."""
    return _minimal_manager.get_status()


# Environment interceptor
_original_getenv = os.getenv

def _smart_getenv(key: str, default=None):
    """Smart getenv with failover support."""
    # Check if it's an API key we manage
    if any(key == pattern or key.startswith(pattern + '_') for pattern in PROVIDER_MAP):
        api_key = get_api_key(key)
        if api_key:
            return api_key
    
    # Fallback to original
    return _original_getenv(key, default)


def enable_global_failover():
    """Enable global failover by replacing os.getenv."""
    os.getenv = _smart_getenv
    logger.info("Global API key failover enabled")


def disable_global_failover():
    """Disable global failover."""
    os.getenv = _original_getenv
    logger.info("Global API key failover disabled")


# Log status on import
if __name__ != "__main__":
    status = get_api_status()
    logger.info(f"Minimal global API manager initialized: {status.get('active_providers', 0)}/{status.get('total_providers', 0)} providers")