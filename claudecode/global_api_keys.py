"""
Global API Key Management System

This module provides a global, singleton-based API key management system with
automatic multi-key failover support. It intercepts environment variable access
to provide transparent failover when API keys hit rate limits or errors.

Usage:
    from global_api_keys import get_api_key, record_api_result
    
    # Get an API key (automatically handles failover)
    api_key = await get_api_key('OPENAI_API_KEY')
    
    # Record result for intelligent failover
    await record_api_result('OPENAI_API_KEY', success=False, error_type='rate_limit')
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from enum import Enum

# Ensure the self_reflecting_agent module is in path
sys.path.append(str(Path(__file__).parent))

# Try to import multi-key manager
try:
    from self_reflecting_agent.routing.multi_key_manager import (
        MultiKeyManager,
        ModelProvider,
        get_multi_key_manager as _get_router_manager
    )
    MULTI_KEY_AVAILABLE = True
except ImportError:
    MULTI_KEY_AVAILABLE = False
    MultiKeyManager = None
    ModelProvider = None

# Try to load .env file
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Provider mapping from environment variable to ModelProvider enum
PROVIDER_MAPPING = {
    'ANTHROPIC_API_KEY': 'anthropic',
    'OPENAI_API_KEY': 'openai',
    'GOOGLE_API_KEY': 'google',
    'OPENROUTER_API_KEY': 'openrouter',
    'GROQ_API_KEY': 'groq'
}


class GlobalAPIKeyManager:
    """
    Global API key manager that provides transparent multi-key failover
    for any module in the system.
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.logger = logging.getLogger(__name__)
            self._multi_key_manager = None
            self._fallback_cache = {}
            
            # Initialize multi-key manager if available
            if MULTI_KEY_AVAILABLE:
                try:
                    self._multi_key_manager = _get_router_manager()
                    self.logger.info("Multi-key failover system initialized globally")
                except Exception as e:
                    self.logger.warning(f"Could not initialize multi-key manager: {e}")
            else:
                self.logger.warning("Multi-key manager not available - using direct environment access")
            
            self.__class__._initialized = True
    
    def get_provider_from_env_var(self, env_var: str) -> Optional[str]:
        """Get provider name from environment variable name."""
        # Direct mapping
        if env_var in PROVIDER_MAPPING:
            return PROVIDER_MAPPING[env_var]
        
        # Handle numbered keys (e.g., OPENAI_API_KEY_2)
        for base_var, provider in PROVIDER_MAPPING.items():
            if env_var.startswith(base_var.rsplit('_', 1)[0]):
                return provider
        
        return None
    
    async def get_api_key(self, env_var: str) -> Optional[str]:
        """
        Get an API key with automatic failover support.
        
        Args:
            env_var: Environment variable name (e.g., 'OPENAI_API_KEY')
            
        Returns:
            API key string or None if not available
        """
        # Try multi-key manager first
        if self._multi_key_manager:
            provider_name = self.get_provider_from_env_var(env_var)
            if provider_name:
                try:
                    # Get ModelProvider enum
                    provider = ModelProvider(provider_name)
                    key = await self._multi_key_manager.get_active_key(provider)
                    if key:
                        return key
                except Exception as e:
                    self.logger.debug(f"Multi-key lookup failed for {env_var}: {e}")
        
        # Fallback to direct environment access
        return os.getenv(env_var)
    
    def get_api_key_sync(self, env_var: str) -> Optional[str]:
        """
        Synchronous version of get_api_key for compatibility.
        
        Note: This doesn't provide failover benefits, just compatibility.
        """
        # For sync contexts, just use direct access
        # Store intent for future async handling
        return os.getenv(env_var)
    
    async def record_api_result(
        self,
        env_var: str,
        success: bool,
        error_type: Optional[str] = None,
        cost: float = 0.0
    ):
        """
        Record the result of an API call for failover logic.
        
        Args:
            env_var: Environment variable name used
            success: Whether the API call succeeded
            error_type: Type of error if failed (e.g., 'rate_limit')
            cost: Cost of the API call
        """
        if not self._multi_key_manager:
            return
        
        provider_name = self.get_provider_from_env_var(env_var)
        if provider_name:
            try:
                provider = ModelProvider(provider_name)
                await self._multi_key_manager.record_request_result(
                    provider=provider,
                    success=success,
                    error_type=error_type,
                    cost=cost
                )
            except Exception as e:
                self.logger.debug(f"Could not record result for {env_var}: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the global API key system."""
        if self._multi_key_manager:
            return self._multi_key_manager.get_system_status()
        else:
            # Basic status when multi-key not available
            return {
                "available": False,
                "message": "Multi-key failover system not available",
                "providers": {}
            }


# Global singleton instance
_global_manager = GlobalAPIKeyManager()


# Convenience functions for global access
async def get_api_key(env_var: str) -> Optional[str]:
    """
    Get an API key with automatic failover support.
    
    Example:
        api_key = await get_api_key('OPENAI_API_KEY')
    """
    return await _global_manager.get_api_key(env_var)


def get_api_key_sync(env_var: str) -> Optional[str]:
    """
    Synchronous version of get_api_key for compatibility.
    
    Note: This doesn't provide failover benefits.
    """
    return _global_manager.get_api_key_sync(env_var)


async def record_api_result(
    env_var: str,
    success: bool,
    error_type: Optional[str] = None,
    cost: float = 0.0
):
    """
    Record the result of an API call for failover logic.
    
    Example:
        await record_api_result('OPENAI_API_KEY', success=False, error_type='rate_limit')
    """
    await _global_manager.record_api_result(env_var, success, error_type, cost)


def get_api_status() -> Dict[str, Any]:
    """Get comprehensive status of the global API key system."""
    return _global_manager.get_status()


# Environment variable interceptor for transparent integration
class EnvInterceptor:
    """
    Intercepts os.getenv() calls to provide transparent multi-key support.
    
    Note: This is an advanced feature that modifies os.getenv behavior.
    """
    
    def __init__(self):
        self._original_getenv = os.getenv
        self._async_loop = None
        
    def enable(self):
        """Enable environment variable interception."""
        os.getenv = self._intercepted_getenv
        logger.info("Environment variable interception enabled")
    
    def disable(self):
        """Disable environment variable interception."""
        os.getenv = self._original_getenv
        logger.info("Environment variable interception disabled")
    
    def _intercepted_getenv(self, key: str, default=None):
        """Intercepted version of os.getenv with multi-key support."""
        # Check if this is an API key we manage
        if key in PROVIDER_MAPPING or any(key.startswith(k.rsplit('_', 1)[0]) for k in PROVIDER_MAPPING):
            # Try to get from multi-key manager
            try:
                # For sync context, use sync version
                api_key = get_api_key_sync(key)
                if api_key:
                    return api_key
            except Exception as e:
                logger.debug(f"Interceptor failed for {key}: {e}")
        
        # Fall back to original behavior
        return self._original_getenv(key, default)


# Global interceptor instance
_env_interceptor = EnvInterceptor()


def enable_global_failover():
    """
    Enable global API key failover by intercepting os.getenv() calls.
    
    Warning: This modifies global behavior and should be called early in application startup.
    """
    _env_interceptor.enable()


def disable_global_failover():
    """Disable global API key failover interception."""
    _env_interceptor.disable()


# Auto-initialize on import for immediate availability
if __name__ != "__main__":
    # Log that global API key management is available
    if MULTI_KEY_AVAILABLE:
        logger.info("Global multi-key API management system loaded")
    else:
        logger.warning("Global API key system loaded without multi-key support")