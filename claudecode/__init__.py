"""
ClaudeCode Global Initialization

This module ensures that global systems like multi-key API management
are available throughout the entire codebase.
"""

import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Try to initialize global API key management
# First try minimal version (no dependencies)
try:
    from minimal_global_api_keys import enable_global_failover, get_api_status
    
    # Enable global failover by default
    # This intercepts os.getenv() calls to provide transparent multi-key support
    enable_global_failover()
    
    # Log initialization status
    status = get_api_status()
    if status.get('available', False):
        logging.info(f"Global multi-key API management initialized: {status.get('active_providers', 0)} providers active")
    else:
        logging.info("Global API key management initialized (without multi-key support)")
        
except ImportError as e:
    # Try full version as fallback
    try:
        from global_api_keys import enable_global_failover, get_api_status
        enable_global_failover()
        status = get_api_status()
        logging.info(f"Full global API management initialized: {status.get('active_providers', 0)} providers")
    except:
        logging.warning(f"Could not initialize global API key management: {e}")
except Exception as e:
    logging.error(f"Error initializing global systems: {e}")

# Export key functions for easy access
__all__ = ['get_api_key', 'record_api_result', 'get_api_status']

try:
    from minimal_global_api_keys import get_api_key, record_api_result, get_api_status
except ImportError:
    try:
        from global_api_keys import get_api_key as _async_get, get_api_key_sync as get_api_key, record_api_result as _async_record, get_api_status
        
        # Wrap async functions for compatibility
        def record_api_result(env_var: str, success: bool, error_type=None, cost=0.0):
            import asyncio
            asyncio.create_task(_async_record(env_var, success, error_type, cost))
            
    except ImportError:
        # Provide fallback implementations
        import os
        
        def get_api_key(env_var: str):
            return os.getenv(env_var)
        
        def record_api_result(env_var: str, success: bool, error_type=None, cost=0.0):
            pass
        
        def get_api_status():
            return {"available": False, "message": "Global API management not available"}