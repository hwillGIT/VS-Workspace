"""
Domain-specific agents for specialized tasks.

This package organizes agents by domain expertise to enable focused,
specialized capabilities while maintaining the core system architecture.
"""

from .domain_manager import DomainManager
from .software_development import *
# Future domains can be imported here as they are implemented

__all__ = ["DomainManager"]