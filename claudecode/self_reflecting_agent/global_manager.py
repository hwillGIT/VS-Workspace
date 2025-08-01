"""
Global Agent Manager

Manages Self-Reflecting Agent instances across different projects and directories,
with persistent configuration and automatic project context detection.
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import hashlib

from .main import SelfReflectingAgent


class GlobalAgentManager:
    """
    Manages agent instances globally across different projects.
    
    Features:
    - Project-specific agent instances with persistent state
    - Automatic configuration based on project type
    - Global configuration management
    - Agent instance caching and reuse
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Global configuration directory
        self.global_config_dir = self._get_global_config_dir()
        self.global_config_file = self.global_config_dir / 'global_config.yaml'
        self.projects_db_file = self.global_config_dir / 'projects.json'
        
        # Agent instance cache
        self.agent_instances: Dict[str, SelfReflectingAgent] = {}
        
        # Ensure global config directory exists
        self.global_config_dir.mkdir(parents=True, exist_ok=True)
        
        # Load global configuration
        self.global_config = self._load_global_config()
        
        # Load projects database
        self.projects_db = self._load_projects_db()
    
    def _get_global_config_dir(self) -> Path:
        """Get the global configuration directory."""
        
        # Use user's home directory for global config
        if os.name == 'nt':  # Windows
            config_dir = Path(os.environ.get('APPDATA', Path.home())) / 'SelfReflectingAgent'
        else:  # Unix-like
            config_dir = Path.home() / '.self_reflecting_agent'
        
        return config_dir
    
    def _load_global_config(self) -> Dict[str, Any]:
        """Load global configuration."""
        
        if self.global_config_file.exists():
            try:
                with open(self.global_config_file, 'r') as f:
                    config = yaml.safe_load(f)
                return config or {}
            except Exception as e:
                self.logger.warning(f"Could not load global config: {e}")
        
        # Create default global configuration
        default_config = {
            'version': '1.0.0',
            'logging': {
                'level': 'INFO',
                'file_logging': True
            },
            'agent_defaults': {
                'enable_memory': True,
                'enable_self_improvement': True,
                'context_management': True
            },
            'project_types': {
                'python': {
                    'domains': ['software_development'],
                    'preferred_workflows': ['architecture_review', 'code_quality_audit']
                },
                'javascript': {
                    'domains': ['software_development'],
                    'preferred_workflows': ['web_application_planning', 'performance_optimization']
                },
                'general': {
                    'domains': ['software_development'],
                    'preferred_workflows': ['comprehensive_project_planning']
                }
            },
            'claude_md_integration': {
                'enabled': True,
                'auto_parse': True,
                'update_context': True
            }
        }
        
        # Save default configuration
        self._save_global_config(default_config)
        return default_config
    
    def _save_global_config(self, config: Dict[str, Any]) -> None:
        """Save global configuration."""
        
        try:
            with open(self.global_config_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
        except Exception as e:
            self.logger.error(f"Could not save global config: {e}")
    
    def _load_projects_db(self) -> Dict[str, Any]:
        """Load projects database."""
        
        if self.projects_db_file.exists():
            try:
                with open(self.projects_db_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Could not load projects database: {e}")
        
        return {'projects': {}}
    
    def _save_projects_db(self) -> None:
        """Save projects database."""
        
        try:
            with open(self.projects_db_file, 'w') as f:
                json.dump(self.projects_db, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Could not save projects database: {e}")
    
    def _get_project_hash(self, project_path: str) -> str:
        """Get a unique hash for a project path."""
        
        return hashlib.md5(str(Path(project_path).resolve()).encode()).hexdigest()[:12]
    
    def _create_project_config(
        self, 
        project_path: str, 
        project_type: str, 
        claude_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create configuration for a specific project."""
        
        # Start with global defaults
        base_config = self._get_base_agent_config()
        
        # Apply project type specific configuration
        project_type_config = self.global_config.get('project_types', {}).get(project_type, {})
        
        # Update domain configuration based on project type
        if 'domains' in project_type_config:
            base_config['domain_agents'] = {}
            for domain in project_type_config['domains']:
                base_config['domain_agents'][domain] = {
                    'enabled': True,
                    'config_path': f'domains/{domain}/config.yaml',
                    'agents_path': f'domains/{domain}/agents'
                }
        
        # Integrate CLAUDE.md configuration if available
        if claude_config:
            # Enhance configuration based on CLAUDE.md content
            if claude_config.get('technologies'):
                base_config['project_context'] = {
                    'technologies': claude_config['technologies'],
                    'description': claude_config.get('project_description', ''),
                    'requirements': claude_config.get('requirements', []),
                    'constraints': claude_config.get('constraints', [])
                }
        
        return base_config
    
    def _get_base_agent_config(self) -> Dict[str, Any]:
        """Get base agent configuration."""
        
        # Load the base config from the package
        package_dir = Path(__file__).parent
        base_config_file = package_dir / 'config.yaml'
        
        if base_config_file.exists():
            try:
                with open(base_config_file, 'r') as f:
                    return yaml.safe_load(f)
            except Exception as e:
                self.logger.warning(f"Could not load base config: {e}")
        
        # Fallback minimal configuration
        return {
            'agents': {
                'manager': {'model': 'gpt-4o', 'temperature': 0.1},
                'coder': {'model': 'gpt-4o', 'temperature': 0.2},
                'reviewer': {'model': 'gpt-4o', 'temperature': 0.1},
                'researcher': {'model': 'gpt-4o', 'temperature': 0.3}
            },
            'domain_agents': {
                'software_development': {
                    'enabled': True,
                    'config_path': 'domains/software_development/config.yaml',
                    'agents_path': 'domains/software_development/agents'
                }
            },
            'workflows': {
                'development': {'max_iterations': 10, 'enable_parallel_execution': True}
            },
            'memory': {'enabled': True, 'provider': 'mem0'},
            'rag': {'enabled': True, 'hybrid': {'bm25_weight': 0.3, 'vector_weight': 0.7}},
            'context': {'max_context_length': 32000, 'compression_enabled': True},
            'evaluation': {'llm_as_judge': {'enabled': True}}
        }
    
    async def get_agent_for_project(
        self,
        project_path: str,
        project_type: str = 'general',
        claude_config: Optional[Dict[str, Any]] = None
    ) -> SelfReflectingAgent:
        """Get or create an agent instance for a specific project."""
        
        project_hash = self._get_project_hash(project_path)
        
        # Check if we already have an agent instance for this project
        if project_hash in self.agent_instances:
            return self.agent_instances[project_hash]
        
        try:
            # Create project-specific configuration
            project_config = self._create_project_config(project_path, project_type, claude_config)
            
            # Create agent instance
            self.logger.info(f"Creating agent instance for project: {project_path}")
            
            agent = SelfReflectingAgent(
                project_path=project_path,
                enable_memory=self.global_config.get('agent_defaults', {}).get('enable_memory', True),
                enable_self_improvement=self.global_config.get('agent_defaults', {}).get('enable_self_improvement', True)
            )
            
            # Override the config with our project-specific config
            agent.config = project_config
            
            # Initialize the agent
            await agent.initialize()
            
            # Cache the agent instance
            self.agent_instances[project_hash] = agent
            
            # Update projects database
            self.projects_db['projects'][project_hash] = {
                'path': str(Path(project_path).resolve()),
                'type': project_type,
                'created_at': str(Path(project_path).stat().st_ctime) if Path(project_path).exists() else None,
                'last_accessed': str(Path(project_path).stat().st_atime) if Path(project_path).exists() else None,
                'has_claude_md': claude_config is not None,
                'technologies': claude_config.get('technologies', []) if claude_config else []
            }
            self._save_projects_db()
            
            self.logger.info(f"Agent instance created and cached for project: {project_path}")
            
            return agent
            
        except Exception as e:
            self.logger.error(f"Failed to create agent for project {project_path}: {e}")
            raise
    
    def list_projects(self) -> Dict[str, Any]:
        """List all registered projects."""
        
        return self.projects_db.get('projects', {})
    
    def get_project_info(self, project_path: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific project."""
        
        project_hash = self._get_project_hash(project_path)
        return self.projects_db.get('projects', {}).get(project_hash)
    
    async def cleanup_inactive_agents(self, max_inactive_hours: int = 24) -> int:
        """Clean up agent instances that haven't been used recently."""
        
        # This would implement cleanup logic based on last access time
        # For now, just clear all cached instances
        
        cleanup_count = len(self.agent_instances)
        
        # Shutdown all cached agents
        for agent in self.agent_instances.values():
            try:
                if hasattr(agent, 'shutdown'):
                    await agent.shutdown()
            except Exception as e:
                self.logger.warning(f"Error shutting down agent: {e}")
        
        # Clear the cache
        self.agent_instances.clear()
        
        self.logger.info(f"Cleaned up {cleanup_count} agent instances")
        return cleanup_count
    
    def update_global_config(self, updates: Dict[str, Any]) -> None:
        """Update global configuration."""
        
        def deep_update(base_dict, update_dict):
            for key, value in update_dict.items():
                if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        deep_update(self.global_config, updates)
        self._save_global_config(self.global_config)
        
        self.logger.info("Global configuration updated")
    
    def get_global_stats(self) -> Dict[str, Any]:
        """Get global usage statistics."""
        
        return {
            'total_projects': len(self.projects_db.get('projects', {})),
            'active_agents': len(self.agent_instances),
            'global_config_dir': str(self.global_config_dir),
            'projects_by_type': self._get_projects_by_type(),
            'projects_with_claude_md': sum(
                1 for project in self.projects_db.get('projects', {}).values()
                if project.get('has_claude_md', False)
            )
        }
    
    def _get_projects_by_type(self) -> Dict[str, int]:
        """Get project count by type."""
        
        type_counts = {}
        for project in self.projects_db.get('projects', {}).values():
            project_type = project.get('type', 'unknown')
            type_counts[project_type] = type_counts.get(project_type, 0) + 1
        
        return type_counts
    
    async def shutdown(self) -> None:
        """Shutdown the global manager and all agent instances."""
        
        try:
            # Shutdown all cached agents
            for agent in self.agent_instances.values():
                if hasattr(agent, 'shutdown'):
                    await agent.shutdown()
            
            # Clear the cache
            self.agent_instances.clear()
            
            self.logger.info("Global agent manager shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during global manager shutdown: {e}")