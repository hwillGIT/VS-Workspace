"""
Global Agent Activation System

Provides automatic activation and integration of the Self-Reflecting Agent system
when Claude Code starts in any directory.
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Import agent components
try:
    from .global_manager import GlobalAgentManager
    from .hooks import CLAUDEMDIntegration, apply_claude_md_pre_execution
    from .main import SelfReflectingAgent
except ImportError:
    # Handle case where package isn't installed yet
    GlobalAgentManager = None
    CLAUDEMDIntegration = None
    SelfReflectingAgent = None


class GlobalActivationSystem:
    """
    System for automatically activating and configuring agents based on
    directory context and CLAUDE.md files.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.manager = None
        self.claude_integration = None
        
        # Initialize components if available
        if GlobalAgentManager:
            self.manager = GlobalAgentManager()
        if CLAUDEMDIntegration:
            self.claude_integration = CLAUDEMDIntegration()
    
    def is_available(self) -> bool:
        """Check if the agent system is available."""
        return self.manager is not None and self.claude_integration is not None
    
    def detect_project_context(self, directory: Path) -> Dict[str, Any]:
        """
        Detect project context from directory structure and files.
        
        Args:
            directory: Directory to analyze
            
        Returns:
            Dictionary with project context information
        """
        
        context = {
            'project_root': str(directory),
            'project_type': 'general',
            'has_claude_md': False,
            'claude_config': None,
            'technologies': [],
            'project_indicators': []
        }
        
        # Check for CLAUDE.md
        claude_md_path = directory / 'CLAUDE.md'
        if claude_md_path.exists():
            context['has_claude_md'] = True
            if self.claude_integration:
                context['claude_config'] = self.claude_integration.parser.parse_claude_md(claude_md_path)
                context['technologies'] = context['claude_config'].get('technologies', [])
        
        # Detect project type from files
        project_indicators = {
            'python': ['requirements.txt', 'pyproject.toml', 'setup.py', 'Pipfile', 'poetry.lock'],
            'javascript': ['package.json', 'yarn.lock', 'npm-shrinkwrap.json'],
            'typescript': ['tsconfig.json', 'package.json'],
            'rust': ['Cargo.toml', 'Cargo.lock'],
            'go': ['go.mod', 'go.sum'],
            'java': ['pom.xml', 'build.gradle', 'gradle.properties'],
            'csharp': ['*.csproj', '*.sln', 'Directory.Build.props'],
            'php': ['composer.json', 'composer.lock'],
            'ruby': ['Gemfile', 'Gemfile.lock'],
            'docker': ['Dockerfile', 'docker-compose.yml', 'docker-compose.yaml'],
            'k8s': ['*.yaml', 'kustomization.yaml'],
            'terraform': ['*.tf', 'terraform.tfvars']
        }
        
        detected_types = []
        found_indicators = []
        
        for project_type, indicators in project_indicators.items():
            for indicator in indicators:
                if indicator.startswith('*.'):
                    # Glob pattern
                    extension = indicator[2:]
                    if any(f.suffix == f'.{extension}' for f in directory.glob('**/*') if f.is_file()):
                        detected_types.append(project_type)
                        found_indicators.append(f'*{extension} files')
                        break
                else:
                    # Exact file match
                    if (directory / indicator).exists():
                        detected_types.append(project_type)
                        found_indicators.append(indicator)
                        break
        
        # Prioritize project types
        if 'python' in detected_types:
            context['project_type'] = 'python'
        elif 'javascript' in detected_types or 'typescript' in detected_types:
            context['project_type'] = 'javascript'
        elif detected_types:
            context['project_type'] = detected_types[0]
        
        context['project_indicators'] = found_indicators
        
        # Check for git repository
        if (directory / '.git').exists():
            context['is_git_repo'] = True
            context['project_indicators'].append('.git')
        else:
            context['is_git_repo'] = False
        
        return context
    
    async def activate_for_directory(self, directory: Path) -> Optional[Dict[str, Any]]:
        """
        Activate the agent system for a specific directory.
        
        Args:
            directory: Directory to activate agents for
            
        Returns:
            Activation result with agent information
        """
        
        if not self.is_available():
            self.logger.warning("Agent system not available - package may not be installed")
            return None
        
        try:
            # Detect project context
            context = self.detect_project_context(directory)
            
            # Get or create agent for this project
            agent = await self.manager.get_agent_for_project(
                project_path=str(directory),
                project_type=context['project_type'],
                claude_config=context.get('claude_config')
            )
            
            # Apply any pre-execution hooks
            if context['has_claude_md']:
                enhanced_context = await apply_claude_md_pre_execution(
                    str(directory), 
                    context
                )
                context.update(enhanced_context)
            
            activation_result = {
                'status': 'activated',
                'project_context': context,
                'available_domains': agent.list_available_domains() if hasattr(agent, 'list_available_domains') else [],
                'agent_id': id(agent),
                'capabilities': self._get_agent_capabilities(agent, context)
            }
            
            self.logger.info(f"Agent system activated for {directory}")
            return activation_result
            
        except Exception as e:
            self.logger.error(f"Failed to activate agent system for {directory}: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'project_context': self.detect_project_context(directory)
            }
    
    def _get_agent_capabilities(self, agent: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get capabilities information for the activated agent."""
        
        capabilities = {
            'core_tasks': [
                'execute_task',
                'code_generation',
                'code_review',
                'architecture_analysis'
            ],
            'domains': [],
            'workflows': [],
            'specialized_for': context.get('technologies', [])
        }
        
        # Get domain-specific capabilities
        if hasattr(agent, 'list_available_domains'):
            try:
                domains = agent.list_available_domains()
                capabilities['domains'] = domains
                
                # Get workflows for each domain
                for domain in domains:
                    if hasattr(agent, 'list_domain_workflows'):
                        workflows = agent.list_domain_workflows(domain)
                        capabilities['workflows'].extend([f"{domain}.{wf}" for wf in workflows])
                        
            except Exception as e:
                self.logger.warning(f"Could not get domain capabilities: {e}")
        
        # Add CLAUDE.md specific capabilities
        claude_config = context.get('claude_config', {})
        if claude_config:
            preferred_workflows = claude_config.get('workflows', [])
            if preferred_workflows:
                capabilities['preferred_workflows'] = [
                    wf.get('name') if isinstance(wf, dict) else wf 
                    for wf in preferred_workflows
                ]
        
        return capabilities
    
    def get_activation_status(self, directory: Path) -> Dict[str, Any]:
        """
        Get the current activation status for a directory.
        
        Args:
            directory: Directory to check
            
        Returns:
            Status information
        """
        
        status = {
            'is_activated': False,
            'agent_available': self.is_available(),
            'project_context': self.detect_project_context(directory)
        }
        
        if self.manager:
            # Check if we have an active agent for this project
            project_info = self.manager.get_project_info(str(directory))
            if project_info:
                status['is_activated'] = True
                status['project_info'] = project_info
        
        return status
    
    def suggest_next_actions(self, context: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Suggest next actions based on project context.
        
        Args:
            context: Project context information
            
        Returns:
            List of suggested actions
        """
        
        suggestions = []
        
        # Basic suggestions based on project type
        project_type = context.get('project_type', 'general')
        has_claude_md = context.get('has_claude_md', False)
        
        if not has_claude_md:
            suggestions.append({
                'action': 'create_claude_md',
                'description': 'Create CLAUDE.md file to configure agent behavior for this project',
                'command': 'sra task "Create a CLAUDE.md configuration file for this project"'
            })
        
        if project_type == 'python':
            suggestions.extend([
                {
                    'action': 'architecture_review',
                    'description': 'Review Python project architecture and structure',
                    'command': 'sra workflow software_development architecture_review "Review Python project architecture"'
                },
                {
                    'action': 'code_quality_audit',
                    'description': 'Audit code quality and adherence to Python best practices',
                    'command': 'sra workflow software_development code_quality_audit "Audit Python code quality"'
                }
            ])
        elif project_type == 'javascript':
            suggestions.extend([
                {
                    'action': 'web_app_planning',
                    'description': 'Plan web application architecture and features',
                    'command': 'sra workflow software_development web_application_planning "Plan web application"'
                },
                {
                    'action': 'performance_audit',
                    'description': 'Analyze JavaScript application performance',
                    'command': 'sra workflow software_development system_analysis "Analyze application performance"'
                }
            ])
        
        # General suggestions
        suggestions.extend([
            {
                'action': 'comprehensive_planning',
                'description': 'Create comprehensive project plan with multi-perspective analysis',
                'command': 'sra workflow software_development comprehensive_project_planning "Plan project development"'
            },
            {
                'action': 'system_info',
                'description': 'Show detailed information about available agents and capabilities',
                'command': 'sra info'
            }
        ])
        
        return suggestions[:5]  # Return top 5 suggestions


# Global instance
_global_activation_system = None


def get_activation_system() -> GlobalActivationSystem:
    """Get the global activation system instance."""
    global _global_activation_system
    if _global_activation_system is None:
        _global_activation_system = GlobalActivationSystem()
    return _global_activation_system


async def auto_activate_for_current_directory() -> Dict[str, Any]:
    """
    Automatically activate the agent system for the current working directory.
    
    Returns:
        Activation result
    """
    
    current_dir = Path.cwd()
    activation_system = get_activation_system()
    
    return await activation_system.activate_for_directory(current_dir)


def print_activation_info(activation_result: Dict[str, Any]) -> None:
    """
    Print activation information to the user.
    
    Args:
        activation_result: Result from activation process
    """
    
    if activation_result.get('status') == 'activated':
        context = activation_result.get('project_context', {})
        capabilities = activation_result.get('capabilities', {})
        
        print("🤖 Self-Reflecting Agent System - ACTIVATED")
        print("=" * 50)
        print(f"📁 Project: {context.get('project_root', 'Unknown')}")
        print(f"🏷️  Type: {context.get('project_type', 'general')}")
        print(f"📄 CLAUDE.md: {'✅ Found' if context.get('has_claude_md') else '❌ Not found'}")
        
        if context.get('technologies'):
            print(f"🔧 Technologies: {', '.join(context['technologies'])}")
        
        domains = capabilities.get('domains', [])
        if domains:
            print(f"🌐 Available Domains: {', '.join(domains)}")
        
        workflows = capabilities.get('preferred_workflows', [])
        if workflows:
            print(f"⚙️  Preferred Workflows: {', '.join(workflows[:3])}{'...' if len(workflows) > 3 else ''}")
        
        # Show suggestions
        activation_system = get_activation_system()
        suggestions = activation_system.suggest_next_actions(context)
        
        if suggestions:
            print(f"\n💡 Suggested Actions:")
            for i, suggestion in enumerate(suggestions[:3], 1):
                print(f"  {i}. {suggestion['description']}")
                print(f"     Command: {suggestion['command']}")
        
        print(f"\n📚 Use 'sra info' for detailed system information")
        print(f"📖 Use 'sra --help' for available commands")
        
    elif activation_result.get('status') == 'failed':
        print("❌ Agent System Activation Failed")
        print(f"Error: {activation_result.get('error', 'Unknown error')}")
        
        # Still show project context
        context = activation_result.get('project_context', {})
        if context:
            print(f"\n📁 Project Context:")
            print(f"  Root: {context.get('project_root')}")
            print(f"  Type: {context.get('project_type')}")
            print(f"  CLAUDE.md: {'Found' if context.get('has_claude_md') else 'Not found'}")
    
    else:
        print("⚠️ Agent system not available")
        print("Run 'python install.py' to install the agent system globally")


# CLI integration function
def integrate_with_claude_code():
    """
    Integrate with Claude Code startup process.
    This function can be called when Claude Code starts in a new directory.
    """
    
    try:
        # Check if we're in a directory that could benefit from agent activation
        current_dir = Path.cwd()
        activation_system = get_activation_system()
        
        if not activation_system.is_available():
            # Don't spam if the system isn't installed
            return
        
        status = activation_system.get_activation_status(current_dir)
        
        # Only show info if this looks like a development project
        context = status.get('project_context', {})
        if (context.get('project_indicators') or 
            context.get('has_claude_md') or 
            context.get('project_type') != 'general'):
            
            print("\n" + "="*60)
            print("🔍 Self-Reflecting Agent System Detected Project")
            print("="*60)
            
            # Run activation asynchronously
            activation_result = asyncio.run(
                activation_system.activate_for_directory(current_dir)
            )
            
            print_activation_info(activation_result)
            print("="*60)
    
    except Exception as e:
        # Fail silently to not interfere with Claude Code startup
        logger = logging.getLogger(__name__)
        logger.debug(f"Agent activation failed: {e}")


if __name__ == "__main__":
    # For testing
    asyncio.run(auto_activate_for_current_directory())