"""
CLAUDE.md Integration Hooks

Provides automatic integration with CLAUDE.md files for enhanced project context
and agent configuration based on project-specific requirements.
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import yaml


class CLAUDEMDParser:
    """
    Parser for CLAUDE.md files that extracts structured information
    for agent configuration and project context.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def parse_claude_md(self, file_path: Path) -> Dict[str, Any]:
        """
        Parse a CLAUDE.md file and extract structured information.
        
        Args:
            file_path: Path to the CLAUDE.md file
            
        Returns:
            Dictionary with parsed information including project context,
            agent preferences, workflows, and constraints.
        """
        
        if not file_path.exists():
            return {}
        
        try:
            content = file_path.read_text(encoding='utf-8')
            return self._parse_content(content)
        except Exception as e:
            self.logger.warning(f"Could not parse CLAUDE.md: {e}")
            return {}
    
    def _parse_content(self, content: str) -> Dict[str, Any]:
        """Parse the content of a CLAUDE.md file."""
        
        sections = self._extract_sections(content)
        
        parsed_data = {
            'raw_content': content,
            'project_description': self._extract_project_description(sections),
            'technologies': self._extract_technologies(sections),
            'capabilities': self._extract_capabilities(sections),
            'workflows': self._extract_preferred_workflows(sections),
            'domain_agents': self._extract_domain_agents(sections),
            'guidelines': self._extract_guidelines(sections),
            'constraints': self._extract_constraints(sections),
            'integration_hooks': self._extract_integration_hooks(sections),
            'optimization_preferences': self._extract_optimization_preferences(sections),
            'usage_examples': self._extract_usage_examples(sections)
        }
        
        return parsed_data
    
    def _extract_sections(self, content: str) -> Dict[str, str]:
        """Extract sections from markdown content."""
        
        sections = {}
        current_section = None
        current_content = []
        
        for line in content.split('\n'):
            if line.startswith('#'):
                # Save previous section
                if current_section:
                    sections[current_section] = '\n'.join(current_content)
                
                # Start new section
                current_section = line.strip('#').strip().lower().replace(' ', '_')
                current_content = []
            else:
                if current_section:
                    current_content.append(line)
        
        # Save last section
        if current_section:
            sections[current_section] = '\n'.join(current_content)
        
        return sections
    
    def _extract_project_description(self, sections: Dict[str, str]) -> str:
        """Extract project description from sections."""
        
        description_sections = ['project_context', 'project_description', 'overview']
        
        for section_name in description_sections:
            if section_name in sections:
                content = sections[section_name].strip()
                # Extract key information from the section
                lines = [line.strip() for line in content.split('\n') if line.strip()]
                
                # Look for project type and description
                description_parts = []
                for line in lines:
                    if line.startswith('**') and line.endswith('**'):
                        # Extract description from bold text
                        desc = line.strip('*').split(':')
                        if len(desc) > 1:
                            description_parts.append(desc[1].strip())
                    elif not line.startswith('**') and len(line) > 20:
                        description_parts.append(line)
                
                return ' '.join(description_parts)
        
        return ""
    
    def _extract_technologies(self, sections: Dict[str, str]) -> List[str]:
        """Extract technologies from sections."""
        
        technologies = []
        tech_sections = ['project_context', 'technologies', 'tech_stack']
        
        for section_name in tech_sections:
            if section_name in sections:
                content = sections[section_name]
                
                # Look for technology mentions
                tech_patterns = [
                    r'\*\*Technologies\*\*:\s*([^\n]+)',
                    r'- ([A-Za-z][A-Za-z0-9\+\#\.\-]+)',
                    r'\b(Python|JavaScript|TypeScript|Java|Go|Rust|C\+\+|C#|PHP|Ruby|Swift|Kotlin|Scala|Clojure|Elixir|Erlang|Haskell|OCaml|F#|Julia|R|MATLAB|Perl|Lua|Shell|PowerShell|Bash|Zsh|Fish)\b',
                    r'\b(React|Vue|Angular|Django|Flask|FastAPI|Express|Spring|Rails|Laravel|Symfony|ASP\.NET|Gin|Echo|Actix|Rocket|Axum|Warp)\b',
                    r'\b(PostgreSQL|MySQL|SQLite|MongoDB|Redis|Elasticsearch|Cassandra|DynamoDB|Firebase|Supabase)\b',
                    r'\b(Docker|Kubernetes|AWS|GCP|Azure|Terraform|Ansible|Vagrant|Jenkins|GitLab|GitHub)\b'
                ]
                
                for pattern in tech_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    technologies.extend(matches)
        
        # Clean and deduplicate
        technologies = list(set([tech.strip() for tech in technologies if tech.strip()]))
        return technologies
    
    def _extract_capabilities(self, sections: Dict[str, str]) -> List[str]:
        """Extract core capabilities from sections."""
        
        capabilities = []
        
        if 'core_capabilities' in sections:
            content = sections['core_capabilities']
            # Look for bullet points or bold items
            for line in content.split('\n'):
                line = line.strip()
                if line.startswith('-') or line.startswith('*'):
                    capability = line.lstrip('-*').strip()
                    if capability.startswith('**') and '**' in capability[2:]:
                        # Extract just the bold part
                        capability = capability.split('**')[1]
                    capabilities.append(capability)
        
        return capabilities
    
    def _extract_preferred_workflows(self, sections: Dict[str, str]) -> List[str]:
        """Extract preferred workflows from sections."""
        
        workflows = []
        
        if 'agent_configuration' in sections:
            content = sections['agent_configuration']
            
            # Look for workflow mentions
            workflow_pattern = r'`([a-z_]+)`\s*-\s*([^\n]+)'
            matches = re.findall(workflow_pattern, content)
            
            for workflow_name, description in matches:
                workflows.append({
                    'name': workflow_name,
                    'description': description.strip()
                })
        
        return workflows
    
    def _extract_domain_agents(self, sections: Dict[str, str]) -> List[Dict[str, str]]:
        """Extract domain agent information from sections."""
        
        agents = []
        
        if 'agent_configuration' in sections:
            content = sections['agent_configuration']
            
            # Look for agent mentions
            agent_pattern = r'\*\*([a-z_]+)\*\*:\s*([^\n]+)'
            matches = re.findall(agent_pattern, content)
            
            for agent_name, description in matches:
                agents.append({
                    'name': agent_name,
                    'description': description.strip()
                })
        
        return agents
    
    def _extract_guidelines(self, sections: Dict[str, str]) -> Dict[str, List[str]]:
        """Extract development guidelines from sections."""
        
        guidelines = {}
        guideline_sections = ['development_guidelines', 'code_quality_standards', 'architecture_principles']
        
        for section_name in guideline_sections:
            if section_name in sections:
                content = sections[section_name]
                guidelines[section_name] = []
                
                for line in content.split('\n'):
                    line = line.strip()
                    if line.startswith('-') or line.startswith('*'):
                        guideline = line.lstrip('-*').strip()
                        guidelines[section_name].append(guideline)
        
        return guidelines
    
    def _extract_constraints(self, sections: Dict[str, str]) -> Dict[str, List[str]]:
        """Extract project constraints from sections."""
        
        constraints = {}
        
        if 'project_constraints' in sections:
            content = sections['project_constraints']
            
            current_category = None
            for line in content.split('\n'):
                line = line.strip()
                if line.startswith('-') and line.endswith(':'):
                    # Category header
                    current_category = line.strip('-:').strip().lower().replace(' ', '_')
                    constraints[current_category] = []
                elif line.startswith('-') and current_category:
                    # Constraint item
                    constraint = line.lstrip('-').strip()
                    constraints[current_category].append(constraint)
        
        return constraints
    
    def _extract_integration_hooks(self, sections: Dict[str, str]) -> Dict[str, List[str]]:
        """Extract integration hooks configuration."""
        
        hooks = {}
        
        if 'integration_hooks' in sections:
            content = sections['integration_hooks']
            
            current_hook_type = None
            for line in content.split('\n'):
                line = line.strip()
                if line.startswith('###'):
                    current_hook_type = line.strip('#').strip().lower().replace(' ', '_').replace('-', '_')
                    hooks[current_hook_type] = []
                elif line.startswith('-') and current_hook_type:
                    hook_item = line.lstrip('-').strip()
                    hooks[current_hook_type].append(hook_item)
        
        return hooks
    
    def _extract_optimization_preferences(self, sections: Dict[str, str]) -> Dict[str, List[str]]:
        """Extract optimization preferences from sections."""
        
        preferences = {}
        
        if 'optimization_preferences' in sections:
            content = sections['optimization_preferences']
            
            current_category = None
            for line in content.split('\n'):
                line = line.strip()
                if line.startswith('###'):
                    current_category = line.strip('#').strip().lower().replace(' ', '_')
                    preferences[current_category] = []
                elif line.startswith('-') and current_category:
                    preference = line.lstrip('-').strip()
                    preferences[current_category].append(preference)
        
        return preferences
    
    def _extract_usage_examples(self, sections: Dict[str, str]) -> List[str]:
        """Extract usage examples from sections."""
        
        examples = []
        
        if 'usage_examples' in sections:
            content = sections['usage_examples']
            
            # Look for code blocks or command examples
            in_code_block = False
            current_example = []
            
            for line in content.split('\n'):
                if line.strip().startswith('```'):
                    if in_code_block:
                        # End of code block
                        if current_example:
                            examples.append('\n'.join(current_example))
                            current_example = []
                        in_code_block = False
                    else:
                        # Start of code block
                        in_code_block = True
                elif in_code_block:
                    current_example.append(line)
        
        return examples


class CLAUDEMDIntegration:
    """
    Integration system that applies CLAUDE.md configuration to agent behavior.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.parser = CLAUDEMDParser()
    
    def apply_claude_md_config(self, agent_config: Dict[str, Any], claude_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply CLAUDE.md configuration to agent configuration.
        
        Args:
            agent_config: Base agent configuration
            claude_data: Parsed CLAUDE.md data
            
        Returns:
            Enhanced agent configuration
        """
        
        enhanced_config = agent_config.copy()
        
        # Apply project context
        enhanced_config['project_context'] = {
            'description': claude_data.get('project_description', ''),
            'technologies': claude_data.get('technologies', []),
            'capabilities': claude_data.get('capabilities', []),
            'constraints': claude_data.get('constraints', {})
        }
        
        # Apply workflow preferences
        if claude_data.get('workflows'):
            enhanced_config['preferred_workflows'] = claude_data['workflows']
        
        # Apply domain agent preferences
        if claude_data.get('domain_agents'):
            enhanced_config['domain_agent_preferences'] = claude_data['domain_agents']
        
        # Apply optimization preferences
        if claude_data.get('optimization_preferences'):
            enhanced_config['optimization'] = claude_data['optimization_preferences']
        
        # Apply integration hooks
        if claude_data.get('integration_hooks'):
            enhanced_config['hooks'] = claude_data['integration_hooks']
        
        return enhanced_config
    
    def get_suggested_workflows(self, claude_data: Dict[str, Any], task_description: str) -> List[str]:
        """
        Suggest appropriate workflows based on CLAUDE.md config and task.
        
        Args:
            claude_data: Parsed CLAUDE.md data
            task_description: Description of the task to be performed
            
        Returns:
            List of suggested workflow names
        """
        
        suggested_workflows = []
        
        # Get preferred workflows from CLAUDE.md
        preferred_workflows = claude_data.get('workflows', [])
        
        # Analyze task description to suggest appropriate workflows
        task_lower = task_description.lower()
        
        workflow_keywords = {
            'comprehensive_project_planning': ['plan', 'design', 'architecture', 'system', 'project'],
            'architecture_review': ['review', 'architecture', 'design', 'system', 'scalability'],
            'code_quality_audit': ['quality', 'code', 'audit', 'review', 'patterns', 'solid'],
            'system_analysis': ['analyze', 'analysis', 'security', 'performance', 'dependencies'],
            'migration_planning': ['migrate', 'migration', 'upgrade', 'modernize', 'legacy']
        }
        
        # Score workflows based on task keywords
        workflow_scores = {}
        for workflow, keywords in workflow_keywords.items():
            score = sum(1 for keyword in keywords if keyword in task_lower)
            if score > 0:
                workflow_scores[workflow] = score
        
        # Sort by score and add to suggestions
        sorted_workflows = sorted(workflow_scores.items(), key=lambda x: x[1], reverse=True)
        suggested_workflows.extend([workflow for workflow, score in sorted_workflows[:3]])
        
        # Add preferred workflows from CLAUDE.md if not already included
        for workflow_info in preferred_workflows:
            workflow_name = workflow_info.get('name') if isinstance(workflow_info, dict) else workflow_info
            if workflow_name and workflow_name not in suggested_workflows:
                suggested_workflows.append(workflow_name)
        
        return suggested_workflows[:5]  # Return top 5 suggestions
    
    def get_context_enhancements(self, claude_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get context enhancements based on CLAUDE.md configuration.
        
        Args:
            claude_data: Parsed CLAUDE.md data
            
        Returns:
            Dictionary of context enhancements
        """
        
        enhancements = {}
        
        # Technology-specific context
        technologies = claude_data.get('technologies', [])
        if technologies:
            enhancements['technology_context'] = {
                'primary_technologies': technologies,
                'technology_preferences': self._get_technology_preferences(technologies)
            }
        
        # Guidelines and constraints
        guidelines = claude_data.get('guidelines', {})
        if guidelines:
            enhancements['development_guidelines'] = guidelines
        
        constraints = claude_data.get('constraints', {})
        if constraints:
            enhancements['project_constraints'] = constraints
        
        # Optimization preferences
        optimization_prefs = claude_data.get('optimization_preferences', {})
        if optimization_prefs:
            enhancements['optimization_preferences'] = optimization_prefs
        
        return enhancements
    
    def _get_technology_preferences(self, technologies: List[str]) -> Dict[str, Any]:
        """Get technology-specific preferences and best practices."""
        
        preferences = {}
        
        # Python-specific preferences
        if any(tech.lower() in ['python', 'django', 'flask', 'fastapi'] for tech in technologies):
            preferences['python'] = {
                'style_guide': 'PEP 8',
                'type_hints': 'required',
                'async_preferred': 'fastapi' in [t.lower() for t in technologies],
                'testing_framework': 'pytest'
            }
        
        # JavaScript/TypeScript preferences
        if any(tech.lower() in ['javascript', 'typescript', 'node.js', 'react', 'vue', 'angular'] for tech in technologies):
            preferences['javascript'] = {
                'style_guide': 'ESLint + Prettier',
                'prefer_typescript': 'typescript' in [t.lower() for t in technologies],
                'testing_framework': 'Jest',
                'package_manager': 'npm'
            }
        
        # Database preferences
        db_techs = [tech for tech in technologies if tech.lower() in ['postgresql', 'mysql', 'mongodb', 'redis', 'sqlite']]
        if db_techs:
            preferences['database'] = {
                'primary_db': db_techs[0],
                'migration_strategy': 'incremental',
                'connection_pooling': 'recommended'
            }
        
        return preferences


def create_claude_md_hooks() -> Dict[str, Any]:
    """
    Create standard CLAUDE.md integration hooks for the agent system.
    
    Returns:
        Dictionary of hook configurations
    """
    
    integration = CLAUDEMDIntegration()
    
    hooks = {
        'pre_task_execution': {
            'name': 'claude_md_pre_execution',
            'description': 'Load and apply CLAUDE.md configuration before task execution',
            'handler': 'apply_claude_md_pre_execution'
        },
        'context_enhancement': {
            'name': 'claude_md_context_enhancement',
            'description': 'Enhance task context with CLAUDE.md information',
            'handler': 'enhance_context_with_claude_md'
        },
        'workflow_suggestion': {
            'name': 'claude_md_workflow_suggestion',
            'description': 'Suggest appropriate workflows based on CLAUDE.md preferences',
            'handler': 'suggest_workflows_from_claude_md'
        },
        'post_task_execution': {
            'name': 'claude_md_post_execution',
            'description': 'Update CLAUDE.md or project context after task completion',
            'handler': 'apply_claude_md_post_execution'
        }
    }
    
    return hooks


# Hook handler functions
async def apply_claude_md_pre_execution(project_path: str, task_context: Dict[str, Any]) -> Dict[str, Any]:
    """Pre-execution hook to apply CLAUDE.md configuration."""
    
    claude_md_path = Path(project_path) / 'CLAUDE.md'
    if not claude_md_path.exists():
        return task_context
    
    integration = CLAUDEMDIntegration()
    claude_data = integration.parser.parse_claude_md(claude_md_path)
    
    # Enhance task context with CLAUDE.md data
    enhanced_context = task_context.copy()
    enhanced_context.update(integration.get_context_enhancements(claude_data))
    
    return enhanced_context


async def enhance_context_with_claude_md(project_path: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """Context enhancement hook using CLAUDE.md information."""
    
    claude_md_path = Path(project_path) / 'CLAUDE.md'
    if not claude_md_path.exists():
        return context
    
    integration = CLAUDEMDIntegration()
    claude_data = integration.parser.parse_claude_md(claude_md_path)
    
    enhanced_context = context.copy()
    enhanced_context['claude_md_data'] = claude_data
    enhanced_context['project_preferences'] = integration.get_context_enhancements(claude_data)
    
    return enhanced_context


async def suggest_workflows_from_claude_md(project_path: str, task_description: str) -> List[str]:
    """Workflow suggestion hook based on CLAUDE.md preferences."""
    
    claude_md_path = Path(project_path) / 'CLAUDE.md'
    if not claude_md_path.exists():
        return []
    
    integration = CLAUDEMDIntegration()
    claude_data = integration.parser.parse_claude_md(claude_md_path)
    
    return integration.get_suggested_workflows(claude_data, task_description)


async def apply_claude_md_post_execution(project_path: str, execution_result: Dict[str, Any]) -> None:
    """Post-execution hook to update project context based on results."""
    
    # This could update CLAUDE.md with new learnings, update project statistics, etc.
    # For now, we'll just log the completion
    logger = logging.getLogger(__name__)
    logger.info(f"Task completed for project {project_path} with CLAUDE.md integration")