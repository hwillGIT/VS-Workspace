"""
Domain Manager for loading and managing domain-specific agents.

Handles dynamic loading of agent domains, configuration management,
and agent lifecycle for specialized domains like software development,
financial trading, data science, etc.
"""

import asyncio
import logging
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Type
from importlib import import_module

from ..agents.base_agent import BaseAgent


class DomainManager:
    """
    Manager for domain-specific agents and configurations.
    
    Handles:
    - Dynamic loading of domain configurations
    - Agent instantiation from domain-specific modules
    - Domain-specific workflow coordination
    - Cross-domain agent collaboration
    """
    
    def __init__(self, base_config: Dict[str, Any], domains_path: str = "domains"):
        self.base_config = base_config
        self.domains_path = Path(domains_path)
        self.logger = logging.getLogger(__name__)
        
        # Domain registry
        self.loaded_domains: Dict[str, Dict[str, Any]] = {}
        self.domain_agents: Dict[str, Dict[str, BaseAgent]] = {}
        self.domain_configs: Dict[str, Dict[str, Any]] = {}
        
        # Domain-specific workflows
        self.domain_workflows: Dict[str, Dict[str, Any]] = {}
        
        self.initialized = False
    
    async def initialize(self) -> bool:
        """Initialize the domain manager and load enabled domains."""
        
        try:
            self.logger.info("Initializing domain manager")
            
            # Get domain configuration from base config
            domain_config = self.base_config.get("domain_agents", {})
            
            # Load each enabled domain
            for domain_name, config in domain_config.items():
                if config.get("enabled", False):
                    await self._load_domain(domain_name, config)
            
            self.initialized = True
            self.logger.info(f"Domain manager initialized with {len(self.loaded_domains)} domains")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize domain manager: {e}")
            return False
    
    async def _load_domain(self, domain_name: str, domain_config: Dict[str, Any]) -> bool:
        """Load a specific domain and its agents."""
        
        try:
            self.logger.info(f"Loading domain: {domain_name}")
            
            # Load domain configuration
            config_path = Path(domain_config.get("config_path", f"domains/{domain_name}/config.yaml"))
            if config_path.exists():
                with open(config_path, 'r') as f:
                    domain_specific_config = yaml.safe_load(f)
            else:
                self.logger.warning(f"Domain config not found: {config_path}")
                domain_specific_config = {}
            
            self.domain_configs[domain_name] = domain_specific_config
            
            # Load domain agents
            agents_path = domain_config.get("agents_path", f"domains/{domain_name}/agents")
            domain_agents = await self._load_domain_agents(domain_name, domain_specific_config, agents_path)
            
            # Store domain information
            self.loaded_domains[domain_name] = {
                "config": domain_specific_config,
                "agents_path": agents_path,
                "agent_count": len(domain_agents)
            }
            
            self.domain_agents[domain_name] = domain_agents
            
            # Load domain workflows
            workflows = domain_specific_config.get("workflows", {})
            self.domain_workflows[domain_name] = workflows
            
            self.logger.info(f"Domain {domain_name} loaded with {len(domain_agents)} agents")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load domain {domain_name}: {e}")
            return False
    
    async def _load_domain_agents(
        self, 
        domain_name: str, 
        domain_config: Dict[str, Any], 
        agents_path: str
    ) -> Dict[str, BaseAgent]:
        """Load agents for a specific domain."""
        
        agents = {}
        agent_configs = domain_config.get("agents", {})
        
        try:
            for agent_name, agent_config in agent_configs.items():
                agent = await self._instantiate_domain_agent(
                    domain_name, agent_name, agent_config
                )
                if agent:
                    agents[agent_name] = agent
                    self.logger.debug(f"Loaded agent: {domain_name}.{agent_name}")
                else:
                    self.logger.warning(f"Failed to load agent: {domain_name}.{agent_name}")
            
            return agents
            
        except Exception as e:
            self.logger.error(f"Failed to load agents for domain {domain_name}: {e}")
            return {}
    
    async def _instantiate_domain_agent(
        self, 
        domain_name: str, 
        agent_name: str, 
        agent_config: Dict[str, Any]
    ) -> Optional[BaseAgent]:
        """Instantiate a domain-specific agent."""
        
        try:
            # Try to import the agent class dynamically
            agent_class_name = self._get_agent_class_name(agent_name)
            module_path = f"domains.{domain_name}.agents.{agent_name}_agent"
            
            try:
                # Import the agent module
                agent_module = import_module(module_path)
                agent_class = getattr(agent_module, agent_class_name)
                
                # Instantiate the agent
                agent = agent_class(agent_config)
                
                # Initialize the agent
                await agent.initialize()
                
                return agent
                
            except (ImportError, AttributeError) as e:
                self.logger.warning(f"Could not import {module_path}.{agent_class_name}: {e}")
                
                # Fallback: create a generic agent with domain-specific configuration
                return await self._create_generic_domain_agent(domain_name, agent_name, agent_config)
                
        except Exception as e:
            self.logger.error(f"Failed to instantiate agent {domain_name}.{agent_name}: {e}")
            return None
    
    def _get_agent_class_name(self, agent_name: str) -> str:
        """Convert agent name to class name convention."""
        # Convert snake_case to PascalCase and add Agent suffix
        words = agent_name.split('_')
        class_name = ''.join(word.capitalize() for word in words) + 'Agent'
        return class_name
    
    async def _create_generic_domain_agent(
        self, 
        domain_name: str, 
        agent_name: str, 
        agent_config: Dict[str, Any]
    ) -> BaseAgent:
        """Create a generic agent with domain-specific configuration."""
        
        # Use base agent with domain-specific ID and configuration
        agent = BaseAgent(
            agent_id=f"{domain_name}.{agent_name}",
            agent_type=f"{domain_name}.{agent_name}",
            config=agent_config
        )
        
        await agent.initialize()
        return agent
    
    def get_domain_agent(self, domain_name: str, agent_name: str) -> Optional[BaseAgent]:
        """Get a specific domain agent."""
        
        domain_agents = self.domain_agents.get(domain_name, {})
        return domain_agents.get(agent_name)
    
    def get_all_domain_agents(self, domain_name: str) -> Dict[str, BaseAgent]:
        """Get all agents for a specific domain."""
        
        return self.domain_agents.get(domain_name, {})
    
    def list_available_domains(self) -> List[str]:
        """List all loaded domains."""
        
        return list(self.loaded_domains.keys())
    
    def list_domain_agents(self, domain_name: str) -> List[str]:
        """List all agents in a specific domain."""
        
        domain_agents = self.domain_agents.get(domain_name, {})
        return list(domain_agents.keys())
    
    async def execute_domain_workflow(
        self, 
        domain_name: str, 
        workflow_name: str, 
        task_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a domain-specific workflow."""
        
        try:
            # Get workflow configuration
            domain_workflows = self.domain_workflows.get(domain_name, {})
            workflow_config = domain_workflows.get(workflow_name)
            
            if not workflow_config:
                raise ValueError(f"Workflow {workflow_name} not found in domain {domain_name}")
            
            # Check if this is a multi-perspective planning workflow
            if workflow_config.get("type") == "multi_perspective":
                return await self._execute_multi_perspective_workflow(
                    domain_name, workflow_name, workflow_config, task_context
                )
            
            # Execute traditional agent-based workflow
            required_agents = workflow_config.get("agents", [])
            sequence = workflow_config.get("sequence", "sequential")
            aggregation = workflow_config.get("aggregation", "simple")
            
            # Execute workflow
            if sequence == "parallel":
                results = await self._execute_parallel_workflow(
                    domain_name, required_agents, task_context
                )
            else:
                results = await self._execute_sequential_workflow(
                    domain_name, required_agents, task_context
                )
            
            # Aggregate results
            aggregated_result = await self._aggregate_workflow_results(
                results, aggregation
            )
            
            return {
                "workflow": workflow_name,
                "domain": domain_name,
                "results": aggregated_result,
                "execution_summary": {
                    "agents_used": required_agents,
                    "sequence": sequence,
                    "aggregation": aggregation
                }
            }
            
        except Exception as e:
            self.logger.error(f"Domain workflow execution failed: {e}")
            return {"error": str(e)}
    
    async def _execute_multi_perspective_workflow(
        self,
        domain_name: str,
        workflow_name: str, 
        workflow_config: Dict[str, Any],
        task_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a multi-perspective planning workflow."""
        
        try:
            # Import the perspective planner (only for software_development domain for now)
            if domain_name == "software_development":
                from .software_development.planning.perspective_planner import (
                    PerspectivePlanner, PlanningContext
                )
                
                # Create planning context
                planning_context = PlanningContext(
                    project_type=task_context.get("project_type", "software_project"),
                    problem_description=task_context.get("task", "Project planning task"),
                    requirements=task_context.get("requirements", {}),
                    constraints=task_context.get("constraints", {}),
                    stakeholders=task_context.get("stakeholders", ["development_team"]),
                    timeline=task_context.get("timeline"),
                    budget=task_context.get("budget"),
                    target_audience=task_context.get("target_audience")
                )
                
                # Create perspective planner
                planner = PerspectivePlanner(self, workflow_config)
                
                # Execute multi-perspective planning
                synthesized_plan = await planner.create_multi_perspective_plan(
                    planning_context,
                    selected_perspectives=workflow_config.get("perspectives")
                )
                
                return {
                    "workflow": workflow_name,
                    "domain": domain_name,
                    "type": "multi_perspective_planning",
                    "synthesized_plan": {
                        "executive_summary": synthesized_plan.executive_summary,
                        "technical_specification": synthesized_plan.technical_specification,
                        "implementation_roadmap": synthesized_plan.implementation_roadmap,
                        "validation_results": synthesized_plan.validation_results,
                        "success_criteria": synthesized_plan.success_criteria
                    },
                    "perspective_plans": [
                        {
                            "perspective": plan.perspective_name,
                            "agent": plan.agent_name,
                            "confidence": plan.confidence_score,
                            "recommendations": plan.recommendations,
                            "concerns": plan.concerns
                        }
                        for plan in synthesized_plan.perspective_plans
                    ],
                    "execution_summary": {
                        "perspectives_used": workflow_config.get("perspectives", []),
                        "synthesis_strategy": workflow_config.get("synthesis_strategy"),
                        "conflict_resolution": workflow_config.get("conflict_resolution"),
                        "validation_score": synthesized_plan.validation_results.get("overall_score", 0.0)
                    }
                }
            else:
                raise ValueError(f"Multi-perspective planning not yet supported for domain {domain_name}")
                
        except Exception as e:
            self.logger.error(f"Multi-perspective workflow execution failed: {e}")
            return {"error": str(e), "workflow_type": "multi_perspective_planning"}
    
    async def _execute_parallel_workflow(
        self, 
        domain_name: str, 
        agent_names: List[str], 
        task_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute agents in parallel."""
        
        tasks = []
        for agent_name in agent_names:
            agent = self.get_domain_agent(domain_name, agent_name)
            if agent:
                task = agent.process_task(task_context.get("task", ""), task_context)
                tasks.append((agent_name, task))
        
        # Execute all tasks concurrently
        results = {}
        if tasks:
            task_results = await asyncio.gather(
                *[task for _, task in tasks], 
                return_exceptions=True
            )
            
            for (agent_name, _), result in zip(tasks, task_results):
                if isinstance(result, Exception):
                    results[agent_name] = {"error": str(result)}
                else:
                    results[agent_name] = result
        
        return results
    
    async def _execute_sequential_workflow(
        self, 
        domain_name: str, 
        agent_names: List[str], 
        task_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute agents sequentially."""
        
        results = {}
        current_context = task_context.copy()
        
        for agent_name in agent_names:
            agent = self.get_domain_agent(domain_name, agent_name)
            if agent:
                try:
                    result = await agent.process_task(
                        current_context.get("task", ""), 
                        current_context
                    )
                    results[agent_name] = result
                    
                    # Update context with previous results for next agent
                    current_context["previous_results"] = results
                    
                except Exception as e:
                    results[agent_name] = {"error": str(e)}
                    self.logger.error(f"Agent {agent_name} failed: {e}")
            else:
                results[agent_name] = {"error": f"Agent {agent_name} not found"}
        
        return results
    
    async def _aggregate_workflow_results(
        self, 
        results: Dict[str, Any], 
        aggregation_method: str
    ) -> Dict[str, Any]:
        """Aggregate workflow results based on method."""
        
        if aggregation_method == "consensus":
            return await self._consensus_aggregation(results)
        elif aggregation_method == "comprehensive":
            return await self._comprehensive_aggregation(results)
        elif aggregation_method == "report":
            return await self._report_aggregation(results)
        elif aggregation_method == "plan":
            return await self._plan_aggregation(results)
        else:  # simple
            return results
    
    async def _consensus_aggregation(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate results by finding consensus."""
        return {"aggregation_type": "consensus", "results": results}
    
    async def _comprehensive_aggregation(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive aggregation of all results."""
        return {"aggregation_type": "comprehensive", "results": results}
    
    async def _report_aggregation(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a structured report from results."""
        return {"aggregation_type": "report", "results": results}
    
    async def _plan_aggregation(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate an actionable plan from results."""
        return {"aggregation_type": "plan", "results": results}
    
    def get_domain_statistics(self) -> Dict[str, Any]:
        """Get statistics about loaded domains."""
        
        stats = {
            "loaded_domains": len(self.loaded_domains),
            "total_agents": sum(
                len(agents) for agents in self.domain_agents.values()
            ),
            "domains": {}
        }
        
        for domain_name, domain_info in self.loaded_domains.items():
            stats["domains"][domain_name] = {
                "agent_count": domain_info["agent_count"],
                "workflows": len(self.domain_workflows.get(domain_name, {})),
                "agents": list(self.domain_agents.get(domain_name, {}).keys())
            }
        
        return stats
    
    async def shutdown_domain(self, domain_name: str) -> bool:
        """Shutdown a specific domain and its agents."""
        
        try:
            domain_agents = self.domain_agents.get(domain_name, {})
            
            # Shutdown all agents in the domain
            for agent in domain_agents.values():
                if hasattr(agent, 'shutdown'):
                    await agent.shutdown()
            
            # Remove from registries
            self.loaded_domains.pop(domain_name, None)
            self.domain_agents.pop(domain_name, None)
            self.domain_configs.pop(domain_name, None)
            self.domain_workflows.pop(domain_name, None)
            
            self.logger.info(f"Domain {domain_name} shutdown successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to shutdown domain {domain_name}: {e}")
            return False
    
    async def shutdown(self) -> None:
        """Shutdown the domain manager and all domains."""
        
        try:
            # Shutdown all domains
            for domain_name in list(self.loaded_domains.keys()):
                await self.shutdown_domain(domain_name)
            
            self.initialized = False
            self.logger.info("Domain manager shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during domain manager shutdown: {e}")