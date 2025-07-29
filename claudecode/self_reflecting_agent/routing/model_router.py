"""
Intelligent Model Router System

Routes agent tasks to optimal AI models based on task characteristics,
model availability, performance history, and cost considerations.
"""

import asyncio
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import yaml
from datetime import datetime, timedelta
import statistics


class TaskType(Enum):
    """Task types that require different model capabilities."""
    ORCHESTRATION = "orchestration"       # High-level planning and coordination
    CODE_GENERATION = "code_generation"   # Writing new code
    CODE_REVIEW = "code_review"          # Analyzing and reviewing code
    DEBUGGING = "debugging"              # Finding and fixing bugs
    ARCHITECTURE = "architecture"        # System design and planning
    DOCUMENTATION = "documentation"      # Writing documentation
    TESTING = "testing"                  # Creating tests
    REFACTORING = "refactoring"         # Code restructuring
    RESEARCH = "research"               # Information gathering
    ANALYSIS = "analysis"               # Data/code analysis
    CONVERSATION = "conversation"        # General interaction
    REASONING = "reasoning"             # Complex logical reasoning


class ModelProvider(Enum):
    """Supported AI model providers."""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"
    LOCAL = "local"


@dataclass
class ModelCapabilities:
    """Model capabilities and characteristics."""
    context_length: int
    supports_code: bool
    supports_reasoning: bool
    supports_vision: bool
    supports_function_calling: bool
    latency_ms: Optional[int] = None
    cost_per_1k_tokens: Optional[float] = None
    quality_score: Optional[float] = None  # 0-1 scale


@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    name: str
    provider: ModelProvider
    api_key_env: str
    capabilities: ModelCapabilities
    base_url: Optional[str] = None
    model_params: Optional[Dict[str, Any]] = None
    enabled: bool = True
    priority: int = 0  # Higher priority = preferred


@dataclass
class TaskContext:
    """Context information for routing decisions."""
    task_type: TaskType
    complexity: str  # "low", "medium", "high"
    estimated_tokens: int
    requires_code: bool = False
    requires_reasoning: bool = False
    requires_vision: bool = False
    requires_function_calling: bool = False
    latency_sensitive: bool = False
    cost_sensitive: bool = False
    context_data: Optional[Dict[str, Any]] = None


@dataclass
class RoutingDecision:
    """Result of a routing decision."""
    selected_model: str
    provider: ModelProvider
    reasoning: str
    fallback_models: List[str]
    estimated_cost: Optional[float] = None
    expected_latency: Optional[int] = None


@dataclass
class ModelPerformance:
    """Performance metrics for a model."""
    success_rate: float
    avg_latency_ms: float
    avg_cost_per_request: float
    total_requests: int
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    failure_reasons: List[str] = None

    def __post_init__(self):
        if self.failure_reasons is None:
            self.failure_reasons = []


class ModelAvailabilityChecker:
    """Checks model availability and health."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.cache_duration = 300  # 5 minutes
        self.availability_cache: Dict[str, Tuple[bool, datetime]] = {}
    
    async def check_model_availability(self, model_config: ModelConfig) -> bool:
        """
        Check if a model is currently available.
        
        Args:
            model_config: Configuration for the model to check
            
        Returns:
            True if model is available, False otherwise
        """
        
        cache_key = f"{model_config.provider.value}:{model_config.name}"
        
        # Check cache first
        if cache_key in self.availability_cache:
            is_available, check_time = self.availability_cache[cache_key]
            if datetime.now() - check_time < timedelta(seconds=self.cache_duration):
                return is_available
        
        # Perform actual availability check
        try:
            is_available = await self._perform_availability_check(model_config)
            self.availability_cache[cache_key] = (is_available, datetime.now())
            return is_available
            
        except Exception as e:
            self.logger.warning(f"Availability check failed for {cache_key}: {e}")
            return False
    
    async def _perform_availability_check(self, model_config: ModelConfig) -> bool:
        """Perform the actual availability check."""
        
        # Check if API key is available
        api_key = os.getenv(model_config.api_key_env)
        if not api_key:
            return False
        
        # Perform provider-specific checks
        if model_config.provider == ModelProvider.ANTHROPIC:
            return await self._check_anthropic_availability(model_config, api_key)
        elif model_config.provider == ModelProvider.OPENAI:
            return await self._check_openai_availability(model_config, api_key)
        elif model_config.provider == ModelProvider.GOOGLE:
            return await self._check_google_availability(model_config, api_key)
        else:
            return True  # Assume local models are available
    
    async def _check_anthropic_availability(self, config: ModelConfig, api_key: str) -> bool:
        """Check Anthropic Claude availability."""
        try:
            import anthropic
            
            client = anthropic.Anthropic(api_key=api_key)
            
            # Simple test request
            response = client.messages.create(
                model=config.name,
                max_tokens=10,
                messages=[{"role": "user", "content": "Hi"}]
            )
            
            return True
            
        except anthropic.RateLimitError:
            self.logger.warning(f"Anthropic rate limit hit for {config.name}")
            return False
        except anthropic.AuthenticationError:
            self.logger.error(f"Anthropic authentication failed for {config.name}")
            return False
        except Exception as e:
            self.logger.debug(f"Anthropic availability check failed: {e}")
            return False
    
    async def _check_openai_availability(self, config: ModelConfig, api_key: str) -> bool:
        """Check OpenAI GPT availability."""
        try:
            import openai
            
            client = openai.OpenAI(api_key=api_key)
            
            # Simple test request
            response = client.chat.completions.create(
                model=config.name,
                max_tokens=10,
                messages=[{"role": "user", "content": "Hi"}]
            )
            
            return True
            
        except openai.RateLimitError:
            self.logger.warning(f"OpenAI rate limit hit for {config.name}")
            return False
        except openai.AuthenticationError:
            self.logger.error(f"OpenAI authentication failed for {config.name}")
            return False
        except Exception as e:
            self.logger.debug(f"OpenAI availability check failed: {e}")
            return False
    
    async def _check_google_availability(self, config: ModelConfig, api_key: str) -> bool:
        """Check Google Gemini availability."""
        try:
            import google.generativeai as genai
            
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(config.name)
            
            # Simple test request
            response = model.generate_content("Hi")
            
            return True
            
        except Exception as e:
            self.logger.debug(f"Google availability check failed: {e}")
            return False


class PerformanceTracker:
    """Tracks model performance metrics."""
    
    def __init__(self, data_file: Optional[Path] = None):
        self.logger = logging.getLogger(__name__)
        self.data_file = data_file or Path.home() / '.self_reflecting_agent' / 'model_performance.json'
        self.data_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.performance_data: Dict[str, ModelPerformance] = {}
        self.load_performance_data()
    
    def load_performance_data(self):
        """Load performance data from disk."""
        if self.data_file.exists():
            try:
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    
                for model_name, perf_data in data.items():
                    # Convert datetime strings back to datetime objects
                    if perf_data.get('last_success'):
                        perf_data['last_success'] = datetime.fromisoformat(perf_data['last_success'])
                    if perf_data.get('last_failure'):
                        perf_data['last_failure'] = datetime.fromisoformat(perf_data['last_failure'])
                    
                    self.performance_data[model_name] = ModelPerformance(**perf_data)
                    
            except Exception as e:
                self.logger.warning(f"Could not load performance data: {e}")
    
    def save_performance_data(self):
        """Save performance data to disk."""
        try:
            # Convert datetime objects to strings for JSON serialization
            data_to_save = {}
            for model_name, perf in self.performance_data.items():
                perf_dict = asdict(perf)
                if perf_dict.get('last_success'):
                    perf_dict['last_success'] = perf_dict['last_success'].isoformat()
                if perf_dict.get('last_failure'):
                    perf_dict['last_failure'] = perf_dict['last_failure'].isoformat()
                data_to_save[model_name] = perf_dict
            
            with open(self.data_file, 'w') as f:
                json.dump(data_to_save, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Could not save performance data: {e}")
    
    def record_success(self, model_name: str, latency_ms: int, cost: float):
        """Record a successful model interaction."""
        if model_name not in self.performance_data:
            self.performance_data[model_name] = ModelPerformance(
                success_rate=0.0,
                avg_latency_ms=0.0,
                avg_cost_per_request=0.0,
                total_requests=0
            )
        
        perf = self.performance_data[model_name]
        
        # Update metrics
        total_requests = perf.total_requests + 1
        successful_requests = int(perf.success_rate * perf.total_requests) + 1
        
        perf.success_rate = successful_requests / total_requests
        perf.avg_latency_ms = ((perf.avg_latency_ms * perf.total_requests) + latency_ms) / total_requests
        perf.avg_cost_per_request = ((perf.avg_cost_per_request * perf.total_requests) + cost) / total_requests
        perf.total_requests = total_requests
        perf.last_success = datetime.now()
        
        self.save_performance_data()
    
    def record_failure(self, model_name: str, reason: str):
        """Record a failed model interaction."""
        if model_name not in self.performance_data:
            self.performance_data[model_name] = ModelPerformance(
                success_rate=0.0,
                avg_latency_ms=0.0,
                avg_cost_per_request=0.0,
                total_requests=0
            )
        
        perf = self.performance_data[model_name]
        
        # Update metrics
        total_requests = perf.total_requests + 1
        successful_requests = int(perf.success_rate * perf.total_requests)
        
        perf.success_rate = successful_requests / total_requests
        perf.total_requests = total_requests
        perf.last_failure = datetime.now()
        perf.failure_reasons.append(reason)
        
        # Keep only last 10 failure reasons
        if len(perf.failure_reasons) > 10:
            perf.failure_reasons = perf.failure_reasons[-10:]
        
        self.save_performance_data()
    
    def get_model_score(self, model_name: str, task_context: TaskContext) -> float:
        """
        Calculate a score for a model based on performance and task context.
        
        Returns:
            Score from 0.0 to 1.0, higher is better
        """
        if model_name not in self.performance_data:
            return 0.5  # Neutral score for unknown models
        
        perf = self.performance_data[model_name]
        
        if perf.total_requests == 0:
            return 0.5  # Neutral score for untested models
        
        # Base score from success rate
        score = perf.success_rate
        
        # Adjust for latency if latency-sensitive
        if task_context.latency_sensitive and perf.avg_latency_ms > 0:
            # Penalize high latency (assume good latency is < 5000ms)
            latency_penalty = min(perf.avg_latency_ms / 5000.0, 0.5)
            score -= latency_penalty * 0.2
        
        # Adjust for cost if cost-sensitive
        if task_context.cost_sensitive and perf.avg_cost_per_request > 0:
            # Penalize high cost (assume good cost is < $0.10)
            cost_penalty = min(perf.avg_cost_per_request / 0.10, 0.5)
            score -= cost_penalty * 0.2
        
        # Penalize recent failures
        if perf.last_failure:
            hours_since_failure = (datetime.now() - perf.last_failure).total_seconds() / 3600
            if hours_since_failure < 1:  # Recent failure
                score -= 0.3
            elif hours_since_failure < 24:  # Failure within 24 hours
                score -= 0.1
        
        return max(0.0, min(1.0, score))


class ModelRouter:
    """
    Intelligent model router that selects optimal models for tasks.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self.config_path = config_path or Path(__file__).parent / 'router_config.yaml'
        self.config = self._load_config()
        
        # Initialize components
        self.availability_checker = ModelAvailabilityChecker()
        self.performance_tracker = PerformanceTracker()
        
        # Model registry
        self.models: Dict[str, ModelConfig] = {}
        self.routing_rules: Dict[TaskType, List[str]] = {}
        
        self._initialize_models()
        self._initialize_routing_rules()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load router configuration."""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    return yaml.safe_load(f)
            except Exception as e:
                self.logger.warning(f"Could not load router config: {e}")
        
        return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default router configuration."""
        return {
            'models': {
                'claude-3-5-sonnet-20241022': {
                    'provider': 'anthropic',
                    'api_key_env': 'ANTHROPIC_API_KEY',
                    'capabilities': {
                        'context_length': 200000,
                        'supports_code': True,
                        'supports_reasoning': True,
                        'supports_vision': True,
                        'supports_function_calling': True,
                        'cost_per_1k_tokens': 0.003,
                        'quality_score': 0.95
                    },
                    'priority': 10,
                    'enabled': True
                },
                'claude-3-haiku-20240307': {
                    'provider': 'anthropic',
                    'api_key_env': 'ANTHROPIC_API_KEY',
                    'capabilities': {
                        'context_length': 200000,
                        'supports_code': True,
                        'supports_reasoning': True,
                        'supports_vision': False,
                        'supports_function_calling': True,
                        'cost_per_1k_tokens': 0.00025,
                        'quality_score': 0.85
                    },
                    'priority': 5,
                    'enabled': True
                },
                'gpt-4o': {
                    'provider': 'openai',
                    'api_key_env': 'OPENAI_API_KEY',
                    'capabilities': {
                        'context_length': 128000,
                        'supports_code': True,
                        'supports_reasoning': True,
                        'supports_vision': True,
                        'supports_function_calling': True,
                        'cost_per_1k_tokens': 0.005,
                        'quality_score': 0.92
                    },
                    'priority': 8,
                    'enabled': True
                },
                'gpt-4o-mini': {
                    'provider': 'openai',
                    'api_key_env': 'OPENAI_API_KEY',
                    'capabilities': {
                        'context_length': 128000,
                        'supports_code': True,
                        'supports_reasoning': True,
                        'supports_vision': True,
                        'supports_function_calling': True,
                        'cost_per_1k_tokens': 0.00015,
                        'quality_score': 0.8
                    },
                    'priority': 3,
                    'enabled': True
                },
                'gemini-2.0-flash-exp': {
                    'provider': 'google',
                    'api_key_env': 'GOOGLE_API_KEY',
                    'capabilities': {
                        'context_length': 1000000,
                        'supports_code': True,
                        'supports_reasoning': True,
                        'supports_vision': True,
                        'supports_function_calling': True,
                        'cost_per_1k_tokens': 0.00075,
                        'quality_score': 0.88
                    },
                    'priority': 7,
                    'enabled': True
                }
            },
            'routing_rules': {
                'orchestration': ['claude-3-5-sonnet-20241022', 'gpt-4o', 'gemini-2.0-flash-exp'],
                'code_generation': ['claude-3-5-sonnet-20241022', 'gpt-4o', 'gemini-2.0-flash-exp'],
                'debugging': ['gemini-2.0-flash-exp', 'claude-3-5-sonnet-20241022', 'gpt-4o'],
                'code_review': ['claude-3-5-sonnet-20241022', 'gpt-4o', 'gemini-2.0-flash-exp'],
                'architecture': ['claude-3-5-sonnet-20241022', 'gpt-4o', 'gemini-2.0-flash-exp'],
                'documentation': ['claude-haiku-20240307', 'gpt-4o-mini', 'gemini-2.0-flash-exp'],
                'testing': ['claude-3-5-sonnet-20241022', 'gpt-4o', 'gemini-2.0-flash-exp'],
                'conversation': ['claude-haiku-20240307', 'gpt-4o-mini', 'gemini-2.0-flash-exp']
            }
        }
    
    def _initialize_models(self):
        """Initialize model registry from configuration."""
        models_config = self.config.get('models', {})
        
        for model_name, model_data in models_config.items():
            try:
                capabilities = ModelCapabilities(**model_data['capabilities'])
                
                model_config = ModelConfig(
                    name=model_name,
                    provider=ModelProvider(model_data['provider']),
                    api_key_env=model_data['api_key_env'],
                    capabilities=capabilities,
                    base_url=model_data.get('base_url'),
                    model_params=model_data.get('model_params', {}),
                    enabled=model_data.get('enabled', True),
                    priority=model_data.get('priority', 0)
                )
                
                self.models[model_name] = model_config
                
            except Exception as e:
                self.logger.error(f"Could not initialize model {model_name}: {e}")
    
    def _initialize_routing_rules(self):
        """Initialize routing rules from configuration."""
        rules_config = self.config.get('routing_rules', {})
        
        for task_type_name, model_list in rules_config.items():
            try:
                task_type = TaskType(task_type_name)
                self.routing_rules[task_type] = model_list
            except ValueError:
                self.logger.warning(f"Unknown task type in routing rules: {task_type_name}")
    
    async def route_task(self, task_context: TaskContext) -> RoutingDecision:
        """
        Route a task to the optimal model.
        
        Args:
            task_context: Context information for the task
            
        Returns:
            Routing decision with selected model and reasoning
        """
        
        # Get candidate models for this task type
        candidate_models = self._get_candidate_models(task_context)
        
        if not candidate_models:
            raise ValueError(f"No models available for task type: {task_context.task_type}")
        
        # Check availability and score models
        scored_models = []
        for model_name in candidate_models:
            if model_name not in self.models:
                continue
                
            model_config = self.models[model_name]
            
            # Check if model meets requirements
            if not self._model_meets_requirements(model_config, task_context):
                continue
            
            # Check availability
            is_available = await self.availability_checker.check_model_availability(model_config)
            if not is_available:
                continue
            
            # Calculate score
            performance_score = self.performance_tracker.get_model_score(model_name, task_context)
            capability_score = self._calculate_capability_score(model_config, task_context)
            priority_score = model_config.priority / 10.0  # Normalize to 0-1
            
            # Weighted final score
            final_score = (
                performance_score * 0.4 +
                capability_score * 0.4 +
                priority_score * 0.2
            )
            
            scored_models.append((model_name, final_score, model_config))
        
        if not scored_models:
            raise RuntimeError("No models are currently available for this task")
        
        # Sort by score (highest first)
        scored_models.sort(key=lambda x: x[1], reverse=True)
        
        # Select best model
        selected_model_name, score, selected_config = scored_models[0]
        fallback_models = [name for name, _, _ in scored_models[1:6]]  # Top 5 fallbacks
        
        # Generate reasoning
        reasoning = self._generate_routing_reasoning(
            selected_model_name, 
            selected_config, 
            task_context, 
            score
        )
        
        return RoutingDecision(
            selected_model=selected_model_name,
            provider=selected_config.provider,
            reasoning=reasoning,
            fallback_models=fallback_models,
            estimated_cost=self._estimate_cost(selected_config, task_context),
            expected_latency=selected_config.capabilities.latency_ms
        )
    
    def _get_candidate_models(self, task_context: TaskContext) -> List[str]:
        """Get candidate models for a task type."""
        # Get models from routing rules
        rule_models = self.routing_rules.get(task_context.task_type, [])
        
        # If no specific rules, use all enabled models
        if not rule_models:
            rule_models = [name for name, config in self.models.items() if config.enabled]
        
        return rule_models
    
    def _model_meets_requirements(self, model_config: ModelConfig, task_context: TaskContext) -> bool:
        """Check if a model meets the task requirements."""
        capabilities = model_config.capabilities
        
        # Check basic requirements
        if task_context.requires_code and not capabilities.supports_code:
            return False
        
        if task_context.requires_reasoning and not capabilities.supports_reasoning:
            return False
        
        if task_context.requires_vision and not capabilities.supports_vision:
            return False
        
        if task_context.requires_function_calling and not capabilities.supports_function_calling:
            return False
        
        # Check context length
        if task_context.estimated_tokens > capabilities.context_length:
            return False
        
        return True
    
    def _calculate_capability_score(self, model_config: ModelConfig, task_context: TaskContext) -> float:
        """Calculate how well a model's capabilities match the task."""
        capabilities = model_config.capabilities
        score = 0.0
        
        # Base quality score
        if capabilities.quality_score:
            score += capabilities.quality_score * 0.4
        else:
            score += 0.5 * 0.4  # Default quality
        
        # Context length bonus (more is better, up to a point)
        context_ratio = min(capabilities.context_length / task_context.estimated_tokens, 2.0)
        score += (context_ratio / 2.0) * 0.2
        
        # Feature matching bonus
        feature_score = 0.0
        if task_context.requires_code and capabilities.supports_code:
            feature_score += 0.25
        if task_context.requires_reasoning and capabilities.supports_reasoning:
            feature_score += 0.25
        if task_context.requires_vision and capabilities.supports_vision:
            feature_score += 0.25
        if task_context.requires_function_calling and capabilities.supports_function_calling:
            feature_score += 0.25
        
        score += feature_score * 0.4
        
        return min(1.0, score)
    
    def _estimate_cost(self, model_config: ModelConfig, task_context: TaskContext) -> Optional[float]:
        """Estimate the cost for a task."""
        if not model_config.capabilities.cost_per_1k_tokens:
            return None
        
        # Estimate total tokens (input + output, rough approximation)
        total_tokens = task_context.estimated_tokens * 1.5  # Assume 50% more for output
        
        return (total_tokens / 1000.0) * model_config.capabilities.cost_per_1k_tokens
    
    def _generate_routing_reasoning(
        self, 
        model_name: str, 
        model_config: ModelConfig, 
        task_context: TaskContext, 
        score: float
    ) -> str:
        """Generate human-readable reasoning for the routing decision."""
        
        reasons = []
        
        # Task type specific reasoning
        if task_context.task_type == TaskType.ORCHESTRATION:
            reasons.append("Selected for orchestration capabilities")
        elif task_context.task_type == TaskType.DEBUGGING:
            reasons.append("Optimized for debugging and error analysis")
        elif task_context.task_type == TaskType.CODE_GENERATION:
            reasons.append("Strong code generation capabilities")
        
        # Provider specific strengths
        if model_config.provider == ModelProvider.ANTHROPIC:
            reasons.append("Claude excels at thoughtful analysis and code quality")
        elif model_config.provider == ModelProvider.GOOGLE:
            reasons.append("Gemini 2.5 Pro optimized for debugging and large contexts")
        elif model_config.provider == ModelProvider.OPENAI:
            reasons.append("GPT models provide reliable general performance")
        
        # Capability highlights
        if model_config.capabilities.context_length > 100000:
            reasons.append("Large context window for complex tasks")
        
        if task_context.cost_sensitive and model_config.capabilities.cost_per_1k_tokens:
            if model_config.capabilities.cost_per_1k_tokens < 0.001:
                reasons.append("Cost-effective option selected")
        
        # Performance considerations
        if score > 0.8:
            reasons.append("High performance history")
        elif score > 0.6:
            reasons.append("Good performance history")
        
        return f"Score: {score:.2f} - " + "; ".join(reasons)
    
    async def record_task_result(
        self, 
        model_name: str, 
        success: bool, 
        latency_ms: int, 
        cost: float = 0.0, 
        error_reason: Optional[str] = None
    ):
        """Record the result of a task execution."""
        if success:
            self.performance_tracker.record_success(model_name, latency_ms, cost)
        else:
            self.performance_tracker.record_failure(model_name, error_reason or "Unknown error")
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status information for all models."""
        status = {
            'total_models': len(self.models),
            'enabled_models': sum(1 for config in self.models.values() if config.enabled),
            'models': {}
        }
        
        for model_name, config in self.models.items():
            perf = self.performance_tracker.performance_data.get(model_name)
            
            model_status = {
                'provider': config.provider.value,
                'enabled': config.enabled,
                'priority': config.priority,
                'has_api_key': bool(os.getenv(config.api_key_env)),
                'capabilities': asdict(config.capabilities)
            }
            
            if perf:
                model_status['performance'] = {
                    'success_rate': perf.success_rate,
                    'avg_latency_ms': perf.avg_latency_ms,
                    'avg_cost': perf.avg_cost_per_request,
                    'total_requests': perf.total_requests,
                    'last_success': perf.last_success.isoformat() if perf.last_success else None,
                    'last_failure': perf.last_failure.isoformat() if perf.last_failure else None
                }
            
            status['models'][model_name] = model_status
        
        return status
    
    def update_model_config(self, model_name: str, updates: Dict[str, Any]):
        """Update configuration for a specific model."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        config = self.models[model_name]
        
        # Update basic properties
        if 'enabled' in updates:
            config.enabled = updates['enabled']
        if 'priority' in updates:
            config.priority = updates['priority']
        
        # Update capabilities
        if 'capabilities' in updates:
            for key, value in updates['capabilities'].items():
                if hasattr(config.capabilities, key):
                    setattr(config.capabilities, key, value)
        
        self.logger.info(f"Updated configuration for model {model_name}")


# Convenience functions for common routing scenarios

async def route_for_orchestration(router: ModelRouter, complexity: str = "medium", tokens: int = 4000) -> RoutingDecision:
    """Route a task for orchestration/management."""
    context = TaskContext(
        task_type=TaskType.ORCHESTRATION,
        complexity=complexity,
        estimated_tokens=tokens,
        requires_reasoning=True,
        latency_sensitive=False
    )
    return await router.route_task(context)


async def route_for_debugging(router: ModelRouter, complexity: str = "high", tokens: int = 8000) -> RoutingDecision:
    """Route a task for debugging."""
    context = TaskContext(
        task_type=TaskType.DEBUGGING,
        complexity=complexity,
        estimated_tokens=tokens,
        requires_code=True,
        requires_reasoning=True,
        latency_sensitive=True  # Debugging often needs quick iteration
    )
    return await router.route_task(context)


async def route_for_code_generation(router: ModelRouter, complexity: str = "medium", tokens: int = 6000) -> RoutingDecision:
    """Route a task for code generation."""
    context = TaskContext(
        task_type=TaskType.CODE_GENERATION,
        complexity=complexity,
        estimated_tokens=tokens,
        requires_code=True,
        requires_reasoning=True
    )
    return await router.route_task(context)


async def route_for_documentation(router: ModelRouter, tokens: int = 3000) -> RoutingDecision:
    """Route a task for documentation writing."""
    context = TaskContext(
        task_type=TaskType.DOCUMENTATION,
        complexity="low",
        estimated_tokens=tokens,
        requires_code=False,
        cost_sensitive=True  # Documentation can use cheaper models
    )
    return await router.route_task(context)