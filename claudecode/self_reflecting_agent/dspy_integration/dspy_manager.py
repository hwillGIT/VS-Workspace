"""
DSPy Manager for coordinating DSPy integration across agents.

This module manages DSPy configuration, optimization, and integration
with the agent system for improved performance through learning.
"""

import logging
import json
from typing import Any, Dict, List, Optional, Type, Union
from datetime import datetime
from pathlib import Path

try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    # Create mock DSPy classes for fallback
    class MockDSPy:
        class LM:
            def __init__(self, *args, **kwargs):
                pass
        
        class Signature:
            pass
        
        class TypedChainOfThought:
            def __init__(self, signature):
                self.signature = signature
        
        @staticmethod
        def configure(**kwargs):
            pass
    
    dspy = MockDSPy()

from .signatures import AgentSignatures
from .optimization import SignatureOptimizer
from .metrics import DSPyMetrics


class DSPyManager:
    """
    Central manager for DSPy integration across the agent system.
    
    This class handles:
    - DSPy language model configuration
    - Signature management and optimization
    - Performance tracking and metrics
    - Integration with agent workflows
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # DSPy availability
        self.dspy_enabled = DSPY_AVAILABLE and config.get("enable_dspy", True)
        
        if not self.dspy_enabled:
            self.logger.warning("DSPy not available or disabled, using fallback implementations")
        
        # Core components
        self.language_model: Optional[dspy.LM] = None
        self.signatures = AgentSignatures()
        self.optimizer: Optional[SignatureOptimizer] = None
        self.metrics = DSPyMetrics()
        
        # Configuration
        self.model_config = config.get("model", {})
        self.optimization_config = config.get("optimization", {})
        
        # State management
        self.initialized = False
        self.optimization_history: List[Dict[str, Any]] = []
        
        # Initialize if DSPy is available
        if self.dspy_enabled:
            self._initialize_dspy()
    
    def _initialize_dspy(self) -> None:
        """Initialize DSPy components."""
        
        try:
            # Configure language model
            self._setup_language_model()
            
            # Initialize optimizer
            if self.optimization_config.get("enabled", False):
                self.optimizer = SignatureOptimizer(
                    config=self.optimization_config,
                    metrics=self.metrics
                )
            
            self.initialized = True
            self.logger.info("DSPy integration initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize DSPy: {e}")
            self.dspy_enabled = False
    
    def _setup_language_model(self) -> None:
        """Setup the DSPy language model."""
        
        model_name = self.model_config.get("name", "gpt-4o")
        model_params = self.model_config.get("params", {})
        
        try:
            # Configure based on model type
            if "gpt" in model_name.lower():
                # OpenAI GPT models
                self.language_model = dspy.OpenAI(
                    model=model_name,
                    max_tokens=model_params.get("max_tokens", 4000),
                    temperature=model_params.get("temperature", 0.1),
                    **model_params.get("additional_params", {})
                )
            elif "claude" in model_name.lower():
                # Anthropic Claude models
                self.language_model = dspy.Anthropic(
                    model=model_name,
                    max_tokens=model_params.get("max_tokens", 4000),
                    temperature=model_params.get("temperature", 0.1),
                    **model_params.get("additional_params", {})
                )
            else:
                # Generic model setup
                self.language_model = dspy.LM(
                    model=model_name,
                    **model_params
                )
            
            # Configure DSPy with the language model
            dspy.configure(lm=self.language_model)
            
            self.logger.info(f"Configured DSPy with model: {model_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to setup language model: {e}")
            raise
    
    def get_signature(self, signature_name: str) -> Optional[Type[dspy.Signature]]:
        """Get a DSPy signature by name."""
        
        if not self.dspy_enabled:
            return None
        
        return self.signatures.get_signature(signature_name)
    
    def create_module(self, signature_name: str, module_type: str = "chain_of_thought") -> Optional[Any]:
        """Create a DSPy module from a signature."""
        
        if not self.dspy_enabled:
            return None
        
        signature = self.get_signature(signature_name)
        if not signature:
            self.logger.warning(f"Signature not found: {signature_name}")
            return None
        
        try:
            if module_type == "chain_of_thought":
                return dspy.TypedChainOfThought(signature)
            elif module_type == "react":
                return dspy.ReAct(signature)
            elif module_type == "program_of_thought":
                return dspy.ProgramOfThought(signature)
            else:
                self.logger.warning(f"Unknown module type: {module_type}")
                return dspy.TypedChainOfThought(signature)  # Default fallback
                
        except Exception as e:
            self.logger.error(f"Failed to create module for {signature_name}: {e}")
            return None
    
    async def optimize_signature(
        self, 
        signature_name: str,
        training_data: List[Dict[str, Any]],
        validation_data: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """Optimize a signature using training data."""
        
        if not self.dspy_enabled or not self.optimizer:
            self.logger.warning("DSPy optimization not available")
            return False
        
        try:
            # Get the signature
            signature = self.get_signature(signature_name)
            if not signature:
                return False
            
            # Create module for optimization
            module = self.create_module(signature_name)
            if not module:
                return False
            
            # Run optimization
            optimization_result = await self.optimizer.optimize_signature(
                module=module,
                signature_name=signature_name,
                training_data=training_data,
                validation_data=validation_data
            )
            
            # Record optimization
            self.optimization_history.append({
                "signature_name": signature_name,
                "timestamp": datetime.now().isoformat(),
                "result": optimization_result,
                "training_samples": len(training_data),
                "validation_samples": len(validation_data) if validation_data else 0
            })
            
            return optimization_result.get("success", False)
            
        except Exception as e:
            self.logger.error(f"Signature optimization failed for {signature_name}: {e}")
            return False
    
    def record_execution(
        self, 
        signature_name: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        execution_time: float,
        success: bool,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record execution metrics for a signature."""
        
        self.metrics.record_execution(
            signature_name=signature_name,
            inputs=inputs,
            outputs=outputs,
            execution_time=execution_time,
            success=success,
            metadata=metadata or {}
        )
    
    def get_signature_performance(self, signature_name: str) -> Dict[str, Any]:
        """Get performance metrics for a signature."""
        
        return self.metrics.get_signature_metrics(signature_name)
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all DSPy metrics."""
        
        return self.metrics.get_all_metrics()
    
    async def auto_optimize_signatures(
        self, 
        min_executions: int = 10,
        performance_threshold: float = 0.8
    ) -> Dict[str, bool]:
        """Automatically optimize signatures based on performance data."""
        
        if not self.dspy_enabled or not self.optimizer:
            return {}
        
        optimization_results = {}
        signature_metrics = self.metrics.get_all_metrics()
        
        for signature_name, metrics in signature_metrics.items():
            # Check if optimization is needed
            execution_count = metrics.get("execution_count", 0)
            success_rate = metrics.get("success_rate", 0.0)
            
            if (execution_count >= min_executions and 
                success_rate < performance_threshold):
                
                self.logger.info(f"Auto-optimizing signature: {signature_name}")
                
                # Generate training data from execution history
                training_data = self.metrics.get_training_data(signature_name)
                
                if len(training_data) >= 5:  # Minimum training samples
                    result = await self.optimize_signature(
                        signature_name=signature_name,
                        training_data=training_data
                    )
                    optimization_results[signature_name] = result
                else:
                    self.logger.warning(f"Insufficient training data for {signature_name}")
                    optimization_results[signature_name] = False
        
        return optimization_results
    
    def export_configuration(self, export_path: Path) -> bool:
        """Export DSPy configuration and optimization history."""
        
        try:
            export_data = {
                "dspy_enabled": self.dspy_enabled,
                "model_config": self.model_config,
                "optimization_config": self.optimization_config,
                "optimization_history": self.optimization_history,
                "metrics_summary": self.get_all_metrics(),
                "export_timestamp": datetime.now().isoformat()
            }
            
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            self.logger.info(f"DSPy configuration exported to: {export_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export configuration: {e}")
            return False
    
    def import_configuration(self, import_path: Path) -> bool:
        """Import DSPy configuration and optimization history."""
        
        try:
            if not import_path.exists():
                self.logger.error(f"Import file not found: {import_path}")
                return False
            
            with open(import_path, 'r') as f:
                import_data = json.load(f)
            
            # Update configuration
            if "optimization_history" in import_data:
                self.optimization_history = import_data["optimization_history"]
            
            # Import metrics if available
            if "metrics_summary" in import_data:
                self.metrics.import_metrics(import_data["metrics_summary"])
            
            self.logger.info(f"DSPy configuration imported from: {import_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to import configuration: {e}")
            return False
    
    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Get recommendations for signature optimization."""
        
        recommendations = []
        signature_metrics = self.metrics.get_all_metrics()
        
        for signature_name, metrics in signature_metrics.items():
            execution_count = metrics.get("execution_count", 0)
            success_rate = metrics.get("success_rate", 0.0)
            avg_execution_time = metrics.get("avg_execution_time", 0.0)
            
            # Recommend optimization based on various criteria
            if execution_count >= 10:
                if success_rate < 0.8:
                    recommendations.append({
                        "signature_name": signature_name,
                        "recommendation": "optimize_for_accuracy",
                        "reason": f"Success rate ({success_rate:.2f}) below threshold",
                        "priority": "high" if success_rate < 0.6 else "medium"
                    })
                
                if avg_execution_time > 5.0:  # More than 5 seconds
                    recommendations.append({
                        "signature_name": signature_name,
                        "recommendation": "optimize_for_speed",
                        "reason": f"Average execution time ({avg_execution_time:.2f}s) too high",
                        "priority": "medium"
                    })
            
            elif execution_count >= 5:
                recommendations.append({
                    "signature_name": signature_name,
                    "recommendation": "collect_more_data",
                    "reason": f"Only {execution_count} executions recorded",
                    "priority": "low"
                })
        
        return recommendations
    
    def reset_metrics(self) -> None:
        """Reset all DSPy metrics."""
        self.metrics.reset()
        self.optimization_history.clear()
        self.logger.info("DSPy metrics reset")
    
    def is_available(self) -> bool:
        """Check if DSPy is available and initialized."""
        return self.dspy_enabled and self.initialized
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of DSPy integration."""
        
        status = {
            "dspy_available": DSPY_AVAILABLE,
            "dspy_enabled": self.dspy_enabled,
            "initialized": self.initialized,
            "model_config": self.model_config,
            "optimization_enabled": self.optimizer is not None,
            "total_signatures": len(self.signatures.get_all_signatures()),
            "optimizations_performed": len(self.optimization_history),
            "total_executions": sum(
                metrics.get("execution_count", 0) 
                for metrics in self.metrics.get_all_metrics().values()
            )
        }
        
        if self.language_model:
            status["language_model"] = str(type(self.language_model).__name__)
        
        return status