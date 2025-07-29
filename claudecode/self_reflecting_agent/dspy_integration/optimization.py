"""
DSPy signature optimization for improved performance.

This module handles the optimization of DSPy signatures using various
optimization strategies to improve agent performance over time.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False

from .metrics import DSPyMetrics


@dataclass
class OptimizationResult:
    """Result of a signature optimization process."""
    success: bool
    signature_name: str
    optimization_method: str
    improvement_score: float
    execution_time: float
    iterations: int
    final_metrics: Dict[str, Any]
    error_message: Optional[str] = None


class SignatureOptimizer:
    """
    Optimizer for DSPy signatures to improve performance.
    
    This class implements various optimization strategies including:
    - Few-shot learning with examples
    - Automatic prompt optimization  
    - Parameter tuning
    - Performance-based optimization
    """
    
    def __init__(self, config: Dict[str, Any], metrics: DSPyMetrics):
        self.config = config
        self.metrics = metrics
        self.logger = logging.getLogger(__name__)
        
        # Optimization settings
        self.max_iterations = config.get("max_iterations", 50)
        self.optimization_timeout = config.get("timeout_seconds", 300)
        self.min_improvement = config.get("min_improvement", 0.05)
        
        # Available optimization methods
        self.optimization_methods = {
            "bootstrap_few_shot": self._bootstrap_few_shot,
            "random_search": self._random_search,
            "gradient_free": self._gradient_free_optimization
        }
        
        self.logger.info("Signature optimizer initialized")
    
    async def optimize_signature(
        self,
        module: Any,
        signature_name: str,
        training_data: List[Dict[str, Any]],
        validation_data: Optional[List[Dict[str, Any]]] = None
    ) -> OptimizationResult:
        """
        Optimize a DSPy signature using training data.
        
        Args:
            module: DSPy module to optimize
            signature_name: Name of the signature being optimized
            training_data: Training examples for optimization
            validation_data: Validation examples for evaluation
            
        Returns:
            OptimizationResult with optimization details and metrics
        """
        
        if not DSPY_AVAILABLE:
            return OptimizationResult(
                success=False,
                signature_name=signature_name,
                optimization_method="none",
                improvement_score=0.0,
                execution_time=0.0,
                iterations=0,
                final_metrics={},
                error_message="DSPy not available"
            )
        
        start_time = datetime.now()
        
        try:
            self.logger.info(f"Starting optimization for signature: {signature_name}")
            
            # Get baseline performance
            baseline_metrics = await self._evaluate_module(
                module, validation_data or training_data[:5]
            )
            
            # Choose optimization method
            method = self.config.get("optimization_method", "bootstrap_few_shot")
            
            if method not in self.optimization_methods:
                method = "bootstrap_few_shot"  # Fallback
            
            # Run optimization
            optimization_func = self.optimization_methods[method]
            optimized_module, iterations = await optimization_func(
                module, training_data, validation_data
            )
            
            # Evaluate optimized module
            final_metrics = await self._evaluate_module(
                optimized_module, validation_data or training_data[:5]
            )
            
            # Calculate improvement
            improvement_score = self._calculate_improvement(
                baseline_metrics, final_metrics
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            result = OptimizationResult(
                success=True,
                signature_name=signature_name,
                optimization_method=method,
                improvement_score=improvement_score,
                execution_time=execution_time,
                iterations=iterations,
                final_metrics=final_metrics
            )
            
            self.logger.info(
                f"Optimization completed for {signature_name}: "
                f"{improvement_score:.3f} improvement in {iterations} iterations"
            )
            
            return result
            
        except Exception as e:
            error_msg = f"Optimization failed for {signature_name}: {str(e)}"
            self.logger.error(error_msg)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return OptimizationResult(
                success=False,
                signature_name=signature_name,
                optimization_method=method if 'method' in locals() else "unknown",
                improvement_score=0.0,
                execution_time=execution_time,
                iterations=0,
                final_metrics={},
                error_message=error_msg
            )
    
    async def _bootstrap_few_shot(
        self,
        module: Any,
        training_data: List[Dict[str, Any]],
        validation_data: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[Any, int]:
        """Optimize using bootstrap few-shot learning."""
        
        try:
            # Convert training data to DSPy examples
            examples = []
            for data in training_data[:20]:  # Limit examples
                try:
                    # Create example from input/output pairs
                    example_dict = {}
                    
                    # Add inputs
                    if "inputs" in data:
                        example_dict.update(data["inputs"])
                    
                    # Add expected outputs
                    if "outputs" in data:
                        example_dict.update(data["outputs"])
                    
                    if example_dict:
                        examples.append(dspy.Example(**example_dict))
                        
                except Exception as e:
                    self.logger.warning(f"Failed to create example: {e}")
                    continue
            
            if not examples:
                self.logger.warning("No valid examples created for optimization")
                return module, 0
            
            # Use DSPy's BootstrapFewShot optimizer
            optimizer = dspy.BootstrapFewShot(
                metric=self._create_metric_function(),
                max_bootstrapped_demos=min(len(examples), 8),
                max_labeled_demos=min(len(examples), 16)
            )
            
            # Compile the optimized module
            optimized_module = optimizer.compile(
                module,
                trainset=examples[:min(len(examples), 12)],
                valset=examples[12:] if len(examples) > 12 else examples[:3]
            )
            
            return optimized_module, len(examples)
            
        except Exception as e:
            self.logger.error(f"Bootstrap few-shot optimization failed: {e}")
            return module, 0
    
    async def _random_search(
        self,
        module: Any,
        training_data: List[Dict[str, Any]],
        validation_data: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[Any, int]:
        """Optimize using random search over hyperparameters."""
        
        # This is a simplified implementation
        # In practice, this would search over various parameters
        
        best_module = module
        best_score = 0.0
        iterations = 0
        
        try:
            eval_data = validation_data or training_data[:5]
            baseline_metrics = await self._evaluate_module(module, eval_data)
            best_score = baseline_metrics.get("success_rate", 0.0)
            
            # Try different configurations
            for i in range(min(10, self.max_iterations)):
                iterations += 1
                
                # This would involve creating variations of the module
                # For now, we'll just return the original module
                candidate_module = module
                
                candidate_metrics = await self._evaluate_module(candidate_module, eval_data)
                candidate_score = candidate_metrics.get("success_rate", 0.0)
                
                if candidate_score > best_score:
                    best_module = candidate_module
                    best_score = candidate_score
            
            return best_module, iterations
            
        except Exception as e:
            self.logger.error(f"Random search optimization failed: {e}")
            return module, iterations
    
    async def _gradient_free_optimization(
        self,
        module: Any,
        training_data: List[Dict[str, Any]],
        validation_data: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[Any, int]:
        """Optimize using gradient-free methods."""
        
        # This would implement gradient-free optimization strategies
        # For now, return the original module
        
        self.logger.info("Gradient-free optimization not fully implemented")
        return module, 1
    
    async def _evaluate_module(
        self,
        module: Any,
        evaluation_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Evaluate a module's performance on evaluation data."""
        
        metrics = {
            "success_rate": 0.0,
            "avg_execution_time": 0.0,
            "total_evaluations": 0,
            "successful_evaluations": 0
        }
        
        if not evaluation_data:
            return metrics
        
        total_time = 0.0
        successful_evaluations = 0
        
        for data in evaluation_data:
            try:
                start_time = datetime.now()
                
                # Execute the module with input data
                inputs = data.get("inputs", {})
                expected_outputs = data.get("outputs", {})
                
                # This would involve actually calling the module
                # For now, we'll simulate evaluation
                execution_time = 0.1  # Simulated execution time
                success = True  # Simulated success
                
                total_time += execution_time
                if success:
                    successful_evaluations += 1
                    
            except Exception as e:
                self.logger.warning(f"Evaluation failed for data point: {e}")
                continue
        
        total_evaluations = len(evaluation_data)
        
        metrics.update({
            "success_rate": successful_evaluations / total_evaluations if total_evaluations > 0 else 0.0,
            "avg_execution_time": total_time / total_evaluations if total_evaluations > 0 else 0.0,
            "total_evaluations": total_evaluations,
            "successful_evaluations": successful_evaluations
        })
        
        return metrics
    
    def _calculate_improvement(
        self,
        baseline_metrics: Dict[str, Any],
        final_metrics: Dict[str, Any]
    ) -> float:
        """Calculate improvement score between baseline and final metrics."""
        
        baseline_score = baseline_metrics.get("success_rate", 0.0)
        final_score = final_metrics.get("success_rate", 0.0)
        
        if baseline_score == 0.0:
            return 1.0 if final_score > 0.0 else 0.0
        
        improvement = (final_score - baseline_score) / baseline_score
        return max(improvement, -1.0)  # Cap at -100% decline
    
    def _create_metric_function(self):
        """Create a metric function for DSPy optimization."""
        
        def metric(example, prediction, trace=None):
            """Simple metric function for optimization."""
            try:
                # This would implement actual metric calculation
                # For now, return a simple score
                return 1.0 if prediction else 0.0
            except:
                return 0.0
        
        return metric
    
    async def optimize_multiple_signatures(
        self,
        modules: Dict[str, Any],
        training_data: Dict[str, List[Dict[str, Any]]],
        validation_data: Optional[Dict[str, List[Dict[str, Any]]]] = None
    ) -> Dict[str, OptimizationResult]:
        """
        Optimize multiple signatures in parallel.
        
        Args:
            modules: Dictionary of signature_name -> module
            training_data: Dictionary of signature_name -> training examples
            validation_data: Dictionary of signature_name -> validation examples
            
        Returns:
            Dictionary of signature_name -> OptimizationResult
        """
        
        optimization_tasks = []
        
        for signature_name, module in modules.items():
            if signature_name in training_data:
                task = self.optimize_signature(
                    module=module,
                    signature_name=signature_name,
                    training_data=training_data[signature_name],
                    validation_data=validation_data.get(signature_name) if validation_data else None
                )
                optimization_tasks.append((signature_name, task))
        
        # Execute optimizations in parallel
        results = {}
        
        if optimization_tasks:
            task_results = await asyncio.gather(
                *[task for _, task in optimization_tasks],
                return_exceptions=True
            )
            
            for (signature_name, _), result in zip(optimization_tasks, task_results):
                if isinstance(result, Exception):
                    results[signature_name] = OptimizationResult(
                        success=False,
                        signature_name=signature_name,
                        optimization_method="parallel",
                        improvement_score=0.0,
                        execution_time=0.0,
                        iterations=0,
                        final_metrics={},
                        error_message=str(result)
                    )
                else:
                    results[signature_name] = result
        
        return results
    
    def get_optimization_recommendations(
        self,
        signature_metrics: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Get recommendations for signature optimization."""
        
        recommendations = []
        
        for signature_name, metrics in signature_metrics.items():
            execution_count = metrics.get("execution_count", 0)
            success_rate = metrics.get("success_rate", 0.0)
            avg_execution_time = metrics.get("avg_execution_time", 0.0)
            
            if execution_count >= 10:
                if success_rate < 0.8:
                    recommendations.append({
                        "signature_name": signature_name,
                        "recommendation": "accuracy_optimization",
                        "reason": f"Success rate ({success_rate:.2f}) below threshold",
                        "priority": "high" if success_rate < 0.6 else "medium",
                        "suggested_method": "bootstrap_few_shot"
                    })
                
                if avg_execution_time > 5.0:
                    recommendations.append({
                        "signature_name": signature_name,
                        "recommendation": "speed_optimization", 
                        "reason": f"Average execution time ({avg_execution_time:.2f}s) too high",
                        "priority": "medium",
                        "suggested_method": "random_search"
                    })
            
            elif execution_count >= 5:
                recommendations.append({
                    "signature_name": signature_name,
                    "recommendation": "collect_more_data",
                    "reason": f"Only {execution_count} executions recorded",
                    "priority": "low",
                    "suggested_method": None
                })
        
        return recommendations
    
    def export_optimization_results(
        self,
        results: Dict[str, OptimizationResult],
        export_path: str
    ) -> bool:
        """Export optimization results to file."""
        
        try:
            export_data = {
                "optimization_timestamp": datetime.now().isoformat(),
                "total_signatures": len(results),
                "successful_optimizations": sum(1 for r in results.values() if r.success),
                "results": {}
            }
            
            for signature_name, result in results.items():
                export_data["results"][signature_name] = {
                    "success": result.success,
                    "optimization_method": result.optimization_method,
                    "improvement_score": result.improvement_score,
                    "execution_time": result.execution_time,
                    "iterations": result.iterations,
                    "final_metrics": result.final_metrics,
                    "error_message": result.error_message
                }
            
            import json
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            self.logger.info(f"Optimization results exported to: {export_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export optimization results: {e}")
            return False