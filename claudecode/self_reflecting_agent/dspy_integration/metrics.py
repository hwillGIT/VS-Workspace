"""
Metrics tracking for DSPy signature performance.

This module tracks performance metrics for DSPy signatures to enable
optimization and continuous improvement of agent performance.
"""

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class ExecutionRecord:
    """Record of a single signature execution."""
    signature_name: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    execution_time: float
    success: bool
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None


class DSPyMetrics:
    """
    Metrics tracker for DSPy signature performance.
    
    This class tracks execution metrics, performance trends, and provides
    data for optimization processes.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Execution records
        self.execution_records: List[ExecutionRecord] = []
        self.max_records = 10000  # Limit memory usage
        
        # Aggregated metrics by signature
        self.signature_metrics: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "execution_count": 0,
                "success_count": 0,
                "total_execution_time": 0.0,
                "success_rate": 0.0,
                "avg_execution_time": 0.0,
                "min_execution_time": float('inf'),
                "max_execution_time": 0.0,
                "last_execution": None,
                "error_count": 0,
                "recent_trend": "stable"
            }
        )
        
        # Performance tracking
        self.performance_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        self.logger.info("DSPy metrics tracker initialized")
    
    def record_execution(
        self,
        signature_name: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        execution_time: float,
        success: bool,
        metadata: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None
    ) -> None:
        """Record a signature execution."""
        
        timestamp = datetime.now()
        
        # Create execution record
        record = ExecutionRecord(
            signature_name=signature_name,
            inputs=inputs.copy(),
            outputs=outputs.copy(),
            execution_time=execution_time,
            success=success,
            timestamp=timestamp,
            metadata=metadata or {},
            error_message=error_message
        )
        
        # Add to records (with size limit)
        self.execution_records.append(record)
        if len(self.execution_records) > self.max_records:
            self.execution_records.pop(0)  # Remove oldest record
        
        # Update aggregated metrics
        self._update_signature_metrics(record)
        
        # Update performance history
        self._update_performance_history(record)
    
    def _update_signature_metrics(self, record: ExecutionRecord) -> None:
        """Update aggregated metrics for a signature."""
        
        metrics = self.signature_metrics[record.signature_name]
        
        # Update counts
        metrics["execution_count"] += 1
        if record.success:
            metrics["success_count"] += 1
        else:
            metrics["error_count"] += 1
        
        # Update execution time metrics
        metrics["total_execution_time"] += record.execution_time
        metrics["min_execution_time"] = min(metrics["min_execution_time"], record.execution_time)
        metrics["max_execution_time"] = max(metrics["max_execution_time"], record.execution_time)
        
        # Recalculate derived metrics
        metrics["success_rate"] = metrics["success_count"] / metrics["execution_count"]
        metrics["avg_execution_time"] = metrics["total_execution_time"] / metrics["execution_count"]
        metrics["last_execution"] = record.timestamp.isoformat()
        
        # Update trend
        metrics["recent_trend"] = self._calculate_trend(record.signature_name)
    
    def _update_performance_history(self, record: ExecutionRecord) -> None:
        """Update performance history for trend analysis."""
        
        history = self.performance_history[record.signature_name]
        
        # Add current performance point
        history.append({
            "timestamp": record.timestamp.isoformat(),
            "execution_time": record.execution_time,
            "success": record.success,
            "inputs_size": len(str(record.inputs)),
            "outputs_size": len(str(record.outputs))
        })
        
        # Keep only recent history (last 100 executions)
        if len(history) > 100:
            history.pop(0)
    
    def _calculate_trend(self, signature_name: str) -> str:
        """Calculate performance trend for a signature."""
        
        history = self.performance_history[signature_name]
        
        if len(history) < 10:
            return "insufficient_data"
        
        # Analyze recent vs older performance
        recent_history = history[-10:]  # Last 10 executions
        older_history = history[-20:-10] if len(history) >= 20 else history[:-10]
        
        if not older_history:
            return "stable"
        
        # Calculate average success rates
        recent_success_rate = sum(1 for h in recent_history if h["success"]) / len(recent_history)
        older_success_rate = sum(1 for h in older_history if h["success"]) / len(older_history)
        
        # Calculate average execution times
        recent_avg_time = sum(h["execution_time"] for h in recent_history) / len(recent_history)
        older_avg_time = sum(h["execution_time"] for h in older_history) / len(older_history)
        
        # Determine trend
        success_improvement = recent_success_rate - older_success_rate
        time_improvement = older_avg_time - recent_avg_time  # Positive if time decreased
        
        if success_improvement > 0.1 or time_improvement > 0.5:
            return "improving"
        elif success_improvement < -0.1 or time_improvement < -0.5:
            return "declining"
        else:
            return "stable"
    
    def get_signature_metrics(self, signature_name: str) -> Dict[str, Any]:
        """Get metrics for a specific signature."""
        
        if signature_name not in self.signature_metrics:
            return {}
        
        metrics = self.signature_metrics[signature_name].copy()
        
        # Add additional computed metrics
        metrics["performance_score"] = self._calculate_performance_score(signature_name)
        metrics["optimization_recommendation"] = self._get_optimization_recommendation(signature_name)
        
        return metrics
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all signatures."""
        
        return {
            signature_name: self.get_signature_metrics(signature_name)
            for signature_name in self.signature_metrics.keys()
        }
    
    def _calculate_performance_score(self, signature_name: str) -> float:
        """Calculate an overall performance score for a signature."""
        
        metrics = self.signature_metrics[signature_name]
        
        if metrics["execution_count"] == 0:
            return 0.0
        
        # Weight different factors
        success_weight = 0.6
        speed_weight = 0.3
        stability_weight = 0.1
        
        # Success rate component (0-1)
        success_component = metrics["success_rate"]
        
        # Speed component (inverse of execution time, normalized)
        avg_time = metrics["avg_execution_time"]
        speed_component = min(1.0, 5.0 / max(avg_time, 0.1))  # 5 seconds = baseline
        
        # Stability component (based on trend)
        trend = metrics["recent_trend"]
        if trend == "improving":
            stability_component = 1.0
        elif trend == "stable":
            stability_component = 0.8
        elif trend == "declining":
            stability_component = 0.4
        else:  # insufficient_data
            stability_component = 0.6
        
        # Calculate weighted score
        performance_score = (
            success_weight * success_component +
            speed_weight * speed_component +
            stability_weight * stability_component
        )
        
        return min(performance_score, 1.0)
    
    def _get_optimization_recommendation(self, signature_name: str) -> str:
        """Get optimization recommendation for a signature."""
        
        metrics = self.signature_metrics[signature_name]
        
        if metrics["execution_count"] < 5:
            return "collect_more_data"
        
        success_rate = metrics["success_rate"]
        avg_time = metrics["avg_execution_time"]
        trend = metrics["recent_trend"]
        
        if success_rate < 0.6:
            return "optimize_for_accuracy"
        elif success_rate < 0.8 and trend == "declining":
            return "optimize_for_stability"
        elif avg_time > 10.0:
            return "optimize_for_speed"
        elif trend == "declining":
            return "investigate_degradation"
        else:
            return "no_optimization_needed"
    
    def get_training_data(self, signature_name: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get training data for signature optimization."""
        
        # Filter successful executions for the signature
        successful_records = [
            record for record in self.execution_records
            if record.signature_name == signature_name and record.success
        ]
        
        # Sort by recency and limit
        successful_records.sort(key=lambda r: r.timestamp, reverse=True)
        recent_records = successful_records[:limit]
        
        # Convert to training format
        training_data = []
        for record in recent_records:
            training_data.append({
                "inputs": record.inputs,
                "outputs": record.outputs,
                "metadata": record.metadata
            })
        
        return training_data
    
    def get_performance_report(self, signature_name: Optional[str] = None) -> Dict[str, Any]:
        """Generate a performance report."""
        
        report = {
            "report_timestamp": datetime.now().isoformat(),
            "total_executions": len(self.execution_records),
            "total_signatures": len(self.signature_metrics),
            "signatures": {}
        }
        
        # Add signature-specific metrics
        signatures_to_report = [signature_name] if signature_name else list(self.signature_metrics.keys())
        
        for sig_name in signatures_to_report:
            if sig_name in self.signature_metrics:
                report["signatures"][sig_name] = self.get_signature_metrics(sig_name)
        
        # Add overall statistics
        if not signature_name:
            total_executions = sum(m["execution_count"] for m in self.signature_metrics.values())
            total_successes = sum(m["success_count"] for m in self.signature_metrics.values())
            
            report["overall_statistics"] = {
                "total_executions": total_executions,
                "overall_success_rate": total_successes / total_executions if total_executions > 0 else 0.0,
                "average_execution_time": sum(
                    m["avg_execution_time"] * m["execution_count"] 
                    for m in self.signature_metrics.values()
                ) / total_executions if total_executions > 0 else 0.0,
                "signatures_needing_optimization": len([
                    sig for sig, metrics in self.signature_metrics.items()
                    if metrics["success_rate"] < 0.8 and metrics["execution_count"] >= 10
                ])
            }
        
        return report
    
    def get_recent_executions(
        self, 
        signature_name: Optional[str] = None,
        hours: int = 24,
        limit: int = 100
    ) -> List[ExecutionRecord]:
        """Get recent executions within specified timeframe."""
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_records = [
            record for record in self.execution_records
            if record.timestamp >= cutoff_time
        ]
        
        # Filter by signature if specified
        if signature_name:
            recent_records = [
                record for record in recent_records
                if record.signature_name == signature_name
            ]
        
        # Sort by recency and limit
        recent_records.sort(key=lambda r: r.timestamp, reverse=True)
        return recent_records[:limit]
    
    def get_error_analysis(self, signature_name: Optional[str] = None) -> Dict[str, Any]:
        """Analyze errors for debugging purposes."""
        
        error_records = [
            record for record in self.execution_records
            if not record.success and record.error_message
        ]
        
        if signature_name:
            error_records = [
                record for record in error_records
                if record.signature_name == signature_name
            ]
        
        # Categorize errors
        error_categories = defaultdict(list)
        error_frequency = defaultdict(int)
        
        for record in error_records:
            error_msg = record.error_message or "Unknown error"
            
            # Simple error categorization
            if "timeout" in error_msg.lower():
                category = "timeout"
            elif "network" in error_msg.lower() or "connection" in error_msg.lower():
                category = "network"
            elif "parse" in error_msg.lower() or "format" in error_msg.lower():
                category = "parsing"
            else:
                category = "other"
            
            error_categories[category].append(record)
            error_frequency[error_msg] += 1
        
        return {
            "total_errors": len(error_records),
            "error_categories": {
                category: len(records) 
                for category, records in error_categories.items()
            },
            "most_common_errors": sorted(
                error_frequency.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10],
            "recent_errors": error_records[-10:] if error_records else []
        }
    
    def export_metrics(self, export_path: str) -> bool:
        """Export metrics to file."""
        
        try:
            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "metrics": self.get_all_metrics(),
                "performance_report": self.get_performance_report(),
                "error_analysis": self.get_error_analysis()
            }
            
            import json
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            self.logger.info(f"Metrics exported to: {export_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export metrics: {e}")
            return False
    
    def import_metrics(self, metrics_data: Dict[str, Any]) -> bool:
        """Import metrics data."""
        
        try:
            if "metrics" in metrics_data:
                for signature_name, metrics in metrics_data["metrics"].items():
                    # Import signature metrics (excluding computed fields)
                    base_metrics = {
                        k: v for k, v in metrics.items()
                        if k not in ["performance_score", "optimization_recommendation"]
                    }
                    self.signature_metrics[signature_name].update(base_metrics)
            
            self.logger.info("Metrics imported successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to import metrics: {e}")
            return False
    
    def reset(self) -> None:
        """Reset all metrics."""
        
        self.execution_records.clear()
        self.signature_metrics.clear()
        self.performance_history.clear()
        
        self.logger.info("Metrics reset")
    
    def cleanup_old_records(self, days: int = 30) -> int:
        """Clean up old execution records."""
        
        cutoff_time = datetime.now() - timedelta(days=days)
        
        original_count = len(self.execution_records)
        self.execution_records = [
            record for record in self.execution_records
            if record.timestamp >= cutoff_time
        ]
        
        removed_count = original_count - len(self.execution_records)
        
        if removed_count > 0:
            self.logger.info(f"Cleaned up {removed_count} old execution records")
        
        return removed_count