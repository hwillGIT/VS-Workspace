"""
Functional Signal Processing Module for Signal Synthesis Agent

This module implements functional programming patterns for signal processing,
consensus building, and validation. It provides pure functions, immutable data
structures, and composable operations for reliable signal analysis.

Key Features:
- Immutable signal data structures
- Pure validation functions
- Composable consensus algorithms
- Safe error handling with Maybe/Either monads
- Parallel signal processing capabilities
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from enum import Enum
import numpy as np
from datetime import datetime
from functools import partial, reduce

# Import functional programming utilities from global ClaudeCode level
try:
    from ....functional_utils import (
        Maybe, Either, FunctionalList, FunctionalOps, ParallelOps,
        fl, fmap, ffilter, fpartition, fgroupby
    )
    FUNCTIONAL_AVAILABLE = True
except ImportError:
    FUNCTIONAL_AVAILABLE = False
    from typing import NamedTuple
    # Fallback implementations


class SignalType(Enum):
    """Types of trading signals."""
    BUY = 1
    SELL = -1
    HOLD = 0
    STRONG_BUY = 2
    STRONG_SELL = -2


class ConfidenceLevel(Enum):
    """Confidence levels for signals."""
    VERY_LOW = 0.2
    LOW = 0.4
    MEDIUM = 0.6
    HIGH = 0.8
    VERY_HIGH = 0.95


@dataclass(frozen=True)
class Signal:
    """Immutable signal data structure."""
    source: str
    timestamp: datetime
    signal_type: SignalType
    strength: float  # -1.0 to 1.0
    confidence: float  # 0.0 to 1.0
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        if not -1.0 <= self.strength <= 1.0:
            raise ValueError("Signal strength must be between -1.0 and 1.0")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
    
    def normalize(self) -> 'Signal':
        """Normalize signal to standard format."""
        # Ensure signal_type matches strength
        if self.strength > 0.5:
            new_type = SignalType.STRONG_BUY
        elif self.strength > 0.1:
            new_type = SignalType.BUY
        elif self.strength < -0.5:
            new_type = SignalType.STRONG_SELL
        elif self.strength < -0.1:
            new_type = SignalType.SELL
        else:
            new_type = SignalType.HOLD
        
        return Signal(
            source=self.source,
            timestamp=self.timestamp,
            signal_type=new_type,
            strength=self.strength,
            confidence=self.confidence,
            metadata=self.metadata
        )
    
    def weighted_strength(self) -> float:
        """Get confidence-weighted signal strength."""
        return self.strength * self.confidence


@dataclass(frozen=True)
class ConsensusResult:
    """Immutable consensus analysis result."""
    consensus_signal: Optional[Signal]
    participating_signals: FunctionalList[Signal]
    outlier_signals: FunctionalList[Signal]
    consensus_strength: float
    agreement_ratio: float
    is_valid: bool
    validation_details: Dict[str, Any]


@dataclass(frozen=True)
class SignalValidationRule:
    """Immutable validation rule for signals."""
    name: str
    predicate: Callable[[Signal], bool]
    weight: float
    description: str


class FunctionalSignalProcessor:
    """
    Functional signal processing operations using pure functions and immutable data.
    """
    
    # ============================================================================
    # Pure Signal Operations
    # ============================================================================
    
    @staticmethod
    def create_signal(source: str, 
                     strength: float, 
                     confidence: float,
                     timestamp: Optional[datetime] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> Maybe[Signal]:
        """Create a signal safely with validation."""
        if not FUNCTIONAL_AVAILABLE:
            try:
                signal = Signal(
                    source=source,
                    timestamp=timestamp or datetime.now(),
                    signal_type=SignalType.HOLD,  # Will be corrected by normalize
                    strength=strength,
                    confidence=confidence,
                    metadata=metadata or {}
                )
                return type('Maybe', (), {'value': signal.normalize(), 'is_some': lambda: True})()
            except Exception:
                return type('Maybe', (), {'is_some': lambda: False})()
        
        try:
            signal = Signal(
                source=source,
                timestamp=timestamp or datetime.now(),
                signal_type=SignalType.HOLD,  # Will be corrected by normalize
                strength=strength,
                confidence=confidence,
                metadata=metadata or {}
            )
            return Maybe.some(signal.normalize())
        except Exception:
            return Maybe.none()
    
    @staticmethod
    def filter_signals_by_confidence(signals: FunctionalList[Signal], 
                                   min_confidence: float) -> FunctionalList[Signal]:
        """Filter signals by minimum confidence level."""
        return signals.filter(lambda s: s.confidence >= min_confidence)
    
    @staticmethod
    def group_signals_by_source(signals: FunctionalList[Signal]) -> Dict[str, FunctionalList[Signal]]:
        """Group signals by their source."""
        return signals.group_by(lambda s: s.source)
    
    @staticmethod
    def partition_signals_by_direction(signals: FunctionalList[Signal]) -> Tuple[FunctionalList[Signal], FunctionalList[Signal]]:
        """Partition signals into bullish (positive) and bearish (negative)."""
        return signals.partition(lambda s: s.strength > 0)
    
    @staticmethod
    def calculate_weighted_consensus(signals: FunctionalList[Signal]) -> Maybe[float]:
        """Calculate consensus using confidence-weighted average."""
        if signals.is_empty():
            return Maybe.none() if FUNCTIONAL_AVAILABLE else type('Maybe', (), {'is_some': lambda: False})()
        
        weighted_strengths = signals.map(lambda s: s.weighted_strength())
        total_confidence = signals.map(lambda s: s.confidence).reduce(lambda a, b: a + b, 0)
        
        if total_confidence == 0:
            return Maybe.none() if FUNCTIONAL_AVAILABLE else type('Maybe', (), {'is_some': lambda: False})()
        
        consensus = weighted_strengths.reduce(lambda a, b: a + b, 0) / total_confidence
        return Maybe.some(consensus) if FUNCTIONAL_AVAILABLE else type('Maybe', (), {'value': consensus, 'is_some': lambda: True})()
    
    # ============================================================================
    # Outlier Detection
    # ============================================================================
    
    @staticmethod
    def detect_outliers_zscore(signals: FunctionalList[Signal], 
                             threshold: float = 2.0) -> FunctionalList[Signal]:
        """Detect outlier signals using z-score method."""
        if len(signals) < 3:
            return fl([])
        
        strengths = signals.map(lambda s: s.strength)
        mean = sum(strengths) / len(strengths)
        variance = sum((x - mean) ** 2 for x in strengths) / (len(strengths) - 1)
        std = variance ** 0.5
        
        if std == 0:
            return fl([])
        
        return signals.filter(lambda s: abs(s.strength - mean) / std > threshold)
    
    @staticmethod
    def detect_outliers_iqr(signals: FunctionalList[Signal]) -> FunctionalList[Signal]:
        """Detect outlier signals using IQR method."""
        if len(signals) < 4:
            return fl([])
        
        strengths = signals.map(lambda s: s.strength).sort()
        n = len(strengths)
        
        q1_idx = n // 4
        q3_idx = 3 * n // 4
        
        q1 = strengths[q1_idx]
        q3 = strengths[q3_idx]
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        return signals.filter(lambda s: s.strength < lower_bound or s.strength > upper_bound)
    
    # ============================================================================
    # Consensus Building
    # ============================================================================
    
    @staticmethod
    def build_consensus(signals: FunctionalList[Signal],
                       min_sources: int = 3,
                       outlier_method: str = 'zscore',
                       confidence_threshold: float = 0.7) -> ConsensusResult:
        """Build consensus from multiple signals using functional approach."""
        
        # Validate minimum sources
        if len(signals) < min_sources:
            return ConsensusResult(
                consensus_signal=None,
                participating_signals=fl([]),
                outlier_signals=fl([]),
                consensus_strength=0.0,
                agreement_ratio=0.0,
                is_valid=False,
                validation_details={'error': 'Insufficient sources'}
            )
        
        # Detect and remove outliers
        if outlier_method == 'zscore':
            outliers = FunctionalSignalProcessor.detect_outliers_zscore(signals)
        elif outlier_method == 'iqr':
            outliers = FunctionalSignalProcessor.detect_outliers_iqr(signals)
        else:
            outliers = fl([])
        
        # Get participating signals (non-outliers)
        outlier_sources = outliers.map(lambda s: s.source).distinct()
        participating = signals.filter(lambda s: s.source not in outlier_sources._items)
        
        # Check if we still have enough sources
        if len(participating) < min_sources:
            return ConsensusResult(
                consensus_signal=None,
                participating_signals=participating,
                outlier_signals=outliers,
                consensus_strength=0.0,
                agreement_ratio=0.0,
                is_valid=False,
                validation_details={'error': 'Too many outliers'}
            )
        
        # Calculate consensus
        consensus_result = FunctionalSignalProcessor.calculate_weighted_consensus(participating)
        
        if not consensus_result.is_some():
            return ConsensusResult(
                consensus_signal=None,
                participating_signals=participating,
                outlier_signals=outliers,
                consensus_strength=0.0,
                agreement_ratio=0.0,
                is_valid=False,
                validation_details={'error': 'Consensus calculation failed'}
            )
        
        consensus_strength = consensus_result.value
        
        # Calculate agreement ratio
        consensus_direction = 1 if consensus_strength > 0 else -1 if consensus_strength < 0 else 0
        agreeing_signals = participating.filter(
            lambda s: (s.strength > 0 and consensus_direction > 0) or 
                     (s.strength < 0 and consensus_direction < 0) or
                     (abs(s.strength) < 0.1 and consensus_direction == 0)
        )
        
        agreement_ratio = len(agreeing_signals) / len(participating) if len(participating) > 0 else 0
        
        # Determine overall confidence
        avg_confidence = participating.map(lambda s: s.confidence).reduce(lambda a, b: a + b, 0) / len(participating)
        overall_confidence = agreement_ratio * avg_confidence
        
        # Create consensus signal
        consensus_signal = Signal(
            source="consensus",
            timestamp=datetime.now(),
            signal_type=SignalType.HOLD,  # Will be normalized
            strength=consensus_strength,
            confidence=overall_confidence,
            metadata={
                'participating_sources': len(participating),
                'outlier_sources': len(outliers),
                'agreement_ratio': agreement_ratio,
                'method': outlier_method
            }
        ).normalize()
        
        # Validate consensus
        is_valid = (
            agreement_ratio >= 0.6 and
            overall_confidence >= confidence_threshold and
            len(participating) >= min_sources
        )
        
        return ConsensusResult(
            consensus_signal=consensus_signal,
            participating_signals=participating,
            outlier_signals=outliers,
            consensus_strength=abs(consensus_strength),
            agreement_ratio=agreement_ratio,
            is_valid=is_valid,
            validation_details={
                'total_sources': len(signals),
                'participating_sources': len(participating),
                'outlier_sources': len(outliers),
                'average_confidence': avg_confidence,
                'consensus_direction': consensus_direction,
                'method': outlier_method
            }
        )
    
    # ============================================================================
    # Signal Validation
    # ============================================================================
    
    @staticmethod
    def create_validation_rules() -> FunctionalList[SignalValidationRule]:
        """Create standard validation rules for signals."""
        rules = [
            SignalValidationRule(
                name="confidence_check",
                predicate=lambda s: s.confidence >= 0.3,
                weight=1.0,
                description="Signal must have minimum confidence"
            ),
            SignalValidationRule(
                name="strength_range",
                predicate=lambda s: -1.0 <= s.strength <= 1.0,
                weight=1.0,
                description="Signal strength must be in valid range"
            ),
            SignalValidationRule(
                name="recent_timestamp",
                predicate=lambda s: (datetime.now() - s.timestamp).total_seconds() < 3600,
                weight=0.8,
                description="Signal should be recent (within 1 hour)"
            ),
            SignalValidationRule(
                name="source_exists",
                predicate=lambda s: len(s.source.strip()) > 0,
                weight=1.0,
                description="Signal must have valid source"
            )
        ]
        
        return fl(rules)
    
    @staticmethod
    def validate_signal(signal: Signal, 
                       rules: FunctionalList[SignalValidationRule]) -> Dict[str, Any]:
        """Validate signal against rules."""
        results = {}
        total_weight = 0
        passed_weight = 0
        
        for rule in rules:
            passed = rule.predicate(signal)
            results[rule.name] = {
                'passed': passed,
                'description': rule.description,
                'weight': rule.weight
            }
            
            total_weight += rule.weight
            if passed:
                passed_weight += rule.weight
        
        validation_score = passed_weight / total_weight if total_weight > 0 else 0
        
        return {
            'is_valid': validation_score >= 0.8,
            'validation_score': validation_score,
            'rule_results': results,
            'passed_rules': sum(1 for r in results.values() if r['passed']),
            'total_rules': len(rules)
        }
    
    # ============================================================================
    # Parallel Processing
    # ============================================================================
    
    @staticmethod
    def parallel_signal_validation(signals: FunctionalList[Signal],
                                 rules: FunctionalList[SignalValidationRule],
                                 max_workers: Optional[int] = None) -> Dict[str, Dict[str, Any]]:
        """Validate multiple signals in parallel."""
        if not FUNCTIONAL_AVAILABLE:
            # Sequential fallback
            return {
                signal.source: FunctionalSignalProcessor.validate_signal(signal, rules)
                for signal in signals
            }
        
        validation_func = partial(FunctionalSignalProcessor.validate_signal, rules=rules)
        
        try:
            results = ParallelOps.parallel_map(validation_func, signals, max_workers)
            return {
                signal.source: result
                for signal, result in zip(signals, results)
            }
        except Exception:
            # Fallback to sequential
            return {
                signal.source: FunctionalSignalProcessor.validate_signal(signal, rules)
                for signal in signals
            }
    
    # ============================================================================
    # Pipeline Operations
    # ============================================================================
    
    @staticmethod
    def create_signal_processing_pipeline(min_confidence: float = 0.3,
                                        min_sources: int = 3,
                                        outlier_method: str = 'zscore') -> Callable:
        """Create a comprehensive signal processing pipeline."""
        
        if not FUNCTIONAL_AVAILABLE:
            def pipeline(raw_signals: List[Dict[str, Any]]) -> Dict[str, Any]:
                # Simple processing without functional utilities
                try:
                    signals = []
                    for raw in raw_signals:
                        signal = FunctionalSignalProcessor.create_signal(
                            raw.get('source', ''),
                            raw.get('strength', 0),
                            raw.get('confidence', 0),
                            raw.get('timestamp'),
                            raw.get('metadata', {})
                        )
                        if hasattr(signal, 'is_some') and signal.is_some():
                            signals.append(signal.value)
                    
                    # Simple consensus
                    if len(signals) >= min_sources:
                        avg_strength = sum(s.strength * s.confidence for s in signals) / sum(s.confidence for s in signals)
                        return {'consensus_strength': avg_strength, 'valid': True}
                    else:
                        return {'error': 'Insufficient signals', 'valid': False}
                except Exception as e:
                    return {'error': str(e), 'valid': False}
            
            return pipeline
        
        def pipeline(raw_signals: List[Dict[str, Any]]) -> ConsensusResult:
            """Complete signal processing pipeline."""
            
            # Step 1: Create signals from raw data
            signal_maybes = fl([
                FunctionalSignalProcessor.create_signal(
                    raw.get('source', ''),
                    raw.get('strength', 0),
                    raw.get('confidence', 0),
                    raw.get('timestamp'),
                    raw.get('metadata', {})
                )
                for raw in raw_signals
            ])
            
            # Step 2: Extract valid signals
            valid_signals = fl([
                maybe.value for maybe in signal_maybes if maybe.is_some()
            ])
            
            # Step 3: Filter by confidence
            filtered_signals = FunctionalSignalProcessor.filter_signals_by_confidence(
                valid_signals, min_confidence
            )
            
            # Step 4: Build consensus
            consensus_result = FunctionalSignalProcessor.build_consensus(
                filtered_signals, min_sources, outlier_method
            )
            
            return consensus_result
        
        return pipeline
    
    @staticmethod
    def create_real_time_processor(window_size: int = 100) -> Callable:
        """Create real-time signal processor with sliding window."""
        
        signal_buffer = []
        
        def processor(new_signals: List[Dict[str, Any]]) -> ConsensusResult:
            nonlocal signal_buffer
            
            # Add new signals to buffer
            pipeline = FunctionalSignalProcessor.create_signal_processing_pipeline()
            new_consensus = pipeline(new_signals)
            
            # Maintain sliding window
            signal_buffer.extend(new_signals)
            if len(signal_buffer) > window_size:
                signal_buffer = signal_buffer[-window_size:]
            
            # Process entire window for consensus
            return pipeline(signal_buffer)
        
        return processor


# Convenience functions for common operations
def process_signals(raw_signals: List[Dict[str, Any]], 
                   min_confidence: float = 0.3,
                   min_sources: int = 3) -> ConsensusResult:
    """Process raw signals into consensus result."""
    pipeline = FunctionalSignalProcessor.create_signal_processing_pipeline(
        min_confidence, min_sources
    )
    return pipeline(raw_signals)


def validate_signal_quality(signals: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate signal quality using functional patterns."""
    if not FUNCTIONAL_AVAILABLE:
        return {'error': 'Functional utilities not available'}
    
    # Convert to Signal objects
    signal_objects = []
    for raw in signals:
        maybe_signal = FunctionalSignalProcessor.create_signal(
            raw.get('source', ''),
            raw.get('strength', 0),
            raw.get('confidence', 0),
            raw.get('timestamp'),
            raw.get('metadata', {})
        )
        if maybe_signal.is_some():
            signal_objects.append(maybe_signal.value)
    
    if not signal_objects:
        return {'error': 'No valid signals', 'valid_count': 0}
    
    # Validate all signals
    rules = FunctionalSignalProcessor.create_validation_rules()
    validation_results = FunctionalSignalProcessor.parallel_signal_validation(
        fl(signal_objects), rules
    )
    
    # Aggregate results
    valid_count = sum(1 for r in validation_results.values() if r['is_valid'])
    avg_score = sum(r['validation_score'] for r in validation_results.values()) / len(validation_results)
    
    return {
        'total_signals': len(signal_objects),
        'valid_signals': valid_count,
        'validation_rate': valid_count / len(signal_objects),
        'average_score': avg_score,
        'individual_results': validation_results
    }


# Export important classes and functions
__all__ = [
    'Signal', 'SignalType', 'ConfidenceLevel', 'ConsensusResult',
    'SignalValidationRule', 'FunctionalSignalProcessor',
    'process_signals', 'validate_signal_quality'
]