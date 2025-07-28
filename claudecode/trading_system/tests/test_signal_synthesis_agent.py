"""
Comprehensive tests for the Signal Synthesis Agent.
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from agents.synthesis.signal_synthesis_agent import SignalSynthesisAgent
from core.base.exceptions import DataError, ValidationError


@pytest.fixture
def synthesis_agent():
    """Create a SignalSynthesisAgent instance for testing."""
    return SignalSynthesisAgent()


@pytest.fixture
def sample_signals():
    """Create sample signals from different agents for testing."""
    timestamps = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    
    signals_data = {
        'technical_analysis': {
            'timestamps': timestamps,
            'signals': np.random.choice([-1, 0, 1], size=len(timestamps), p=[0.2, 0.6, 0.2]),
            'confidence': np.random.uniform(0.3, 0.9, len(timestamps)),
            'strength': np.random.uniform(0.1, 1.0, len(timestamps))
        },
        'momentum': {
            'timestamps': timestamps,
            'signals': np.random.choice([-1, 0, 1], size=len(timestamps), p=[0.15, 0.7, 0.15]),
            'confidence': np.random.uniform(0.4, 0.8, len(timestamps)),
            'strength': np.random.uniform(0.2, 0.9, len(timestamps))
        },
        'mean_reversion': {
            'timestamps': timestamps,
            'signals': np.random.choice([-1, 0, 1], size=len(timestamps), p=[0.25, 0.5, 0.25]),
            'confidence': np.random.uniform(0.2, 0.7, len(timestamps)),
            'strength': np.random.uniform(0.1, 0.8, len(timestamps))
        },
        'ml_ensemble': {
            'timestamps': timestamps,
            'signals': np.random.choice([-1, 0, 1], size=len(timestamps), p=[0.18, 0.64, 0.18]),
            'confidence': np.random.uniform(0.5, 0.95, len(timestamps)),
            'strength': np.random.uniform(0.3, 1.0, len(timestamps))
        },
        'risk_modeling': {
            'timestamps': timestamps,
            'risk_scores': np.random.uniform(0.1, 0.9, len(timestamps)),
            'volatility_forecast': np.random.uniform(0.05, 0.3, len(timestamps)),
            'var_estimates': np.random.uniform(0.01, 0.05, len(timestamps))
        }
    }
    
    return signals_data


@pytest.fixture
def sample_inputs():
    """Sample inputs for signal synthesis."""
    return {
        "signals_sources": ["technical_analysis", "momentum", "mean_reversion", "ml_ensemble"],
        "synthesis_method": "weighted_average",
        "confidence_threshold": 0.6,
        "correlation_adjustment": True,
        "regime_dependent": True,
        "risk_adjustment": True
    }


class TestSignalSynthesisAgent:
    """Test cases for SignalSynthesisAgent."""
    
    @pytest.mark.unit
    def test_agent_initialization(self, synthesis_agent):
        """Test agent initialization."""
        assert synthesis_agent.name == "SignalSynthesisAgent"
        assert hasattr(synthesis_agent, 'signal_aggregator')
        assert hasattr(synthesis_agent, 'weight_optimizer')
        assert hasattr(synthesis_agent, 'conflict_resolver')
    
    @pytest.mark.unit
    def test_signal_normalization(self, synthesis_agent, sample_signals):
        """Test signal normalization across different sources."""
        raw_signals = sample_signals['technical_analysis']['signals']
        
        normalized = synthesis_agent._normalize_signals(
            raw_signals,
            method='z_score',
            lookback=30
        )
        
        assert isinstance(normalized, np.ndarray)
        assert len(normalized) == len(raw_signals)
        assert abs(normalized[30:].mean()) < 0.1  # Should be approximately zero-mean
    
    @pytest.mark.unit
    def test_signal_weighting(self, synthesis_agent, sample_signals):
        """Test dynamic signal weighting based on performance."""
        # Create mock historical performance data
        performance_data = {
            'technical_analysis': {'sharpe': 0.8, 'hit_rate': 0.55, 'avg_return': 0.02},
            'momentum': {'sharpe': 1.2, 'hit_rate': 0.62, 'avg_return': 0.035},
            'mean_reversion': {'sharpe': 0.6, 'hit_rate': 0.48, 'avg_return': 0.015},
            'ml_ensemble': {'sharpe': 1.5, 'hit_rate': 0.68, 'avg_return': 0.045}
        }
        
        weights = synthesis_agent._calculate_dynamic_weights(
            performance_data,
            method='sharpe_weighted'
        )
        
        assert isinstance(weights, dict)
        assert abs(sum(weights.values()) - 1.0) < 0.01  # Should sum to 1
        assert all(w >= 0 for w in weights.values())  # All weights should be positive
        assert weights['ml_ensemble'] > weights['mean_reversion']  # Better performer should have higher weight
    
    @pytest.mark.unit
    def test_signal_correlation_analysis(self, synthesis_agent, sample_signals):
        """Test signal correlation analysis."""
        signal_matrix = pd.DataFrame({
            source: data['signals'] for source, data in sample_signals.items() 
            if 'signals' in data
        })
        
        correlations = synthesis_agent._analyze_signal_correlations(signal_matrix)
        
        assert isinstance(correlations, pd.DataFrame)
        assert correlations.shape[0] == correlations.shape[1]
        assert (correlations.diagonal() == 1.0).all()
        assert (correlations.values >= -1).all() and (correlations.values <= 1).all()
    
    @pytest.mark.unit
    def test_conflict_resolution(self, synthesis_agent):
        """Test signal conflict resolution."""
        conflicting_signals = {
            'momentum': {'signal': 1, 'confidence': 0.8, 'strength': 0.7},
            'mean_reversion': {'signal': -1, 'confidence': 0.7, 'strength': 0.6},
            'technical_analysis': {'signal': 1, 'confidence': 0.6, 'strength': 0.5}
        }
        
        resolved_signal = synthesis_agent._resolve_signal_conflicts(
            conflicting_signals,
            method='confidence_weighted'
        )
        
        assert isinstance(resolved_signal, dict)
        assert 'final_signal' in resolved_signal
        assert 'confidence' in resolved_signal
        assert 'consensus_strength' in resolved_signal
        assert -1 <= resolved_signal['final_signal'] <= 1
    
    @pytest.mark.unit
    def test_ensemble_voting(self, synthesis_agent, sample_signals):
        """Test ensemble voting mechanism."""
        signals_at_time = {
            'technical_analysis': {'signal': 1, 'confidence': 0.7},
            'momentum': {'signal': 1, 'confidence': 0.8},
            'mean_reversion': {'signal': -1, 'confidence': 0.6},
            'ml_ensemble': {'signal': 1, 'confidence': 0.9}
        }
        
        ensemble_result = synthesis_agent._ensemble_vote(
            signals_at_time,
            voting_method='weighted'
        )
        
        assert isinstance(ensemble_result, dict)
        assert 'consensus_signal' in ensemble_result
        assert 'agreement_score' in ensemble_result
        assert 'minority_dissent' in ensemble_result
        assert 0 <= ensemble_result['agreement_score'] <= 1
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_process_with_mock_data(self, synthesis_agent, sample_inputs, sample_signals):
        """Test processing with mocked data."""
        with patch.object(synthesis_agent, '_fetch_agent_signals', return_value=sample_signals):
            result = await synthesis_agent.process(sample_inputs)
            
            assert isinstance(result, dict)
            assert 'synthesized_signals' in result
            assert 'signal_weights' in result
            assert 'consensus_strength' in result
            assert 'signal_quality' in result
            assert 'confidence' in result
    
    @pytest.mark.unit
    def test_regime_dependent_synthesis(self, synthesis_agent, sample_signals):
        """Test regime-dependent signal synthesis."""
        market_regime = 'trending'  # vs 'mean_reverting', 'volatile', 'crisis'
        
        regime_weights = synthesis_agent._calculate_regime_weights(
            market_regime,
            sample_signals
        )
        
        assert isinstance(regime_weights, dict)
        assert sum(regime_weights.values()) <= 1.1  # Allow for rounding
        
        # In trending regime, momentum should get higher weight
        if 'momentum' in regime_weights and 'mean_reversion' in regime_weights:
            assert regime_weights['momentum'] > regime_weights['mean_reversion']
    
    @pytest.mark.unit
    def test_signal_decay_adjustment(self, synthesis_agent):
        """Test signal decay over time."""
        signal_history = pd.Series([1, 1, 0, -1, -1], 
                                  index=pd.date_range('2023-01-01', periods=5, freq='D'))
        
        decayed_signals = synthesis_agent._apply_signal_decay(
            signal_history,
            decay_rate=0.1  # 10% decay per day
        )
        
        assert isinstance(decayed_signals, pd.Series)
        assert len(decayed_signals) == len(signal_history)
        # Recent signals should have higher weight
        assert abs(decayed_signals.iloc[-1]) > abs(decayed_signals.iloc[0])
    
    @pytest.mark.unit
    def test_confidence_calibration(self, synthesis_agent, sample_signals):
        """Test confidence score calibration."""
        raw_confidence_scores = np.array([0.3, 0.5, 0.7, 0.9, 0.95])
        historical_outcomes = np.array([0, 1, 1, 1, 0])  # Binary outcomes
        
        calibrated_confidence = synthesis_agent._calibrate_confidence(
            raw_confidence_scores,
            historical_outcomes
        )
        
        assert isinstance(calibrated_confidence, np.ndarray)
        assert len(calibrated_confidence) == len(raw_confidence_scores)
        assert (calibrated_confidence >= 0).all() and (calibrated_confidence <= 1).all()
    
    @pytest.mark.unit
    def test_signal_persistence_analysis(self, synthesis_agent):
        """Test signal persistence and momentum."""
        signal_series = pd.Series([1, 1, 1, 0, -1, -1, 1, 1])
        
        persistence_score = synthesis_agent._calculate_signal_persistence(
            signal_series,
            lookback=3
        )
        
        assert isinstance(persistence_score, float)
        assert 0 <= persistence_score <= 1
    
    @pytest.mark.unit
    def test_outlier_detection_and_filtering(self, synthesis_agent, sample_signals):
        """Test outlier signal detection and filtering."""
        # Add some outlier signals
        signals_with_outliers = sample_signals['technical_analysis']['signals'].copy()
        signals_with_outliers[10:15] = 10  # Extreme outliers
        
        filtered_signals = synthesis_agent._filter_outlier_signals(
            signals_with_outliers,
            method='iqr',
            threshold=3.0
        )
        
        assert isinstance(filtered_signals, np.ndarray)
        assert len(filtered_signals) == len(signals_with_outliers)
        assert abs(filtered_signals[10:15]).max() < 10  # Outliers should be capped
    
    @pytest.mark.unit
    def test_adaptive_learning_weights(self, synthesis_agent):
        """Test adaptive learning of signal weights."""
        # Historical performance data over time
        historical_performance = pd.DataFrame({
            'technical_analysis': [0.02, 0.01, -0.01, 0.03, 0.02],
            'momentum': [0.03, 0.02, 0.01, 0.04, 0.03],
            'mean_reversion': [-0.01, 0.02, 0.03, -0.02, 0.01]
        })
        
        adaptive_weights = synthesis_agent._learn_adaptive_weights(
            historical_performance,
            learning_rate=0.1
        )
        
        assert isinstance(adaptive_weights, pd.Series)
        assert abs(adaptive_weights.sum() - 1.0) < 0.01
        assert (adaptive_weights >= 0).all()
    
    @pytest.mark.unit
    def test_signal_timing_adjustment(self, synthesis_agent):
        """Test signal timing and lag adjustment."""
        signals = pd.Series([0, 1, 1, 0, -1, -1, 0])
        market_returns = pd.Series([0.01, 0.02, 0.01, -0.01, -0.02, -0.01, 0.01])
        
        optimal_lag = synthesis_agent._find_optimal_signal_timing(
            signals,
            market_returns,
            max_lag=3
        )
        
        assert isinstance(optimal_lag, int)
        assert 0 <= optimal_lag <= 3
    
    @pytest.mark.unit
    def test_meta_signal_generation(self, synthesis_agent, sample_signals):
        """Test meta-signal generation from signal agreement patterns."""
        signal_agreement = pd.DataFrame({
            'high_agreement': [0.9, 0.8, 0.95, 0.7],
            'medium_agreement': [0.6, 0.7, 0.65, 0.5],
            'low_agreement': [0.3, 0.4, 0.2, 0.45]
        })
        
        meta_signal = synthesis_agent._generate_meta_signal(
            signal_agreement,
            agreement_threshold=0.75
        )
        
        assert isinstance(meta_signal, pd.Series)
        assert (meta_signal.isin([-1, 0, 1])).all()
    
    @pytest.mark.unit
    def test_signal_quality_metrics(self, synthesis_agent):
        """Test signal quality assessment metrics."""
        signals = pd.Series([1, 1, -1, 0, 1, -1, 1])
        returns = pd.Series([0.02, 0.01, -0.015, 0.005, 0.02, -0.01, 0.015])
        
        quality_metrics = synthesis_agent._assess_signal_quality(
            signals,
            returns
        )
        
        assert isinstance(quality_metrics, dict)
        assert 'hit_rate' in quality_metrics
        assert 'information_ratio' in quality_metrics
        assert 'signal_decay' in quality_metrics
        assert 0 <= quality_metrics['hit_rate'] <= 1
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_backtesting_synthesis_strategies(self, synthesis_agent):
        """Test backtesting of different synthesis strategies."""
        backtest_inputs = {
            "strategies": ["equal_weight", "performance_weight", "confidence_weight"],
            "start_date": datetime(2022, 1, 1),
            "end_date": datetime(2023, 1, 1),
            "rebalance_frequency": "weekly"
        }
        
        # Mock historical signal data
        mock_signals = self._generate_mock_historical_signals()
        
        with patch.object(synthesis_agent, '_fetch_historical_signals', return_value=mock_signals):
            backtest_results = await synthesis_agent.backtest(backtest_inputs)
            
            assert 'strategy_performance' in backtest_results
            assert 'signal_contributions' in backtest_results
            assert 'synthesis_stability' in backtest_results
            assert 'optimal_weights' in backtest_results
    
    @pytest.mark.unit
    def test_real_time_signal_processing(self, synthesis_agent):
        """Test real-time signal processing and updates."""
        # Simulate streaming signals
        new_signals = {
            'technical_analysis': {'signal': 1, 'confidence': 0.8, 'timestamp': datetime.now()},
            'momentum': {'signal': 1, 'confidence': 0.7, 'timestamp': datetime.now()},
            'mean_reversion': {'signal': -1, 'confidence': 0.6, 'timestamp': datetime.now()}
        }
        
        processed_signal = synthesis_agent._process_real_time_signals(
            new_signals,
            update_weights=True
        )
        
        assert isinstance(processed_signal, dict)
        assert 'synthesized_signal' in processed_signal
        assert 'processing_latency' in processed_signal
        assert 'signal_freshness' in processed_signal
    
    def _generate_mock_historical_signals(self):
        """Helper to generate mock historical signals."""
        dates = pd.date_range(start='2022-01-01', end='2023-01-01', freq='D')
        
        mock_data = {}
        for source in ['technical_analysis', 'momentum', 'mean_reversion']:
            mock_data[source] = pd.DataFrame({
                'signal': np.random.choice([-1, 0, 1], size=len(dates)),
                'confidence': np.random.uniform(0.3, 0.9, len(dates)),
                'returns': np.random.normal(0, 0.01, len(dates))
            }, index=dates)
        
        return mock_data