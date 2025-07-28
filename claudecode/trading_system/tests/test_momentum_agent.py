"""
Comprehensive tests for the Momentum Trading Agent.
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

from agents.strategies.momentum.momentum_agent import MomentumAgent
from core.base.exceptions import DataError, ValidationError


@pytest.fixture
def momentum_agent():
    """Create a MomentumAgent instance for testing."""
    return MomentumAgent()


@pytest.fixture
def sample_market_data():
    """Create sample market data for testing."""
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)
    
    # Generate trending price data with momentum
    trend = np.linspace(100, 150, len(dates))
    noise = np.random.normal(0, 2, len(dates))
    prices = trend + noise
    
    data = pd.DataFrame({
        'timestamp': dates,
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, len(dates)),
        'returns': pd.Series(prices).pct_change()
    })
    
    return data


@pytest.fixture
def sample_inputs():
    """Sample inputs for momentum strategy."""
    return {
        "symbols": ["AAPL", "GOOGL", "MSFT"],
        "lookback_period": 20,
        "holding_period": 5,
        "rebalance_frequency": "weekly",
        "momentum_type": "relative",
        "risk_limit": 0.02
    }


class TestMomentumAgent:
    """Test cases for MomentumAgent."""
    
    @pytest.mark.unit
    def test_agent_initialization(self, momentum_agent):
        """Test agent initialization."""
        assert momentum_agent.name == "MomentumAgent"
        assert hasattr(momentum_agent, 'momentum_calculator')
        assert hasattr(momentum_agent, 'position_sizer')
    
    @pytest.mark.unit
    def test_calculate_momentum_score(self, momentum_agent, sample_market_data):
        """Test momentum score calculation."""
        score = momentum_agent._calculate_momentum_score(
            sample_market_data, 
            lookback_period=20
        )
        
        assert isinstance(score, float)
        assert -1 <= score <= 1  # Normalized score
    
    @pytest.mark.unit
    def test_relative_momentum_ranking(self, momentum_agent):
        """Test relative momentum ranking across assets."""
        # Create mock data for multiple assets
        assets_data = {
            'AAPL': pd.Series([100, 105, 110, 115, 120]),  # Strong momentum
            'GOOGL': pd.Series([100, 102, 104, 106, 108]),  # Moderate momentum
            'MSFT': pd.Series([100, 99, 98, 97, 96])  # Negative momentum
        }
        
        rankings = momentum_agent._rank_by_momentum(assets_data)
        
        assert isinstance(rankings, pd.Series)
        assert rankings['AAPL'] > rankings['GOOGL']
        assert rankings['GOOGL'] > rankings['MSFT']
    
    @pytest.mark.unit
    def test_position_sizing(self, momentum_agent):
        """Test position sizing based on momentum strength."""
        momentum_scores = pd.Series({
            'AAPL': 0.8,
            'GOOGL': 0.5,
            'MSFT': -0.3
        })
        
        positions = momentum_agent._calculate_position_sizes(
            momentum_scores,
            risk_limit=0.02,
            max_position=0.3
        )
        
        assert isinstance(positions, pd.Series)
        assert positions.sum() <= 1.0  # Total allocation <= 100%
        assert all(positions >= 0)  # No short positions in basic momentum
        assert positions['AAPL'] > positions['GOOGL']  # Higher momentum = larger position
    
    @pytest.mark.unit
    def test_entry_signal_generation(self, momentum_agent, sample_market_data):
        """Test entry signal generation."""
        signals = momentum_agent._generate_entry_signals(
            sample_market_data,
            momentum_threshold=0.5
        )
        
        assert isinstance(signals, pd.Series)
        assert signals.dtype == bool
        assert signals.any()  # Should have at least some signals
    
    @pytest.mark.unit
    def test_exit_signal_generation(self, momentum_agent, sample_market_data):
        """Test exit signal generation."""
        entry_date = datetime(2023, 6, 1)
        holding_period = 20
        
        exit_signal = momentum_agent._calculate_exit_date(
            entry_date,
            holding_period,
            sample_market_data
        )
        
        assert isinstance(exit_signal, datetime)
        assert exit_signal > entry_date
    
    @pytest.mark.unit
    def test_risk_management(self, momentum_agent):
        """Test risk management constraints."""
        positions = pd.Series({
            'AAPL': 0.4,
            'GOOGL': 0.5,
            'MSFT': 0.3
        })
        
        adjusted_positions = momentum_agent._apply_risk_constraints(
            positions,
            max_position=0.3,
            max_leverage=1.0
        )
        
        assert adjusted_positions.max() <= 0.3
        assert adjusted_positions.sum() <= 1.0
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_process_with_mock_data(self, momentum_agent, sample_inputs):
        """Test processing with mocked data."""
        mock_data = {
            'AAPL': pd.DataFrame({'close': np.random.uniform(100, 200, 100)}),
            'GOOGL': pd.DataFrame({'close': np.random.uniform(1000, 2000, 100)}),
            'MSFT': pd.DataFrame({'close': np.random.uniform(200, 300, 100)})
        }
        
        with patch.object(momentum_agent, '_fetch_market_data', side_effect=lambda s: mock_data[s]):
            result = await momentum_agent.process(sample_inputs)
            
            assert isinstance(result, dict)
            assert 'signals' in result
            assert 'positions' in result
            assert 'momentum_scores' in result
            assert 'confidence' in result
    
    @pytest.mark.unit
    def test_time_series_momentum(self, momentum_agent, sample_market_data):
        """Test time series momentum calculation."""
        ts_momentum = momentum_agent._calculate_time_series_momentum(
            sample_market_data,
            lookback_periods=[1, 3, 6, 12]
        )
        
        assert isinstance(ts_momentum, pd.DataFrame)
        assert all(col in ts_momentum.columns for col in ['1M', '3M', '6M', '12M'])
    
    @pytest.mark.unit
    def test_cross_sectional_momentum(self, momentum_agent):
        """Test cross-sectional momentum calculation."""
        returns_data = pd.DataFrame({
            'AAPL': [0.01, 0.02, 0.03, 0.02, 0.01],
            'GOOGL': [0.02, 0.01, 0.02, 0.01, 0.02],
            'MSFT': [-0.01, -0.02, 0.01, 0.02, 0.03]
        })
        
        cs_momentum = momentum_agent._calculate_cross_sectional_momentum(returns_data)
        
        assert isinstance(cs_momentum, pd.Series)
        assert len(cs_momentum) == 3
    
    @pytest.mark.unit
    def test_momentum_crash_detection(self, momentum_agent, sample_market_data):
        """Test momentum crash risk detection."""
        # Create data with momentum crash pattern
        crash_data = sample_market_data.copy()
        crash_data.loc[50:60, 'returns'] = -0.05  # Sudden reversal
        
        crash_risk = momentum_agent._detect_momentum_crash_risk(crash_data)
        
        assert isinstance(crash_risk, float)
        assert 0 <= crash_risk <= 1
    
    @pytest.mark.unit
    def test_dynamic_lookback_adjustment(self, momentum_agent, sample_market_data):
        """Test dynamic lookback period adjustment based on market conditions."""
        volatility = sample_market_data['returns'].std()
        
        adjusted_lookback = momentum_agent._adjust_lookback_period(
            base_lookback=20,
            volatility=volatility,
            market_regime='trending'
        )
        
        assert isinstance(adjusted_lookback, int)
        assert 5 <= adjusted_lookback <= 60  # Reasonable bounds
    
    @pytest.mark.unit
    def test_sector_rotation_momentum(self, momentum_agent):
        """Test sector rotation momentum strategy."""
        sector_data = {
            'Technology': pd.Series([100, 110, 120, 130, 140]),
            'Healthcare': pd.Series([100, 105, 110, 115, 120]),
            'Finance': pd.Series([100, 98, 96, 94, 92]),
            'Energy': pd.Series([100, 102, 104, 103, 101])
        }
        
        rotation_signals = momentum_agent._generate_sector_rotation_signals(
            sector_data,
            top_n=2
        )
        
        assert isinstance(rotation_signals, dict)
        assert len(rotation_signals) == 2  # Only top 2 sectors
        assert 'Technology' in rotation_signals
        assert 'Healthcare' in rotation_signals
    
    @pytest.mark.unit
    def test_momentum_factor_exposure(self, momentum_agent, sample_market_data):
        """Test momentum factor exposure calculation."""
        factor_exposure = momentum_agent._calculate_factor_exposure(
            sample_market_data,
            factor='momentum'
        )
        
        assert isinstance(factor_exposure, float)
        assert -1 <= factor_exposure <= 1
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_backtesting_momentum_strategy(self, momentum_agent):
        """Test backtesting of momentum strategy."""
        backtest_inputs = {
            "symbols": ["AAPL", "GOOGL"],
            "start_date": datetime(2022, 1, 1),
            "end_date": datetime(2023, 1, 1),
            "lookback_period": 20,
            "rebalance_frequency": "monthly"
        }
        
        # Mock historical data
        mock_historical = pd.DataFrame({
            'timestamp': pd.date_range(start='2022-01-01', end='2023-01-01', freq='D'),
            'AAPL': np.random.uniform(150, 180, 366),
            'GOOGL': np.random.uniform(2000, 3000, 366)
        })
        
        with patch.object(momentum_agent, '_fetch_historical_data', return_value=mock_historical):
            backtest_results = await momentum_agent.backtest(backtest_inputs)
            
            assert 'returns' in backtest_results
            assert 'sharpe_ratio' in backtest_results
            assert 'max_drawdown' in backtest_results
            assert 'win_rate' in backtest_results
    
    @pytest.mark.unit
    def test_input_validation(self, momentum_agent):
        """Test input validation."""
        # Test missing required fields
        invalid_inputs = {"symbols": ["AAPL"]}  # Missing lookback_period
        
        with pytest.raises(ValidationError):
            momentum_agent._validate_inputs(invalid_inputs)
        
        # Test invalid lookback period
        invalid_lookback = {
            "symbols": ["AAPL"],
            "lookback_period": -10  # Negative lookback
        }
        
        with pytest.raises(ValidationError):
            momentum_agent._validate_inputs(invalid_lookback)
    
    @pytest.mark.unit
    def test_performance_attribution(self, momentum_agent):
        """Test performance attribution for momentum strategy."""
        returns = pd.Series([0.01, 0.02, -0.01, 0.03, 0.01])
        positions = pd.Series([1, 1, 0, 1, 1])
        
        attribution = momentum_agent._calculate_performance_attribution(
            returns,
            positions
        )
        
        assert 'total_return' in attribution
        assert 'momentum_contribution' in attribution
        assert 'timing_contribution' in attribution