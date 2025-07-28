"""
Comprehensive tests for the Statistical Arbitrage Agent.
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

from agents.strategies.stat_arb.stat_arb_agent import StatArbAgent
from core.base.exceptions import DataError, ValidationError


@pytest.fixture
def stat_arb_agent():
    """Create a StatArbAgent instance for testing."""
    return StatArbAgent()


@pytest.fixture
def cointegrated_pair_data():
    """Create sample cointegrated pair data for testing."""
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)
    
    # Generate cointegrated series
    # X = random walk
    x = 100 + np.cumsum(np.random.normal(0, 1, len(dates)))
    
    # Y = beta * X + stationary spread + noise
    beta = 1.2
    spread = 5 * np.sin(np.arange(len(dates)) * 2 * np.pi / 50)  # Mean-reverting spread
    noise = np.random.normal(0, 0.5, len(dates))
    y = beta * x + spread + noise
    
    data = pd.DataFrame({
        'timestamp': dates,
        'asset1': x,
        'asset2': y
    })
    
    return data


@pytest.fixture
def sample_inputs():
    """Sample inputs for statistical arbitrage."""
    return {
        "pairs": [("AAPL", "MSFT"), ("XOM", "CVX")],
        "lookback_period": 60,
        "z_score_entry": 2.0,
        "z_score_exit": 0.5,
        "max_holding_period": 20,
        "cointegration_pvalue": 0.05
    }


class TestStatArbAgent:
    """Test cases for StatArbAgent."""
    
    @pytest.mark.unit
    def test_agent_initialization(self, stat_arb_agent):
        """Test agent initialization."""
        assert stat_arb_agent.name == "StatArbAgent"
        assert hasattr(stat_arb_agent, 'pair_selector')
        assert hasattr(stat_arb_agent, 'spread_calculator')
    
    @pytest.mark.unit
    def test_cointegration_test(self, stat_arb_agent, cointegrated_pair_data):
        """Test cointegration testing between pairs."""
        asset1 = cointegrated_pair_data['asset1']
        asset2 = cointegrated_pair_data['asset2']
        
        is_cointegrated, pvalue, hedge_ratio = stat_arb_agent._test_cointegration(
            asset1, asset2
        )
        
        assert isinstance(is_cointegrated, bool)
        assert isinstance(pvalue, float)
        assert isinstance(hedge_ratio, float)
        assert 0 <= pvalue <= 1
    
    @pytest.mark.unit
    def test_spread_calculation(self, stat_arb_agent, cointegrated_pair_data):
        """Test spread calculation between pairs."""
        asset1 = cointegrated_pair_data['asset1']
        asset2 = cointegrated_pair_data['asset2']
        hedge_ratio = 1.2
        
        spread = stat_arb_agent._calculate_spread(asset1, asset2, hedge_ratio)
        
        assert isinstance(spread, pd.Series)
        assert len(spread) == len(asset1)
        assert spread.std() < asset1.std()  # Spread should be less volatile
    
    @pytest.mark.unit
    def test_z_score_calculation(self, stat_arb_agent):
        """Test z-score calculation for spread."""
        spread = pd.Series(np.random.normal(0, 1, 100))
        lookback = 20
        
        z_scores = stat_arb_agent._calculate_z_score(spread, lookback)
        
        assert isinstance(z_scores, pd.Series)
        assert len(z_scores) == len(spread)
        assert not z_scores[:lookback].notna().any()  # First values should be NaN
    
    @pytest.mark.unit
    def test_entry_signal_generation(self, stat_arb_agent):
        """Test entry signal generation based on z-scores."""
        z_scores = pd.Series([-2.5, -1.5, 0, 1.5, 2.5, 3.0, 1.0, 0])
        z_entry_threshold = 2.0
        
        long_signals, short_signals = stat_arb_agent._generate_entry_signals(
            z_scores, z_entry_threshold
        )
        
        assert isinstance(long_signals, pd.Series)
        assert isinstance(short_signals, pd.Series)
        assert long_signals[0] == True  # z-score = -2.5, should go long
        assert short_signals[4] == True  # z-score = 2.5, should go short
    
    @pytest.mark.unit
    def test_exit_signal_generation(self, stat_arb_agent):
        """Test exit signal generation."""
        z_scores = pd.Series([2.5, 2.0, 1.5, 1.0, 0.5, 0.3, 0.1])
        z_exit_threshold = 0.5
        positions = pd.Series([1, 1, 1, 1, 1, 1, 1])  # Long position
        
        exit_signals = stat_arb_agent._generate_exit_signals(
            z_scores, z_exit_threshold, positions
        )
        
        assert isinstance(exit_signals, pd.Series)
        assert exit_signals[4] == True  # z-score crosses exit threshold
    
    @pytest.mark.unit
    def test_pair_selection(self, stat_arb_agent):
        """Test pair selection based on cointegration."""
        # Create mock price data for multiple assets
        assets = ['A', 'B', 'C', 'D']
        price_data = pd.DataFrame({
            asset: np.random.uniform(90, 110, 100)
            for asset in assets
        })
        
        selected_pairs = stat_arb_agent._select_cointegrated_pairs(
            price_data,
            min_correlation=0.7,
            max_pvalue=0.05
        )
        
        assert isinstance(selected_pairs, list)
        for pair in selected_pairs:
            assert len(pair) == 2
            assert pair[0] in assets
            assert pair[1] in assets
    
    @pytest.mark.unit
    def test_position_sizing(self, stat_arb_agent):
        """Test position sizing for stat arb trades."""
        z_scores = pd.Series({
            'pair1': 2.5,
            'pair2': -2.0,
            'pair3': 0.5
        })
        
        positions = stat_arb_agent._calculate_position_sizes(
            z_scores,
            max_position_per_pair=0.2,
            z_score_threshold=2.0
        )
        
        assert isinstance(positions, pd.Series)
        assert abs(positions['pair1']) > 0  # Should have position
        assert abs(positions['pair2']) > 0  # Should have position
        assert positions['pair3'] == 0  # Below threshold
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_process_with_mock_data(self, stat_arb_agent, sample_inputs, cointegrated_pair_data):
        """Test processing with mocked data."""
        with patch.object(stat_arb_agent, '_fetch_pair_data', return_value=cointegrated_pair_data):
            result = await stat_arb_agent.process(sample_inputs)
            
            assert isinstance(result, dict)
            assert 'signals' in result
            assert 'positions' in result
            assert 'spreads' in result
            assert 'z_scores' in result
            assert 'confidence' in result
    
    @pytest.mark.unit
    def test_kalman_filter_hedge_ratio(self, stat_arb_agent, cointegrated_pair_data):
        """Test dynamic hedge ratio estimation using Kalman filter."""
        asset1 = cointegrated_pair_data['asset1']
        asset2 = cointegrated_pair_data['asset2']
        
        dynamic_hedge_ratios = stat_arb_agent._estimate_dynamic_hedge_ratio(
            asset1, asset2
        )
        
        assert isinstance(dynamic_hedge_ratios, pd.Series)
        assert len(dynamic_hedge_ratios) == len(asset1)
        assert dynamic_hedge_ratios.std() > 0  # Should vary over time
    
    @pytest.mark.unit
    def test_ornstein_uhlenbeck_parameters(self, stat_arb_agent):
        """Test Ornstein-Uhlenbeck process parameter estimation."""
        # Generate OU process
        n = 1000
        theta = 0.5  # Mean reversion speed
        mu = 0  # Long-term mean
        sigma = 0.1  # Volatility
        
        dt = 1/252
        ou_process = [0]
        for _ in range(n-1):
            dW = np.random.normal(0, np.sqrt(dt))
            dx = theta * (mu - ou_process[-1]) * dt + sigma * dW
            ou_process.append(ou_process[-1] + dx)
        
        spread = pd.Series(ou_process)
        
        estimated_params = stat_arb_agent._estimate_ou_parameters(spread)
        
        assert 'theta' in estimated_params
        assert 'mu' in estimated_params
        assert 'sigma' in estimated_params
        assert estimated_params['theta'] > 0  # Mean reversion speed should be positive
    
    @pytest.mark.unit
    def test_half_life_calculation(self, stat_arb_agent):
        """Test half-life calculation for mean reversion."""
        spread = pd.Series(np.random.normal(0, 1, 100))
        
        half_life = stat_arb_agent._calculate_half_life(spread)
        
        assert isinstance(half_life, float)
        assert half_life > 0
    
    @pytest.mark.unit
    def test_risk_management_stop_loss(self, stat_arb_agent):
        """Test stop-loss implementation for stat arb."""
        positions = pd.Series([1, 1, 1, 1, 1])
        spreads = pd.Series([0, -0.5, -1.0, -2.0, -3.0])
        stop_loss_z_score = 2.5
        
        stop_signals = stat_arb_agent._check_stop_loss(
            positions, spreads, stop_loss_z_score
        )
        
        assert isinstance(stop_signals, pd.Series)
        assert stop_signals[4] == True  # Should trigger stop loss
    
    @pytest.mark.unit
    def test_correlation_breakdown_detection(self, stat_arb_agent, cointegrated_pair_data):
        """Test detection of correlation breakdown."""
        # Create data with correlation breakdown
        data = cointegrated_pair_data.copy()
        # Break correlation in second half
        data.loc[len(data)//2:, 'asset2'] = np.random.uniform(50, 150, len(data)//2)
        
        breakdown_detected = stat_arb_agent._detect_correlation_breakdown(
            data['asset1'], data['asset2'],
            window=30,
            threshold=0.5
        )
        
        assert isinstance(breakdown_detected, pd.Series)
        assert breakdown_detected.dtype == bool
        assert breakdown_detected.iloc[-1] == True  # Should detect breakdown
    
    @pytest.mark.unit
    def test_multi_asset_cointegration(self, stat_arb_agent):
        """Test cointegration testing for multiple assets."""
        # Create cointegrated basket
        n_assets = 4
        n_periods = 100
        
        # Generate cointegrated series
        common_factor = np.cumsum(np.random.normal(0, 1, n_periods))
        assets_data = {}
        
        for i in range(n_assets):
            loading = np.random.uniform(0.8, 1.2)
            noise = np.random.normal(0, 0.5, n_periods)
            assets_data[f'asset_{i}'] = loading * common_factor + noise
        
        basket_weights = stat_arb_agent._find_cointegrating_portfolio(
            pd.DataFrame(assets_data)
        )
        
        assert isinstance(basket_weights, pd.Series)
        assert len(basket_weights) == n_assets
        assert abs(basket_weights.sum()) < 0.1  # Weights should sum close to 0
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_backtesting_stat_arb(self, stat_arb_agent):
        """Test backtesting of statistical arbitrage strategy."""
        backtest_inputs = {
            "pairs": [("GLD", "SLV"), ("XOM", "CVX")],
            "start_date": datetime(2022, 1, 1),
            "end_date": datetime(2023, 1, 1),
            "z_score_entry": 2.0,
            "z_score_exit": 0.5
        }
        
        # Mock historical data with cointegrated pairs
        mock_data = self._generate_mock_cointegrated_data()
        
        with patch.object(stat_arb_agent, '_fetch_historical_data', return_value=mock_data):
            backtest_results = await stat_arb_agent.backtest(backtest_inputs)
            
            assert 'returns' in backtest_results
            assert 'sharpe_ratio' in backtest_results
            assert 'max_drawdown' in backtest_results
            assert 'number_of_trades' in backtest_results
            assert 'win_rate' in backtest_results
    
    @pytest.mark.unit
    def test_transaction_cost_modeling(self, stat_arb_agent):
        """Test transaction cost impact on stat arb profitability."""
        spreads = pd.Series([0, 2.5, 2.0, 0.5, 0, -0.5, -2.0, -2.5, 0])
        positions = pd.Series([0, 1, 1, 1, 0, 0, -1, -1, 0])
        transaction_cost = 0.001  # 10 bps
        
        net_pnl = stat_arb_agent._calculate_net_pnl(
            spreads, positions, transaction_cost
        )
        
        assert isinstance(net_pnl, pd.Series)
        assert net_pnl.sum() < spreads.diff().multiply(positions).sum()  # Should be less due to costs
    
    def _generate_mock_cointegrated_data(self):
        """Helper to generate mock cointegrated data."""
        dates = pd.date_range(start='2022-01-01', end='2023-01-01', freq='D')
        
        # Generate cointegrated pairs
        gld = 100 + np.cumsum(np.random.normal(0, 1, len(dates)))
        slv = 0.5 * gld + 5 * np.sin(np.arange(len(dates)) * 2 * np.pi / 30) + np.random.normal(0, 0.5, len(dates))
        
        xom = 80 + np.cumsum(np.random.normal(0, 0.8, len(dates)))
        cvx = 0.9 * xom + 3 * np.sin(np.arange(len(dates)) * 2 * np.pi / 45) + np.random.normal(0, 0.4, len(dates))
        
        return pd.DataFrame({
            'GLD': gld,
            'SLV': slv,
            'XOM': xom,
            'CVX': cvx
        }, index=dates)