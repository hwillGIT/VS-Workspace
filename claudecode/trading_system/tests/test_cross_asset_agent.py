"""
Comprehensive tests for the Cross-Asset Trading Agent.
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

from agents.strategies.cross_asset.cross_asset_agent import CrossAssetAgent
from core.base.exceptions import DataError, ValidationError


@pytest.fixture
def cross_asset_agent():
    """Create a CrossAssetAgent instance for testing."""
    return CrossAssetAgent()


@pytest.fixture
def multi_asset_data():
    """Create sample multi-asset data for testing."""
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)
    
    # Generate correlated asset returns
    n_periods = len(dates)
    
    # Common factor
    market_factor = np.random.normal(0, 0.01, n_periods)
    
    # Individual asset returns with different exposures to market factor
    asset_data = {}
    
    # Equities - high correlation with market
    asset_data['SPY'] = market_factor * 1.2 + np.random.normal(0, 0.008, n_periods)
    asset_data['QQQ'] = market_factor * 1.4 + np.random.normal(0, 0.012, n_periods)
    
    # Bonds - negative correlation with equities
    asset_data['TLT'] = -market_factor * 0.5 + np.random.normal(0, 0.005, n_periods)
    asset_data['HYG'] = market_factor * 0.3 + np.random.normal(0, 0.004, n_periods)
    
    # Commodities - low correlation
    asset_data['GLD'] = market_factor * 0.1 + np.random.normal(0, 0.01, n_periods)
    asset_data['USO'] = market_factor * 0.8 + np.random.normal(0, 0.015, n_periods)
    
    # Currencies - mixed correlations
    asset_data['UUP'] = -market_factor * 0.3 + np.random.normal(0, 0.006, n_periods)
    asset_data['EWJ'] = market_factor * 0.7 + np.random.normal(0, 0.009, n_periods)
    
    # Convert to prices
    for asset in asset_data:
        prices = [100]
        for ret in asset_data[asset]:
            prices.append(prices[-1] * (1 + ret))
        asset_data[asset] = prices[1:]
    
    df = pd.DataFrame(asset_data, index=dates)
    return df


@pytest.fixture
def sample_inputs():
    """Sample inputs for cross-asset strategy."""
    return {
        "asset_classes": ["equity", "fixed_income", "commodity", "currency"],
        "assets": {
            "equity": ["SPY", "QQQ"],
            "fixed_income": ["TLT", "HYG"],
            "commodity": ["GLD", "USO"],
            "currency": ["UUP", "EWJ"]
        },
        "rebalance_frequency": "monthly",
        "risk_budget": 0.1,
        "correlation_lookback": 60,
        "min_correlation": 0.3
    }


class TestCrossAssetAgent:
    """Test cases for CrossAssetAgent."""
    
    @pytest.mark.unit
    def test_agent_initialization(self, cross_asset_agent):
        """Test agent initialization."""
        assert cross_asset_agent.name == "CrossAssetAgent"
        assert hasattr(cross_asset_agent, 'correlation_analyzer')
        assert hasattr(cross_asset_agent, 'regime_detector')
        assert hasattr(cross_asset_agent, 'risk_parity_calculator')
    
    @pytest.mark.unit
    def test_correlation_matrix_calculation(self, cross_asset_agent, multi_asset_data):
        """Test correlation matrix calculation."""
        returns = multi_asset_data.pct_change().dropna()
        
        corr_matrix = cross_asset_agent._calculate_correlation_matrix(
            returns,
            method='pearson',
            window=60
        )
        
        assert isinstance(corr_matrix, pd.DataFrame)
        assert corr_matrix.shape[0] == corr_matrix.shape[1]
        assert (corr_matrix.diagonal() == 1.0).all()
        assert (corr_matrix.values >= -1).all() and (corr_matrix.values <= 1).all()
    
    @pytest.mark.unit
    def test_regime_detection(self, cross_asset_agent, multi_asset_data):
        """Test market regime detection."""
        returns = multi_asset_data.pct_change().dropna()
        
        regime = cross_asset_agent._detect_market_regime(
            returns,
            lookback_window=30
        )
        
        assert isinstance(regime, dict)
        assert 'regime_type' in regime
        assert 'confidence' in regime
        assert 'characteristics' in regime
        assert regime['regime_type'] in ['risk_on', 'risk_off', 'neutral', 'crisis']
    
    @pytest.mark.unit
    def test_asset_allocation_optimization(self, cross_asset_agent, multi_asset_data):
        """Test asset allocation optimization."""
        returns = multi_asset_data.pct_change().dropna()
        
        allocation = cross_asset_agent._optimize_allocation(
            returns,
            method='risk_parity',
            target_vol=0.1
        )
        
        assert isinstance(allocation, pd.Series)
        assert abs(allocation.sum() - 1.0) < 0.01  # Should sum to 1
        assert (allocation >= 0).all()  # Long-only for this test
    
    @pytest.mark.unit
    def test_carry_trade_strategy(self, cross_asset_agent):
        """Test carry trade strategy for currencies."""
        currency_data = {
            'USD_JPY': {'spot': 110, 'interest_rate': 0.25},
            'USD_CHF': {'spot': 0.92, 'interest_rate': -0.75},
            'GBP_USD': {'spot': 1.35, 'interest_rate': 0.50},
            'AUD_USD': {'spot': 0.75, 'interest_rate': 1.50}
        }
        
        carry_signals = cross_asset_agent._generate_carry_signals(currency_data)
        
        assert isinstance(carry_signals, dict)
        for pair in carry_signals:
            assert 'signal' in carry_signals[pair]
            assert 'carry_yield' in carry_signals[pair]
            assert carry_signals[pair]['signal'] in ['long', 'short', 'neutral']
    
    @pytest.mark.unit
    def test_flight_to_quality_detection(self, cross_asset_agent, multi_asset_data):
        """Test flight-to-quality event detection."""
        returns = multi_asset_data.pct_change().dropna()
        
        flight_events = cross_asset_agent._detect_flight_to_quality(
            returns,
            equity_cols=['SPY', 'QQQ'],
            safe_haven_cols=['TLT', 'UUP']
        )
        
        assert isinstance(flight_events, pd.Series)
        assert flight_events.dtype == bool
    
    @pytest.mark.unit
    def test_momentum_across_assets(self, cross_asset_agent, multi_asset_data):
        """Test momentum strategy across asset classes."""
        momentum_scores = cross_asset_agent._calculate_cross_asset_momentum(
            multi_asset_data,
            lookback_periods=[1, 3, 6, 12]
        )
        
        assert isinstance(momentum_scores, pd.DataFrame)
        assert momentum_scores.shape[1] == len(multi_asset_data.columns)
        assert (momentum_scores.abs() <= 1).all().all()  # Normalized scores
    
    @pytest.mark.unit
    def test_volatility_targeting(self, cross_asset_agent, multi_asset_data):
        """Test volatility targeting across assets."""
        returns = multi_asset_data.pct_change().dropna()
        
        vol_targets = cross_asset_agent._calculate_volatility_targets(
            returns,
            target_vol=0.1,
            lookback=30
        )
        
        assert isinstance(vol_targets, pd.Series)
        assert (vol_targets > 0).all()
        assert vol_targets.sum() <= 1.1  # Allow for rounding
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_process_with_mock_data(self, cross_asset_agent, sample_inputs, multi_asset_data):
        """Test processing with mocked data."""
        with patch.object(cross_asset_agent, '_fetch_multi_asset_data', return_value=multi_asset_data):
            result = await cross_asset_agent.process(sample_inputs)
            
            assert isinstance(result, dict)
            assert 'allocations' in result
            assert 'regime_analysis' in result
            assert 'correlation_matrix' in result
            assert 'risk_metrics' in result
            assert 'rebalancing_signals' in result
            assert 'confidence' in result
    
    @pytest.mark.unit
    def test_currency_hedging(self, cross_asset_agent):
        """Test currency hedging for international portfolios."""
        portfolio = {
            'domestic': {'value': 1000000, 'currency': 'USD'},
            'international': [
                {'value': 500000, 'currency': 'EUR'},
                {'value': 300000, 'currency': 'JPY'},
                {'value': 200000, 'currency': 'GBP'}
            ]
        }
        
        hedge_ratios = cross_asset_agent._calculate_currency_hedge_ratios(
            portfolio,
            hedge_ratio=0.5,  # 50% hedge
            hedge_method='forward'
        )
        
        assert isinstance(hedge_ratios, dict)
        assert 'EUR_USD' in hedge_ratios
        assert 'JPY_USD' in hedge_ratios
        assert 'GBP_USD' in hedge_ratios
    
    @pytest.mark.unit
    def test_sector_rotation_signals(self, cross_asset_agent, multi_asset_data):
        """Test sector rotation signals based on macro factors."""
        macro_factors = {
            'yield_curve_slope': 1.5,
            'credit_spreads': 1.2,
            'dollar_strength': 0.3,
            'oil_price_change': 0.05
        }
        
        rotation_signals = cross_asset_agent._generate_sector_rotation_signals(
            macro_factors,
            multi_asset_data
        )
        
        assert isinstance(rotation_signals, dict)
        assert 'recommended_overweight' in rotation_signals
        assert 'recommended_underweight' in rotation_signals
    
    @pytest.mark.unit
    def test_cross_asset_arbitrage(self, cross_asset_agent):
        """Test cross-asset arbitrage opportunities."""
        asset_prices = {
            'SPY': 400,  # S&P 500 ETF
            'SPX_futures': 4010,  # S&P 500 futures
            'dividend_yield': 0.015,
            'risk_free_rate': 0.04,
            'days_to_expiry': 30
        }
        
        arbitrage_opportunity = cross_asset_agent._detect_arbitrage_opportunity(
            asset_prices,
            threshold=0.5  # 0.5% threshold
        )
        
        assert isinstance(arbitrage_opportunity, dict)
        assert 'opportunity_exists' in arbitrage_opportunity
        assert 'expected_profit' in arbitrage_opportunity
        assert 'risk_score' in arbitrage_opportunity
    
    @pytest.mark.unit
    def test_risk_parity_allocation(self, cross_asset_agent, multi_asset_data):
        """Test risk parity allocation across asset classes."""
        returns = multi_asset_data.pct_change().dropna()
        
        risk_parity_weights = cross_asset_agent._calculate_risk_parity_weights(
            returns,
            lookback=60
        )
        
        assert isinstance(risk_parity_weights, pd.Series)
        assert abs(risk_parity_weights.sum() - 1.0) < 0.01
        assert (risk_parity_weights >= 0).all()
        
        # Check that risk contributions are approximately equal
        covariance_matrix = returns.cov()
        risk_contributions = cross_asset_agent._calculate_risk_contributions(
            risk_parity_weights, covariance_matrix
        )
        
        assert risk_contributions.std() < 0.1  # Risk contributions should be similar
    
    @pytest.mark.unit
    def test_mean_reversion_strategy(self, cross_asset_agent, multi_asset_data):
        """Test mean reversion strategy across asset classes."""
        returns = multi_asset_data.pct_change().dropna()
        
        mean_reversion_signals = cross_asset_agent._generate_mean_reversion_signals(
            returns,
            lookback=20,
            z_score_threshold=2.0
        )
        
        assert isinstance(mean_reversion_signals, pd.DataFrame)
        assert mean_reversion_signals.shape[1] == len(multi_asset_data.columns)
        assert set(mean_reversion_signals.values.flatten()) <= {-1, 0, 1}
    
    @pytest.mark.unit
    def test_macro_factor_exposure(self, cross_asset_agent, multi_asset_data):
        """Test macro factor exposure analysis."""
        returns = multi_asset_data.pct_change().dropna()
        
        factor_exposures = cross_asset_agent._calculate_factor_exposures(
            returns,
            factors=['market', 'size', 'value', 'momentum']
        )
        
        assert isinstance(factor_exposures, pd.DataFrame)
        assert 'market' in factor_exposures.columns
        assert factor_exposures.shape[0] == len(multi_asset_data.columns)
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_dynamic_hedging(self, cross_asset_agent):
        """Test dynamic hedging across asset classes."""
        portfolio = {
            'equity_exposure': 0.6,
            'bond_exposure': 0.3,
            'commodity_exposure': 0.1
        }
        
        hedge_strategy = await cross_asset_agent._calculate_dynamic_hedge(
            portfolio,
            hedge_horizon=30,
            confidence_level=0.95
        )
        
        assert isinstance(hedge_strategy, dict)
        assert 'hedge_instruments' in hedge_strategy
        assert 'hedge_ratios' in hedge_strategy
        assert 'expected_protection' in hedge_strategy
    
    @pytest.mark.unit
    def test_tail_risk_hedging(self, cross_asset_agent, multi_asset_data):
        """Test tail risk hedging strategies."""
        returns = multi_asset_data.pct_change().dropna()
        
        tail_hedge = cross_asset_agent._design_tail_risk_hedge(
            returns,
            protection_level=0.95,
            cost_budget=0.02
        )
        
        assert isinstance(tail_hedge, dict)
        assert 'hedge_instruments' in tail_hedge
        assert 'cost' in tail_hedge
        assert 'protection_ratio' in tail_hedge
        assert tail_hedge['cost'] <= 0.02
    
    @pytest.mark.unit
    def test_liquidity_analysis(self, cross_asset_agent):
        """Test liquidity analysis across asset classes."""
        volume_data = {
            'SPY': [50000000, 45000000, 60000000],
            'TLT': [15000000, 12000000, 18000000],
            'GLD': [8000000, 7500000, 9000000]
        }
        
        liquidity_scores = cross_asset_agent._analyze_liquidity(
            volume_data,
            lookback=30
        )
        
        assert isinstance(liquidity_scores, pd.Series)
        assert (liquidity_scores >= 0).all()
        assert (liquidity_scores <= 1).all()
        assert liquidity_scores['SPY'] > liquidity_scores['GLD']  # SPY should be more liquid