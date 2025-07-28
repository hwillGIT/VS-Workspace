"""
Comprehensive tests for the Options Trading Agent.
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

from agents.strategies.options.options_agent import OptionsAgent
from core.base.exceptions import DataError, ValidationError


@pytest.fixture
def options_agent():
    """Create an OptionsAgent instance for testing."""
    return OptionsAgent()


@pytest.fixture
def sample_options_chain():
    """Create sample options chain data for testing."""
    strikes = np.arange(90, 111, 5)
    expiries = [datetime.now() + timedelta(days=d) for d in [7, 30, 60, 90]]
    
    options_data = []
    underlying_price = 100
    
    for expiry in expiries:
        days_to_expiry = (expiry - datetime.now()).days
        for strike in strikes:
            # Calculate theoretical IV based on moneyness and time
            moneyness = strike / underlying_price
            base_iv = 0.2 + 0.1 * abs(1 - moneyness)
            
            call_delta = 0.5 + 0.4 * (underlying_price - strike) / underlying_price
            put_delta = call_delta - 1
            
            options_data.extend([
                {
                    'symbol': 'AAPL',
                    'strike': strike,
                    'expiry': expiry,
                    'option_type': 'call',
                    'bid': max(0, underlying_price - strike) + np.random.uniform(0.1, 0.5),
                    'ask': max(0, underlying_price - strike) + np.random.uniform(0.5, 1.0),
                    'volume': np.random.randint(100, 5000),
                    'open_interest': np.random.randint(1000, 20000),
                    'implied_volatility': base_iv,
                    'delta': call_delta,
                    'gamma': 0.02,
                    'theta': -0.05,
                    'vega': 0.15
                },
                {
                    'symbol': 'AAPL',
                    'strike': strike,
                    'expiry': expiry,
                    'option_type': 'put',
                    'bid': max(0, strike - underlying_price) + np.random.uniform(0.1, 0.5),
                    'ask': max(0, strike - underlying_price) + np.random.uniform(0.5, 1.0),
                    'volume': np.random.randint(100, 5000),
                    'open_interest': np.random.randint(1000, 20000),
                    'implied_volatility': base_iv,
                    'delta': put_delta,
                    'gamma': 0.02,
                    'theta': -0.05,
                    'vega': 0.15
                }
            ])
    
    return pd.DataFrame(options_data)


@pytest.fixture
def sample_inputs():
    """Sample inputs for options strategies."""
    return {
        "symbols": ["AAPL", "GOOGL"],
        "strategies": ["covered_call", "iron_condor", "straddle"],
        "max_days_to_expiry": 45,
        "min_volume": 100,
        "iv_rank_threshold": 50,
        "target_return": 0.02,
        "max_risk": 0.05
    }


class TestOptionsAgent:
    """Test cases for OptionsAgent."""
    
    @pytest.mark.unit
    def test_agent_initialization(self, options_agent):
        """Test agent initialization."""
        assert options_agent.name == "OptionsAgent"
        assert hasattr(options_agent, 'strategy_selector')
        assert hasattr(options_agent, 'greek_calculator')
        assert hasattr(options_agent, 'volatility_analyzer')
    
    @pytest.mark.unit
    def test_implied_volatility_analysis(self, options_agent, sample_options_chain):
        """Test implied volatility surface analysis."""
        iv_surface = options_agent._analyze_iv_surface(sample_options_chain)
        
        assert isinstance(iv_surface, dict)
        assert 'term_structure' in iv_surface
        assert 'skew' in iv_surface
        assert 'iv_rank' in iv_surface
        assert 0 <= iv_surface['iv_rank'] <= 100
    
    @pytest.mark.unit
    def test_options_pricing_model(self, options_agent):
        """Test Black-Scholes options pricing."""
        price = options_agent._calculate_option_price(
            S=100,  # Underlying price
            K=105,  # Strike
            T=0.25,  # Time to expiry (years)
            r=0.05,  # Risk-free rate
            sigma=0.2,  # Volatility
            option_type='call'
        )
        
        assert isinstance(price, float)
        assert price > 0
        assert price < 100  # Call can't be worth more than underlying
    
    @pytest.mark.unit
    def test_greek_calculations(self, options_agent):
        """Test options Greeks calculations."""
        greeks = options_agent._calculate_greeks(
            S=100,
            K=100,
            T=0.25,
            r=0.05,
            sigma=0.2,
            option_type='call'
        )
        
        assert isinstance(greeks, dict)
        assert 'delta' in greeks
        assert 'gamma' in greeks
        assert 'theta' in greeks
        assert 'vega' in greeks
        assert 'rho' in greeks
        assert 0 <= greeks['delta'] <= 1  # Call delta between 0 and 1
    
    @pytest.mark.unit
    def test_covered_call_strategy(self, options_agent, sample_options_chain):
        """Test covered call strategy generation."""
        underlying_price = 100
        
        strategy = options_agent._generate_covered_call(
            options_chain=sample_options_chain,
            underlying_price=underlying_price,
            target_return=0.02
        )
        
        assert isinstance(strategy, dict)
        assert strategy['strategy_type'] == 'covered_call'
        assert 'legs' in strategy
        assert len(strategy['legs']) == 2  # Stock + short call
        assert strategy['max_profit'] > 0
        assert strategy['break_even'] > 0
    
    @pytest.mark.unit
    def test_iron_condor_strategy(self, options_agent, sample_options_chain):
        """Test iron condor strategy generation."""
        strategy = options_agent._generate_iron_condor(
            options_chain=sample_options_chain,
            underlying_price=100,
            target_credit=1.5,
            wing_width=10
        )
        
        assert isinstance(strategy, dict)
        assert strategy['strategy_type'] == 'iron_condor'
        assert len(strategy['legs']) == 4  # 4 options legs
        assert strategy['max_profit'] > 0
        assert strategy['max_loss'] < 0
        assert len(strategy['break_evens']) == 2
    
    @pytest.mark.unit
    def test_straddle_strategy(self, options_agent, sample_options_chain):
        """Test straddle strategy generation."""
        strategy = options_agent._generate_straddle(
            options_chain=sample_options_chain,
            underlying_price=100,
            iv_threshold=0.3
        )
        
        assert isinstance(strategy, dict)
        assert strategy['strategy_type'] == 'straddle'
        assert len(strategy['legs']) == 2  # Long call + long put
        assert strategy['max_loss'] < 0
        assert strategy['break_evens'][0] < 100 < strategy['break_evens'][1]
    
    @pytest.mark.unit
    def test_butterfly_spread(self, options_agent, sample_options_chain):
        """Test butterfly spread strategy."""
        strategy = options_agent._generate_butterfly(
            options_chain=sample_options_chain,
            underlying_price=100,
            spread_width=5
        )
        
        assert isinstance(strategy, dict)
        assert strategy['strategy_type'] == 'butterfly'
        assert len(strategy['legs']) == 3  # 3 strike prices
        assert strategy['max_profit'] > 0
        assert strategy['max_loss'] < 0
    
    @pytest.mark.unit
    def test_calendar_spread(self, options_agent, sample_options_chain):
        """Test calendar spread strategy."""
        strategy = options_agent._generate_calendar_spread(
            options_chain=sample_options_chain,
            underlying_price=100,
            strike=100
        )
        
        assert isinstance(strategy, dict)
        assert strategy['strategy_type'] == 'calendar_spread'
        assert len(strategy['legs']) == 2
        # Verify different expiries
        assert strategy['legs'][0]['expiry'] != strategy['legs'][1]['expiry']
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_process_with_mock_data(self, options_agent, sample_inputs, sample_options_chain):
        """Test processing with mocked data."""
        with patch.object(options_agent, '_fetch_options_chain', return_value=sample_options_chain):
            result = await options_agent.process(sample_inputs)
            
            assert isinstance(result, dict)
            assert 'strategies' in result
            assert 'risk_metrics' in result
            assert 'greek_exposure' in result
            assert 'iv_analysis' in result
            assert 'confidence' in result
    
    @pytest.mark.unit
    def test_volatility_arbitrage(self, options_agent, sample_options_chain):
        """Test volatility arbitrage opportunity detection."""
        vol_arb = options_agent._detect_volatility_arbitrage(
            options_chain=sample_options_chain,
            historical_vol=0.15
        )
        
        assert isinstance(vol_arb, list)
        for opportunity in vol_arb:
            assert 'strike' in opportunity
            assert 'expiry' in opportunity
            assert 'iv_vs_hv' in opportunity
            assert 'expected_profit' in opportunity
    
    @pytest.mark.unit
    def test_portfolio_hedging(self, options_agent):
        """Test portfolio hedging with options."""
        portfolio = {
            'AAPL': {'shares': 1000, 'value': 100000},
            'GOOGL': {'shares': 50, 'value': 150000}
        }
        
        hedge_strategy = options_agent._calculate_portfolio_hedge(
            portfolio=portfolio,
            protection_level=0.9,  # 10% downside protection
            time_horizon=30  # days
        )
        
        assert isinstance(hedge_strategy, dict)
        assert 'hedge_ratio' in hedge_strategy
        assert 'put_options' in hedge_strategy
        assert 'cost' in hedge_strategy
        assert hedge_strategy['cost'] > 0
    
    @pytest.mark.unit
    def test_delta_neutral_strategy(self, options_agent, sample_options_chain):
        """Test delta-neutral strategy construction."""
        strategy = options_agent._build_delta_neutral_position(
            options_chain=sample_options_chain,
            underlying_price=100,
            target_vega=100  # Target vega exposure
        )
        
        assert isinstance(strategy, dict)
        assert abs(strategy['net_delta']) < 0.01  # Should be close to zero
        assert strategy['net_vega'] > 0  # Positive vega for volatility play
    
    @pytest.mark.unit
    def test_risk_reversal_strategy(self, options_agent, sample_options_chain):
        """Test risk reversal (synthetic long) strategy."""
        strategy = options_agent._generate_risk_reversal(
            options_chain=sample_options_chain,
            underlying_price=100,
            bullish=True
        )
        
        assert isinstance(strategy, dict)
        assert strategy['strategy_type'] == 'risk_reversal'
        assert len(strategy['legs']) == 2  # Long call + short put
        assert strategy['net_cost'] < 5  # Should be cheap or credit
    
    @pytest.mark.unit
    def test_options_market_making(self, options_agent, sample_options_chain):
        """Test options market making strategy."""
        mm_params = options_agent._calculate_market_making_params(
            options_chain=sample_options_chain,
            option_strike=100,
            option_expiry=datetime.now() + timedelta(days=30)
        )
        
        assert isinstance(mm_params, dict)
        assert 'bid_price' in mm_params
        assert 'ask_price' in mm_params
        assert 'hedge_ratio' in mm_params
        assert mm_params['bid_price'] < mm_params['ask_price']
    
    @pytest.mark.unit
    def test_volatility_smile_fitting(self, options_agent, sample_options_chain):
        """Test volatility smile curve fitting."""
        expiry = sample_options_chain['expiry'].iloc[0]
        expiry_chain = sample_options_chain[sample_options_chain['expiry'] == expiry]
        
        smile_params = options_agent._fit_volatility_smile(
            options_chain=expiry_chain,
            model='sabr'  # SABR model for smile
        )
        
        assert isinstance(smile_params, dict)
        assert 'alpha' in smile_params  # SABR parameters
        assert 'beta' in smile_params
        assert 'rho' in smile_params
        assert 'nu' in smile_params
    
    @pytest.mark.unit
    def test_early_exercise_detection(self, options_agent):
        """Test early exercise detection for American options."""
        should_exercise = options_agent._check_early_exercise(
            option_type='call',
            S=110,  # Underlying price
            K=100,  # Strike
            T=0.1,  # Time to expiry
            r=0.05,  # Risk-free rate
            q=0.02  # Dividend yield
        )
        
        assert isinstance(should_exercise, bool)
        # Deep ITM call with dividend might be exercised early
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_options_backtesting(self, options_agent):
        """Test backtesting of options strategies."""
        backtest_inputs = {
            "symbol": "SPY",
            "strategy": "iron_condor",
            "start_date": datetime(2022, 1, 1),
            "end_date": datetime(2023, 1, 1),
            "dte_target": 45,
            "profit_target": 0.5,
            "stop_loss": 2.0
        }
        
        # Mock historical options data
        mock_options_history = self._generate_mock_options_history()
        
        with patch.object(options_agent, '_fetch_historical_options', return_value=mock_options_history):
            backtest_results = await options_agent.backtest(backtest_inputs)
            
            assert 'total_trades' in backtest_results
            assert 'win_rate' in backtest_results
            assert 'average_profit' in backtest_results
            assert 'max_drawdown' in backtest_results
            assert 'sharpe_ratio' in backtest_results
    
    @pytest.mark.unit
    def test_pin_risk_assessment(self, options_agent):
        """Test pin risk assessment near expiration."""
        pin_risk = options_agent._assess_pin_risk(
            underlying_price=100,
            strike=100,
            days_to_expiry=1,
            gamma=0.5,
            position_size=10
        )
        
        assert isinstance(pin_risk, dict)
        assert 'risk_score' in pin_risk
        assert 'recommended_action' in pin_risk
        assert 0 <= pin_risk['risk_score'] <= 1
    
    def _generate_mock_options_history(self):
        """Helper to generate mock historical options data."""
        dates = pd.date_range(start='2022-01-01', end='2023-01-01', freq='D')
        
        options_history = []
        for date in dates:
            underlying_price = 100 + 10 * np.sin(dates.get_loc(date) * 2 * np.pi / 252)
            
            for strike in [90, 95, 100, 105, 110]:
                for dte in [30, 45, 60]:
                    iv = 0.2 + 0.05 * abs(strike - underlying_price) / underlying_price
                    
                    options_history.append({
                        'date': date,
                        'underlying_price': underlying_price,
                        'strike': strike,
                        'dte': dte,
                        'call_price': max(0, underlying_price - strike) + np.random.uniform(0.5, 2),
                        'put_price': max(0, strike - underlying_price) + np.random.uniform(0.5, 2),
                        'implied_volatility': iv
                    })
        
        return pd.DataFrame(options_history)