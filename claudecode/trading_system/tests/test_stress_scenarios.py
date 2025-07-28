"""
Stress Testing Scenarios for Risk Management.
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

from main import TradingSystem
from core.base.exceptions import RiskLimitExceeded, LiquidityError


@pytest.fixture
def stress_test_system():
    """Create a trading system configured for stress testing."""
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    return TradingSystem(config_path)


@pytest.fixture
def market_crash_scenario():
    """Generate market crash scenario data."""
    dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
    symbols = ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM']
    
    crash_data = {}
    np.random.seed(42)
    
    for symbol in symbols:
        # Normal market for first 10 days
        normal_returns = np.random.normal(0.001, 0.015, 10)
        
        # Crash day with -20% drop
        crash_returns = [-0.20]
        
        # Recovery period with high volatility
        recovery_returns = np.random.normal(-0.02, 0.05, 19)
        
        all_returns = np.concatenate([normal_returns, crash_returns, recovery_returns])
        
        # Convert to prices
        prices = [100]
        for ret in all_returns:
            prices.append(prices[-1] * (1 + ret))
        
        crash_data[symbol] = pd.DataFrame({
            'timestamp': dates,
            'close': prices[1:],
            'volume': np.random.randint(50000000, 200000000, len(dates)),  # High volume during crash
            'returns': all_returns
        })
    
    return crash_data


@pytest.fixture
def liquidity_crisis_scenario():
    """Generate liquidity crisis scenario data."""
    dates = pd.date_range(start='2023-01-01', periods=20, freq='D')
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
    
    liquidity_data = {}
    
    for symbol in symbols:
        # Normal volume for first 10 days
        normal_volume = np.random.randint(30000000, 50000000, 10)
        
        # Liquidity crisis - volume drops to 10% of normal
        crisis_volume = np.random.randint(3000000, 8000000, 10)
        
        all_volume = np.concatenate([normal_volume, crisis_volume])
        
        # Wide bid-ask spreads during crisis
        normal_spreads = np.random.uniform(0.01, 0.05, 10)
        crisis_spreads = np.random.uniform(0.20, 1.00, 10)
        spreads = np.concatenate([normal_spreads, crisis_spreads])
        
        prices = 100 + np.random.randn(len(dates)).cumsum()
        
        liquidity_data[symbol] = pd.DataFrame({
            'timestamp': dates,
            'close': prices,
            'volume': all_volume,
            'bid_ask_spread': spreads,
            'returns': np.diff(prices, prepend=prices[0]) / prices[:-1]
        })
    
    return liquidity_data


@pytest.fixture
def volatility_spike_scenario():
    """Generate volatility spike scenario."""
    dates = pd.date_range(start='2023-01-01', periods=25, freq='D')
    symbols = ['VIX', 'SPY', 'TLT', 'GLD', 'USD']
    
    vol_data = {}
    
    for symbol in symbols:
        if symbol == 'VIX':
            # VIX spikes from 15 to 80
            normal_vix = np.random.uniform(12, 18, 15)
            spike_vix = np.random.uniform(75, 85, 5)
            settling_vix = np.random.uniform(35, 45, 5)
            vix_values = np.concatenate([normal_vix, spike_vix, settling_vix])
            
            vol_data[symbol] = pd.DataFrame({
                'timestamp': dates,
                'close': vix_values,
                'volume': np.random.randint(100000, 500000, len(dates))
            })
        else:
            # Other assets show increased volatility
            normal_vol = 0.015
            spike_vol = 0.08
            
            normal_returns = np.random.normal(0, normal_vol, 15)
            spike_returns = np.random.normal(0, spike_vol, 5)
            settling_returns = np.random.normal(0, normal_vol * 2, 5)
            
            all_returns = np.concatenate([normal_returns, spike_returns, settling_returns])
            
            prices = [100]
            for ret in all_returns:
                prices.append(prices[-1] * (1 + ret))
            
            vol_data[symbol] = pd.DataFrame({
                'timestamp': dates,
                'close': prices[1:],
                'volume': np.random.randint(20000000, 80000000, len(dates)),
                'returns': all_returns
            })
    
    return vol_data


class TestStressScenarios:
    """Stress testing scenarios for risk management."""
    
    @pytest.mark.stress
    @pytest.mark.asyncio
    async def test_market_crash_stress(self, stress_test_system, market_crash_scenario):
        """Test system behavior during market crash."""
        await stress_test_system.initialize()
        
        # Set up initial portfolio
        initial_portfolio = {
            'SPY': 0.4,
            'QQQ': 0.3,
            'IWM': 0.2,
            'EFA': 0.1
        }
        
        with patch.object(stress_test_system, '_fetch_market_data', return_value=market_crash_scenario):
            
            # Run stress test
            stress_result = await stress_test_system.run_stress_test(
                scenario_type='market_crash',
                portfolio=initial_portfolio,
                shock_magnitude=0.20  # 20% market drop
            )
            
            assert 'portfolio_impact' in stress_result
            assert 'var_impact' in stress_result
            assert 'drawdown' in stress_result
            assert 'liquidity_impact' in stress_result
            
            # Verify risk management responses
            assert stress_result['portfolio_impact'] < -0.15  # Significant loss expected
            assert stress_result['drawdown'] > 0.15
            
            # Check if risk limits were triggered
            if stress_result['var_impact'] > stress_test_system.config['risk_limits']['max_var']:
                assert 'risk_limit_breach' in stress_result
                assert stress_result['risk_limit_breach'] == True
    
    @pytest.mark.stress
    @pytest.mark.asyncio
    async def test_liquidity_crisis_stress(self, stress_test_system, liquidity_crisis_scenario):
        """Test system behavior during liquidity crisis."""
        await stress_test_system.initialize()
        
        portfolio = {
            'AAPL': 0.25,
            'GOOGL': 0.25,
            'MSFT': 0.25,
            'TSLA': 0.25
        }
        
        with patch.object(stress_test_system, '_fetch_market_data', return_value=liquidity_crisis_scenario):
            
            stress_result = await stress_test_system.run_liquidity_stress_test(
                portfolio=portfolio,
                liquidation_horizon=5  # days
            )
            
            assert 'liquidation_cost' in stress_result
            assert 'time_to_liquidate' in stress_result
            assert 'market_impact' in stress_result
            
            # During liquidity crisis, costs should be high
            assert stress_result['liquidation_cost'] > 0.05  # > 5% cost
            assert stress_result['time_to_liquidate'] > 3  # > 3 days
            
            # Check if alternative liquidation strategies are suggested
            if stress_result['liquidation_cost'] > 0.10:
                assert 'alternative_strategies' in stress_result
                assert len(stress_result['alternative_strategies']) > 0
    
    @pytest.mark.stress
    @pytest.mark.asyncio
    async def test_volatility_spike_stress(self, stress_test_system, volatility_spike_scenario):
        """Test system behavior during volatility spike."""
        await stress_test_system.initialize()
        
        portfolio = {
            'SPY': 0.6,
            'TLT': 0.3,
            'GLD': 0.1
        }
        
        with patch.object(stress_test_system, '_fetch_market_data', return_value=volatility_spike_scenario):
            
            stress_result = await stress_test_system.run_volatility_stress_test(
                portfolio=portfolio,
                vol_shock_multiplier=4  # 4x normal volatility
            )
            
            assert 'portfolio_volatility' in stress_result
            assert 'var_impact' in stress_result
            assert 'hedging_cost' in stress_result
            
            # High volatility should trigger hedging
            if stress_result['portfolio_volatility'] > 0.3:  # 30% volatility
                assert 'hedging_recommendation' in stress_result
                assert stress_result['hedging_cost'] > 0
    
    @pytest.mark.stress
    @pytest.mark.asyncio
    async def test_correlation_breakdown_stress(self, stress_test_system):
        """Test system behavior when correlations break down."""
        await stress_test_system.initialize()
        
        # Generate data where correlations break down
        dates = pd.date_range(start='2023-01-01', periods=60, freq='D')
        
        # First 30 days: high correlation
        common_factor = np.random.normal(0, 0.02, 30)
        asset1_returns_1 = common_factor + np.random.normal(0, 0.005, 30)
        asset2_returns_1 = common_factor + np.random.normal(0, 0.005, 30)
        
        # Last 30 days: correlation breaks down
        asset1_returns_2 = np.random.normal(0.01, 0.03, 30)  # Different regime
        asset2_returns_2 = np.random.normal(-0.01, 0.04, 30)  # Opposite direction
        
        correlation_data = {
            'ASSET1': pd.DataFrame({
                'timestamp': dates,
                'returns': np.concatenate([asset1_returns_1, asset1_returns_2])
            }),
            'ASSET2': pd.DataFrame({
                'timestamp': dates,
                'returns': np.concatenate([asset2_returns_1, asset2_returns_2])
            })
        }
        
        with patch.object(stress_test_system, '_fetch_market_data', return_value=correlation_data):
            
            stress_result = await stress_test_system.test_correlation_breakdown(
                assets=['ASSET1', 'ASSET2'],
                lookback_window=30
            )
            
            assert 'correlation_change' in stress_result
            assert 'diversification_impact' in stress_result
            assert 'portfolio_risk_change' in stress_result
            
            # Should detect correlation breakdown
            assert abs(stress_result['correlation_change']) > 0.5
    
    @pytest.mark.stress
    @pytest.mark.asyncio
    async def test_concentration_risk_stress(self, stress_test_system):
        """Test system behavior under concentration risk scenarios."""
        await stress_test_system.initialize()
        
        # Highly concentrated portfolio
        concentrated_portfolio = {
            'AAPL': 0.6,  # Very large position
            'GOOGL': 0.2,
            'MSFT': 0.15,
            'CASH': 0.05
        }
        
        stress_result = await stress_test_system.test_concentration_risk(
            portfolio=concentrated_portfolio,
            sector_concentration=True
        )
        
        assert 'concentration_score' in stress_result
        assert 'single_name_risk' in stress_result
        assert 'sector_risk' in stress_result
        
        # Should flag high concentration
        assert stress_result['concentration_score'] > 0.7
        assert stress_result['single_name_risk']['AAPL'] > 0.5
        
        # Should recommend diversification
        assert 'diversification_recommendations' in stress_result
    
    @pytest.mark.stress
    @pytest.mark.asyncio
    async def test_model_failure_stress(self, stress_test_system):
        """Test system behavior when ML models fail."""
        await stress_test_system.initialize()
        
        # Simulate model failure scenarios
        failure_scenarios = [
            {'agent': 'ml_ensemble_agent', 'failure_type': 'prediction_error'},
            {'agent': 'risk_modeling_agent', 'failure_type': 'var_estimation_failure'},
            {'agent': 'technical_analysis_agent', 'failure_type': 'indicator_calculation_error'}
        ]
        
        for scenario in failure_scenarios:
            stress_result = await stress_test_system.simulate_model_failure(scenario)
            
            assert 'affected_components' in stress_result
            assert 'fallback_activated' in stress_result
            assert 'performance_degradation' in stress_result
            
            # System should activate fallback mechanisms
            assert stress_result['fallback_activated'] == True
            
            # Performance degradation should be quantified
            assert 0 <= stress_result['performance_degradation'] <= 1
    
    @pytest.mark.stress
    @pytest.mark.asyncio
    async def test_extreme_market_conditions(self, stress_test_system):
        """Test system under extreme market conditions."""
        await stress_test_system.initialize()
        
        extreme_scenarios = [
            {
                'name': 'flash_crash',
                'price_shock': -0.10,  # 10% drop in minutes
                'volume_spike': 10,     # 10x normal volume
                'duration_minutes': 30
            },
            {
                'name': 'circuit_breaker',
                'price_shock': -0.07,  # 7% drop triggering circuit breaker
                'trading_halt': True,
                'duration_minutes': 15
            },
            {
                'name': 'gap_opening',
                'overnight_gap': -0.15,  # 15% gap down
                'liquidity_shortage': True
            }
        ]
        
        for scenario in extreme_scenarios:
            stress_result = await stress_test_system.test_extreme_scenario(scenario)
            
            assert 'scenario_impact' in stress_result
            assert 'risk_management_response' in stress_result
            assert 'recovery_time' in stress_result
            
            # System should respond appropriately to extreme conditions
            if scenario['name'] == 'circuit_breaker':
                assert 'trading_halt_response' in stress_result['risk_management_response']
            
            if 'liquidity_shortage' in scenario and scenario['liquidity_shortage']:
                assert 'liquidity_management' in stress_result['risk_management_response']
    
    @pytest.mark.stress
    @pytest.mark.asyncio
    async def test_counterparty_risk_stress(self, stress_test_system):
        """Test system behavior under counterparty risk scenarios."""
        await stress_test_system.initialize()
        
        # Define counterparty exposures
        counterparty_exposures = {
            'PRIME_BROKER_A': {'exposure': 5000000, 'credit_rating': 'A+'},
            'EXCHANGE_B': {'exposure': 2000000, 'credit_rating': 'AA'},
            'CLEARINGHOUSE_C': {'exposure': 10000000, 'credit_rating': 'AAA'}
        }
        
        # Test counterparty default scenario
        stress_result = await stress_test_system.test_counterparty_default(
            counterparty='PRIME_BROKER_A',
            exposures=counterparty_exposures,
            recovery_rate=0.4  # 40% recovery
        )
        
        assert 'direct_loss' in stress_result
        assert 'indirect_impact' in stress_result
        assert 'mitigation_actions' in stress_result
        
        # Calculate expected loss
        expected_loss = counterparty_exposures['PRIME_BROKER_A']['exposure'] * (1 - 0.4)
        assert abs(stress_result['direct_loss'] - expected_loss) < 100000  # Allow for calculation differences
    
    @pytest.mark.stress
    @pytest.mark.asyncio
    async def test_system_capacity_limits(self, stress_test_system):
        """Test system behavior at capacity limits."""
        await stress_test_system.initialize()
        
        # Test increasing load until system limits
        load_levels = [100, 500, 1000, 2000, 5000]  # Number of simultaneous operations
        
        for load in load_levels:
            start_time = datetime.now()
            
            try:
                # Simulate high load
                tasks = []
                for i in range(load):
                    task = stress_test_system.process_market_update({
                        'symbol': f'STOCK_{i % 100}',
                        'price': 100 + np.random.normal(0, 5),
                        'timestamp': datetime.now()
                    })
                    tasks.append(task)
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                end_time = datetime.now()
                processing_time = (end_time - start_time).total_seconds()
                
                success_count = sum(1 for r in results if not isinstance(r, Exception))
                failure_count = load - success_count
                
                print(f"Load {load}: {success_count} successes, {failure_count} failures, {processing_time:.2f}s")
                
                # Record performance degradation
                if failure_count > load * 0.1:  # More than 10% failures
                    print(f"System capacity limit reached at load {load}")
                    break
                    
            except Exception as e:
                print(f"System failed at load {load}: {e}")
                break