"""
Comprehensive Integration Tests for the Multi-Agent Trading System.
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import json

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from main import TradingSystem
from core.base.exceptions import DataError, ValidationError, SystemError


@pytest.fixture
async def trading_system():
    """Create a TradingSystem instance for testing."""
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    system = TradingSystem(config_path)
    await system.initialize()
    return system


@pytest.fixture
def comprehensive_market_data():
    """Create comprehensive market data for testing."""
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
    
    market_data = {}
    np.random.seed(42)
    
    for symbol in symbols:
        # Generate realistic price data with trends and volatility
        base_price = np.random.uniform(50, 200)
        returns = np.random.normal(0.0005, 0.02, len(dates))
        
        # Add some momentum periods
        momentum_periods = np.random.choice(len(dates), size=50, replace=False)
        returns[momentum_periods] *= 2
        
        # Calculate prices
        prices = [base_price]
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        
        market_data[symbol] = pd.DataFrame({
            'timestamp': dates,
            'open': prices[:-1],
            'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices[:-1]],
            'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices[:-1]],
            'close': prices[1:],
            'volume': np.random.randint(1000000, 50000000, len(dates)),
            'returns': returns
        })
    
    return market_data


@pytest.fixture
def sample_events():
    """Create sample market events for testing."""
    events = [
        {
            'timestamp': datetime(2023, 1, 15, 9, 30),
            'event_type': 'earnings',
            'symbol': 'AAPL',
            'impact_score': 0.8,
            'expected_move': 0.05
        },
        {
            'timestamp': datetime(2023, 3, 22, 14, 0),
            'event_type': 'fed_announcement',
            'symbol': 'SPY',
            'impact_score': 0.9,
            'expected_move': -0.03
        },
        {
            'timestamp': datetime(2023, 7, 10, 10, 0),
            'event_type': 'merger',
            'symbol': 'MSFT',
            'impact_score': 0.7,
            'expected_move': 0.08
        }
    ]
    return events


class TestSystemIntegration:
    """Integration tests for the complete trading system."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_complete_system_initialization(self, trading_system):
        """Test complete system initialization."""
        assert trading_system is not None
        assert hasattr(trading_system, 'agents')
        assert len(trading_system.agents) > 0
        
        # Check that all core agents are initialized
        expected_agents = [
            'data_universe_agent',
            'technical_analysis_agent',
            'momentum_agent',
            'mean_reversion_agent',
            'ml_ensemble_agent',
            'risk_modeling_agent',
            'signal_synthesis_agent',
            'recommendation_agent'
        ]
        
        for agent_name in expected_agents:
            assert agent_name in trading_system.agents
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_end_to_end_signal_pipeline(self, trading_system, comprehensive_market_data):
        """Test end-to-end signal generation pipeline."""
        # Mock market data
        with patch.object(trading_system, '_fetch_market_data', return_value=comprehensive_market_data):
            
            # Run complete signal generation pipeline
            pipeline_input = {
                'symbols': ['AAPL', 'GOOGL', 'MSFT'],
                'start_date': datetime(2023, 1, 1),
                'end_date': datetime(2023, 12, 31),
                'strategies': ['momentum', 'mean_reversion', 'technical_analysis']
            }
            
            result = await trading_system.run_pipeline(pipeline_input)
            
            assert isinstance(result, dict)
            assert 'final_recommendations' in result
            assert 'risk_assessment' in result
            assert 'portfolio_allocation' in result
            assert 'execution_plan' in result
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_multi_agent_coordination(self, trading_system, comprehensive_market_data):
        """Test coordination between multiple agents."""
        with patch.object(trading_system, '_fetch_market_data', return_value=comprehensive_market_data):
            
            # Test data flow between agents
            symbol = 'AAPL'
            
            # 1. Data Universe Agent processes raw data
            universe_result = await trading_system.agents['data_universe_agent'].process({
                'symbols': [symbol],
                'data_types': ['price', 'volume', 'fundamentals']
            })
            
            # 2. Technical Analysis Agent processes the data
            tech_result = await trading_system.agents['technical_analysis_agent'].process({
                'symbols': [symbol],
                'indicators': ['sma', 'rsi', 'macd'],
                'data': universe_result['processed_data']
            })
            
            # 3. ML Ensemble Agent uses technical indicators
            ml_result = await trading_system.agents['ml_ensemble_agent'].process({
                'symbols': [symbol],
                'features': tech_result['indicators'],
                'target': 'price_direction'
            })
            
            # 4. Signal Synthesis Agent combines all signals
            synthesis_result = await trading_system.agents['signal_synthesis_agent'].process({
                'signals': {
                    'technical': tech_result,
                    'ml_ensemble': ml_result
                }
            })
            
            # Verify data flows correctly through the pipeline
            assert 'processed_data' in universe_result
            assert 'indicators' in tech_result
            assert 'predictions' in ml_result
            assert 'synthesized_signals' in synthesis_result
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_risk_management_integration(self, trading_system, comprehensive_market_data):
        """Test integration of risk management across all agents."""
        with patch.object(trading_system, '_fetch_market_data', return_value=comprehensive_market_data):
            
            # Generate portfolio with risk constraints
            portfolio_input = {
                'symbols': ['AAPL', 'GOOGL', 'MSFT', 'TSLA'],
                'target_return': 0.12,
                'max_volatility': 0.15,
                'max_position_size': 0.25,
                'var_limit': 0.02
            }
            
            result = await trading_system.generate_portfolio(portfolio_input)
            
            assert 'portfolio_weights' in result
            assert 'risk_metrics' in result
            assert 'var_estimate' in result['risk_metrics']
            assert 'volatility' in result['risk_metrics']
            
            # Verify risk constraints are satisfied
            weights = result['portfolio_weights']
            assert all(w <= 0.25 for w in weights.values())  # Position size constraint
            assert result['risk_metrics']['volatility'] <= 0.16  # Allow small tolerance
            assert result['risk_metrics']['var_estimate'] <= 0.021  # Allow small tolerance
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_time_processing(self, trading_system):
        """Test real-time market data processing."""
        # Simulate real-time data stream
        real_time_data = {
            'AAPL': {
                'price': 175.50,
                'volume': 1000000,
                'timestamp': datetime.now(),
                'bid': 175.48,
                'ask': 175.52
            }
        }
        
        # Process real-time update
        result = await trading_system.process_real_time_update(real_time_data)
        
        assert isinstance(result, dict)
        assert 'updated_signals' in result
        assert 'position_adjustments' in result
        assert 'processing_latency' in result
        assert result['processing_latency'] < 1.0  # Should process within 1 second
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_event_driven_response(self, trading_system, sample_events):
        """Test system response to market events."""
        with patch.object(trading_system, '_fetch_events', return_value=sample_events):
            
            event_response = await trading_system.handle_market_event(sample_events[0])
            
            assert isinstance(event_response, dict)
            assert 'pre_event_positioning' in event_response
            assert 'expected_impact' in event_response
            assert 'hedging_strategy' in event_response
            assert 'monitoring_plan' in event_response
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_portfolio_rebalancing(self, trading_system, comprehensive_market_data):
        """Test portfolio rebalancing workflow."""
        # Current portfolio
        current_portfolio = {
            'AAPL': 0.3,
            'GOOGL': 0.25,
            'MSFT': 0.2,
            'TSLA': 0.15,
            'NVDA': 0.1
        }
        
        with patch.object(trading_system, '_fetch_market_data', return_value=comprehensive_market_data):
            
            rebalancing_plan = await trading_system.rebalance_portfolio(
                current_portfolio,
                rebalance_trigger='monthly'
            )
            
            assert 'new_weights' in rebalancing_plan
            assert 'trades_required' in rebalancing_plan
            assert 'expected_costs' in rebalancing_plan
            assert 'impact_analysis' in rebalancing_plan
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_stress_testing(self, trading_system, comprehensive_market_data):
        """Test system performance under stress scenarios."""
        stress_scenarios = [
            {'name': 'market_crash', 'shock': -0.2},
            {'name': 'volatility_spike', 'vol_multiplier': 3},
            {'name': 'liquidity_crisis', 'volume_reduction': 0.5}
        ]
        
        with patch.object(trading_system, '_fetch_market_data', return_value=comprehensive_market_data):
            
            stress_results = await trading_system.run_stress_tests(stress_scenarios)
            
            assert isinstance(stress_results, dict)
            for scenario in stress_scenarios:
                assert scenario['name'] in stress_results
                assert 'portfolio_impact' in stress_results[scenario['name']]
                assert 'var_impact' in stress_results[scenario['name']]
                assert 'liquidity_impact' in stress_results[scenario['name']]
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_performance_attribution(self, trading_system):
        """Test performance attribution across agents and strategies."""
        # Mock historical performance data
        historical_returns = pd.DataFrame({
            'portfolio': np.random.normal(0.0008, 0.01, 252),
            'benchmark': np.random.normal(0.0005, 0.008, 252)
        })
        
        attribution_result = await trading_system.calculate_performance_attribution(
            historical_returns,
            attribution_factors=['momentum', 'mean_reversion', 'technical', 'ml']
        )
        
        assert 'factor_contributions' in attribution_result
        assert 'alpha' in attribution_result
        assert 'tracking_error' in attribution_result
        assert 'information_ratio' in attribution_result
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_system_monitoring_and_alerts(self, trading_system):
        """Test system monitoring and alert generation."""
        # Simulate system metrics
        system_metrics = {
            'processing_latency': 0.5,
            'memory_usage': 0.75,
            'error_rate': 0.02,
            'signal_quality': 0.8
        }
        
        alerts = await trading_system.check_system_health(system_metrics)
        
        assert isinstance(alerts, list)
        # Should not generate alerts for normal metrics
        critical_alerts = [alert for alert in alerts if alert['severity'] == 'critical']
        assert len(critical_alerts) == 0
    
    @pytest.mark.integration
    @pytest.mark.slow
    async def test_backtesting_framework(self, trading_system, comprehensive_market_data):
        """Test comprehensive backtesting framework."""
        backtest_config = {
            'start_date': datetime(2023, 1, 1),
            'end_date': datetime(2023, 6, 30),
            'initial_capital': 1000000,
            'rebalance_frequency': 'weekly',
            'transaction_costs': 0.001,
            'strategies': ['momentum', 'mean_reversion']
        }
        
        with patch.object(trading_system, '_fetch_historical_data', return_value=comprehensive_market_data):
            
            backtest_results = await trading_system.run_backtest(backtest_config)
            
            assert 'total_return' in backtest_results
            assert 'sharpe_ratio' in backtest_results
            assert 'max_drawdown' in backtest_results
            assert 'win_rate' in backtest_results
            assert 'trade_analytics' in backtest_results
            assert 'risk_metrics' in backtest_results
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, trading_system):
        """Test system error handling and recovery mechanisms."""
        # Simulate various error conditions
        error_scenarios = [
            {'type': 'data_feed_failure', 'agent': 'data_universe_agent'},
            {'type': 'model_prediction_error', 'agent': 'ml_ensemble_agent'},
            {'type': 'risk_limit_breach', 'agent': 'risk_modeling_agent'}
        ]
        
        for scenario in error_scenarios:
            recovery_plan = await trading_system.handle_system_error(scenario)
            
            assert 'error_type' in recovery_plan
            assert 'affected_components' in recovery_plan
            assert 'recovery_actions' in recovery_plan
            assert 'fallback_strategy' in recovery_plan
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_configuration_management(self, trading_system):
        """Test dynamic configuration updates."""
        # Test configuration update
        new_config = {
            'risk_limits': {
                'max_position_size': 0.15,  # Reduce from 0.25
                'max_portfolio_var': 0.015  # Reduce from 0.02
            }
        }
        
        update_result = await trading_system.update_configuration(new_config)
        
        assert update_result['success'] == True
        assert 'updated_components' in update_result
        
        # Verify new limits are applied
        current_config = trading_system.get_current_configuration()
        assert current_config['risk_limits']['max_position_size'] == 0.15
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_compliance_and_reporting(self, trading_system):
        """Test compliance monitoring and reporting."""
        # Generate compliance report
        compliance_report = await trading_system.generate_compliance_report(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 3, 31)
        )
        
        assert 'risk_limit_breaches' in compliance_report
        assert 'position_limit_checks' in compliance_report
        assert 'transaction_analysis' in compliance_report
        assert 'performance_metrics' in compliance_report
        assert 'regulatory_metrics' in compliance_report
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_scalability_performance(self, trading_system):
        """Test system scalability with increased load."""
        # Simulate processing multiple symbols simultaneously
        large_universe = [f"STOCK_{i:03d}" for i in range(100)]
        
        start_time = datetime.now()
        
        # Process large universe
        batch_result = await trading_system.process_large_universe(
            symbols=large_universe,
            batch_size=20
        )
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        assert 'processed_symbols' in batch_result
        assert len(batch_result['processed_symbols']) == 100
        assert processing_time < 60  # Should complete within 60 seconds
        assert 'performance_metrics' in batch_result
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_data_quality_monitoring(self, trading_system, comprehensive_market_data):
        """Test data quality monitoring and validation."""
        # Introduce data quality issues
        corrupted_data = comprehensive_market_data.copy()
        
        # Add missing values
        corrupted_data['AAPL'].loc[10:20, 'close'] = np.nan
        
        # Add outliers
        corrupted_data['GOOGL'].loc[50, 'close'] *= 10
        
        with patch.object(trading_system, '_fetch_market_data', return_value=corrupted_data):
            
            quality_report = await trading_system.assess_data_quality()
            
            assert 'data_completeness' in quality_report
            assert 'outlier_detection' in quality_report
            assert 'data_freshness' in quality_report
            assert 'quality_score' in quality_report
            
            # Should detect the data quality issues
            assert quality_report['outlier_detection']['GOOGL']['outliers_detected'] > 0
            assert quality_report['data_completeness']['AAPL']['missing_percentage'] > 0