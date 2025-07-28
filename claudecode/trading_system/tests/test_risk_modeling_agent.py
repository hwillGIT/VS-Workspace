"""
Comprehensive tests for the Risk Modeling Agent.
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

from agents.risk_management.risk_modeling_agent import RiskModelingAgent
from core.base.exceptions import RiskError, ValidationError


@pytest.fixture
def risk_agent():
    """Create a RiskModelingAgent instance for testing."""
    return RiskModelingAgent()


@pytest.fixture
def sample_portfolio_data():
    """Create sample portfolio data for testing."""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=252, freq='D')  # One year of trading days
    
    # Generate correlated returns for 5 assets
    correlation_matrix = np.array([
        [1.0, 0.3, 0.2, 0.1, 0.15],
        [0.3, 1.0, 0.4, 0.2, 0.25],
        [0.2, 0.4, 1.0, 0.3, 0.1],
        [0.1, 0.2, 0.3, 1.0, 0.2],
        [0.15, 0.25, 0.1, 0.2, 1.0]
    ])
    
    # Generate random returns with correlation
    returns = np.random.multivariate_normal(
        mean=[0.0005, 0.0008, 0.0003, 0.0006, 0.0004],  # Different expected returns
        cov=correlation_matrix * 0.0001,  # Scale for daily volatility
        size=len(dates)
    )
    
    portfolio_data = pd.DataFrame(
        returns,
        index=dates,
        columns=['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
    )
    
    return portfolio_data


@pytest.fixture
def sample_positions():
    """Create sample portfolio positions."""
    return {
        'AAPL': {'quantity': 100, 'price': 150.0, 'weight': 0.3},
        'GOOGL': {'quantity': 50, 'price': 2500.0, 'weight': 0.25},
        'MSFT': {'quantity': 80, 'price': 300.0, 'weight': 0.2},
        'TSLA': {'quantity': 30, 'price': 800.0, 'weight': 0.15},
        'AMZN': {'quantity': 20, 'price': 3000.0, 'weight': 0.1}
    }


@pytest.fixture
def sample_inputs():
    """Sample inputs for risk modeling."""
    return {
        "portfolio": {
            "positions": {
                'AAPL': {'quantity': 100, 'price': 150.0},
                'GOOGL': {'quantity': 50, 'price': 2500.0}
            }
        },
        "confidence_level": 0.95,
        "time_horizon": 1,
        "risk_metrics": ["var", "cvar", "max_drawdown", "sharpe"],
        "scenario_analysis": True
    }


class TestRiskModelingAgent:
    """Test cases for RiskModelingAgent."""
    
    @pytest.mark.unit
    def test_agent_initialization(self, risk_agent):
        """Test agent initialization."""
        assert risk_agent.name == "RiskModelingAgent"
        assert risk_agent.config_section == "risk_modeling"
        assert hasattr(risk_agent, 'risk_models')
    
    @pytest.mark.unit
    def test_var_calculation_historical(self, risk_agent, sample_portfolio_data):
        """Test historical VaR calculation."""
        portfolio_returns = sample_portfolio_data.sum(axis=1)  # Equal weighted
        
        var_95 = risk_agent._calculate_var_historical(portfolio_returns, confidence_level=0.95)
        var_99 = risk_agent._calculate_var_historical(portfolio_returns, confidence_level=0.99)
        
        assert isinstance(var_95, float)
        assert isinstance(var_99, float)
        assert var_95 < 0  # VaR should be negative (loss)
        assert var_99 < var_95  # 99% VaR should be more extreme than 95% VaR
    
    @pytest.mark.unit
    def test_var_calculation_parametric(self, risk_agent, sample_portfolio_data):
        """Test parametric VaR calculation."""
        portfolio_returns = sample_portfolio_data.sum(axis=1)
        
        var_parametric = risk_agent._calculate_var_parametric(
            portfolio_returns, confidence_level=0.95
        )
        
        assert isinstance(var_parametric, float)
        assert var_parametric < 0
    
    @pytest.mark.unit
    def test_cvar_calculation(self, risk_agent, sample_portfolio_data):
        """Test Conditional VaR (Expected Shortfall) calculation."""
        portfolio_returns = sample_portfolio_data.sum(axis=1)
        
        var_95 = risk_agent._calculate_var_historical(portfolio_returns, confidence_level=0.95)
        cvar_95 = risk_agent._calculate_cvar(portfolio_returns, confidence_level=0.95)
        
        assert isinstance(cvar_95, float)
        assert cvar_95 < var_95  # CVaR should be more extreme than VaR
    
    @pytest.mark.unit
    def test_maximum_drawdown(self, risk_agent, sample_portfolio_data):
        """Test maximum drawdown calculation."""
        # Convert returns to cumulative returns
        cumulative_returns = (1 + sample_portfolio_data.sum(axis=1)).cumprod()
        
        max_dd = risk_agent._calculate_maximum_drawdown(cumulative_returns)
        
        assert isinstance(max_dd, float)
        assert max_dd <= 0  # Drawdown should be negative or zero
        assert max_dd >= -1  # Maximum possible drawdown is -100%
    
    @pytest.mark.unit
    def test_sharpe_ratio(self, risk_agent, sample_portfolio_data):
        """Test Sharpe ratio calculation."""
        portfolio_returns = sample_portfolio_data.sum(axis=1)
        
        sharpe = risk_agent._calculate_sharpe_ratio(portfolio_returns, risk_free_rate=0.02)
        
        assert isinstance(sharpe, float)
        # Sharpe ratio can be positive or negative
    
    @pytest.mark.unit
    def test_beta_calculation(self, risk_agent, sample_portfolio_data):
        """Test beta calculation against market."""
        asset_returns = sample_portfolio_data['AAPL']
        market_returns = sample_portfolio_data.mean(axis=1)  # Use average as market proxy
        
        beta = risk_agent._calculate_beta(asset_returns, market_returns)
        
        assert isinstance(beta, float)
        assert beta > 0  # Generally expect positive beta
    
    @pytest.mark.unit
    def test_correlation_matrix(self, risk_agent, sample_portfolio_data):
        """Test correlation matrix calculation."""
        corr_matrix = risk_agent._calculate_correlation_matrix(sample_portfolio_data)
        
        assert isinstance(corr_matrix, pd.DataFrame)
        assert corr_matrix.shape == (5, 5)  # 5x5 for 5 assets
        assert np.allclose(np.diag(corr_matrix), 1.0)  # Diagonal should be 1
        assert (corr_matrix >= -1).all().all() and (corr_matrix <= 1).all().all()
    
    @pytest.mark.unit
    def test_portfolio_volatility(self, risk_agent, sample_portfolio_data, sample_positions):
        """Test portfolio volatility calculation."""
        weights = np.array([pos['weight'] for pos in sample_positions.values()])
        
        portfolio_vol = risk_agent._calculate_portfolio_volatility(
            sample_portfolio_data, weights
        )
        
        assert isinstance(portfolio_vol, float)
        assert portfolio_vol > 0
    
    @pytest.mark.unit
    def test_monte_carlo_simulation(self, risk_agent, sample_portfolio_data):
        """Test Monte Carlo simulation for risk estimation."""
        portfolio_returns = sample_portfolio_data.sum(axis=1)
        
        simulated_returns = risk_agent._monte_carlo_simulation(
            portfolio_returns, n_simulations=1000, time_horizon=5
        )
        
        assert isinstance(simulated_returns, np.ndarray)
        assert simulated_returns.shape == (1000, 5)  # 1000 simulations, 5 days
    
    @pytest.mark.unit
    def test_stress_testing(self, risk_agent, sample_portfolio_data, sample_positions):
        """Test stress testing scenarios."""
        scenarios = {
            'market_crash': {'AAPL': -0.2, 'GOOGL': -0.25, 'MSFT': -0.18, 'TSLA': -0.3, 'AMZN': -0.22},
            'sector_rotation': {'AAPL': 0.1, 'GOOGL': 0.05, 'MSFT': 0.08, 'TSLA': -0.15, 'AMZN': 0.03}
        }
        
        stress_results = risk_agent._stress_test_portfolio(sample_positions, scenarios)
        
        assert isinstance(stress_results, dict)
        assert 'market_crash' in stress_results
        assert 'sector_rotation' in stress_results
        
        for scenario, result in stress_results.items():
            assert 'portfolio_pnl' in result
            assert 'portfolio_return' in result
    
    @pytest.mark.unit
    def test_risk_attribution(self, risk_agent, sample_portfolio_data, sample_positions):
        """Test risk attribution analysis."""
        weights = np.array([pos['weight'] for pos in sample_positions.values()])
        
        risk_attribution = risk_agent._calculate_risk_attribution(
            sample_portfolio_data, weights
        )
        
        assert isinstance(risk_attribution, dict)
        assert 'individual_var' in risk_attribution
        assert 'marginal_var' in risk_attribution
        assert 'component_var' in risk_attribution
        
        # Check that component VaRs sum to total portfolio VaR (approximately)
        total_component_var = sum(risk_attribution['component_var'].values())
        portfolio_var = risk_agent._calculate_var_historical(
            (sample_portfolio_data * weights).sum(axis=1), confidence_level=0.95
        )
        
        assert abs(total_component_var - portfolio_var) < 0.001  # Small tolerance for numerical errors
    
    @pytest.mark.unit
    def test_liquidity_risk_assessment(self, risk_agent, sample_positions):
        """Test liquidity risk assessment."""
        # Add liquidity metrics to positions
        enhanced_positions = sample_positions.copy()
        for symbol in enhanced_positions:
            enhanced_positions[symbol]['avg_daily_volume'] = np.random.randint(1000000, 10000000)
            enhanced_positions[symbol]['bid_ask_spread'] = np.random.uniform(0.01, 0.05)
        
        liquidity_risk = risk_agent._assess_liquidity_risk(enhanced_positions)
        
        assert isinstance(liquidity_risk, dict)
        assert 'overall_liquidity_score' in liquidity_risk
        assert 'position_liquidity' in liquidity_risk
        assert 0 <= liquidity_risk['overall_liquidity_score'] <= 1
    
    @pytest.mark.unit
    def test_concentration_risk(self, risk_agent, sample_positions):
        """Test concentration risk measurement."""
        concentration_metrics = risk_agent._calculate_concentration_risk(sample_positions)
        
        assert isinstance(concentration_metrics, dict)
        assert 'herfindahl_index' in concentration_metrics
        assert 'max_weight' in concentration_metrics
        assert 'effective_number_positions' in concentration_metrics
        
        assert 0 <= concentration_metrics['herfindahl_index'] <= 1
        assert 0 <= concentration_metrics['max_weight'] <= 1
    
    @pytest.mark.unit
    def test_tail_risk_measures(self, risk_agent, sample_portfolio_data):
        """Test tail risk measures beyond VaR."""
        portfolio_returns = sample_portfolio_data.sum(axis=1)
        
        tail_measures = risk_agent._calculate_tail_risk_measures(portfolio_returns)
        
        assert isinstance(tail_measures, dict)
        assert 'skewness' in tail_measures
        assert 'kurtosis' in tail_measures
        assert 'tail_ratio' in tail_measures
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_process_with_mock_data(self, risk_agent, sample_inputs):
        """Test processing with mocked data."""
        with patch.object(risk_agent, '_fetch_market_data') as mock_fetch:
            mock_fetch.return_value = sample_portfolio_data
            
            result = await risk_agent.process(sample_inputs)
            
            assert isinstance(result, dict)
            assert 'risk_metrics' in result
            assert 'scenario_analysis' in result
            assert 'confidence' in result
    
    @pytest.mark.unit
    def test_input_validation(self, risk_agent):
        """Test input validation."""
        # Test missing portfolio
        invalid_inputs = {"confidence_level": 0.95}
        
        with pytest.raises(ValidationError):
            risk_agent._validate_inputs(invalid_inputs)
        
        # Test invalid confidence level
        invalid_confidence = {
            "portfolio": {"positions": {"AAPL": {"quantity": 100, "price": 150}}},
            "confidence_level": 1.5  # Invalid confidence level
        }
        
        with pytest.raises(ValidationError):
            risk_agent._validate_inputs(invalid_confidence)
    
    @pytest.mark.unit
    def test_portfolio_valuation(self, risk_agent, sample_positions):
        """Test portfolio valuation."""
        portfolio_value = risk_agent._calculate_portfolio_value(sample_positions)
        
        expected_value = sum(pos['quantity'] * pos['price'] for pos in sample_positions.values())
        
        assert isinstance(portfolio_value, float)
        assert abs(portfolio_value - expected_value) < 0.01
    
    @pytest.mark.unit
    def test_risk_budgeting(self, risk_agent, sample_portfolio_data, sample_positions):
        """Test risk budgeting allocation."""
        target_risk_budget = {
            'AAPL': 0.25, 'GOOGL': 0.20, 'MSFT': 0.20, 'TSLA': 0.20, 'AMZN': 0.15
        }
        
        optimal_weights = risk_agent._calculate_risk_budget_weights(
            sample_portfolio_data, target_risk_budget
        )
        
        assert isinstance(optimal_weights, dict)
        assert abs(sum(optimal_weights.values()) - 1.0) < 0.01  # Should sum to 1
        assert all(w >= 0 for w in optimal_weights.values())  # No short positions
    
    @pytest.mark.unit
    def test_factor_model_risk(self, risk_agent, sample_portfolio_data):
        """Test factor model-based risk decomposition."""
        # Create mock factor loadings
        factor_loadings = pd.DataFrame({
            'market': [1.2, 0.9, 1.1, 1.5, 1.0],
            'size': [0.3, -0.2, 0.1, 0.8, -0.1],
            'value': [-0.1, 0.4, 0.2, -0.3, 0.1]
        }, index=sample_portfolio_data.columns)
        
        # Mock factor returns
        factor_returns = pd.DataFrame({
            'market': np.random.normal(0, 0.01, len(sample_portfolio_data)),
            'size': np.random.normal(0, 0.005, len(sample_portfolio_data)),
            'value': np.random.normal(0, 0.003, len(sample_portfolio_data))
        }, index=sample_portfolio_data.index)
        
        factor_risk = risk_agent._calculate_factor_risk(
            factor_loadings, factor_returns, [0.2, 0.2, 0.2, 0.2, 0.2]
        )
        
        assert isinstance(factor_risk, dict)
        assert 'total_factor_risk' in factor_risk
        assert 'idiosyncratic_risk' in factor_risk
        assert 'factor_contributions' in factor_risk
    
    @pytest.mark.unit
    def test_dynamic_var_calculation(self, risk_agent, sample_portfolio_data):
        """Test dynamic VaR using GARCH model."""
        portfolio_returns = sample_portfolio_data.sum(axis=1)
        
        # Test with a subset of data for faster execution
        returns_subset = portfolio_returns.iloc[:100]
        
        dynamic_var = risk_agent._calculate_dynamic_var(returns_subset, confidence_level=0.95)
        
        assert isinstance(dynamic_var, pd.Series)
        assert len(dynamic_var) <= len(returns_subset)
        assert all(var_val < 0 for var_val in dynamic_var.dropna())
    
    @pytest.mark.integration
    async def test_real_time_risk_monitoring(self, risk_agent, sample_positions):
        """Test real-time risk monitoring."""
        # Mock real-time price feeds
        price_updates = {
            'AAPL': 155.0,  # 3.33% increase
            'GOOGL': 2400.0,  # 4% decrease
            'MSFT': 310.0,  # 3.33% increase
            'TSLA': 750.0,  # 6.25% decrease
            'AMZN': 3100.0  # 3.33% increase
        }
        
        updated_risk = risk_agent._update_risk_metrics_realtime(
            sample_positions, price_updates
        )
        
        assert isinstance(updated_risk, dict)
        assert 'current_portfolio_value' in updated_risk
        assert 'intraday_pnl' in updated_risk
        assert 'updated_var' in updated_risk
    
    @pytest.mark.slow
    async def test_comprehensive_risk_report(self, risk_agent, sample_inputs, sample_portfolio_data):
        """Test generation of comprehensive risk report."""
        with patch.object(risk_agent, '_fetch_market_data', return_value=sample_portfolio_data):
            comprehensive_report = await risk_agent._generate_comprehensive_report(sample_inputs)
            
            assert isinstance(comprehensive_report, dict)
            
            # Check all major sections are present
            expected_sections = [
                'executive_summary', 'var_analysis', 'stress_testing',
                'concentration_analysis', 'liquidity_analysis',
                'tail_risk_analysis', 'scenario_analysis', 'recommendations'
            ]
            
            for section in expected_sections:
                assert section in comprehensive_report
    
    @pytest.mark.unit
    def test_risk_limit_monitoring(self, risk_agent):
        """Test risk limit monitoring and alerting."""
        risk_limits = {
            'portfolio_var_limit': 0.05,
            'concentration_limit': 0.25,
            'sector_limit': 0.40,
            'max_drawdown_limit': 0.15
        }
        
        current_metrics = {
            'portfolio_var': 0.06,  # Exceeds limit
            'max_concentration': 0.30,  # Exceeds limit
            'sector_concentration': {'tech': 0.35},  # Within limit
            'current_drawdown': 0.12  # Within limit
        }
        
        limit_breaches = risk_agent._monitor_risk_limits(current_metrics, risk_limits)
        
        assert isinstance(limit_breaches, list)
        assert len(limit_breaches) == 2  # VaR and concentration limits breached
        
        for breach in limit_breaches:
            assert 'metric' in breach
            assert 'current_value' in breach
            assert 'limit' in breach
            assert 'severity' in breach