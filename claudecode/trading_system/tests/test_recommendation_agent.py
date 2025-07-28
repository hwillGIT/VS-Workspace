"""
Comprehensive tests for the Recommendation Agent.
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

from agents.output.recommendation_agent import RecommendationAgent
from core.base.exceptions import RecommendationError, ValidationError


@pytest.fixture
def rec_agent():
    """Create a RecommendationAgent instance for testing."""
    return RecommendationAgent()


@pytest.fixture
def sample_analysis_results():
    """Create sample analysis results from other agents."""
    return {
        'technical_analysis': {
            'signals': {'buy': ['AAPL'], 'sell': ['TSLA'], 'hold': ['GOOGL']},
            'indicators': {
                'AAPL': {'rsi': 30, 'macd': 0.5, 'sma_trend': 'bullish'},
                'TSLA': {'rsi': 75, 'macd': -0.3, 'sma_trend': 'bearish'},
                'GOOGL': {'rsi': 55, 'macd': 0.1, 'sma_trend': 'neutral'}
            },
            'confidence': 0.75
        },
        'ml_predictions': {
            'predictions': {
                'AAPL': {'return_forecast': 0.05, 'confidence': 0.8},
                'TSLA': {'return_forecast': -0.03, 'confidence': 0.7},
                'GOOGL': {'return_forecast': 0.01, 'confidence': 0.6}
            },
            'model_performance': {'accuracy': 0.68, 'sharpe': 1.2}
        },
        'risk_analysis': {
            'portfolio_var': 0.02,
            'individual_vars': {'AAPL': 0.015, 'TSLA': 0.025, 'GOOGL': 0.012},
            'correlations': {
                ('AAPL', 'GOOGL'): 0.3,
                ('AAPL', 'TSLA'): 0.2,
                ('GOOGL', 'TSLA'): 0.4
            },
            'risk_adjusted_returns': {
                'AAPL': 0.8, 'TSLA': -0.2, 'GOOGL': 0.3
            }
        },
        'fundamental_analysis': {
            'valuations': {
                'AAPL': 'fair_value',
                'TSLA': 'overvalued',
                'GOOGL': 'undervalued'
            },
            'financial_health': {
                'AAPL': 'strong',
                'TSLA': 'moderate',
                'GOOGL': 'strong'
            }
        }
    }


@pytest.fixture
def sample_portfolio():
    """Create sample portfolio data."""
    return {
        'AAPL': {'quantity': 100, 'avg_cost': 145.0, 'current_price': 150.0},
        'TSLA': {'quantity': 50, 'avg_cost': 800.0, 'current_price': 750.0},
        'GOOGL': {'quantity': 25, 'avg_cost': 2400.0, 'current_price': 2450.0}
    }


@pytest.fixture
def sample_user_preferences():
    """Create sample user preferences."""
    return {
        'risk_tolerance': 'moderate',
        'investment_horizon': 'medium_term',  # 1-3 years
        'preferred_sectors': ['technology', 'healthcare'],
        'esg_preferences': True,
        'max_position_size': 0.2,  # 20% max position
        'rebalancing_frequency': 'monthly',
        'target_allocation': {
            'equities': 0.7,
            'bonds': 0.2,
            'alternatives': 0.1
        }
    }


@pytest.fixture
def sample_inputs():
    """Sample inputs for recommendation generation."""
    return {
        "portfolio": {
            'AAPL': {'quantity': 100, 'avg_cost': 145.0},
            'GOOGL': {'quantity': 25, 'avg_cost': 2400.0}
        },
        "analysis_results": {},
        "user_preferences": {
            'risk_tolerance': 'moderate',
            'investment_horizon': 'medium_term'
        },
        "market_context": {
            'market_regime': 'normal',
            'volatility_level': 'moderate'
        }
    }


class TestRecommendationAgent:
    """Test cases for RecommendationAgent."""
    
    @pytest.mark.unit
    def test_agent_initialization(self, rec_agent):
        """Test agent initialization."""
        assert rec_agent.name == "RecommendationAgent"
        assert rec_agent.config_section == "recommendation"
        assert hasattr(rec_agent, 'recommendation_engine')
    
    @pytest.mark.unit
    def test_signal_aggregation(self, rec_agent, sample_analysis_results):
        """Test aggregation of signals from multiple sources."""
        aggregated_signals = rec_agent._aggregate_signals(sample_analysis_results)
        
        assert isinstance(aggregated_signals, dict)
        assert 'AAPL' in aggregated_signals
        assert 'TSLA' in aggregated_signals
        assert 'GOOGL' in aggregated_signals
        
        for symbol, signal in aggregated_signals.items():
            assert 'action' in signal  # buy/sell/hold
            assert 'strength' in signal  # signal strength
            assert 'confidence' in signal  # overall confidence
    
    @pytest.mark.unit
    def test_risk_adjusted_scoring(self, rec_agent, sample_analysis_results):
        """Test risk-adjusted scoring of recommendations."""
        risk_adjusted_scores = rec_agent._calculate_risk_adjusted_scores(
            sample_analysis_results
        )
        
        assert isinstance(risk_adjusted_scores, dict)
        
        for symbol, score in risk_adjusted_scores.items():
            assert isinstance(score, float)
            assert -1 <= score <= 1  # Normalized score
    
    @pytest.mark.unit
    def test_position_sizing(self, rec_agent, sample_portfolio, sample_user_preferences):
        """Test position sizing recommendations."""
        target_positions = {
            'AAPL': 'increase',
            'TSLA': 'decrease',
            'GOOGL': 'hold',
            'MSFT': 'new'  # New position
        }
        
        position_sizes = rec_agent._calculate_position_sizes(
            target_positions, sample_portfolio, sample_user_preferences
        )
        
        assert isinstance(position_sizes, dict)
        
        for symbol, size_info in position_sizes.items():
            assert 'target_weight' in size_info
            assert 'current_weight' in size_info or symbol == 'MSFT'  # New position
            assert 'action' in size_info
            assert 'shares_to_trade' in size_info
    
    @pytest.mark.unit
    def test_diversification_analysis(self, rec_agent, sample_portfolio):
        """Test portfolio diversification analysis."""
        # Add sector information
        sector_map = {
            'AAPL': 'technology',
            'TSLA': 'automotive',
            'GOOGL': 'technology'
        }
        
        diversification = rec_agent._analyze_diversification(sample_portfolio, sector_map)
        
        assert isinstance(diversification, dict)
        assert 'sector_concentration' in diversification
        assert 'diversification_score' in diversification
        assert 'recommendations' in diversification
        
        assert 0 <= diversification['diversification_score'] <= 1
    
    @pytest.mark.unit
    def test_timing_analysis(self, rec_agent):
        """Test market timing analysis."""
        market_indicators = {
            'vix': 18.5,  # Moderate volatility
            'yield_curve': 0.02,  # Normal curve
            'market_trend': 'upward',
            'momentum': 0.05
        }
        
        timing_score = rec_agent._analyze_market_timing(market_indicators)
        
        assert isinstance(timing_score, dict)
        assert 'overall_timing_score' in timing_score
        assert 'factors' in timing_score
        assert -1 <= timing_score['overall_timing_score'] <= 1
    
    @pytest.mark.unit
    def test_rebalancing_recommendations(self, rec_agent, sample_portfolio, sample_user_preferences):
        """Test portfolio rebalancing recommendations."""
        target_allocation = {
            'AAPL': 0.4,
            'TSLA': 0.2,
            'GOOGL': 0.3,
            'MSFT': 0.1
        }
        
        rebalancing_plan = rec_agent._generate_rebalancing_plan(
            sample_portfolio, target_allocation, sample_user_preferences
        )
        
        assert isinstance(rebalancing_plan, dict)
        assert 'trades' in rebalancing_plan
        assert 'costs' in rebalancing_plan
        assert 'expected_improvement' in rebalancing_plan
        
        # Check that weights approximately sum to 1
        total_target_weight = sum(target_allocation.values())
        assert abs(total_target_weight - 1.0) < 0.01
    
    @pytest.mark.unit
    def test_recommendation_filtering(self, rec_agent, sample_user_preferences):
        """Test filtering recommendations based on user preferences."""
        raw_recommendations = [
            {'symbol': 'AAPL', 'action': 'buy', 'sector': 'technology', 'esg_score': 85},
            {'symbol': 'XOM', 'action': 'buy', 'sector': 'energy', 'esg_score': 20},
            {'symbol': 'JNJ', 'action': 'buy', 'sector': 'healthcare', 'esg_score': 90},
            {'symbol': 'TSLA', 'action': 'sell', 'sector': 'automotive', 'esg_score': 75},
        ]
        
        filtered_recs = rec_agent._filter_recommendations(
            raw_recommendations, sample_user_preferences
        )
        
        assert isinstance(filtered_recs, list)
        assert len(filtered_recs) <= len(raw_recommendations)
        
        # Check ESG filtering
        if sample_user_preferences['esg_preferences']:
            for rec in filtered_recs:
                if 'esg_score' in rec:
                    assert rec['esg_score'] >= 70  # Assume 70 is ESG threshold
    
    @pytest.mark.unit
    def test_confidence_calculation(self, rec_agent, sample_analysis_results):
        """Test overall confidence calculation."""
        confidence_score = rec_agent._calculate_overall_confidence(sample_analysis_results)
        
        assert isinstance(confidence_score, float)
        assert 0 <= confidence_score <= 1
    
    @pytest.mark.unit
    def test_risk_budget_allocation(self, rec_agent, sample_analysis_results):
        """Test risk budget-based allocation."""
        risk_budget = 0.15  # 15% portfolio risk budget
        
        allocations = rec_agent._allocate_risk_budget(
            sample_analysis_results['risk_analysis'], risk_budget
        )
        
        assert isinstance(allocations, dict)
        
        for symbol, allocation in allocations.items():
            assert 'weight' in allocation
            assert 'risk_contribution' in allocation
            assert allocation['weight'] >= 0
    
    @pytest.mark.unit
    def test_scenario_based_recommendations(self, rec_agent):
        """Test scenario-based recommendation adjustments."""
        scenarios = {
            'bull_market': {'probability': 0.3, 'equity_return': 0.15},
            'bear_market': {'probability': 0.2, 'equity_return': -0.1},
            'normal_market': {'probability': 0.5, 'equity_return': 0.08}
        }
        
        base_recommendations = {
            'AAPL': {'action': 'buy', 'weight': 0.3, 'expected_return': 0.12},
            'TSLA': {'action': 'hold', 'weight': 0.2, 'expected_return': 0.05}
        }
        
        scenario_adjusted = rec_agent._adjust_for_scenarios(
            base_recommendations, scenarios
        )
        
        assert isinstance(scenario_adjusted, dict)
        
        for symbol, rec in scenario_adjusted.items():
            assert 'scenario_adjusted_return' in rec
            assert 'scenario_risk' in rec
    
    @pytest.mark.unit
    def test_tax_optimization(self, rec_agent, sample_portfolio):
        """Test tax-optimized recommendations."""
        # Add tax lot information
        tax_lots = {
            'AAPL': [
                {'quantity': 50, 'cost_basis': 140, 'purchase_date': '2023-01-15'},
                {'quantity': 50, 'cost_basis': 150, 'purchase_date': '2023-06-01'}
            ],
            'TSLA': [
                {'quantity': 50, 'cost_basis': 800, 'purchase_date': '2023-03-01'}
            ]
        }
        
        tax_optimized = rec_agent._optimize_for_taxes(
            sample_portfolio, tax_lots, current_date=datetime(2023, 12, 1)
        )
        
        assert isinstance(tax_optimized, dict)
        assert 'tax_loss_harvest' in tax_optimized
        assert 'wash_sale_warnings' in tax_optimized
        assert 'ltcg_optimization' in tax_optimized
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_process_with_complete_data(self, rec_agent, sample_inputs, sample_analysis_results):
        """Test processing with complete analysis data."""
        sample_inputs['analysis_results'] = sample_analysis_results
        
        result = await rec_agent.process(sample_inputs)
        
        assert isinstance(result, dict)
        assert 'recommendations' in result
        assert 'portfolio_analysis' in result
        assert 'risk_assessment' in result
        assert 'confidence' in result
        assert 'execution_plan' in result
    
    @pytest.mark.unit
    def test_input_validation(self, rec_agent):
        """Test input validation."""
        # Test missing portfolio
        invalid_inputs = {"user_preferences": {"risk_tolerance": "moderate"}}
        
        with pytest.raises(ValidationError):
            rec_agent._validate_inputs(invalid_inputs)
        
        # Test invalid risk tolerance
        invalid_risk = {
            "portfolio": {"AAPL": {"quantity": 100}},
            "user_preferences": {"risk_tolerance": "invalid_level"}
        }
        
        with pytest.raises(ValidationError):
            rec_agent._validate_inputs(invalid_risk)
    
    @pytest.mark.unit
    def test_execution_cost_estimation(self, rec_agent):
        """Test execution cost estimation."""
        trades = [
            {'symbol': 'AAPL', 'action': 'buy', 'quantity': 100, 'price': 150.0},
            {'symbol': 'TSLA', 'action': 'sell', 'quantity': 25, 'price': 750.0},
            {'symbol': 'GOOGL', 'action': 'buy', 'quantity': 10, 'price': 2450.0}
        ]
        
        execution_costs = rec_agent._estimate_execution_costs(trades)
        
        assert isinstance(execution_costs, dict)
        assert 'total_cost' in execution_costs
        assert 'breakdown' in execution_costs
        
        for trade in trades:
            symbol = trade['symbol']
            assert symbol in execution_costs['breakdown']
            assert 'commission' in execution_costs['breakdown'][symbol]
            assert 'spread_cost' in execution_costs['breakdown'][symbol]
    
    @pytest.mark.unit
    def test_performance_attribution(self, rec_agent):
        """Test performance attribution of recommendations."""
        historical_recommendations = [
            {
                'date': '2023-01-01',
                'symbol': 'AAPL',
                'action': 'buy',
                'price': 140.0,
                'confidence': 0.8
            },
            {
                'date': '2023-02-01',
                'symbol': 'TSLA',
                'action': 'sell',
                'price': 850.0,
                'confidence': 0.7
            }
        ]
        
        current_prices = {'AAPL': 155.0, 'TSLA': 750.0}
        
        attribution = rec_agent._analyze_recommendation_performance(
            historical_recommendations, current_prices
        )
        
        assert isinstance(attribution, dict)
        assert 'overall_performance' in attribution
        assert 'individual_performance' in attribution
        assert 'accuracy_rate' in attribution
    
    @pytest.mark.unit
    def test_sentiment_integration(self, rec_agent):
        """Test integration of market sentiment data."""
        sentiment_data = {
            'AAPL': {'news_sentiment': 0.6, 'social_sentiment': 0.4, 'analyst_sentiment': 0.7},
            'TSLA': {'news_sentiment': -0.2, 'social_sentiment': 0.8, 'analyst_sentiment': 0.1},
            'GOOGL': {'news_sentiment': 0.3, 'social_sentiment': 0.2, 'analyst_sentiment': 0.5}
        }
        
        sentiment_scores = rec_agent._integrate_sentiment(sentiment_data)
        
        assert isinstance(sentiment_scores, dict)
        
        for symbol, score in sentiment_scores.items():
            assert 'composite_sentiment' in score
            assert -1 <= score['composite_sentiment'] <= 1
    
    @pytest.mark.unit
    def test_recommendation_explanation(self, rec_agent):
        """Test generation of recommendation explanations."""
        recommendation = {
            'symbol': 'AAPL',
            'action': 'buy',
            'target_weight': 0.25,
            'confidence': 0.8,
            'factors': {
                'technical': 0.7,
                'fundamental': 0.6,
                'sentiment': 0.5,
                'risk_adjusted': 0.8
            }
        }
        
        explanation = rec_agent._generate_explanation(recommendation)
        
        assert isinstance(explanation, dict)
        assert 'summary' in explanation
        assert 'key_factors' in explanation
        assert 'risks' in explanation
        assert 'reasoning' in explanation
    
    @pytest.mark.integration
    async def test_dynamic_rebalancing(self, rec_agent, sample_portfolio):
        """Test dynamic rebalancing based on market conditions."""
        market_conditions = {
            'volatility_regime': 'high',
            'trend': 'downward',
            'correlation_increase': True
        }
        
        dynamic_allocation = await rec_agent._dynamic_rebalancing(
            sample_portfolio, market_conditions
        )
        
        assert isinstance(dynamic_allocation, dict)
        assert 'adjusted_weights' in dynamic_allocation
        assert 'rationale' in dynamic_allocation
        assert 'risk_impact' in dynamic_allocation
    
    @pytest.mark.slow
    async def test_comprehensive_recommendation_generation(self, rec_agent):
        """Test comprehensive recommendation generation."""
        comprehensive_inputs = {
            "portfolio": {
                'AAPL': {'quantity': 100, 'avg_cost': 145.0, 'current_price': 150.0},
                'GOOGL': {'quantity': 25, 'avg_cost': 2400.0, 'current_price': 2450.0}
            },
            "analysis_results": {
                'technical_analysis': {
                    'signals': {'buy': ['MSFT'], 'sell': [], 'hold': ['AAPL', 'GOOGL']},
                    'confidence': 0.7
                },
                'ml_predictions': {
                    'predictions': {
                        'AAPL': {'return_forecast': 0.03, 'confidence': 0.6},
                        'GOOGL': {'return_forecast': 0.02, 'confidence': 0.7},
                        'MSFT': {'return_forecast': 0.06, 'confidence': 0.8}
                    }
                },
                'risk_analysis': {
                    'portfolio_var': 0.018,
                    'individual_vars': {'AAPL': 0.015, 'GOOGL': 0.012, 'MSFT': 0.020}
                }
            },
            "user_preferences": {
                'risk_tolerance': 'moderate',
                'investment_horizon': 'long_term',
                'max_position_size': 0.3,
                'rebalancing_frequency': 'quarterly'
            },
            "market_context": {
                'market_regime': 'bull',
                'volatility_level': 'low',
                'economic_indicators': 'positive'
            }
        }
        
        comprehensive_result = await rec_agent.process(comprehensive_inputs)
        
        # Validate comprehensive result structure
        assert isinstance(comprehensive_result, dict)
        
        expected_sections = [
            'recommendations', 'portfolio_analysis', 'risk_assessment',
            'execution_plan', 'performance_projections', 'confidence'
        ]
        
        for section in expected_sections:
            assert section in comprehensive_result
        
        # Validate recommendations structure
        recommendations = comprehensive_result['recommendations']
        assert isinstance(recommendations, list)
        
        for rec in recommendations:
            assert 'symbol' in rec
            assert 'action' in rec
            assert 'rationale' in rec
            assert 'confidence' in rec