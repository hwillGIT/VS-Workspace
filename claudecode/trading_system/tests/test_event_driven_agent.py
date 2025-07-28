"""
Comprehensive tests for the Event-Driven Trading Agent.
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

from agents.strategies.event_driven.event_driven_agent import EventDrivenAgent
from core.base.exceptions import DataError, ValidationError


@pytest.fixture
def event_driven_agent():
    """Create an EventDrivenAgent instance for testing."""
    return EventDrivenAgent()


@pytest.fixture
def sample_event_data():
    """Create sample event data for testing."""
    events = [
        {
            'timestamp': datetime(2023, 6, 1, 9, 30),
            'event_type': 'earnings',
            'symbol': 'AAPL',
            'impact_score': 0.8,
            'sentiment': 0.6,
            'expected_move': 0.03
        },
        {
            'timestamp': datetime(2023, 6, 15, 14, 0),
            'event_type': 'fed_announcement',
            'symbol': 'SPY',
            'impact_score': 0.9,
            'sentiment': -0.3,
            'expected_move': -0.02
        },
        {
            'timestamp': datetime(2023, 7, 1, 10, 0),
            'event_type': 'merger',
            'symbol': 'MSFT',
            'impact_score': 0.7,
            'sentiment': 0.4,
            'expected_move': 0.05
        }
    ]
    
    return pd.DataFrame(events)


@pytest.fixture
def sample_price_data():
    """Create sample price data around events."""
    dates = pd.date_range(start='2023-05-01', end='2023-08-01', freq='H')
    
    # Generate price data with event impacts
    base_price = 100
    prices = [base_price]
    
    for i in range(1, len(dates)):
        # Add event impact at specific times
        if dates[i].date() == datetime(2023, 6, 1).date():
            shock = 0.03  # 3% move for earnings
        elif dates[i].date() == datetime(2023, 6, 15).date():
            shock = -0.02  # -2% move for Fed
        else:
            shock = 0
        
        noise = np.random.normal(0, 0.001)
        new_price = prices[-1] * (1 + shock + noise)
        prices.append(new_price)
    
    return pd.DataFrame({
        'timestamp': dates,
        'close': prices,
        'volume': np.random.randint(1000000, 5000000, len(dates))
    })


@pytest.fixture
def sample_inputs():
    """Sample inputs for event-driven strategy."""
    return {
        "symbols": ["AAPL", "GOOGL", "MSFT"],
        "event_types": ["earnings", "merger", "fed_announcement"],
        "lookback_window": 30,
        "event_impact_threshold": 0.5,
        "holding_period": 5,
        "risk_per_event": 0.02
    }


class TestEventDrivenAgent:
    """Test cases for EventDrivenAgent."""
    
    @pytest.mark.unit
    def test_agent_initialization(self, event_driven_agent):
        """Test agent initialization."""
        assert event_driven_agent.name == "EventDrivenAgent"
        assert hasattr(event_driven_agent, 'event_classifier')
        assert hasattr(event_driven_agent, 'impact_predictor')
    
    @pytest.mark.unit
    def test_event_classification(self, event_driven_agent, sample_event_data):
        """Test event classification and categorization."""
        classified_events = event_driven_agent._classify_events(sample_event_data)
        
        assert isinstance(classified_events, pd.DataFrame)
        assert 'category' in classified_events.columns
        assert 'priority' in classified_events.columns
        assert all(classified_events['priority'] >= 0)
    
    @pytest.mark.unit
    def test_event_impact_prediction(self, event_driven_agent, sample_event_data):
        """Test event impact prediction."""
        event = sample_event_data.iloc[0]
        
        predicted_impact = event_driven_agent._predict_event_impact(
            event_type=event['event_type'],
            symbol=event['symbol'],
            sentiment=event['sentiment']
        )
        
        assert isinstance(predicted_impact, dict)
        assert 'expected_return' in predicted_impact
        assert 'confidence' in predicted_impact
        assert 'duration' in predicted_impact
        assert -1 <= predicted_impact['expected_return'] <= 1
    
    @pytest.mark.unit
    def test_pre_event_positioning(self, event_driven_agent, sample_event_data):
        """Test pre-event positioning logic."""
        upcoming_event = sample_event_data.iloc[0]
        current_time = upcoming_event['timestamp'] - timedelta(days=1)
        
        position = event_driven_agent._calculate_pre_event_position(
            event=upcoming_event,
            current_time=current_time,
            max_position=0.1
        )
        
        assert isinstance(position, float)
        assert -0.1 <= position <= 0.1
    
    @pytest.mark.unit
    def test_post_event_analysis(self, event_driven_agent, sample_price_data):
        """Test post-event price analysis."""
        event_time = datetime(2023, 6, 1, 9, 30)
        
        post_event_metrics = event_driven_agent._analyze_post_event_movement(
            price_data=sample_price_data,
            event_time=event_time,
            window_hours=24
        )
        
        assert isinstance(post_event_metrics, dict)
        assert 'immediate_impact' in post_event_metrics
        assert 'drift' in post_event_metrics
        assert 'volatility' in post_event_metrics
    
    @pytest.mark.unit
    def test_earnings_event_strategy(self, event_driven_agent):
        """Test earnings event specific strategy."""
        earnings_data = {
            'symbol': 'AAPL',
            'expected_eps': 1.50,
            'actual_eps': 1.65,
            'revenue_surprise': 0.02,
            'guidance': 'positive'
        }
        
        signal = event_driven_agent._generate_earnings_signal(earnings_data)
        
        assert isinstance(signal, dict)
        assert 'direction' in signal
        assert 'strength' in signal
        assert signal['direction'] in ['long', 'short', 'neutral']
        assert 0 <= signal['strength'] <= 1
    
    @pytest.mark.unit
    def test_merger_arbitrage_strategy(self, event_driven_agent):
        """Test merger arbitrage positioning."""
        merger_data = {
            'target': 'ABC',
            'acquirer': 'XYZ',
            'offer_price': 50,
            'current_price': 48,
            'deal_probability': 0.8,
            'expected_close_date': datetime.now() + timedelta(days=90)
        }
        
        arb_position = event_driven_agent._calculate_merger_arb_position(merger_data)
        
        assert isinstance(arb_position, dict)
        assert 'target_position' in arb_position
        assert 'acquirer_position' in arb_position
        assert 'expected_return' in arb_position
    
    @pytest.mark.unit
    def test_macro_event_impact(self, event_driven_agent):
        """Test macro event impact assessment."""
        macro_event = {
            'event_type': 'fed_rate_decision',
            'expected_change': 0.25,
            'actual_change': 0.50,
            'market_surprise': True
        }
        
        market_impact = event_driven_agent._assess_macro_impact(macro_event)
        
        assert isinstance(market_impact, dict)
        assert 'equity_impact' in market_impact
        assert 'bond_impact' in market_impact
        assert 'dollar_impact' in market_impact
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_process_with_mock_data(self, event_driven_agent, sample_inputs, sample_event_data):
        """Test processing with mocked data."""
        with patch.object(event_driven_agent, '_fetch_events', return_value=sample_event_data):
            result = await event_driven_agent.process(sample_inputs)
            
            assert isinstance(result, dict)
            assert 'signals' in result
            assert 'positions' in result
            assert 'upcoming_events' in result
            assert 'risk_allocation' in result
            assert 'confidence' in result
    
    @pytest.mark.unit
    def test_event_clustering(self, event_driven_agent, sample_event_data):
        """Test event clustering for correlated events."""
        # Add more events for clustering
        additional_events = sample_event_data.copy()
        additional_events['timestamp'] = additional_events['timestamp'] + timedelta(hours=1)
        all_events = pd.concat([sample_event_data, additional_events])
        
        clusters = event_driven_agent._cluster_related_events(
            all_events,
            time_window_hours=24
        )
        
        assert isinstance(clusters, list)
        for cluster in clusters:
            assert isinstance(cluster, list)
            assert len(cluster) > 0
    
    @pytest.mark.unit
    def test_sentiment_analysis_integration(self, event_driven_agent):
        """Test sentiment analysis for event context."""
        news_data = {
            'headline': 'Company beats earnings expectations by 10%',
            'body': 'Strong revenue growth driven by new product launches...',
            'source': 'Reuters',
            'timestamp': datetime.now()
        }
        
        sentiment_score = event_driven_agent._analyze_news_sentiment(news_data)
        
        assert isinstance(sentiment_score, float)
        assert -1 <= sentiment_score <= 1
    
    @pytest.mark.unit
    def test_options_strategy_for_events(self, event_driven_agent):
        """Test options strategy generation for events."""
        event_data = {
            'symbol': 'AAPL',
            'event_type': 'earnings',
            'expected_move': 0.05,
            'iv_percentile': 80,
            'days_to_event': 3
        }
        
        options_strategy = event_driven_agent._generate_options_strategy(event_data)
        
        assert isinstance(options_strategy, dict)
        assert 'strategy_type' in options_strategy
        assert 'legs' in options_strategy
        assert options_strategy['strategy_type'] in ['straddle', 'strangle', 'iron_condor', 'butterfly']
    
    @pytest.mark.unit
    def test_risk_management_for_events(self, event_driven_agent):
        """Test risk management for event-driven trades."""
        positions = {
            'AAPL': {'size': 0.1, 'event_type': 'earnings', 'risk_score': 0.7},
            'MSFT': {'size': 0.05, 'event_type': 'merger', 'risk_score': 0.5},
            'SPY': {'size': 0.15, 'event_type': 'fed', 'risk_score': 0.9}
        }
        
        risk_adjusted = event_driven_agent._apply_event_risk_limits(
            positions,
            max_event_exposure=0.2,
            max_total_exposure=0.3
        )
        
        assert isinstance(risk_adjusted, dict)
        total_exposure = sum(pos['size'] for pos in risk_adjusted.values())
        assert total_exposure <= 0.3
    
    @pytest.mark.unit
    def test_event_timing_optimization(self, event_driven_agent):
        """Test optimal entry/exit timing for events."""
        event = {
            'timestamp': datetime(2023, 6, 15, 14, 0),
            'event_type': 'fed_announcement',
            'historical_pattern': 'pre_event_drift'
        }
        
        timing = event_driven_agent._optimize_event_timing(
            event,
            lookback_days=30
        )
        
        assert 'entry_time' in timing
        assert 'exit_time' in timing
        assert timing['entry_time'] < event['timestamp']
        assert timing['exit_time'] > event['timestamp']
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_historical_event_backtesting(self, event_driven_agent):
        """Test backtesting of event-driven strategies."""
        backtest_inputs = {
            "symbols": ["AAPL", "GOOGL"],
            "event_types": ["earnings"],
            "start_date": datetime(2022, 1, 1),
            "end_date": datetime(2023, 1, 1),
            "event_impact_threshold": 0.5
        }
        
        # Mock historical events and prices
        mock_events = self._generate_mock_historical_events()
        mock_prices = self._generate_mock_price_data_with_events()
        
        with patch.object(event_driven_agent, '_fetch_historical_events', return_value=mock_events):
            with patch.object(event_driven_agent, '_fetch_price_data', return_value=mock_prices):
                backtest_results = await event_driven_agent.backtest(backtest_inputs)
                
                assert 'returns' in backtest_results
                assert 'event_hit_rate' in backtest_results
                assert 'average_event_return' in backtest_results
                assert 'event_sharpe' in backtest_results
    
    @pytest.mark.unit
    def test_event_correlation_analysis(self, event_driven_agent):
        """Test correlation analysis between different event types."""
        historical_events = pd.DataFrame({
            'event_type': ['earnings', 'fed', 'earnings', 'merger', 'fed'],
            'impact': [0.03, -0.02, 0.02, 0.05, -0.01],
            'symbol': ['AAPL', 'SPY', 'GOOGL', 'MSFT', 'SPY']
        })
        
        correlations = event_driven_agent._analyze_event_correlations(historical_events)
        
        assert isinstance(correlations, pd.DataFrame)
        assert correlations.shape[0] == correlations.shape[1]
        assert all(correlations.diagonal() == 1.0)
    
    def _generate_mock_historical_events(self):
        """Helper to generate mock historical events."""
        events = []
        symbols = ['AAPL', 'GOOGL']
        
        for month in range(1, 13):
            for symbol in symbols:
                events.append({
                    'timestamp': datetime(2022, month, 15, 9, 30),
                    'event_type': 'earnings',
                    'symbol': symbol,
                    'impact_score': np.random.uniform(0.5, 0.9),
                    'actual_impact': np.random.uniform(-0.05, 0.05)
                })
        
        return pd.DataFrame(events)
    
    def _generate_mock_price_data_with_events(self):
        """Helper to generate price data with event impacts."""
        dates = pd.date_range(start='2022-01-01', end='2023-01-01', freq='H')
        prices = [100]
        
        for i in range(1, len(dates)):
            # Add event impact on the 15th of each month
            if dates[i].day == 15 and dates[i].hour == 10:
                impact = np.random.uniform(-0.03, 0.03)
            else:
                impact = 0
            
            noise = np.random.normal(0, 0.0005)
            new_price = prices[-1] * (1 + impact + noise)
            prices.append(new_price)
        
        return pd.DataFrame({
            'timestamp': dates,
            'close': prices
        })