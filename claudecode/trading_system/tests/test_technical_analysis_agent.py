"""
Comprehensive tests for the Technical Analysis Agent.
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

from agents.feature_engineering.technical_analysis_agent import TechnicalAnalysisAgent
from core.base.exceptions import DataError, ValidationError


@pytest.fixture
def tech_agent():
    """Create a TechnicalAnalysisAgent instance for testing."""
    return TechnicalAnalysisAgent()


@pytest.fixture
def sample_price_data():
    """Create sample price data for testing."""
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)  # For reproducible tests
    
    # Generate realistic price data
    base_price = 100
    price_changes = np.random.normal(0, 0.02, len(dates))
    prices = [base_price]
    
    for change in price_changes[1:]:
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 1))  # Ensure positive prices
    
    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': np.random.randint(1000, 10000, len(dates))
    })
    
    return data


@pytest.fixture
def sample_inputs():
    """Sample inputs for technical analysis."""
    return {
        "symbols": ["AAPL", "GOOGL"],
        "start_date": datetime.now() - timedelta(days=365),
        "end_date": datetime.now(),
        "indicators": ["sma", "rsi", "macd", "bollinger"]
    }


class TestTechnicalAnalysisAgent:
    """Test cases for TechnicalAnalysisAgent."""
    
    @pytest.mark.unit
    def test_agent_initialization(self, tech_agent):
        """Test agent initialization."""
        assert tech_agent.name == "TechnicalAnalysisAgent"
        assert tech_agent.config_section == "technical_analysis"
        assert hasattr(tech_agent, 'indicators')
    
    @pytest.mark.unit
    def test_calculate_sma(self, tech_agent, sample_price_data):
        """Test Simple Moving Average calculation."""
        sma = tech_agent._calculate_sma(sample_price_data['close'], window=20)
        
        assert isinstance(sma, pd.Series)
        assert len(sma) == len(sample_price_data)
        assert not sma.iloc[-1].isna()  # Last value should not be NaN
        assert sma.iloc[0].isna()  # First values should be NaN
    
    @pytest.mark.unit
    def test_calculate_rsi(self, tech_agent, sample_price_data):
        """Test RSI calculation."""
        rsi = tech_agent._calculate_rsi(sample_price_data['close'], window=14)
        
        assert isinstance(rsi, pd.Series)
        assert len(rsi) == len(sample_price_data)
        assert 0 <= rsi.dropna().max() <= 100
        assert 0 <= rsi.dropna().min() <= 100
    
    @pytest.mark.unit
    def test_calculate_macd(self, tech_agent, sample_price_data):
        """Test MACD calculation."""
        macd_line, signal_line, histogram = tech_agent._calculate_macd(sample_price_data['close'])
        
        assert isinstance(macd_line, pd.Series)
        assert isinstance(signal_line, pd.Series)
        assert isinstance(histogram, pd.Series)
        assert len(macd_line) == len(sample_price_data)
    
    @pytest.mark.unit
    def test_calculate_bollinger_bands(self, tech_agent, sample_price_data):
        """Test Bollinger Bands calculation."""
        upper, middle, lower = tech_agent._calculate_bollinger_bands(
            sample_price_data['close'], window=20, std_dev=2
        )
        
        assert isinstance(upper, pd.Series)
        assert isinstance(middle, pd.Series)
        assert isinstance(lower, pd.Series)
        
        # Upper band should be above middle, middle above lower
        valid_data = upper.dropna()
        middle_valid = middle.dropna()
        lower_valid = lower.dropna()
        
        assert (upper >= middle).all()
        assert (middle >= lower).all()
    
    @pytest.mark.unit
    def test_detect_patterns(self, tech_agent, sample_price_data):
        """Test pattern detection."""
        patterns = tech_agent._detect_patterns(sample_price_data)
        
        assert isinstance(patterns, dict)
        assert 'support_resistance' in patterns
        assert 'trend_lines' in patterns
        assert 'chart_patterns' in patterns
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_process_with_mock_data(self, tech_agent, sample_inputs, sample_price_data):
        """Test processing with mocked data."""
        with patch.object(tech_agent, '_fetch_price_data', return_value=sample_price_data):
            result = await tech_agent.process(sample_inputs)
            
            assert isinstance(result, dict)
            assert 'indicators' in result
            assert 'patterns' in result
            assert 'signals' in result
            assert 'confidence' in result
    
    @pytest.mark.unit
    def test_input_validation(self, tech_agent):
        """Test input validation."""
        # Test missing required fields
        invalid_inputs = {"symbols": ["AAPL"]}  # Missing dates
        
        with pytest.raises(ValidationError):
            tech_agent._validate_inputs(invalid_inputs)
        
        # Test invalid date range
        invalid_dates = {
            "symbols": ["AAPL"],
            "start_date": datetime.now(),
            "end_date": datetime.now() - timedelta(days=1)  # End before start
        }
        
        with pytest.raises(ValidationError):
            tech_agent._validate_inputs(invalid_dates)
    
    @pytest.mark.unit
    def test_empty_data_handling(self, tech_agent):
        """Test handling of empty data."""
        empty_data = pd.DataFrame()
        
        with pytest.raises(DataError):
            tech_agent._calculate_sma(empty_data.get('close', pd.Series()), window=20)
    
    @pytest.mark.unit
    def test_generate_signals(self, tech_agent, sample_price_data):
        """Test signal generation."""
        # Create mock indicators
        indicators = {
            'sma_20': tech_agent._calculate_sma(sample_price_data['close'], 20),
            'rsi': tech_agent._calculate_rsi(sample_price_data['close'], 14),
            'macd_line': tech_agent._calculate_macd(sample_price_data['close'])[0]
        }
        
        signals = tech_agent._generate_signals(sample_price_data, indicators)
        
        assert isinstance(signals, dict)
        assert 'buy_signals' in signals
        assert 'sell_signals' in signals
        assert 'strength' in signals
    
    @pytest.mark.unit
    def test_confidence_calculation(self, tech_agent):
        """Test confidence score calculation."""
        # Mock indicators with various signal strengths
        indicators = {
            'rsi': pd.Series([30, 70, 50]),  # Oversold, overbought, neutral
            'sma_20': pd.Series([100, 105, 102]),
            'price': pd.Series([98, 108, 103])
        }
        
        confidence = tech_agent._calculate_confidence(indicators)
        
        assert isinstance(confidence, float)
        assert 0 <= confidence <= 1
    
    @pytest.mark.unit
    def test_momentum_indicators(self, tech_agent, sample_price_data):
        """Test momentum indicators calculation."""
        momentum = tech_agent._calculate_momentum_indicators(sample_price_data)
        
        assert isinstance(momentum, dict)
        assert 'rsi' in momentum
        assert 'stoch' in momentum
        assert 'williams_r' in momentum
    
    @pytest.mark.unit
    def test_trend_indicators(self, tech_agent, sample_price_data):
        """Test trend indicators calculation."""
        trend = tech_agent._calculate_trend_indicators(sample_price_data)
        
        assert isinstance(trend, dict)
        assert 'sma' in trend
        assert 'ema' in trend
        assert 'adx' in trend
    
    @pytest.mark.unit
    def test_volatility_indicators(self, tech_agent, sample_price_data):
        """Test volatility indicators calculation."""
        volatility = tech_agent._calculate_volatility_indicators(sample_price_data)
        
        assert isinstance(volatility, dict)
        assert 'bollinger' in volatility
        assert 'atr' in volatility
        assert 'keltner' in volatility
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_multiple_symbols_processing(self, tech_agent, sample_price_data):
        """Test processing multiple symbols."""
        symbols = ["AAPL", "GOOGL", "MSFT"]
        
        with patch.object(tech_agent, '_fetch_price_data', return_value=sample_price_data):
            results = []
            for symbol in symbols:
                inputs = {
                    "symbols": [symbol],
                    "start_date": datetime.now() - timedelta(days=90),
                    "end_date": datetime.now()
                }
                result = await tech_agent.process(inputs)
                results.append(result)
            
            assert len(results) == 3
            for result in results:
                assert 'indicators' in result
                assert 'confidence' in result
    
    @pytest.mark.unit
    def test_error_handling_invalid_indicator(self, tech_agent, sample_price_data):
        """Test error handling for invalid indicators."""
        with pytest.raises(ValueError):
            tech_agent._calculate_sma(sample_price_data['close'], window=-1)
    
    @pytest.mark.unit
    def test_data_quality_validation(self, tech_agent):
        """Test data quality validation."""
        # Test data with missing values
        data_with_gaps = pd.DataFrame({
            'close': [100, np.nan, 102, 103, np.nan],
            'volume': [1000, 1100, np.nan, 1300, 1400]
        })
        
        quality_score = tech_agent._validate_data_quality(data_with_gaps)
        assert isinstance(quality_score, float)
        assert 0 <= quality_score <= 1
    
    @pytest.mark.integration
    async def test_end_to_end_processing(self, tech_agent):
        """Test complete end-to-end processing."""
        inputs = {
            "symbols": ["AAPL"],
            "start_date": datetime.now() - timedelta(days=30),
            "end_date": datetime.now(),
            "indicators": ["sma", "rsi", "macd"]
        }
        
        # Mock the data fetching to avoid external dependencies
        sample_data = pd.DataFrame({
            'timestamp': pd.date_range(start=inputs['start_date'], end=inputs['end_date'], freq='D'),
            'open': np.random.uniform(90, 110, 31),
            'high': np.random.uniform(95, 115, 31),
            'low': np.random.uniform(85, 105, 31),
            'close': np.random.uniform(90, 110, 31),
            'volume': np.random.randint(1000, 10000, 31)
        })
        
        with patch.object(tech_agent, '_fetch_price_data', return_value=sample_data):
            result = await tech_agent.process(inputs)
            
            # Validate complete result structure
            assert isinstance(result, dict)
            assert 'indicators' in result
            assert 'patterns' in result
            assert 'signals' in result
            assert 'confidence' in result
            assert 'metadata' in result
            
            # Validate indicators
            indicators = result['indicators']
            assert 'sma' in indicators
            assert 'rsi' in indicators
            assert 'macd' in indicators
            
            # Validate confidence is reasonable
            assert 0 <= result['confidence'] <= 1