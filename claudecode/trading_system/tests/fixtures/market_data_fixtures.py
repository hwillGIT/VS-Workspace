"""
Market Data Fixtures and Mocks for Testing.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json


class MarketDataGenerator:
    """Generate realistic market data for testing."""
    
    def __init__(self, seed: int = 42):
        """Initialize with random seed for reproducible results."""
        np.random.seed(seed)
        self.seed = seed
    
    def generate_stock_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        initial_price: float = 100.0,
        volatility: float = 0.02,
        trend: float = 0.0005,
        freq: str = 'D'
    ) -> pd.DataFrame:
        """Generate realistic stock price data."""
        dates = pd.date_range(start=start_date, end=end_date, freq=freq)
        n_periods = len(dates)
        
        # Generate returns with trend and volatility
        returns = np.random.normal(trend, volatility, n_periods)
        
        # Add some autocorrelation to make it more realistic
        for i in range(1, n_periods):
            returns[i] += 0.1 * returns[i-1]
        
        # Calculate prices
        prices = [initial_price]
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        
        # Generate OHLC data
        opens = prices[:-1]
        closes = prices[1:]
        
        # High and low based on close with some noise
        highs = [c * (1 + abs(np.random.normal(0, 0.005))) for c in closes]
        lows = [c * (1 - abs(np.random.normal(0, 0.005))) for c in closes]
        
        # Ensure high >= close >= low and high >= open >= low
        for i in range(len(closes)):
            highs[i] = max(highs[i], closes[i], opens[i])
            lows[i] = min(lows[i], closes[i], opens[i])
        
        # Volume with higher volume on large price moves
        base_volume = 1000000
        volume_multiplier = 1 + 2 * np.abs(returns)
        volumes = (base_volume * volume_multiplier).astype(int)
        
        return pd.DataFrame({
            'timestamp': dates,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes,
            'returns': returns
        })
    
    def generate_options_chain(
        self,
        underlying_symbol: str,
        underlying_price: float,
        expiry_dates: List[datetime],
        strike_range: tuple = (0.8, 1.2),
        strike_step: float = 5.0
    ) -> pd.DataFrame:
        """Generate realistic options chain data."""
        options_data = []
        
        for expiry in expiry_dates:
            days_to_expiry = (expiry - datetime.now()).days
            time_to_expiry = days_to_expiry / 365.0
            
            # Generate strikes around underlying price
            min_strike = underlying_price * strike_range[0]
            max_strike = underlying_price * strike_range[1]
            strikes = np.arange(
                int(min_strike / strike_step) * strike_step,
                int(max_strike / strike_step + 1) * strike_step,
                strike_step
            )
            
            for strike in strikes:
                # Calculate moneyness and base IV
                moneyness = strike / underlying_price
                base_iv = self._calculate_implied_volatility(moneyness, time_to_expiry)
                
                # Calculate theoretical option prices (simplified Black-Scholes)
                call_price = self._black_scholes_call(
                    underlying_price, strike, time_to_expiry, 0.05, base_iv
                )
                put_price = self._black_scholes_put(
                    underlying_price, strike, time_to_expiry, 0.05, base_iv
                )
                
                # Add bid-ask spread
                spread_pct = 0.02 + 0.05 * abs(1 - moneyness)  # Wider spreads for OTM
                
                # Call option
                options_data.append({
                    'symbol': underlying_symbol,
                    'underlying_price': underlying_price,
                    'strike': strike,
                    'expiry': expiry,
                    'days_to_expiry': days_to_expiry,
                    'option_type': 'call',
                    'bid': call_price * (1 - spread_pct/2),
                    'ask': call_price * (1 + spread_pct/2),
                    'last': call_price,
                    'volume': np.random.randint(0, 5000),
                    'open_interest': np.random.randint(100, 20000),
                    'implied_volatility': base_iv,
                    'delta': self._calculate_delta('call', underlying_price, strike, time_to_expiry, 0.05, base_iv),
                    'gamma': self._calculate_gamma(underlying_price, strike, time_to_expiry, 0.05, base_iv),
                    'theta': self._calculate_theta('call', underlying_price, strike, time_to_expiry, 0.05, base_iv),
                    'vega': self._calculate_vega(underlying_price, strike, time_to_expiry, 0.05, base_iv)
                })
                
                # Put option
                options_data.append({
                    'symbol': underlying_symbol,
                    'underlying_price': underlying_price,
                    'strike': strike,
                    'expiry': expiry,
                    'days_to_expiry': days_to_expiry,
                    'option_type': 'put',
                    'bid': put_price * (1 - spread_pct/2),
                    'ask': put_price * (1 + spread_pct/2),
                    'last': put_price,
                    'volume': np.random.randint(0, 5000),
                    'open_interest': np.random.randint(100, 20000),
                    'implied_volatility': base_iv,
                    'delta': self._calculate_delta('put', underlying_price, strike, time_to_expiry, 0.05, base_iv),
                    'gamma': self._calculate_gamma(underlying_price, strike, time_to_expiry, 0.05, base_iv),
                    'theta': self._calculate_theta('put', underlying_price, strike, time_to_expiry, 0.05, base_iv),
                    'vega': self._calculate_vega(underlying_price, strike, time_to_expiry, 0.05, base_iv)
                })
        
        return pd.DataFrame(options_data)
    
    def generate_economic_data(
        self,
        start_date: datetime,
        end_date: datetime,
        indicators: List[str] = None
    ) -> pd.DataFrame:
        """Generate economic indicator data."""
        if indicators is None:
            indicators = ['GDP_GROWTH', 'UNEMPLOYMENT', 'INFLATION', 'FED_RATE', 'YIELD_CURVE_SLOPE']
        
        dates = pd.date_range(start=start_date, end=end_date, freq='M')  # Monthly data
        
        data = {'date': dates}
        
        for indicator in indicators:
            if indicator == 'GDP_GROWTH':
                # GDP growth rate (quarterly, annualized)
                data[indicator] = np.random.normal(2.5, 1.0, len(dates))
            elif indicator == 'UNEMPLOYMENT':
                # Unemployment rate
                data[indicator] = np.random.normal(4.5, 0.8, len(dates))
            elif indicator == 'INFLATION':
                # Inflation rate (CPI YoY)
                data[indicator] = np.random.normal(2.0, 0.5, len(dates))
            elif indicator == 'FED_RATE':
                # Federal funds rate
                data[indicator] = np.random.normal(3.0, 1.0, len(dates))
            elif indicator == 'YIELD_CURVE_SLOPE':
                # 10Y - 2Y yield spread
                data[indicator] = np.random.normal(1.5, 0.8, len(dates))
        
        return pd.DataFrame(data)
    
    def generate_news_sentiment_data(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Generate news sentiment data."""
        dates = pd.date_range(start=start_date, end=end_date, freq='H')  # Hourly news
        
        news_data = []
        
        for date in dates:
            # Random chance of news for each symbol
            for symbol in symbols:
                if np.random.random() < 0.1:  # 10% chance of news per hour
                    sentiment = np.random.normal(0, 0.3)  # Neutral bias with noise
                    sentiment = np.clip(sentiment, -1, 1)  # Clip to valid range
                    
                    news_data.append({
                        'timestamp': date,
                        'symbol': symbol,
                        'sentiment_score': sentiment,
                        'headline': f"News about {symbol}",
                        'source': np.random.choice(['Reuters', 'Bloomberg', 'WSJ', 'CNBC']),
                        'relevance_score': np.random.uniform(0.5, 1.0)
                    })
        
        return pd.DataFrame(news_data)
    
    def generate_earnings_data(
        self,
        symbols: List[str],
        year: int = 2023
    ) -> pd.DataFrame:
        """Generate earnings announcement data."""
        earnings_data = []
        
        for symbol in symbols:
            # Quarterly earnings (4 per year)
            for quarter in range(1, 5):
                # Earnings typically announced in specific months
                if quarter == 1:
                    announcement_month = np.random.choice([1, 2])
                elif quarter == 2:
                    announcement_month = np.random.choice([4, 5])
                elif quarter == 3:
                    announcement_month = np.random.choice([7, 8])
                else:
                    announcement_month = np.random.choice([10, 11])
                
                announcement_date = datetime(year, announcement_month, np.random.randint(1, 29))
                
                # Generate earnings data
                expected_eps = np.random.uniform(1.0, 3.0)
                actual_eps = expected_eps + np.random.normal(0, 0.2)
                
                earnings_data.append({
                    'symbol': symbol,
                    'announcement_date': announcement_date,
                    'quarter': quarter,
                    'year': year,
                    'expected_eps': expected_eps,
                    'actual_eps': actual_eps,
                    'eps_surprise': actual_eps - expected_eps,
                    'revenue_millions': np.random.uniform(1000, 50000),
                    'guidance': np.random.choice(['positive', 'neutral', 'negative'])
                })
        
        return pd.DataFrame(earnings_data)
    
    def _calculate_implied_volatility(self, moneyness: float, time_to_expiry: float) -> float:
        """Calculate implied volatility with smile effect."""
        # Base volatility
        base_vol = 0.2
        
        # Volatility smile (higher IV for OTM options)
        smile_effect = 0.1 * abs(1 - moneyness)
        
        # Term structure (higher IV for shorter expiries)
        if time_to_expiry < 0.1:  # Less than 1 month
            term_effect = 0.1
        else:
            term_effect = 0
        
        return base_vol + smile_effect + term_effect
    
    def _black_scholes_call(self, S, K, T, r, sigma):
        """Simplified Black-Scholes call option pricing."""
        if T <= 0:
            return max(S - K, 0)
        
        from scipy.stats import norm
        import math
        
        d1 = (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
        d2 = d1 - sigma*math.sqrt(T)
        
        call_price = S*norm.cdf(d1) - K*math.exp(-r*T)*norm.cdf(d2)
        return max(call_price, 0)
    
    def _black_scholes_put(self, S, K, T, r, sigma):
        """Simplified Black-Scholes put option pricing."""
        if T <= 0:
            return max(K - S, 0)
        
        call_price = self._black_scholes_call(S, K, T, r, sigma)
        put_price = call_price - S + K * np.exp(-r * T)  # Put-call parity
        return max(put_price, 0)
    
    def _calculate_delta(self, option_type, S, K, T, r, sigma):
        """Calculate option delta."""
        if T <= 0:
            if option_type == 'call':
                return 1.0 if S > K else 0.0
            else:
                return -1.0 if S < K else 0.0
        
        from scipy.stats import norm
        import math
        
        d1 = (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
        
        if option_type == 'call':
            return norm.cdf(d1)
        else:
            return norm.cdf(d1) - 1
    
    def _calculate_gamma(self, S, K, T, r, sigma):
        """Calculate option gamma."""
        if T <= 0:
            return 0.0
        
        from scipy.stats import norm
        import math
        
        d1 = (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
        return norm.pdf(d1) / (S * sigma * math.sqrt(T))
    
    def _calculate_theta(self, option_type, S, K, T, r, sigma):
        """Calculate option theta."""
        if T <= 0:
            return 0.0
        
        from scipy.stats import norm
        import math
        
        d1 = (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
        d2 = d1 - sigma*math.sqrt(T)
        
        theta = (-S*norm.pdf(d1)*sigma/(2*math.sqrt(T)) - 
                r*K*math.exp(-r*T)*norm.cdf(d2 if option_type == 'call' else -d2))
        
        if option_type == 'put':
            theta += r*K*math.exp(-r*T)
        
        return theta / 365  # Convert to daily theta
    
    def _calculate_vega(self, S, K, T, r, sigma):
        """Calculate option vega."""
        if T <= 0:
            return 0.0
        
        from scipy.stats import norm
        import math
        
        d1 = (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
        return S * norm.pdf(d1) * math.sqrt(T) / 100  # Per 1% vol change


class MockDataProvider:
    """Mock data provider for testing."""
    
    def __init__(self):
        self.generator = MarketDataGenerator()
        self._cache = {}
    
    async def get_stock_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get mock stock data."""
        cache_key = f"{symbol}_{start_date}_{end_date}"
        
        if cache_key not in self._cache:
            # Different stocks have different characteristics
            if symbol == 'AAPL':
                data = self.generator.generate_stock_data(
                    symbol, start_date, end_date, 
                    initial_price=150, volatility=0.025, trend=0.0008
                )
            elif symbol == 'GOOGL':
                data = self.generator.generate_stock_data(
                    symbol, start_date, end_date,
                    initial_price=2500, volatility=0.03, trend=0.0006
                )
            elif symbol == 'TSLA':
                data = self.generator.generate_stock_data(
                    symbol, start_date, end_date,
                    initial_price=200, volatility=0.05, trend=0.001
                )
            else:
                data = self.generator.generate_stock_data(
                    symbol, start_date, end_date
                )
            
            self._cache[cache_key] = data
        
        return self._cache[cache_key]
    
    async def get_options_chain(self, symbol: str, expiry_dates: List[datetime]) -> pd.DataFrame:
        """Get mock options chain."""
        # Get current stock price
        current_data = await self.get_stock_data(
            symbol, 
            datetime.now() - timedelta(days=1), 
            datetime.now()
        )
        current_price = current_data['close'].iloc[-1]
        
        return self.generator.generate_options_chain(symbol, current_price, expiry_dates)
    
    async def get_economic_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get mock economic data."""
        return self.generator.generate_economic_data(start_date, end_date)
    
    async def get_news_sentiment(self, symbols: List[str], start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get mock news sentiment data."""
        return self.generator.generate_news_sentiment_data(symbols, start_date, end_date)
    
    async def get_earnings_data(self, symbols: List[str], year: int = 2023) -> pd.DataFrame:
        """Get mock earnings data."""
        return self.generator.generate_earnings_data(symbols, year)


# Create singleton instances for easy import
market_data_generator = MarketDataGenerator()
mock_data_provider = MockDataProvider()


def create_test_portfolio_data():
    """Create test portfolio data for backtesting."""
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2023, 12, 31)
    
    portfolio_data = {}
    
    for symbol in symbols:
        portfolio_data[symbol] = market_data_generator.generate_stock_data(
            symbol, start_date, end_date
        )
    
    return portfolio_data


def create_correlation_test_data():
    """Create correlated asset data for testing."""
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    n_periods = len(dates)
    
    # Common market factor
    market_factor = np.random.normal(0, 0.01, n_periods)
    
    # Assets with different exposures to market factor
    assets = {
        'HIGH_BETA': market_factor * 1.5 + np.random.normal(0, 0.008, n_periods),
        'MEDIUM_BETA': market_factor * 1.0 + np.random.normal(0, 0.006, n_periods),
        'LOW_BETA': market_factor * 0.5 + np.random.normal(0, 0.004, n_periods),
        'DEFENSIVE': -market_factor * 0.3 + np.random.normal(0, 0.003, n_periods)
    }
    
    # Convert to prices
    price_data = {}
    for asset, returns in assets.items():
        prices = [100]
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        
        price_data[asset] = pd.DataFrame({
            'timestamp': dates,
            'close': prices[1:],
            'returns': returns
        })
    
    return price_data