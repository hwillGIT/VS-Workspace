"""
Market data API interface for multiple providers.
"""

import asyncio
import aiohttp
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
import requests
import ccxt
import ccxt.async_support as ccxt_async
from loguru import logger

from ..base.config import config
from ..base.exceptions import APIError, DataError


class BaseMarketDataProvider(ABC):
    """Abstract base class for market data providers."""
    
    @abstractmethod
    async def get_historical_data(self, symbol: str, start_date: datetime, 
                                end_date: datetime, interval: str) -> pd.DataFrame:
        """Get historical OHLCV data."""
        pass
    
    @abstractmethod
    async def get_real_time_quote(self, symbol: str) -> Dict:
        """Get real-time quote data."""
        pass
    
    @abstractmethod
    async def get_intraday_data(self, symbol: str, interval: str) -> pd.DataFrame:
        """Get intraday data."""
        pass


class YahooFinanceProvider(BaseMarketDataProvider):
    """Yahoo Finance data provider."""
    
    def __init__(self):
        self.name = "Yahoo Finance"
        self.rate_limit = config.get("market_data.yahoo_finance.rate_limit", 2000)
        self.logger = logger.bind(provider="yahoo_finance")
    
    async def get_historical_data(self, symbol: str, start_date: datetime, 
                                end_date: datetime, interval: str = "1d") -> pd.DataFrame:
        """
        Get historical data from Yahoo Finance.
        
        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            interval: Data interval
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=True,
                prepost=True
            )
            
            if data.empty:
                raise DataError(f"No data found for symbol {symbol}")
            
            # Standardize column names
            data.columns = data.columns.str.lower()
            data.index.name = 'timestamp'
            
            self.logger.info(f"Retrieved {len(data)} records for {symbol}")
            return data
            
        except Exception as e:
            raise APIError(f"Yahoo Finance API error for {symbol}: {str(e)}")
    
    async def get_real_time_quote(self, symbol: str) -> Dict:
        """Get real-time quote from Yahoo Finance."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            quote = {
                'symbol': symbol,
                'price': info.get('regularMarketPrice', 0),
                'change': info.get('regularMarketChange', 0),
                'change_percent': info.get('regularMarketChangePercent', 0),
                'volume': info.get('regularMarketVolume', 0),
                'timestamp': datetime.utcnow()
            }
            
            return quote
            
        except Exception as e:
            raise APIError(f"Yahoo Finance quote error for {symbol}: {str(e)}")
    
    async def get_intraday_data(self, symbol: str, interval: str = "1m") -> pd.DataFrame:
        """Get intraday data from Yahoo Finance."""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d", interval=interval)
            
            if data.empty:
                raise DataError(f"No intraday data found for symbol {symbol}")
            
            data.columns = data.columns.str.lower()
            data.index.name = 'timestamp'
            
            return data
            
        except Exception as e:
            raise APIError(f"Yahoo Finance intraday error for {symbol}: {str(e)}")


class AlphaVantageProvider(BaseMarketDataProvider):
    """Alpha Vantage data provider."""
    
    def __init__(self):
        self.name = "Alpha Vantage"
        self.api_key = config.alpha_vantage_api_key
        self.base_url = config.get("market_data.alpha_vantage.base_url")
        self.rate_limit = config.get("market_data.alpha_vantage.rate_limit", 5)
        self.logger = logger.bind(provider="alpha_vantage")
        
        if not self.api_key:
            raise APIError("Alpha Vantage API key not configured")
    
    async def get_historical_data(self, symbol: str, start_date: datetime, 
                                end_date: datetime, interval: str = "daily") -> pd.DataFrame:
        """Get historical data from Alpha Vantage."""
        try:
            # Map interval to Alpha Vantage function
            function_map = {
                "daily": "TIME_SERIES_DAILY",
                "weekly": "TIME_SERIES_WEEKLY",
                "monthly": "TIME_SERIES_MONTHLY"
            }
            
            function = function_map.get(interval, "TIME_SERIES_DAILY")
            
            params = {
                'function': function,
                'symbol': symbol,
                'apikey': self.api_key,
                'outputsize': 'full',
                'datatype': 'json'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params) as response:
                    data = await response.json()
            
            # Extract time series data
            time_series_key = None
            for key in data.keys():
                if 'Time Series' in key:
                    time_series_key = key
                    break
            
            if not time_series_key or time_series_key not in data:
                raise DataError(f"No time series data found for {symbol}")
            
            time_series = data[time_series_key]
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(time_series, orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            # Standardize column names
            column_mapping = {
                '1. open': 'open',
                '2. high': 'high',
                '3. low': 'low',
                '4. close': 'close',
                '5. volume': 'volume'
            }
            df = df.rename(columns=column_mapping)
            df = df.astype(float)
            
            # Filter by date range
            df = df[(df.index >= start_date) & (df.index <= end_date)]
            
            self.logger.info(f"Retrieved {len(df)} records for {symbol}")
            return df
            
        except Exception as e:
            raise APIError(f"Alpha Vantage API error for {symbol}: {str(e)}")
    
    async def get_real_time_quote(self, symbol: str) -> Dict:
        """Get real-time quote from Alpha Vantage."""
        try:
            params = {
                'function': 'GLOBAL_QUOTE',
                'symbol': symbol,
                'apikey': self.api_key
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params) as response:
                    data = await response.json()
            
            if 'Global Quote' not in data:
                raise DataError(f"No quote data found for {symbol}")
            
            quote_data = data['Global Quote']
            
            quote = {
                'symbol': symbol,
                'price': float(quote_data.get('05. price', 0)),
                'change': float(quote_data.get('09. change', 0)),
                'change_percent': float(quote_data.get('10. change percent', '0').rstrip('%')),
                'volume': int(quote_data.get('06. volume', 0)),
                'timestamp': datetime.utcnow()
            }
            
            return quote
            
        except Exception as e:
            raise APIError(f"Alpha Vantage quote error for {symbol}: {str(e)}")
    
    async def get_intraday_data(self, symbol: str, interval: str = "1min") -> pd.DataFrame:
        """Get intraday data from Alpha Vantage."""
        try:
            params = {
                'function': 'TIME_SERIES_INTRADAY',
                'symbol': symbol,
                'interval': interval,
                'apikey': self.api_key,
                'outputsize': 'full'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params) as response:
                    data = await response.json()
            
            # Find the time series key
            time_series_key = f'Time Series ({interval})'
            if time_series_key not in data:
                raise DataError(f"No intraday data found for {symbol}")
            
            time_series = data[time_series_key]
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(time_series, orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            # Standardize column names
            column_mapping = {
                '1. open': 'open',
                '2. high': 'high',
                '3. low': 'low',
                '4. close': 'close',
                '5. volume': 'volume'
            }
            df = df.rename(columns=column_mapping)
            df = df.astype(float)
            
            return df
            
        except Exception as e:
            raise APIError(f"Alpha Vantage intraday error for {symbol}: {str(e)}")


class CryptoDataProvider(BaseMarketDataProvider):
    """Cryptocurrency data provider using CCXT."""
    
    def __init__(self, exchange_name: str = 'binance'):
        self.name = f"CCXT-{exchange_name.capitalize()}"
        self.exchange_name = exchange_name
        self.logger = logger.bind(provider=f"ccxt_{exchange_name}")
        
        # Initialize exchange
        try:
            exchange_class = getattr(ccxt, exchange_name)
            self.exchange = exchange_class({
                'apiKey': config.get(f"crypto.{exchange_name}.api_key", ""),
                'secret': config.get(f"crypto.{exchange_name}.secret", ""),
                'timeout': 30000,
                'enableRateLimit': True,
                'sandbox': config.get(f"crypto.{exchange_name}.sandbox", False)
            })
            
            # Async exchange for concurrent operations
            async_exchange_class = getattr(ccxt_async, exchange_name)
            self.async_exchange = async_exchange_class({
                'apiKey': config.get(f"crypto.{exchange_name}.api_key", ""),
                'secret': config.get(f"crypto.{exchange_name}.secret", ""),
                'timeout': 30000,
                'enableRateLimit': True,
                'sandbox': config.get(f"crypto.{exchange_name}.sandbox", False)
            })
            
        except Exception as e:
            raise APIError(f"Failed to initialize {exchange_name} exchange: {str(e)}")
    
    async def get_historical_data(self, symbol: str, start_date: datetime, 
                                end_date: datetime, interval: str = "1d") -> pd.DataFrame:
        """
        Get historical OHLCV data for cryptocurrency.
        
        Args:
            symbol: Crypto trading pair (e.g., 'BTC/USDT')
            start_date: Start date
            end_date: End date
            interval: Timeframe (1m, 5m, 15m, 1h, 4h, 1d, 1w)
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Convert datetime to milliseconds timestamp
            since = int(start_date.timestamp() * 1000)
            
            # Map interval to exchange timeframe
            timeframe_map = {
                "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
                "1h": "1h", "4h": "4h", "1d": "1d", "1w": "1w"
            }
            timeframe = timeframe_map.get(interval, "1d")
            
            # Fetch OHLCV data
            ohlcv_data = await self.async_exchange.fetch_ohlcv(
                symbol, timeframe, since, limit=1000
            )
            
            if not ohlcv_data:
                raise DataError(f"No data found for symbol {symbol}")
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Filter by end date
            df = df[df.index <= end_date]
            
            self.logger.info(f"Retrieved {len(df)} crypto records for {symbol}")
            return df
            
        except Exception as e:
            raise APIError(f"CCXT {self.exchange_name} API error for {symbol}: {str(e)}")
        finally:
            if hasattr(self, 'async_exchange'):
                await self.async_exchange.close()
    
    async def get_real_time_quote(self, symbol: str) -> Dict:
        """Get real-time quote for cryptocurrency."""
        try:
            ticker = await self.async_exchange.fetch_ticker(symbol)
            
            quote = {
                'symbol': symbol,
                'price': ticker.get('last', 0),
                'change': ticker.get('change', 0),
                'change_percent': ticker.get('percentage', 0),
                'volume': ticker.get('baseVolume', 0),
                'bid': ticker.get('bid', 0),
                'ask': ticker.get('ask', 0),
                'timestamp': datetime.utcnow()
            }
            
            return quote
            
        except Exception as e:
            raise APIError(f"CCXT {self.exchange_name} quote error for {symbol}: {str(e)}")
        finally:
            if hasattr(self, 'async_exchange'):
                await self.async_exchange.close()
    
    async def get_intraday_data(self, symbol: str, interval: str = "1m") -> pd.DataFrame:
        """Get intraday crypto data."""
        # For crypto, just get recent data with specified interval
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(hours=24)  # Last 24 hours
        
        return await self.get_historical_data(symbol, start_date, end_date, interval)
    
    async def get_crypto_markets(self) -> List[Dict]:
        """Get list of available cryptocurrency markets."""
        try:
            markets = await self.async_exchange.load_markets()
            
            crypto_markets = []
            for symbol, market in markets.items():
                if market.get('active', True):
                    crypto_markets.append({
                        'symbol': symbol,
                        'base': market.get('base'),
                        'quote': market.get('quote'),
                        'type': market.get('type', 'spot'),
                        'limits': market.get('limits', {})
                    })
            
            return crypto_markets
            
        except Exception as e:
            raise APIError(f"Failed to get crypto markets: {str(e)}")
        finally:
            if hasattr(self, 'async_exchange'):
                await self.async_exchange.close()
    
    async def get_orderbook(self, symbol: str, limit: int = 20) -> Dict:
        """Get order book for cryptocurrency pair."""
        try:
            orderbook = await self.async_exchange.fetch_order_book(symbol, limit)
            
            return {
                'symbol': symbol,
                'bids': orderbook.get('bids', []),
                'asks': orderbook.get('asks', []),
                'timestamp': datetime.utcnow()
            }
            
        except Exception as e:
            raise APIError(f"CCXT orderbook error for {symbol}: {str(e)}")
        finally:
            if hasattr(self, 'async_exchange'):
                await self.async_exchange.close()


class MarketDataAPI:
    """
    Unified market data API that aggregates multiple providers.
    """
    
    def __init__(self):
        self.providers = {
            'yahoo': YahooFinanceProvider(),
            'alpha_vantage': AlphaVantageProvider() if config.alpha_vantage_api_key else None,
            'crypto_binance': CryptoDataProvider('binance'),
            'crypto_coinbase': CryptoDataProvider('coinbasepro'),
            'crypto_kraken': CryptoDataProvider('kraken')
        }
        
        # Remove None providers
        self.providers = {k: v for k, v in self.providers.items() if v is not None}
        
        self.primary_provider = config.get("market_data.primary_provider", "yahoo")
        self.crypto_provider = config.get("market_data.crypto_provider", "crypto_binance")
        self.logger = logger.bind(service="market_data_api")
        
        if not self.providers:
            raise APIError("No market data providers configured")
    
    async def get_historical_data(self, symbol: str, start_date: datetime, 
                                end_date: datetime, interval: str = "1d",
                                provider: Optional[str] = None) -> pd.DataFrame:
        """
        Get historical data with provider fallback.
        
        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            interval: Data interval
            provider: Specific provider to use
            
        Returns:
            DataFrame with OHLCV data
        """
        provider_name = provider or self.primary_provider
        
        if provider_name not in self.providers:
            raise APIError(f"Provider {provider_name} not available")
        
        try:
            return await self.providers[provider_name].get_historical_data(
                symbol, start_date, end_date, interval
            )
        except Exception as e:
            self.logger.error(f"Primary provider {provider_name} failed: {e}")
            
            # Try fallback providers
            for fallback_name, fallback_provider in self.providers.items():
                if fallback_name != provider_name:
                    try:
                        self.logger.info(f"Trying fallback provider: {fallback_name}")
                        return await fallback_provider.get_historical_data(
                            symbol, start_date, end_date, interval
                        )
                    except Exception as fallback_error:
                        self.logger.error(f"Fallback provider {fallback_name} failed: {fallback_error}")
                        continue
            
            raise APIError(f"All providers failed for symbol {symbol}")
    
    async def get_multiple_symbols(self, symbols: List[str], start_date: datetime,
                                 end_date: datetime, interval: str = "1d") -> Dict[str, pd.DataFrame]:
        """
        Get historical data for multiple symbols concurrently.
        
        Args:
            symbols: List of symbols
            start_date: Start date
            end_date: End date
            interval: Data interval
            
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        tasks = []
        for symbol in symbols:
            task = self.get_historical_data(symbol, start_date, end_date, interval)
            tasks.append((symbol, task))
        
        results = {}
        completed_tasks = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
        
        for (symbol, _), result in zip(tasks, completed_tasks):
            if isinstance(result, Exception):
                self.logger.error(f"Failed to get data for {symbol}: {result}")
                results[symbol] = pd.DataFrame()  # Empty DataFrame for failed symbols
            else:
                results[symbol] = result
        
        return results
    
    async def get_real_time_quotes(self, symbols: List[str]) -> Dict[str, Dict]:
        """Get real-time quotes for multiple symbols."""
        provider = self.providers[self.primary_provider]
        
        tasks = []
        for symbol in symbols:
            task = provider.get_real_time_quote(symbol)
            tasks.append((symbol, task))
        
        results = {}
        completed_tasks = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
        
        for (symbol, _), result in zip(tasks, completed_tasks):
            if isinstance(result, Exception):
                self.logger.error(f"Failed to get quote for {symbol}: {result}")
                results[symbol] = {}
            else:
                results[symbol] = result
        
        return results
    
    def get_supported_symbols(self, exchange: str = "NYSE") -> List[str]:
        """
        Get list of supported symbols for an exchange.
        Note: This is a simplified implementation. 
        In production, you'd fetch this from the exchange or a reference data provider.
        """
        # Sample symbols - in production, fetch from exchange listings
        sample_symbols = {
            "NYSE": ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NVDA", "JPM", "JNJ", "V"],
            "NASDAQ": ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NVDA", "NFLX", "ADBE", "CRM"],
        }
        
        return sample_symbols.get(exchange.upper(), [])
    
    async def get_crypto_data(self, symbol: str, start_date: datetime, 
                            end_date: datetime, interval: str = "1d",
                            exchange: str = "binance") -> pd.DataFrame:
        """
        Get cryptocurrency data from specified exchange.
        
        Args:
            symbol: Crypto trading pair (e.g., 'BTC/USDT')
            start_date: Start date
            end_date: End date
            interval: Data interval
            exchange: Exchange name (binance, coinbasepro, kraken)
            
        Returns:
            DataFrame with OHLCV data
        """
        provider_name = f"crypto_{exchange}"
        
        if provider_name not in self.providers:
            raise APIError(f"Crypto provider {provider_name} not available")
        
        return await self.providers[provider_name].get_historical_data(
            symbol, start_date, end_date, interval
        )
    
    async def get_crypto_markets(self, exchange: str = "binance") -> List[Dict]:
        """Get available cryptocurrency markets from exchange."""
        provider_name = f"crypto_{exchange}"
        
        if provider_name not in self.providers:
            raise APIError(f"Crypto provider {provider_name} not available")
        
        provider = self.providers[provider_name]
        if hasattr(provider, 'get_crypto_markets'):
            return await provider.get_crypto_markets()
        else:
            raise APIError(f"Provider {provider_name} does not support market listing")
    
    async def get_crypto_orderbook(self, symbol: str, exchange: str = "binance", 
                                 limit: int = 20) -> Dict:
        """Get order book for cryptocurrency pair."""
        provider_name = f"crypto_{exchange}"
        
        if provider_name not in self.providers:
            raise APIError(f"Crypto provider {provider_name} not available")
        
        provider = self.providers[provider_name]
        if hasattr(provider, 'get_orderbook'):
            return await provider.get_orderbook(symbol, limit)
        else:
            raise APIError(f"Provider {provider_name} does not support order book data")
    
    def is_crypto_symbol(self, symbol: str) -> bool:
        """Check if symbol is a cryptocurrency pair."""
        # Common crypto pair patterns
        crypto_patterns = [
            '/', '-', '_',  # BTC/USDT, BTC-USD, BTC_USDT
        ]
        
        # Common crypto base currencies
        crypto_bases = [
            'BTC', 'ETH', 'BNB', 'ADA', 'SOL', 'DOT', 'LINK', 'MATIC',
            'AVAX', 'UNI', 'LTC', 'XRP', 'DOGE', 'SHIB'
        ]
        
        # Check if symbol contains common separators
        has_separator = any(pattern in symbol for pattern in crypto_patterns)
        
        # Check if symbol starts with known crypto base
        starts_with_crypto = any(symbol.upper().startswith(base) for base in crypto_bases)
        
        return has_separator or starts_with_crypto
    
    async def get_data_auto_detect(self, symbol: str, start_date: datetime,
                                 end_date: datetime, interval: str = "1d") -> pd.DataFrame:
        """
        Automatically detect if symbol is crypto or traditional and fetch data.
        """
        if self.is_crypto_symbol(symbol):
            # Try crypto providers
            for provider_name in ['crypto_binance', 'crypto_coinbase', 'crypto_kraken']:
                if provider_name in self.providers:
                    try:
                        return await self.providers[provider_name].get_historical_data(
                            symbol, start_date, end_date, interval
                        )
                    except Exception as e:
                        self.logger.warning(f"Crypto provider {provider_name} failed for {symbol}: {e}")
                        continue
            
            raise APIError(f"All crypto providers failed for symbol {symbol}")
        else:
            # Use traditional stock providers
            return await self.get_historical_data(symbol, start_date, end_date, interval)