"""
Performance Optimizations for Trading System

This module provides optimized implementations for computationally intensive operations
identified by the System Architect analysis.
"""

import asyncio
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import lru_cache, wraps
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Callable
import time
import psutil
from dataclasses import dataclass
from enum import Enum

from loguru import logger


class OptimizationLevel(Enum):
    """Optimization levels for different scenarios."""
    NONE = "none"
    BASIC = "basic"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


@dataclass
class PerformanceConfig:
    """Configuration for performance optimizations."""
    max_workers: int = min(32, (psutil.cpu_count() or 1) + 4)
    chunk_size: int = 1000
    cache_size: int = 1024
    optimization_level: OptimizationLevel = OptimizationLevel.MODERATE
    enable_parallel: bool = True
    enable_caching: bool = True
    memory_threshold: float = 0.8  # 80% memory usage threshold


class PerformanceMonitor:
    """Monitor and track performance metrics."""
    
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
    
    def start_timer(self, operation: str):
        """Start timing an operation."""
        self.start_times[operation] = time.time()
    
    def end_timer(self, operation: str) -> float:
        """End timing and record duration."""
        if operation in self.start_times:
            duration = time.time() - self.start_times[operation]
            self.metrics[operation] = self.metrics.get(operation, []) + [duration]
            return duration
        return 0.0
    
    def get_stats(self, operation: str) -> Dict[str, float]:
        """Get statistics for an operation."""
        times = self.metrics.get(operation, [])
        if not times:
            return {}
        
        return {
            'count': len(times),
            'total': sum(times),
            'average': sum(times) / len(times),
            'min': min(times),
            'max': max(times)
        }


def performance_timer(func):
    """Decorator to time function execution."""
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            duration = time.time() - start_time
            logger.debug(f"{func.__name__} took {duration:.3f}s")
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"{func.__name__} failed after {duration:.3f}s: {e}")
            raise
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            logger.debug(f"{func.__name__} took {duration:.3f}s")
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"{func.__name__} failed after {duration:.3f}s: {e}")
            raise
    
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper


def adaptive_cache(maxsize: int = 1024, typed: bool = False):
    """Adaptive LRU cache that adjusts based on memory usage."""
    def decorator(func):
        cached_func = lru_cache(maxsize=maxsize, typed=typed)(func)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Check memory usage and clear cache if necessary
            memory_percent = psutil.virtual_memory().percent
            if memory_percent > 85:  # Clear cache if memory usage > 85%
                cached_func.cache_clear()
                logger.warning(f"Cache cleared due to high memory usage: {memory_percent:.1f}%")
            
            return cached_func(*args, **kwargs)
        
        # Expose cache methods
        wrapper.cache_info = cached_func.cache_info
        wrapper.cache_clear = cached_func.cache_clear
        
        return wrapper
    return decorator


class ParallelProcessor:
    """High-performance parallel processing utilities."""
    
    def __init__(self, config: PerformanceConfig = None):
        self.config = config or PerformanceConfig()
        logger.info(f"ParallelProcessor initialized with {self.config.max_workers} workers")
    
    async def process_parallel_async(self, 
                                   items: List[Any], 
                                   func: Callable,
                                   *args, **kwargs) -> List[Any]:
        """Process items in parallel using asyncio."""
        if not self.config.enable_parallel or len(items) < 10:
            # For small datasets, parallel processing overhead isn't worth it
            return [await func(item, *args, **kwargs) for item in items]
        
        semaphore = asyncio.Semaphore(self.config.max_workers)
        
        async def bounded_func(item):
            async with semaphore:
                return await func(item, *args, **kwargs)
        
        tasks = [bounded_func(item) for item in items]
        return await asyncio.gather(*tasks, return_exceptions=False)
    
    def process_parallel_cpu(self, 
                           items: List[Any], 
                           func: Callable,
                           *args, **kwargs) -> List[Any]:
        """Process CPU-intensive tasks in parallel using ProcessPoolExecutor."""
        if not self.config.enable_parallel or len(items) < self.config.chunk_size:
            return [func(item, *args, **kwargs) for item in items]
        
        # Split items into chunks for better load balancing
        chunks = self._create_chunks(items, self.config.chunk_size)
        
        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            chunk_results = list(executor.map(
                self._process_chunk, 
                [(chunk, func, args, kwargs) for chunk in chunks]
            ))
        
        # Flatten results
        return [item for chunk_result in chunk_results for item in chunk_result]
    
    def process_parallel_io(self, 
                          items: List[Any], 
                          func: Callable,
                          *args, **kwargs) -> List[Any]:
        """Process I/O-bound tasks in parallel using ThreadPoolExecutor."""
        if not self.config.enable_parallel or len(items) < 10:
            return [func(item, *args, **kwargs) for item in items]
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            results = list(executor.map(
                lambda item: func(item, *args, **kwargs), 
                items
            ))
        
        return results
    
    @staticmethod
    def _create_chunks(items: List[Any], chunk_size: int) -> List[List[Any]]:
        """Create chunks from items."""
        return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
    
    @staticmethod
    def _process_chunk(args):
        """Process a chunk of items."""
        chunk, func, f_args, f_kwargs = args
        return [func(item, *f_args, **f_kwargs) for item in chunk]


class OptimizedDataProcessor:
    """Optimized data processing utilities for financial data."""
    
    def __init__(self, config: PerformanceConfig = None):
        self.config = config or PerformanceConfig()
        self.parallel_processor = ParallelProcessor(config)
    
    @adaptive_cache(maxsize=512)
    def calculate_returns_optimized(self, prices: pd.Series) -> pd.Series:
        """Optimized returns calculation using numpy."""
        if len(prices) < 2:
            return pd.Series(dtype=float)
        
        # Use numpy for faster computation
        price_values = prices.values
        returns = np.diff(price_values) / price_values[:-1]
        
        return pd.Series(returns, index=prices.index[1:])
    
    @adaptive_cache(maxsize=256)
    def calculate_rolling_stats_optimized(self, 
                                        data: pd.Series, 
                                        window: int) -> Dict[str, pd.Series]:
        """Optimized rolling statistics calculation."""
        if len(data) < window:
            return {}
        
        # Use pandas' optimized rolling functions
        rolling = data.rolling(window=window, min_periods=window)
        
        return {
            'mean': rolling.mean(),
            'std': rolling.std(),
            'min': rolling.min(),
            'max': rolling.max(),
            'median': rolling.median()
        }
    
    def calculate_correlation_matrix_optimized(self, 
                                             data: pd.DataFrame,
                                             method: str = 'pearson') -> pd.DataFrame:
        """Optimized correlation matrix calculation."""
        if data.empty:
            return pd.DataFrame()
        
        # Use numpy for faster correlation calculation
        if method == 'pearson':
            # Remove NaN values for cleaner correlation
            clean_data = data.dropna()
            if clean_data.empty:
                return pd.DataFrame()
            
            # Use numpy's corrcoef for speed
            corr_matrix = np.corrcoef(clean_data.T)
            return pd.DataFrame(corr_matrix, 
                              index=clean_data.columns, 
                              columns=clean_data.columns)
        else:
            return data.corr(method=method)
    
    async def batch_process_symbols(self, 
                                  symbols: List[str],
                                  processing_func: Callable,
                                  batch_size: int = None) -> Dict[str, Any]:
        """Process symbols in optimized batches."""
        batch_size = batch_size or self.config.chunk_size
        
        # Split symbols into batches
        batches = [symbols[i:i + batch_size] 
                  for i in range(0, len(symbols), batch_size)]
        
        results = {}
        
        for batch in batches:
            batch_results = await self.parallel_processor.process_parallel_async(
                batch, processing_func
            )
            
            # Combine results
            for symbol, result in zip(batch, batch_results):
                results[symbol] = result
        
        return results


class MemoryOptimizer:
    """Memory optimization utilities."""
    
    @staticmethod
    def optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage by downcasting numeric types."""
        optimized_df = df.copy()
        
        for col in optimized_df.columns:
            col_type = optimized_df[col].dtype
            
            if col_type != 'object':
                c_min = optimized_df[col].min()
                c_max = optimized_df[col].max()
                
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        optimized_df[col] = optimized_df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        optimized_df[col] = optimized_df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        optimized_df[col] = optimized_df[col].astype(np.int32)
                
                elif str(col_type)[:5] == 'float':
                    if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        optimized_df[col] = optimized_df[col].astype(np.float32)
        
        return optimized_df
    
    @staticmethod
    def memory_usage_mb(obj) -> float:
        """Get memory usage in MB."""
        if hasattr(obj, 'memory_usage'):
            return obj.memory_usage(deep=True).sum() / 1024 / 1024
        else:
            return psutil.Process().memory_info().rss / 1024 / 1024
    
    @staticmethod
    def clear_memory_if_needed(threshold: float = 0.8):
        """Clear memory if usage exceeds threshold."""
        memory_percent = psutil.virtual_memory().percent / 100
        
        if memory_percent > threshold:
            import gc
            gc.collect()
            logger.warning(f"Memory cleared due to high usage: {memory_percent:.1%}")


class QueryOptimizer:
    """Database and data query optimization utilities."""
    
    @staticmethod
    def optimize_pandas_query(df: pd.DataFrame, 
                            conditions: List[str],
                            columns: List[str] = None) -> pd.DataFrame:
        """Optimize pandas queries using query() method."""
        if not conditions:
            result = df
        else:
            # Combine conditions with 'and'
            query_string = ' and '.join(f'({condition})' for condition in conditions)
            result = df.query(query_string)
        
        if columns:
            result = result[columns]
        
        return result
    
    @staticmethod
    def create_optimized_index(df: pd.DataFrame, 
                             columns: List[str]) -> pd.DataFrame:
        """Create optimized index for faster lookups."""
        if len(columns) == 1:
            return df.set_index(columns[0])
        else:
            return df.set_index(columns)


# Global performance configuration
_global_config = PerformanceConfig()
_global_monitor = PerformanceMonitor()


def get_performance_config() -> PerformanceConfig:
    """Get global performance configuration."""
    return _global_config


def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor."""
    return _global_monitor


def set_performance_config(config: PerformanceConfig):
    """Set global performance configuration."""
    global _global_config
    _global_config = config
    logger.info(f"Performance config updated: {config}")


# Example usage and optimization patterns
class OptimizedTechnicalIndicators:
    """Optimized implementations of common technical indicators."""
    
    def __init__(self, config: PerformanceConfig = None):
        self.config = config or get_performance_config()
        self.data_processor = OptimizedDataProcessor(config)
    
    @performance_timer
    @adaptive_cache(maxsize=1024)
    def sma_optimized(self, prices: pd.Series, window: int) -> pd.Series:
        """Optimized Simple Moving Average."""
        if len(prices) < window:
            return pd.Series(dtype=float)
        
        # Use pandas rolling mean (optimized in C)
        return prices.rolling(window=window, min_periods=window).mean()
    
    @performance_timer
    @adaptive_cache(maxsize=512)
    def rsi_optimized(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Optimized RSI calculation using numpy."""
        if len(prices) < window + 1:
            return pd.Series(dtype=float)
        
        # Calculate price changes
        delta = prices.diff()
        
        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        # Calculate average gains and losses using exponential moving average
        avg_gains = gains.ewm(span=window, adjust=False).mean()
        avg_losses = losses.ewm(span=window, adjust=False).mean()
        
        # Calculate RSI
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @performance_timer
    def macd_optimized(self, prices: pd.Series, 
                      fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
        """Optimized MACD calculation."""
        if len(prices) < slow:
            return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)
        
        # Use pandas EWM for optimized exponential moving averages
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram


# Performance testing utilities
def benchmark_function(func: Callable, *args, iterations: int = 10, **kwargs):
    """Benchmark a function's performance."""
    times = []
    
    for _ in range(iterations):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        times.append(end_time - start_time)
    
    return {
        'average_time': sum(times) / len(times),
        'min_time': min(times),
        'max_time': max(times),
        'total_time': sum(times),
        'iterations': iterations
    }


if __name__ == "__main__":
    # Example usage
    print("Performance Optimization Module")
    print("="*50)
    
    # Test configuration
    config = PerformanceConfig(
        max_workers=4,
        optimization_level=OptimizationLevel.MODERATE
    )
    
    # Example data
    dates = pd.date_range('2023-01-01', periods=1000, freq='D')
    prices = pd.Series(
        100 + np.random.randn(1000).cumsum(),
        index=dates
    )
    
    # Test optimized indicators
    indicators = OptimizedTechnicalIndicators(config)
    
    # Benchmark SMA
    sma_result = benchmark_function(
        indicators.sma_optimized,
        prices, 20,
        iterations=5
    )
    
    print(f"SMA Benchmark: {sma_result['average_time']:.4f}s average")
    
    # Test memory optimization
    df = pd.DataFrame({
        'price': prices,
        'volume': np.random.randint(1000, 100000, len(prices))
    })
    
    original_memory = MemoryOptimizer.memory_usage_mb(df)
    optimized_df = MemoryOptimizer.optimize_dataframe_memory(df)
    optimized_memory = MemoryOptimizer.memory_usage_mb(optimized_df)
    
    print(f"Memory optimization: {original_memory:.2f}MB -> {optimized_memory:.2f}MB")
    print(f"Memory saved: {((original_memory - optimized_memory) / original_memory * 100):.1f}%")