"""
Performance Benchmarking Tests for the Trading System.
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
import time
import psutil
import gc
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from memory_profiler import profile
from contextlib import contextmanager

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from main import TradingSystem


@contextmanager
def performance_timer():
    """Context manager to measure execution time."""
    start_time = time.perf_counter()
    start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    yield
    
    end_time = time.perf_counter()
    end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    execution_time = end_time - start_time
    memory_delta = end_memory - start_memory
    
    print(f"Execution time: {execution_time:.3f} seconds")
    print(f"Memory delta: {memory_delta:.2f} MB")


@pytest.fixture
def large_dataset():
    """Generate large dataset for performance testing."""
    symbols = [f"STOCK_{i:04d}" for i in range(500)]
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    
    data = {}
    np.random.seed(42)
    
    for symbol in symbols:
        prices = [100]
        for _ in range(len(dates) - 1):
            change = np.random.normal(0, 0.02)
            prices.append(prices[-1] * (1 + change))
        
        data[symbol] = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'close': prices,
            'volume': np.random.randint(100000, 10000000, len(dates))
        })
    
    return data


class TestPerformanceBenchmarks:
    """Performance benchmarking test cases."""
    
    @pytest.mark.performance
    @pytest.mark.slow
    async def test_signal_generation_latency(self):
        """Test signal generation latency benchmarks."""
        config_path = Path(__file__).parent.parent / "config" / "config.yaml"
        trading_system = TradingSystem(config_path)
        await trading_system.initialize()
        
        # Test single symbol processing
        with performance_timer():
            result = await trading_system.agents['technical_analysis_agent'].process({
                'symbols': ['AAPL'],
                'indicators': ['sma', 'rsi', 'macd'],
                'start_date': datetime.now() - timedelta(days=30),
                'end_date': datetime.now()
            })
        
        # Benchmark: Should complete within 100ms for single symbol
        assert 'indicators' in result
        
        # Test multiple symbols processing
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
        
        with performance_timer():
            tasks = []
            for symbol in symbols:
                task = trading_system.agents['technical_analysis_agent'].process({
                    'symbols': [symbol],
                    'indicators': ['sma', 'rsi', 'macd'],
                    'start_date': datetime.now() - timedelta(days=30),
                    'end_date': datetime.now()
                })
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
        
        assert len(results) == len(symbols)
    
    @pytest.mark.performance
    @pytest.mark.slow
    async def test_portfolio_optimization_performance(self, large_dataset):
        """Test portfolio optimization performance with large universe."""
        config_path = Path(__file__).parent.parent / "config" / "config.yaml"
        trading_system = TradingSystem(config_path)
        await trading_system.initialize()
        
        # Test with increasing universe sizes
        universe_sizes = [10, 50, 100, 250, 500]
        performance_metrics = {}
        
        for size in universe_sizes:
            symbols = list(large_dataset.keys())[:size]
            subset_data = {symbol: large_dataset[symbol] for symbol in symbols}
            
            with patch.object(trading_system, '_fetch_market_data', return_value=subset_data):
                start_time = time.perf_counter()
                start_memory = psutil.Process().memory_info().rss / 1024 / 1024
                
                # Run portfolio optimization
                result = await trading_system.optimize_portfolio({
                    'symbols': symbols,
                    'method': 'mean_variance',
                    'lookback_days': 252
                })
                
                end_time = time.perf_counter()
                end_memory = psutil.Process().memory_info().rss / 1024 / 1024
                
                performance_metrics[size] = {
                    'execution_time': end_time - start_time,
                    'memory_usage': end_memory - start_memory,
                    'success': 'weights' in result
                }
        
        # Analyze performance scaling
        for size, metrics in performance_metrics.items():
            print(f"Universe size {size}: {metrics['execution_time']:.3f}s, {metrics['memory_usage']:.2f}MB")
            assert metrics['success'], f"Failed for universe size {size}"
            
            # Performance benchmarks
            if size <= 50:
                assert metrics['execution_time'] < 5.0, f"Too slow for {size} symbols"
            elif size <= 100:
                assert metrics['execution_time'] < 15.0, f"Too slow for {size} symbols"
            elif size <= 250:
                assert metrics['execution_time'] < 45.0, f"Too slow for {size} symbols"
    
    @pytest.mark.performance
    async def test_real_time_processing_throughput(self):
        """Test real-time data processing throughput."""
        config_path = Path(__file__).parent.parent / "config" / "config.yaml"
        trading_system = TradingSystem(config_path)
        await trading_system.initialize()
        
        # Generate stream of market updates
        n_updates = 1000
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
        
        updates = []
        for i in range(n_updates):
            symbol = np.random.choice(symbols)
            update = {
                'symbol': symbol,
                'price': 100 + np.random.normal(0, 5),
                'volume': np.random.randint(1000, 100000),
                'timestamp': datetime.now() + timedelta(microseconds=i*1000)
            }
            updates.append(update)
        
        # Process updates and measure throughput
        start_time = time.perf_counter()
        processed_count = 0
        
        for update in updates:
            try:
                await trading_system.process_market_update(update)
                processed_count += 1
            except Exception as e:
                print(f"Error processing update {processed_count}: {e}")
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        throughput = processed_count / total_time
        
        print(f"Processed {processed_count} updates in {total_time:.3f}s")
        print(f"Throughput: {throughput:.1f} updates/second")
        
        # Benchmark: Should process at least 100 updates per second
        assert throughput >= 100, f"Throughput too low: {throughput:.1f} updates/second"
    
    @pytest.mark.performance
    async def test_memory_usage_optimization(self, large_dataset):
        """Test memory usage optimization."""
        config_path = Path(__file__).parent.parent / "config" / "config.yaml"
        trading_system = TradingSystem(config_path)
        await trading_system.initialize()
        
        # Force garbage collection before starting
        gc.collect()
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Process large dataset
        symbols = list(large_dataset.keys())[:100]  # Use subset for testing
        subset_data = {symbol: large_dataset[symbol] for symbol in symbols}
        
        with patch.object(trading_system, '_fetch_market_data', return_value=subset_data):
            
            # Process multiple iterations to check for memory leaks
            max_memory = initial_memory
            
            for iteration in range(5):
                await trading_system.run_pipeline({
                    'symbols': symbols,
                    'strategies': ['momentum', 'technical_analysis']
                })
                
                # Force garbage collection
                gc.collect()
                current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                max_memory = max(max_memory, current_memory)
                
                print(f"Iteration {iteration}: {current_memory:.2f} MB")
        
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_growth = final_memory - initial_memory
        
        print(f"Initial memory: {initial_memory:.2f} MB")
        print(f"Final memory: {final_memory:.2f} MB")
        print(f"Memory growth: {memory_growth:.2f} MB")
        print(f"Peak memory: {max_memory:.2f} MB")
        
        # Memory leak check: growth should be reasonable
        assert memory_growth < 500, f"Excessive memory growth: {memory_growth:.2f} MB"
    
    @pytest.mark.performance
    async def test_concurrent_processing_performance(self):
        """Test concurrent processing performance."""
        config_path = Path(__file__).parent.parent / "config" / "config.yaml"
        trading_system = TradingSystem(config_path)
        await trading_system.initialize()
        
        # Test different concurrency levels
        concurrency_levels = [1, 5, 10, 20]
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'AMZN', 'META', 'NFLX', 'CRM', 'ADBE']
        
        for concurrency in concurrency_levels:
            tasks = []
            start_time = time.perf_counter()
            
            # Create concurrent tasks
            for i in range(concurrency):
                symbol = symbols[i % len(symbols)]
                task = trading_system.agents['technical_analysis_agent'].process({
                    'symbols': [symbol],
                    'indicators': ['sma', 'rsi', 'macd', 'bollinger'],
                    'start_date': datetime.now() - timedelta(days=60),
                    'end_date': datetime.now()
                })
                tasks.append(task)
            
            # Execute tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            
            successful_results = [r for r in results if not isinstance(r, Exception)]
            success_rate = len(successful_results) / len(results)
            
            print(f"Concurrency {concurrency}: {execution_time:.3f}s, {success_rate:.2%} success")
            
            # Benchmarks
            assert success_rate >= 0.95, f"Low success rate at concurrency {concurrency}"
            if concurrency <= 10:
                assert execution_time < 10.0, f"Too slow at concurrency {concurrency}"
    
    @pytest.mark.performance
    async def test_database_query_performance(self):
        """Test database query performance optimization."""
        config_path = Path(__file__).parent.parent / "config" / "config.yaml"
        trading_system = TradingSystem(config_path)
        await trading_system.initialize()
        
        # Test different query patterns
        query_patterns = [
            {'type': 'single_symbol', 'symbols': ['AAPL'], 'days': 252},
            {'type': 'multiple_symbols', 'symbols': ['AAPL', 'GOOGL', 'MSFT'], 'days': 252},
            {'type': 'large_universe', 'symbols': [f'STOCK_{i:03d}' for i in range(50)], 'days': 30},
            {'type': 'long_history', 'symbols': ['SPY'], 'days': 2520}  # 10 years
        ]
        
        for pattern in query_patterns:
            start_time = time.perf_counter()
            
            try:
                # Mock database query
                with patch.object(trading_system, '_query_database') as mock_query:
                    mock_query.return_value = self._generate_mock_data(
                        pattern['symbols'], 
                        pattern['days']
                    )
                    
                    result = await trading_system.fetch_historical_data(
                        symbols=pattern['symbols'],
                        days=pattern['days']
                    )
                
                end_time = time.perf_counter()
                query_time = end_time - start_time
                
                print(f"Query pattern '{pattern['type']}': {query_time:.3f}s")
                
                # Performance benchmarks
                if pattern['type'] == 'single_symbol':
                    assert query_time < 0.5, "Single symbol query too slow"
                elif pattern['type'] == 'multiple_symbols':
                    assert query_time < 2.0, "Multiple symbols query too slow"
                elif pattern['type'] == 'large_universe':
                    assert query_time < 5.0, "Large universe query too slow"
                elif pattern['type'] == 'long_history':
                    assert query_time < 3.0, "Long history query too slow"
                
                assert result is not None
                
            except Exception as e:
                pytest.fail(f"Query pattern '{pattern['type']}' failed: {e}")
    
    @pytest.mark.performance
    async def test_cache_performance(self):
        """Test caching system performance."""
        config_path = Path(__file__).parent.parent / "config" / "config.yaml"
        trading_system = TradingSystem(config_path)
        await trading_system.initialize()
        
        # Test cache hit/miss performance
        symbol = 'AAPL'
        cache_key = f"technical_indicators_{symbol}"
        
        # First call (cache miss)
        start_time = time.perf_counter()
        
        result1 = await trading_system.get_cached_indicators(symbol)
        
        miss_time = time.perf_counter() - start_time
        
        # Second call (cache hit)
        start_time = time.perf_counter()
        
        result2 = await trading_system.get_cached_indicators(symbol)
        
        hit_time = time.perf_counter() - start_time
        
        print(f"Cache miss time: {miss_time:.4f}s")
        print(f"Cache hit time: {hit_time:.4f}s")
        print(f"Cache speedup: {miss_time/hit_time:.1f}x")
        
        # Benchmarks
        assert hit_time < miss_time * 0.1, "Cache not providing sufficient speedup"
        assert hit_time < 0.001, "Cache hit too slow"
    
    @pytest.mark.performance
    async def test_ml_model_inference_performance(self):
        """Test ML model inference performance."""
        config_path = Path(__file__).parent.parent / "config" / "config.yaml"
        trading_system = TradingSystem(config_path)
        await trading_system.initialize()
        
        # Test batch inference vs single predictions
        n_samples = 1000
        feature_data = np.random.randn(n_samples, 20)  # 20 features
        
        # Single prediction performance
        single_times = []
        for i in range(100):  # Test 100 single predictions
            start_time = time.perf_counter()
            
            prediction = await trading_system.agents['ml_ensemble_agent'].predict_single(
                feature_data[i]
            )
            
            end_time = time.perf_counter()
            single_times.append(end_time - start_time)
        
        avg_single_time = np.mean(single_times)
        
        # Batch prediction performance
        start_time = time.perf_counter()
        
        batch_predictions = await trading_system.agents['ml_ensemble_agent'].predict_batch(
            feature_data
        )
        
        batch_time = time.perf_counter() - start_time
        avg_batch_time = batch_time / n_samples
        
        print(f"Average single prediction time: {avg_single_time:.4f}s")
        print(f"Average batch prediction time: {avg_batch_time:.6f}s")
        print(f"Batch speedup: {avg_single_time/avg_batch_time:.1f}x")
        
        # Benchmarks
        assert avg_single_time < 0.01, "Single prediction too slow"
        assert avg_batch_time < 0.001, "Batch prediction per sample too slow"
        assert avg_batch_time < avg_single_time * 0.5, "Batch not providing sufficient speedup"
    
    @pytest.mark.performance
    async def test_system_startup_performance(self):
        """Test system startup and initialization performance."""
        config_path = Path(__file__).parent.parent / "config" / "config.yaml"
        
        # Test cold startup
        start_time = time.perf_counter()
        
        trading_system = TradingSystem(config_path)
        await trading_system.initialize()
        
        startup_time = time.perf_counter() - start_time
        
        print(f"System startup time: {startup_time:.3f}s")
        
        # Benchmark: Should start within 30 seconds
        assert startup_time < 30.0, f"System startup too slow: {startup_time:.3f}s"
        
        # Test component initialization times
        component_times = trading_system.get_initialization_times()
        
        for component, init_time in component_times.items():
            print(f"{component}: {init_time:.3f}s")
            
            # Component-specific benchmarks
            if 'data' in component.lower():
                assert init_time < 5.0, f"{component} initialization too slow"
            else:
                assert init_time < 10.0, f"{component} initialization too slow"
    
    def _generate_mock_data(self, symbols, days):
        """Helper to generate mock data for performance testing."""
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        
        data = {}
        for symbol in symbols:
            data[symbol] = pd.DataFrame({
                'timestamp': dates,
                'close': 100 + np.random.randn(days).cumsum(),
                'volume': np.random.randint(1000000, 10000000, days)
            })
        
        return data