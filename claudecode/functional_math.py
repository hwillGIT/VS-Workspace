"""
Functional Mathematical Operations for Trading and Finance

This module provides functional programming patterns for mathematical operations,
building on the functional_utils module to create pure, composable mathematical
functions for trading system use.

Key features:
- Pure functions for all mathematical operations
- Immutable data handling
- Composable transformation pipelines
- Lazy evaluation for performance
- Safe operations with Maybe/Either monads
"""

import numpy as np
import pandas as pd
from typing import Callable, List, Tuple, Union, Optional, Dict, Any
from functools import partial, reduce
from itertools import accumulate, pairwise
import math
from dataclasses import dataclass
from enum import Enum

from .functional_utils import Maybe, Either, FunctionalList, FunctionalOps, fl, fmap, ffilter


class ReturnType(Enum):
    """Types of return calculations."""
    SIMPLE = "simple"
    LOG = "log"
    COMPOUND = "compound"


@dataclass(frozen=True)
class PricePoint:
    """Immutable price point with timestamp."""
    timestamp: pd.Timestamp
    price: float
    volume: Optional[float] = None
    
    def __post_init__(self):
        if self.price <= 0:
            raise ValueError("Price must be positive")


@dataclass(frozen=True)
class ReturnPoint:
    """Immutable return point."""
    timestamp: pd.Timestamp
    return_value: float
    return_type: ReturnType


@dataclass(frozen=True)
class RiskMetrics:
    """Immutable risk metrics container."""
    volatility: float
    var_95: float
    cvar_95: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    
    def __post_init__(self):
        if self.volatility < 0:
            raise ValueError("Volatility cannot be negative")


class FunctionalMath:
    """
    Functional mathematical operations for trading and finance.
    All functions are pure and return new immutable data structures.
    """
    
    # ============================================================================
    # Pure Price Operations
    # ============================================================================
    
    @staticmethod
    def create_price_series(prices: List[float], 
                          timestamps: Optional[List[pd.Timestamp]] = None,
                          volumes: Optional[List[float]] = None) -> FunctionalList[PricePoint]:
        """Create immutable price series from raw data."""
        if timestamps is None:
            timestamps = [pd.Timestamp.now() + pd.Timedelta(minutes=i) for i in range(len(prices))]
        
        if volumes is None:
            volumes = [None] * len(prices)
        
        return fl([
            PricePoint(ts, price, vol) 
            for ts, price, vol in zip(timestamps, prices, volumes)
        ])
    
    @staticmethod
    def extract_prices(price_series: FunctionalList[PricePoint]) -> FunctionalList[float]:
        """Extract prices from price series."""
        return price_series.map(lambda p: p.price)
    
    @staticmethod
    def extract_timestamps(price_series: FunctionalList[PricePoint]) -> FunctionalList[pd.Timestamp]:
        """Extract timestamps from price series."""
        return price_series.map(lambda p: p.timestamp)
    
    # ============================================================================
    # Pure Return Calculations
    # ============================================================================
    
    @staticmethod
    def calculate_simple_return(prev_price: float, curr_price: float) -> Maybe[float]:
        """Calculate simple return between two prices."""
        if prev_price <= 0:
            return Maybe.none()
        return Maybe.some((curr_price - prev_price) / prev_price)
    
    @staticmethod
    def calculate_log_return(prev_price: float, curr_price: float) -> Maybe[float]:
        """Calculate log return between two prices."""
        if prev_price <= 0 or curr_price <= 0:
            return Maybe.none()
        return Maybe.some(math.log(curr_price / prev_price))
    
    @staticmethod
    def price_series_to_returns(price_series: FunctionalList[PricePoint], 
                               return_type: ReturnType = ReturnType.SIMPLE) -> FunctionalList[ReturnPoint]:
        """Convert price series to return series functionally."""
        prices = FunctionalMath.extract_prices(price_series)
        timestamps = FunctionalMath.extract_timestamps(price_series)
        
        if len(prices) < 2:
            return fl([])
        
        # Create pairs of consecutive prices
        price_pairs = fl(list(zip(prices._items[:-1], prices._items[1:])))
        return_timestamps = timestamps.drop(1)  # Skip first timestamp
        
        # Calculate returns based on type
        if return_type == ReturnType.SIMPLE:
            calc_func = lambda pair: FunctionalMath.calculate_simple_return(pair[0], pair[1])
        elif return_type == ReturnType.LOG:
            calc_func = lambda pair: FunctionalMath.calculate_log_return(pair[0], pair[1])
        else:
            raise ValueError(f"Unsupported return type: {return_type}")
        
        # Map calculation over price pairs
        maybe_returns = price_pairs.map(calc_func)
        
        # Filter out None values and create return points
        return_points = []
        for maybe_ret, ts in zip(maybe_returns, return_timestamps):
            if maybe_ret.is_some():
                return_points.append(ReturnPoint(ts, maybe_ret.value, return_type))
        
        return fl(return_points)
    
    @staticmethod
    def extract_return_values(return_series: FunctionalList[ReturnPoint]) -> FunctionalList[float]:
        """Extract return values from return series."""
        return return_series.map(lambda r: r.return_value)
    
    # ============================================================================
    # Pure Statistical Operations
    # ============================================================================
    
    @staticmethod
    def safe_mean(values: FunctionalList[float]) -> Maybe[float]:
        """Calculate mean safely."""
        if values.is_empty():
            return Maybe.none()
        return Maybe.some(sum(values) / len(values))
    
    @staticmethod
    def safe_variance(values: FunctionalList[float], ddof: int = 1) -> Maybe[float]:
        """Calculate variance safely."""
        if len(values) <= ddof:
            return Maybe.none()
        
        mean_val = FunctionalMath.safe_mean(values)
        if mean_val.is_none():
            return Maybe.none()
        
        mean = mean_val.value
        squared_diffs = values.map(lambda x: (x - mean) ** 2)
        variance = sum(squared_diffs) / (len(values) - ddof)
        return Maybe.some(variance)
    
    @staticmethod
    def safe_std(values: FunctionalList[float], ddof: int = 1) -> Maybe[float]:
        """Calculate standard deviation safely."""
        return FunctionalMath.safe_variance(values, ddof).map(math.sqrt)
    
    @staticmethod
    def safe_percentile(values: FunctionalList[float], percentile: float) -> Maybe[float]:
        """Calculate percentile safely."""
        if values.is_empty():
            return Maybe.none()
        
        sorted_vals = values.sort()
        n = len(sorted_vals)
        index = (percentile / 100) * (n - 1)
        
        if index == int(index):
            return Maybe.some(sorted_vals[int(index)])
        
        # Linear interpolation
        lower_idx = int(index)
        upper_idx = lower_idx + 1
        
        if upper_idx >= n:
            return Maybe.some(sorted_vals[-1])
        
        lower_val = sorted_vals[lower_idx]
        upper_val = sorted_vals[upper_idx]
        weight = index - lower_idx
        
        return Maybe.some(lower_val + weight * (upper_val - lower_val))
    
    # ============================================================================
    # Pure Risk Calculations
    # ============================================================================
    
    @staticmethod
    def calculate_volatility(returns: FunctionalList[float], 
                           annualize: bool = True, 
                           periods_per_year: int = 252) -> Maybe[float]:
        """Calculate volatility functionally."""
        std_result = FunctionalMath.safe_std(returns)
        if std_result.is_none():
            return Maybe.none()
        
        vol = std_result.value
        if annualize:
            vol *= math.sqrt(periods_per_year)
        
        return Maybe.some(vol)
    
    @staticmethod
    def calculate_var(returns: FunctionalList[float], confidence_level: float = 0.05) -> Maybe[float]:
        """Calculate Value at Risk functionally."""
        return FunctionalMath.safe_percentile(returns, confidence_level * 100)
    
    @staticmethod
    def calculate_cvar(returns: FunctionalList[float], confidence_level: float = 0.05) -> Maybe[float]:
        """Calculate Conditional VaR functionally."""
        var_result = FunctionalMath.calculate_var(returns, confidence_level)
        if var_result.is_none():
            return Maybe.none()
        
        var_value = var_result.value
        tail_returns = returns.filter(lambda r: r <= var_value)
        
        return FunctionalMath.safe_mean(tail_returns)
    
    @staticmethod
    def calculate_sharpe_ratio(returns: FunctionalList[float], 
                             risk_free_rate: float = 0.0,
                             periods_per_year: int = 252) -> Maybe[float]:
        """Calculate Sharpe ratio functionally."""
        mean_result = FunctionalMath.safe_mean(returns)
        std_result = FunctionalMath.safe_std(returns)
        
        if mean_result.is_none() or std_result.is_none():
            return Maybe.none()
        
        annual_return = mean_result.value * periods_per_year
        annual_vol = std_result.value * math.sqrt(periods_per_year)
        
        if annual_vol == 0:
            return Maybe.some(0.0)
        
        return Maybe.some((annual_return - risk_free_rate) / annual_vol)
    
    @staticmethod
    def calculate_sortino_ratio(returns: FunctionalList[float], 
                              risk_free_rate: float = 0.0,
                              periods_per_year: int = 252) -> Maybe[float]:
        """Calculate Sortino ratio functionally."""
        mean_result = FunctionalMath.safe_mean(returns)
        if mean_result.is_none():
            return Maybe.none()
        
        downside_returns = returns.filter(lambda r: r < 0)
        if downside_returns.is_empty():
            return Maybe.some(float('inf'))
        
        downside_std_result = FunctionalMath.safe_std(downside_returns)
        if downside_std_result.is_none():
            return Maybe.some(float('inf'))
        
        annual_return = mean_result.value * periods_per_year
        annual_downside_vol = downside_std_result.value * math.sqrt(periods_per_year)
        
        if annual_downside_vol == 0:
            return Maybe.some(float('inf'))
        
        return Maybe.some((annual_return - risk_free_rate) / annual_downside_vol)
    
    @staticmethod
    def calculate_max_drawdown(prices: FunctionalList[float]) -> Maybe[Tuple[float, int, int]]:
        """Calculate maximum drawdown functionally."""
        if prices.is_empty():
            return Maybe.none()
        
        # Calculate cumulative maximum
        cummax_list = []
        current_max = prices[0]
        
        for price in prices:
            current_max = max(current_max, price)
            cummax_list.append(current_max)
        
        cummax = fl(cummax_list)
        
        # Calculate drawdowns
        drawdowns = prices.zip_with(cummax, lambda price, peak: (price - peak) / peak)
        
        if drawdowns.is_empty():
            return Maybe.none()
        
        # Find maximum drawdown
        min_dd = min(drawdowns)
        end_idx = next(i for i, dd in enumerate(drawdowns) if dd == min_dd)
        
        # Find the peak before the max drawdown
        start_idx = next(i for i, price in enumerate(prices._items[:end_idx+1]) 
                        if price == cummax_list[end_idx])
        
        return Maybe.some((min_dd, start_idx, end_idx))
    
    @staticmethod
    def calculate_comprehensive_risk_metrics(returns: FunctionalList[float],
                                           risk_free_rate: float = 0.0,
                                           periods_per_year: int = 252) -> Maybe[RiskMetrics]:
        """Calculate comprehensive risk metrics functionally."""
        # Calculate all risk metrics
        vol_result = FunctionalMath.calculate_volatility(returns, True, periods_per_year)
        var_result = FunctionalMath.calculate_var(returns)
        cvar_result = FunctionalMath.calculate_cvar(returns)
        sharpe_result = FunctionalMath.calculate_sharpe_ratio(returns, risk_free_rate, periods_per_year)
        sortino_result = FunctionalMath.calculate_sortino_ratio(returns, risk_free_rate, periods_per_year)
        
        # For max drawdown, we need to convert returns to cumulative returns (prices)
        cumulative_returns = fl(list(accumulate(returns, lambda acc, ret: acc * (1 + ret), initial=1.0)))
        max_dd_result = FunctionalMath.calculate_max_drawdown(cumulative_returns)
        
        # Check if all calculations succeeded
        if any(result.is_none() for result in [vol_result, var_result, cvar_result, 
                                              sharpe_result, sortino_result]):
            return Maybe.none()
        
        max_dd = max_dd_result.get_or_else((0.0, 0, 0))[0]
        
        # Calculate Calmar ratio
        mean_result = FunctionalMath.safe_mean(returns)
        if mean_result.is_none():
            return Maybe.none()
        
        annual_return = mean_result.value * periods_per_year
        calmar = annual_return / abs(max_dd) if max_dd != 0 else float('inf')
        
        return Maybe.some(RiskMetrics(
            volatility=vol_result.value,
            var_95=var_result.value,
            cvar_95=cvar_result.value,
            max_drawdown=max_dd,
            sharpe_ratio=sharpe_result.value,
            sortino_ratio=sortino_result.value,
            calmar_ratio=calmar
        ))
    
    # ============================================================================
    # Functional Pipeline Operations
    # ============================================================================
    
    @staticmethod
    def create_risk_analysis_pipeline() -> Callable[[FunctionalList[PricePoint]], Maybe[RiskMetrics]]:
        """Create a functional pipeline for risk analysis."""
        return FunctionalOps.compose(
            FunctionalMath.calculate_comprehensive_risk_metrics,
            FunctionalMath.extract_return_values,
            partial(FunctionalMath.price_series_to_returns, return_type=ReturnType.SIMPLE)
        )
    
    @staticmethod
    def create_return_analysis_pipeline(return_type: ReturnType = ReturnType.SIMPLE) -> Callable:
        """Create a functional pipeline for return analysis."""
        return FunctionalOps.pipe(
            partial(FunctionalMath.price_series_to_returns, return_type=return_type),
            FunctionalMath.extract_return_values,
            lambda returns: {
                'mean': FunctionalMath.safe_mean(returns),
                'volatility': FunctionalMath.calculate_volatility(returns),
                'sharpe': FunctionalMath.calculate_sharpe_ratio(returns),
                'var': FunctionalMath.calculate_var(returns),
                'cvar': FunctionalMath.calculate_cvar(returns)
            }
        )
    
    # ============================================================================
    # Utility Functions
    # ============================================================================
    
    @staticmethod
    def safe_divide(a: float, b: float) -> Maybe[float]:
        """Safe division returning Maybe."""
        if b == 0:
            return Maybe.none()
        return Maybe.some(a / b)
    
    @staticmethod
    def safe_log(x: float) -> Maybe[float]:
        """Safe logarithm returning Maybe."""
        if x <= 0:
            return Maybe.none()
        return Maybe.some(math.log(x))
    
    @staticmethod
    def safe_sqrt(x: float) -> Maybe[float]:
        """Safe square root returning Maybe."""
        if x < 0:
            return Maybe.none()
        return Maybe.some(math.sqrt(x))
    
    @staticmethod
    def clamp(value: float, min_val: float, max_val: float) -> float:
        """Clamp value between min and max."""
        return max(min_val, min(max_val, value))
    
    @staticmethod
    def normalize_weights(weights: FunctionalList[float]) -> Maybe[FunctionalList[float]]:
        """Normalize weights to sum to 1."""
        total = sum(weights)
        if total == 0:
            return Maybe.none()
        return Maybe.some(weights.map(lambda w: w / total))


# Convenience functions for common operations
def safe_returns(prices: List[float], 
                return_type: ReturnType = ReturnType.SIMPLE) -> FunctionalList[float]:
    """Convert prices to returns safely."""
    price_series = FunctionalMath.create_price_series(prices)
    return_series = FunctionalMath.price_series_to_returns(price_series, return_type)
    return FunctionalMath.extract_return_values(return_series)


def risk_pipeline(prices: List[float]) -> Maybe[RiskMetrics]:
    """Complete risk analysis pipeline."""
    price_series = FunctionalMath.create_price_series(prices)
    pipeline = FunctionalMath.create_risk_analysis_pipeline()
    return pipeline(price_series)


def return_stats(prices: List[float], return_type: ReturnType = ReturnType.SIMPLE) -> Dict[str, Maybe]:
    """Calculate return statistics."""
    price_series = FunctionalMath.create_price_series(prices)
    pipeline = FunctionalMath.create_return_analysis_pipeline(return_type)
    return pipeline(price_series)


# Export important classes and functions
__all__ = [
    'PricePoint', 'ReturnPoint', 'RiskMetrics', 'ReturnType',
    'FunctionalMath', 'safe_returns', 'risk_pipeline', 'return_stats'
]