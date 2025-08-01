"""
Mathematical utilities for the trading system.

This module integrates functional programming patterns for enhanced reliability,
composability, and immutability in mathematical operations.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple, List, Callable
from scipy import stats
from scipy.optimize import minimize
import warnings
from functools import partial, reduce

# Import functional programming utilities from global ClaudeCode level
try:
    from ....functional_utils import Maybe, Either, FunctionalList, FunctionalOps, fl, fmap, ffilter
    from ....functional_math import FunctionalMath, safe_returns, risk_pipeline
    FUNCTIONAL_AVAILABLE = True
except ImportError:
    FUNCTIONAL_AVAILABLE = False
    warnings.warn("Functional programming utilities not available. Install ClaudeCode functional modules.")

# Import for backward compatibility
import functools


class MathUtils:
    """
    Mathematical utility functions for trading and finance calculations.
    """
    
    @staticmethod
    def calculate_returns(prices: Union[pd.Series, np.ndarray], 
                         method: str = "simple") -> Union[pd.Series, np.ndarray]:
        """
        Calculate returns from price series.
        
        Args:
            prices: Price series
            method: 'simple' or 'log' returns
            
        Returns:
            Return series
        """
        if isinstance(prices, pd.Series):
            if method == "simple":
                return prices.pct_change()
            elif method == "log":
                return np.log(prices / prices.shift(1))
        else:
            if method == "simple":
                return np.diff(prices) / prices[:-1]
            elif method == "log":
                return np.diff(np.log(prices))
        
        raise ValueError("Method must be 'simple' or 'log'")
    
    @staticmethod
    def calculate_volatility(returns: Union[pd.Series, np.ndarray], 
                           annualize: bool = True, 
                           periods_per_year: int = 252) -> float:
        """
        Calculate volatility from returns.
        
        Args:
            returns: Return series
            annualize: Whether to annualize the volatility
            periods_per_year: Number of periods per year for annualization
            
        Returns:
            Volatility
        """
        if isinstance(returns, pd.Series):
            vol = returns.std()
        else:
            vol = np.std(returns, ddof=1)
        
        if annualize:
            vol *= np.sqrt(periods_per_year)
        
        return vol
    
    @staticmethod
    def calculate_sharpe_ratio(returns: Union[pd.Series, np.ndarray], 
                              risk_free_rate: float = 0.0,
                              periods_per_year: int = 252) -> float:
        """
        Calculate Sharpe ratio.
        
        Args:
            returns: Return series
            risk_free_rate: Risk-free rate (annualized)
            periods_per_year: Number of periods per year
            
        Returns:
            Sharpe ratio
        """
        if isinstance(returns, pd.Series):
            mean_return = returns.mean()
            std_return = returns.std()
        else:
            mean_return = np.mean(returns)
            std_return = np.std(returns, ddof=1)
        
        # Annualize
        annual_return = mean_return * periods_per_year
        annual_vol = std_return * np.sqrt(periods_per_year)
        
        if annual_vol == 0:
            return 0.0
        
        return (annual_return - risk_free_rate) / annual_vol
    
    @staticmethod
    def calculate_sortino_ratio(returns: Union[pd.Series, np.ndarray], 
                               risk_free_rate: float = 0.0,
                               periods_per_year: int = 252) -> float:
        """
        Calculate Sortino ratio (uses downside deviation instead of total volatility).
        
        Args:
            returns: Return series
            risk_free_rate: Risk-free rate (annualized)
            periods_per_year: Number of periods per year
            
        Returns:
            Sortino ratio
        """
        if isinstance(returns, pd.Series):
            mean_return = returns.mean()
            downside_returns = returns[returns < 0]
        else:
            mean_return = np.mean(returns)
            downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0:
            return np.inf
        
        downside_std = np.std(downside_returns, ddof=1)
        
        # Annualize
        annual_return = mean_return * periods_per_year
        annual_downside_vol = downside_std * np.sqrt(periods_per_year)
        
        if annual_downside_vol == 0:
            return np.inf
        
        return (annual_return - risk_free_rate) / annual_downside_vol
    
    @staticmethod
    def calculate_max_drawdown(prices: Union[pd.Series, np.ndarray]) -> Tuple[float, int, int]:
        """
        Calculate maximum drawdown.
        
        Args:
            prices: Price series
            
        Returns:
            Tuple of (max_drawdown, start_idx, end_idx)
        """
        if isinstance(prices, pd.Series):
            cumulative = (1 + prices.pct_change().fillna(0)).cumprod()
            prices_array = cumulative.values
        else:
            prices_array = prices
        
        peak = np.maximum.accumulate(prices_array)
        drawdown = (prices_array - peak) / peak
        
        max_dd = np.min(drawdown)
        end_idx = np.argmin(drawdown)
        
        # Find the peak before the max drawdown
        start_idx = np.argmax(prices_array[:end_idx+1])
        
        return max_dd, start_idx, end_idx
    
    @staticmethod
    def calculate_var(returns: Union[pd.Series, np.ndarray], 
                     confidence_level: float = 0.05) -> float:
        """
        Calculate Value at Risk (VaR).
        
        Args:
            returns: Return series
            confidence_level: Confidence level (e.g., 0.05 for 95% VaR)
            
        Returns:
            VaR value (negative number)
        """
        if isinstance(returns, pd.Series):
            returns_clean = returns.dropna()
        else:
            returns_clean = returns[~np.isnan(returns)]
        
        return np.percentile(returns_clean, confidence_level * 100)
    
    @staticmethod
    def calculate_cvar(returns: Union[pd.Series, np.ndarray], 
                      confidence_level: float = 0.05) -> float:
        """
        Calculate Conditional Value at Risk (CVaR/Expected Shortfall).
        
        Args:
            returns: Return series
            confidence_level: Confidence level
            
        Returns:
            CVaR value
        """
        var = MathUtils.calculate_var(returns, confidence_level)
        
        if isinstance(returns, pd.Series):
            tail_returns = returns[returns <= var]
        else:
            tail_returns = returns[returns <= var]
        
        return np.mean(tail_returns) if len(tail_returns) > 0 else var
    
    @staticmethod
    def calculate_beta(asset_returns: Union[pd.Series, np.ndarray], 
                      market_returns: Union[pd.Series, np.ndarray]) -> float:
        """
        Calculate beta coefficient.
        
        Args:
            asset_returns: Asset return series
            market_returns: Market return series
            
        Returns:
            Beta coefficient
        """
        if isinstance(asset_returns, pd.Series) and isinstance(market_returns, pd.Series):
            # Align the series
            combined = pd.concat([asset_returns, market_returns], axis=1).dropna()
            asset_clean = combined.iloc[:, 0]
            market_clean = combined.iloc[:, 1]
        else:
            # For numpy arrays, assume they're already aligned
            asset_clean = asset_returns
            market_clean = market_returns
        
        covariance = np.cov(asset_clean, market_clean)[0, 1]
        market_variance = np.var(market_clean, ddof=1)
        
        return covariance / market_variance if market_variance != 0 else 0
    
    @staticmethod
    def calculate_correlation(x: Union[pd.Series, np.ndarray], 
                            y: Union[pd.Series, np.ndarray]) -> float:
        """
        Calculate correlation coefficient between two series.
        
        Args:
            x: First series
            y: Second series
            
        Returns:
            Correlation coefficient
        """
        if isinstance(x, pd.Series) and isinstance(y, pd.Series):
            combined = pd.concat([x, y], axis=1).dropna()
            return combined.corr().iloc[0, 1]
        else:
            return np.corrcoef(x, y)[0, 1]
    
    @staticmethod
    def calculate_rolling_correlation(x: pd.Series, y: pd.Series, 
                                    window: int) -> pd.Series:
        """
        Calculate rolling correlation between two series.
        
        Args:
            x: First series
            y: Second series
            window: Rolling window size
            
        Returns:
            Rolling correlation series
        """
        return x.rolling(window).corr(y)
    
    @staticmethod
    def z_score(data: Union[pd.Series, np.ndarray], 
               window: Optional[int] = None) -> Union[pd.Series, np.ndarray]:
        """
        Calculate z-score of data.
        
        Args:
            data: Input data
            window: Rolling window size (if None, use entire series)
            
        Returns:
            Z-score values
        """
        if isinstance(data, pd.Series):
            if window is not None:
                mean = data.rolling(window).mean()
                std = data.rolling(window).std()
                return (data - mean) / std
            else:
                return (data - data.mean()) / data.std()
        else:
            if window is not None:
                # For numpy arrays with rolling window, need to implement manually
                z_scores = np.full_like(data, np.nan)
                for i in range(window-1, len(data)):
                    window_data = data[i-window+1:i+1]
                    z_scores[i] = (data[i] - np.mean(window_data)) / np.std(window_data, ddof=1)
                return z_scores
            else:
                return (data - np.mean(data)) / np.std(data, ddof=1)
    
    @staticmethod
    def calculate_information_ratio(active_returns: Union[pd.Series, np.ndarray]) -> float:
        """
        Calculate information ratio.
        
        Args:
            active_returns: Active returns (portfolio - benchmark)
            
        Returns:
            Information ratio
        """
        if isinstance(active_returns, pd.Series):
            mean_active = active_returns.mean()
            std_active = active_returns.std()
        else:
            mean_active = np.mean(active_returns)
            std_active = np.std(active_returns, ddof=1)
        
        return mean_active / std_active if std_active != 0 else 0
    
    @staticmethod
    def calculate_calmar_ratio(returns: Union[pd.Series, np.ndarray], 
                              periods_per_year: int = 252) -> float:
        """
        Calculate Calmar ratio (Annual Return / Max Drawdown).
        
        Args:
            returns: Return series
            periods_per_year: Number of periods per year
            
        Returns:
            Calmar ratio
        """
        if isinstance(returns, pd.Series):
            annual_return = returns.mean() * periods_per_year
            cumulative_returns = (1 + returns).cumprod()
        else:
            annual_return = np.mean(returns) * periods_per_year
            cumulative_returns = np.cumprod(1 + returns)
        
        max_dd, _, _ = MathUtils.calculate_max_drawdown(cumulative_returns)
        
        return annual_return / abs(max_dd) if max_dd != 0 else np.inf
    
    @staticmethod
    def normalize_weights(weights: np.ndarray) -> np.ndarray:
        """
        Normalize weights to sum to 1.
        
        Args:
            weights: Weight array
            
        Returns:
            Normalized weights
        """
        weight_sum = np.sum(weights)
        return weights / weight_sum if weight_sum != 0 else weights
    
    @staticmethod
    def calculate_portfolio_metrics(weights: np.ndarray, 
                                  expected_returns: np.ndarray,
                                  covariance_matrix: np.ndarray) -> Tuple[float, float]:
        """
        Calculate portfolio expected return and volatility.
        
        Args:
            weights: Portfolio weights
            expected_returns: Expected returns for each asset
            covariance_matrix: Covariance matrix of returns
            
        Returns:
            Tuple of (expected_return, volatility)
        """
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_variance = np.dot(weights.T, np.dot(covariance_matrix, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        return portfolio_return, portfolio_volatility
    
    @staticmethod
    def detect_regime_change(data: Union[pd.Series, np.ndarray], 
                           method: str = "variance") -> np.ndarray:
        """
        Simple regime change detection.
        
        Args:
            data: Input data series
            method: Detection method ('variance' or 'mean')
            
        Returns:
            Array of regime indicators
        """
        if isinstance(data, pd.Series):
            data_array = data.values
        else:
            data_array = data
        
        if method == "variance":
            # Use rolling variance to detect regime changes
            window = min(50, len(data_array) // 4)
            rolling_var = pd.Series(data_array).rolling(window).var()
            threshold = rolling_var.median() * 2
            regimes = (rolling_var > threshold).astype(int)
        elif method == "mean":
            # Use rolling mean to detect regime changes
            window = min(50, len(data_array) // 4)
            rolling_mean = pd.Series(data_array).rolling(window).mean()
            threshold = rolling_mean.std()
            regimes = (rolling_mean > rolling_mean.median() + threshold).astype(int)
        else:
            raise ValueError("Method must be 'variance' or 'mean'")
        
        return regimes.values if isinstance(regimes, pd.Series) else regimes
    
    # ============================================================================
    # Functional Programming Enhanced Methods
    # ============================================================================
    
    @staticmethod
    def safe_calculate_returns(prices: Union[pd.Series, np.ndarray, List[float]], 
                              method: str = "simple") -> 'Maybe[Union[pd.Series, np.ndarray]]':
        """
        Calculate returns safely using functional programming patterns.
        Returns Maybe monad for safe error handling.
        """
        if not FUNCTIONAL_AVAILABLE:
            # Fallback to original method
            try:
                result = MathUtils.calculate_returns(prices, method)
                return type('Maybe', (), {'value': result, 'is_some': lambda: True, 'get_or_else': lambda default: result})()
            except Exception:
                return type('Maybe', (), {'value': None, 'is_some': lambda: False, 'get_or_else': lambda default: default})()
        
        try:
            if isinstance(prices, list):
                prices = np.array(prices)
            
            if len(prices) < 2:
                return Maybe.none()
            
            result = MathUtils.calculate_returns(prices, method)
            return Maybe.some(result)
        except Exception:
            return Maybe.none()
    
    @staticmethod
    def functional_portfolio_optimization(returns: List[List[float]], 
                                        risk_free_rate: float = 0.0) -> 'Maybe[Tuple[np.ndarray, float, float]]':
        """
        Portfolio optimization using functional programming patterns.
        Returns Maybe with (weights, expected_return, volatility) or None if optimization fails.
        """
        if not FUNCTIONAL_AVAILABLE:
            return Maybe.none() if FUNCTIONAL_AVAILABLE else type('Maybe', (), {'is_some': lambda: False})()
        
        try:
            # Convert to functional lists
            return_series = fl([fl(asset_returns) for asset_returns in returns])
            
            # Calculate statistics functionally
            means = return_series.map(lambda asset: FunctionalMath.safe_mean(asset))
            
            # Check if all means calculated successfully
            if any(mean.is_none() for mean in means):
                return Maybe.none()
            
            mean_returns = np.array([mean.value for mean in means])
            
            # Convert back to numpy for covariance calculation
            returns_matrix = np.array(returns).T
            cov_matrix = np.cov(returns_matrix, rowvar=False)
            
            # Optimize portfolio
            n_assets = len(mean_returns)
            weights = np.ones(n_assets) / n_assets  # Equal weights as starting point
            
            def objective(w):
                portfolio_return = np.dot(w, mean_returns)
                portfolio_variance = np.dot(w.T, np.dot(cov_matrix, w))
                sharpe = (portfolio_return - risk_free_rate) / np.sqrt(portfolio_variance) if portfolio_variance > 0 else 0
                return -sharpe  # Minimize negative Sharpe ratio
            
            constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
            bounds = tuple((0, 1) for _ in range(n_assets))
            
            result = minimize(objective, weights, method='SLSQP', bounds=bounds, constraints=constraints)
            
            if result.success:
                optimal_weights = result.x
                expected_return = np.dot(optimal_weights, mean_returns)
                volatility = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
                return Maybe.some((optimal_weights, expected_return, volatility))
            else:
                return Maybe.none()
                
        except Exception:
            return Maybe.none()
    
    @staticmethod
    def compose_risk_calculations(*risk_functions: Callable) -> Callable:
        """
        Compose multiple risk calculation functions into a single pipeline.
        Uses functional composition for clean, reusable risk analysis.
        """
        if not FUNCTIONAL_AVAILABLE:
            # Simple composition fallback
            return lambda data: reduce(lambda result, func: func(result), risk_functions, data)
        
        return FunctionalOps.compose(*risk_functions)
    
    @staticmethod
    def parallel_asset_analysis(price_data: Dict[str, List[float]], 
                               analysis_func: Callable[[List[float]], Any],
                               max_workers: int = None) -> Dict[str, Any]:
        """
        Analyze multiple assets in parallel using functional programming.
        """
        if not FUNCTIONAL_AVAILABLE:
            # Sequential fallback
            return {asset: analysis_func(prices) for asset, prices in price_data.items()}
        
        from ....functional_utils import ParallelOps
        
        assets = list(price_data.keys())
        price_lists = list(price_data.values())
        
        results = ParallelOps.parallel_map(analysis_func, price_lists, max_workers)
        return dict(zip(assets, results))
    
    @staticmethod
    def rolling_functional_analysis(prices: Union[pd.Series, List[float]], 
                                   window_size: int,
                                   analysis_func: Callable[[List[float]], Any]) -> List[Any]:
        """
        Apply functional analysis over rolling windows.
        """
        if isinstance(prices, pd.Series):
            prices = prices.tolist()
        
        if not FUNCTIONAL_AVAILABLE:
            # Simple rolling analysis
            results = []
            for i in range(window_size - 1, len(prices)):
                window_data = prices[i - window_size + 1:i + 1]
                results.append(analysis_func(window_data))
            return results
        
        # Functional approach
        price_list = fl(prices)
        windows = []
        
        for i in range(window_size - 1, len(prices)):
            window = price_list._items[i - window_size + 1:i + 1]
            windows.append(list(window))
        
        return fl(windows).map(analysis_func)._items
    
    @staticmethod
    def chain_transformations(*transformations: Callable) -> Callable:
        """
        Chain multiple data transformations into a single function.
        Useful for preprocessing pipelines.
        """
        if not FUNCTIONAL_AVAILABLE:
            return lambda data: reduce(lambda result, transform: transform(result), transformations, data)
        
        return FunctionalOps.pipe(*transformations)
    
    @staticmethod
    def safe_math_operation(operation: Callable, *args, **kwargs) -> 'Either':
        """
        Perform mathematical operation safely with error handling.
        Returns Either monad with result or error.
        """
        if not FUNCTIONAL_AVAILABLE:
            try:
                result = operation(*args, **kwargs)
                return type('Either', (), {
                    'is_right': lambda: True, 
                    'is_left': lambda: False,
                    'right': result,
                    'get_or_else': lambda default: result
                })()
            except Exception as e:
                return type('Either', (), {
                    'is_right': lambda: False, 
                    'is_left': lambda: True,
                    'left': e,
                    'get_or_else': lambda default: default
                })()
        
        return FunctionalOps.safe_call(operation, *args, **kwargs)
    
    @staticmethod
    def create_risk_analysis_pipeline(risk_free_rate: float = 0.0) -> Callable:
        """
        Create a comprehensive risk analysis pipeline using functional composition.
        """
        if not FUNCTIONAL_AVAILABLE:
            def pipeline(prices):
                try:
                    returns = MathUtils.calculate_returns(prices)
                    return {
                        'volatility': MathUtils.calculate_volatility(returns),
                        'sharpe': MathUtils.calculate_sharpe_ratio(returns, risk_free_rate),
                        'var': MathUtils.calculate_var(returns),
                        'max_drawdown': MathUtils.calculate_max_drawdown(prices)[0]
                    }
                except Exception as e:
                    return {'error': str(e)}
            return pipeline
        
        def pipeline(prices: List[float]) -> Dict[str, Any]:
            """Complete risk analysis pipeline."""
            maybe_result = risk_pipeline(prices)
            
            if maybe_result.is_none():
                return {'error': 'Risk calculation failed'}
            
            risk_metrics = maybe_result.value
            return {
                'volatility': risk_metrics.volatility,
                'var_95': risk_metrics.var_95,
                'cvar_95': risk_metrics.cvar_95,
                'max_drawdown': risk_metrics.max_drawdown,
                'sharpe_ratio': risk_metrics.sharpe_ratio,
                'sortino_ratio': risk_metrics.sortino_ratio,
                'calmar_ratio': risk_metrics.calmar_ratio
            }
        
        return pipeline
    
    @staticmethod
    def batch_process_assets(asset_data: Dict[str, Any], 
                           processors: List[Callable],
                           batch_size: int = 10) -> Dict[str, Dict[str, Any]]:
        """
        Process multiple assets in batches using functional programming patterns.
        """
        if not FUNCTIONAL_AVAILABLE:
            # Simple sequential processing
            results = {}
            for asset, data in asset_data.items():
                asset_results = {}
                for i, processor in enumerate(processors):
                    try:
                        asset_results[f'processor_{i}'] = processor(data)
                    except Exception as e:
                        asset_results[f'processor_{i}'] = {'error': str(e)}
                results[asset] = asset_results
            return results
        
        # Functional batch processing
        assets = list(asset_data.keys())
        data_list = list(asset_data.values())
        
        # Process in batches
        results = {}
        for i in range(0, len(assets), batch_size):
            batch_assets = assets[i:i + batch_size]
            batch_data = data_list[i:i + batch_size]
            
            # Apply all processors to batch
            for asset, data in zip(batch_assets, batch_data):
                asset_results = {}
                processor_pipeline = FunctionalOps.compose(*processors)
                
                result = FunctionalOps.safe_call(processor_pipeline, data)
                if result.is_right():
                    asset_results['combined_result'] = result.right
                else:
                    asset_results['error'] = str(result.left)
                
                results[asset] = asset_results
        
        return results