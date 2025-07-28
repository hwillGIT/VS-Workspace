"""
Empyrical-powered risk metrics engine for professional portfolio analysis.

This module provides institutional-grade risk and performance metrics using
the empyrical library, which is the same library used by Quantopian and Zipline.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import warnings
from loguru import logger

try:
    import empyrical as emp
    EMPYRICAL_AVAILABLE = True
except ImportError:
    EMPYRICAL_AVAILABLE = False
    logger.warning("Empyrical not available. Install with: pip install empyrical")

from ..base.exceptions import CalculationError, ValidationError


class EmpyricalRiskEngine:
    """
    Professional risk metrics engine using empyrical library.
    
    Provides institutional-grade performance and risk analysis including:
    - Return-based metrics (Sharpe, Sortino, Calmar, etc.)
    - Drawdown analysis (max drawdown, underwater periods)
    - Tail risk metrics (VaR, CVaR, downside deviation)
    - Factor analysis and attribution
    - Rolling risk metrics for time-varying analysis
    """
    
    def __init__(self):
        self.logger = logger.bind(component="empyrical_engine")
        
        if not EMPYRICAL_AVAILABLE:
            raise ImportError("Empyrical library is required. Install with: pip install empyrical")
        
        # Risk-free rate sources (default to US Treasury)
        self.default_risk_free_rate = 0.02  # 2% annual
        
        # Benchmark return for relative metrics
        self.default_benchmark_return = 0.10  # 10% annual market return
        
        self.logger.info("Empyrical risk engine initialized")
    
    def calculate_comprehensive_metrics(self, 
                                       returns: pd.Series,
                                       benchmark_returns: Optional[pd.Series] = None,
                                       risk_free_rate: Optional[float] = None,
                                       period: str = 'daily') -> Dict[str, Any]:
        """
        Calculate comprehensive performance and risk metrics.
        
        Args:
            returns: Series of portfolio returns
            benchmark_returns: Optional benchmark returns for relative metrics
            risk_free_rate: Annual risk-free rate (default: 2%)
            period: Return frequency ('daily', 'weekly', 'monthly')
            
        Returns:
            Dictionary with comprehensive metrics
        """
        try:
            if risk_free_rate is None:
                risk_free_rate = self.default_risk_free_rate
            
            # Validate inputs
            self._validate_returns(returns)
            
            # Convert to daily if needed
            daily_returns = self._convert_to_daily(returns, period)
            daily_rf_rate = self._annualize_to_daily(risk_free_rate)
            
            metrics = {}
            
            # Basic return metrics
            metrics.update(self._calculate_return_metrics(daily_returns, period))
            
            # Risk-adjusted metrics
            metrics.update(self._calculate_risk_adjusted_metrics(
                daily_returns, daily_rf_rate, period
            ))
            
            # Drawdown analysis
            metrics.update(self._calculate_drawdown_metrics(daily_returns))
            
            # Tail risk metrics
            metrics.update(self._calculate_tail_risk_metrics(daily_returns))
            
            # Stability and consistency metrics
            metrics.update(self._calculate_stability_metrics(daily_returns, period))
            
            # Benchmark comparison (if provided)
            if benchmark_returns is not None:
                daily_benchmark = self._convert_to_daily(benchmark_returns, period)
                metrics.update(self._calculate_relative_metrics(
                    daily_returns, daily_benchmark, daily_rf_rate
                ))
            
            # Rolling metrics for time-varying analysis
            metrics.update(self._calculate_rolling_metrics(daily_returns, period))
            
            # Overall assessment
            metrics.update(self._calculate_overall_assessment(metrics))
            
            self.logger.info(f"Calculated {len(metrics)} risk metrics")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Comprehensive metrics calculation failed: {str(e)}")
            raise CalculationError(f"Risk metrics calculation failed: {str(e)}")
    
    def _calculate_return_metrics(self, returns: pd.Series, period: str) -> Dict[str, float]:
        """Calculate basic return metrics."""
        metrics = {}
        
        try:
            # Total returns
            metrics['total_return'] = emp.cum_returns_final(returns)
            metrics['annualized_return'] = emp.annual_return(returns)
            
            # Volatility
            metrics['volatility'] = emp.annual_volatility(returns)
            metrics['daily_volatility'] = returns.std()
            
            # Simple statistics
            metrics['mean_return'] = returns.mean()
            metrics['median_return'] = returns.median()
            metrics['std_return'] = returns.std()
            metrics['skewness'] = emp.stats.skew(returns.dropna())
            metrics['kurtosis'] = emp.stats.kurtosis(returns.dropna())
            
            # Win/Loss metrics
            positive_returns = returns[returns > 0]
            negative_returns = returns[returns < 0]
            
            metrics['win_rate'] = len(positive_returns) / len(returns.dropna())
            metrics['loss_rate'] = len(negative_returns) / len(returns.dropna())
            metrics['avg_win'] = positive_returns.mean() if len(positive_returns) > 0 else 0
            metrics['avg_loss'] = negative_returns.mean() if len(negative_returns) > 0 else 0
            
            # Profit factor
            total_gains = positive_returns.sum() if len(positive_returns) > 0 else 0
            total_losses = abs(negative_returns.sum()) if len(negative_returns) > 0 else 1e-10
            metrics['profit_factor'] = total_gains / total_losses
            
        except Exception as e:
            self.logger.warning(f"Basic return metrics calculation failed: {str(e)}")
            metrics['calculation_error'] = str(e)
        
        return metrics
    
    def _calculate_risk_adjusted_metrics(self, returns: pd.Series, 
                                        risk_free_rate: float, 
                                        period: str) -> Dict[str, float]:
        """Calculate risk-adjusted performance metrics."""
        metrics = {}
        
        try:
            # Sharpe ratio
            metrics['sharpe_ratio'] = emp.sharpe_ratio(returns, risk_free_rate)
            
            # Sortino ratio (using downside deviation)
            metrics['sortino_ratio'] = emp.sortino_ratio(returns, risk_free_rate)
            
            # Calmar ratio (return / max drawdown)
            metrics['calmar_ratio'] = emp.calmar_ratio(returns)
            
            # Information ratio (if we had a benchmark, this would be more meaningful)
            metrics['information_ratio'] = emp.annual_return(returns) / emp.annual_volatility(returns)
            
            # Omega ratio
            metrics['omega_ratio'] = emp.omega_ratio(returns, risk_free_rate)
            
            # Downside deviation
            metrics['downside_deviation'] = emp.downside_risk(returns)
            
            # Capture ratios (simplified without benchmark)
            up_capture = len(returns[returns > 0]) / len(returns.dropna())
            down_capture = len(returns[returns < 0]) / len(returns.dropna())
            metrics['up_capture'] = up_capture
            metrics['down_capture'] = down_capture
            
        except Exception as e:
            self.logger.warning(f"Risk-adjusted metrics calculation failed: {str(e)}")
            metrics['risk_adjusted_error'] = str(e)
        
        return metrics
    
    def _calculate_drawdown_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate drawdown analysis metrics."""
        metrics = {}
        
        try:
            # Maximum drawdown
            metrics['max_drawdown'] = emp.max_drawdown(returns)
            
            # Drawdown series for additional analysis
            drawdown_series = emp.utils.to_drawdown_series(emp.cum_returns(returns))
            
            # Underwater periods analysis
            underwater_periods = self._analyze_underwater_periods(drawdown_series)
            metrics.update(underwater_periods)
            
            # Recovery analysis
            recovery_metrics = self._analyze_recovery_periods(returns, drawdown_series)
            metrics.update(recovery_metrics)
            
            # Drawdown frequency
            drawdown_threshold = -0.05  # 5% drawdown threshold
            significant_drawdowns = drawdown_series[drawdown_series < drawdown_threshold]
            metrics['drawdown_frequency'] = len(significant_drawdowns) / len(returns) * 252  # Annualized
            
        except Exception as e:
            self.logger.warning(f"Drawdown metrics calculation failed: {str(e)}")
            metrics['drawdown_error'] = str(e)
        
        return metrics
    
    def _calculate_tail_risk_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate tail risk and extreme value metrics."""
        metrics = {}
        
        try:
            # Value at Risk (VaR)
            metrics['var_95'] = np.percentile(returns.dropna(), 5)  # 95% VaR
            metrics['var_99'] = np.percentile(returns.dropna(), 1)  # 99% VaR
            
            # Conditional Value at Risk (Expected Shortfall)
            metrics['cvar_95'] = emp.conditional_value_at_risk(returns, cutoff=0.05)
            metrics['cvar_99'] = emp.conditional_value_at_risk(returns, cutoff=0.01)
            
            # Tail ratio
            metrics['tail_ratio'] = emp.tail_ratio(returns)
            
            # Common sense ratio
            metrics['common_sense_ratio'] = emp.common_sense_ratio(returns)
            
            # Extreme value analysis
            extreme_returns = returns[abs(returns) > returns.std() * 2]  # 2-sigma events
            metrics['extreme_event_frequency'] = len(extreme_returns) / len(returns.dropna())
            
            # Maximum single period loss
            metrics['max_single_loss'] = returns.min()
            metrics['max_single_gain'] = returns.max()
            
        except Exception as e:
            self.logger.warning(f"Tail risk metrics calculation failed: {str(e)}")
            metrics['tail_risk_error'] = str(e)
        
        return metrics
    
    def _calculate_stability_metrics(self, returns: pd.Series, period: str) -> Dict[str, float]:
        """Calculate stability and consistency metrics."""
        metrics = {}
        
        try:
            # Stability of returns
            metrics['stability_of_timeseries'] = emp.stability_of_timeseries(returns)
            
            # Rolling metrics for consistency analysis
            window = 252 if period == 'daily' else 52 if period == 'weekly' else 12  # 1 year
            
            if len(returns) > window:
                rolling_returns = returns.rolling(window=window).apply(
                    lambda x: emp.cum_returns_final(x) if len(x.dropna()) > 0 else np.nan
                )
                rolling_sharpe = returns.rolling(window=window).apply(
                    lambda x: emp.sharpe_ratio(x) if len(x.dropna()) > 0 else np.nan
                )
                
                # Consistency metrics
                metrics['return_consistency'] = 1 - (rolling_returns.std() / abs(rolling_returns.mean())) if rolling_returns.mean() != 0 else 0
                metrics['sharpe_consistency'] = 1 - (rolling_sharpe.std() / abs(rolling_sharpe.mean())) if rolling_sharpe.mean() != 0 else 0
                
                # Positive periods ratio
                metrics['positive_periods_ratio'] = len(rolling_returns[rolling_returns > 0]) / len(rolling_returns.dropna())
            
        except Exception as e:
            self.logger.warning(f"Stability metrics calculation failed: {str(e)}")
            metrics['stability_error'] = str(e)
        
        return metrics
    
    def _calculate_relative_metrics(self, returns: pd.Series, 
                                   benchmark_returns: pd.Series,
                                   risk_free_rate: float) -> Dict[str, float]:
        """Calculate metrics relative to benchmark."""
        metrics = {}
        
        try:
            # Align series
            aligned_data = pd.DataFrame({
                'portfolio': returns,
                'benchmark': benchmark_returns
            }).dropna()
            
            if len(aligned_data) == 0:
                return {'relative_metrics_error': 'No overlapping data with benchmark'}
            
            portfolio_returns = aligned_data['portfolio']
            benchmark_returns = aligned_data['benchmark']
            
            # Alpha and Beta
            metrics['alpha'] = emp.alpha(portfolio_returns, benchmark_returns, risk_free_rate)
            metrics['beta'] = emp.beta(portfolio_returns, benchmark_returns)
            
            # Tracking error
            excess_returns = portfolio_returns - benchmark_returns
            metrics['tracking_error'] = excess_returns.std() * np.sqrt(252)  # Annualized
            
            # Information ratio
            metrics['information_ratio_vs_benchmark'] = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
            
            # Up/Down capture ratios
            up_periods = benchmark_returns > 0
            down_periods = benchmark_returns < 0
            
            if up_periods.sum() > 0:
                metrics['up_capture_ratio'] = (portfolio_returns[up_periods].mean() / 
                                              benchmark_returns[up_periods].mean())
            
            if down_periods.sum() > 0:
                metrics['down_capture_ratio'] = (portfolio_returns[down_periods].mean() / 
                                                benchmark_returns[down_periods].mean())
            
            # Relative performance
            metrics['outperformance_ratio'] = len(excess_returns[excess_returns > 0]) / len(excess_returns)
            metrics['excess_return_annualized'] = excess_returns.mean() * 252
            
        except Exception as e:
            self.logger.warning(f"Relative metrics calculation failed: {str(e)}")
            metrics['relative_error'] = str(e)
        
        return metrics
    
    def _calculate_rolling_metrics(self, returns: pd.Series, period: str) -> Dict[str, Any]:
        """Calculate rolling metrics for time-varying analysis."""
        metrics = {}
        
        try:
            # Determine appropriate windows
            if period == 'daily':
                short_window = 63  # ~3 months
                long_window = 252  # ~1 year
            elif period == 'weekly':
                short_window = 13  # ~3 months
                long_window = 52   # ~1 year
            else:  # monthly
                short_window = 6   # 6 months
                long_window = 12   # 1 year
            
            if len(returns) > long_window:
                # Rolling Sharpe ratio
                rolling_sharpe = returns.rolling(window=long_window).apply(
                    lambda x: emp.sharpe_ratio(x) if len(x.dropna()) > 0 else np.nan
                )
                
                # Rolling volatility
                rolling_vol = returns.rolling(window=short_window).std() * np.sqrt(252)
                
                # Rolling maximum drawdown
                rolling_max_dd = returns.rolling(window=long_window).apply(
                    lambda x: emp.max_drawdown(x) if len(x.dropna()) > 0 else np.nan
                )
                
                # Summary statistics for rolling metrics
                metrics['rolling_sharpe_mean'] = rolling_sharpe.mean()
                metrics['rolling_sharpe_std'] = rolling_sharpe.std()
                metrics['rolling_sharpe_min'] = rolling_sharpe.min()
                metrics['rolling_sharpe_max'] = rolling_sharpe.max()
                
                metrics['rolling_volatility_mean'] = rolling_vol.mean()
                metrics['rolling_volatility_std'] = rolling_vol.std()
                
                metrics['rolling_max_dd_mean'] = rolling_max_dd.mean()
                metrics['rolling_max_dd_worst'] = rolling_max_dd.min()
                
                # Store rolling series for visualization
                metrics['rolling_metrics_series'] = {
                    'sharpe': rolling_sharpe.dropna(),
                    'volatility': rolling_vol.dropna(),
                    'max_drawdown': rolling_max_dd.dropna()
                }
            
        except Exception as e:
            self.logger.warning(f"Rolling metrics calculation failed: {str(e)}")
            metrics['rolling_error'] = str(e)
        
        return metrics
    
    def _analyze_underwater_periods(self, drawdown_series: pd.Series) -> Dict[str, float]:
        """Analyze underwater periods (consecutive drawdown periods)."""
        try:
            # Find underwater periods (negative drawdown)
            underwater = drawdown_series < 0
            
            # Find transitions
            transitions = underwater.astype(int).diff()
            start_periods = transitions[transitions == 1].index
            end_periods = transitions[transitions == -1].index
            
            # Handle edge cases
            if len(start_periods) == 0:
                return {'avg_underwater_period': 0, 'max_underwater_period': 0, 'underwater_frequency': 0}
            
            if len(end_periods) < len(start_periods):
                end_periods = end_periods.tolist() + [drawdown_series.index[-1]]
            
            # Calculate underwater period lengths
            underwater_lengths = []
            for start, end in zip(start_periods, end_periods):
                length = (end - start).days if hasattr(end - start, 'days') else len(drawdown_series.loc[start:end])
                underwater_lengths.append(length)
            
            return {
                'avg_underwater_period': np.mean(underwater_lengths) if underwater_lengths else 0,
                'max_underwater_period': max(underwater_lengths) if underwater_lengths else 0,
                'underwater_frequency': len(underwater_lengths) / len(drawdown_series) * 252
            }
            
        except Exception as e:
            self.logger.warning(f"Underwater period analysis failed: {str(e)}")
            return {'underwater_analysis_error': str(e)}
    
    def _analyze_recovery_periods(self, returns: pd.Series, drawdown_series: pd.Series) -> Dict[str, float]:
        """Analyze recovery periods from drawdowns."""
        try:
            # Find recovery periods (time to reach new high)
            cumulative_returns = emp.cum_returns(returns)
            rolling_max = cumulative_returns.expanding().max()
            
            # Recovery time analysis
            recovery_times = []
            in_recovery = False
            recovery_start = None
            
            for date in returns.index:
                if cumulative_returns[date] < rolling_max[date] and not in_recovery:
                    # Start of drawdown
                    in_recovery = True
                    recovery_start = date
                elif cumulative_returns[date] >= rolling_max[date] and in_recovery:
                    # End of recovery
                    in_recovery = False
                    if recovery_start:
                        recovery_time = (date - recovery_start).days if hasattr(date - recovery_start, 'days') else 1
                        recovery_times.append(recovery_time)
            
            return {
                'avg_recovery_time': np.mean(recovery_times) if recovery_times else 0,
                'max_recovery_time': max(recovery_times) if recovery_times else 0,
                'recovery_frequency': len(recovery_times) / len(returns) * 252
            }
            
        except Exception as e:
            self.logger.warning(f"Recovery period analysis failed: {str(e)}")
            return {'recovery_analysis_error': str(e)}
    
    def _calculate_overall_assessment(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall portfolio assessment and rating."""
        try:
            assessment = {}
            
            # Performance score (0-100)
            performance_score = 0
            
            # Return component (30 points)
            annual_return = metrics.get('annualized_return', 0)
            if annual_return > 0.15:  # >15%
                performance_score += 30
            elif annual_return > 0.10:  # 10-15%
                performance_score += 20
            elif annual_return > 0.05:  # 5-10%
                performance_score += 10
            
            # Risk-adjusted component (40 points)
            sharpe_ratio = metrics.get('sharpe_ratio', 0)
            if sharpe_ratio > 2.0:
                performance_score += 40
            elif sharpe_ratio > 1.5:
                performance_score += 30
            elif sharpe_ratio > 1.0:
                performance_score += 20
            elif sharpe_ratio > 0.5:
                performance_score += 10
            
            # Risk component (30 points)
            max_drawdown = abs(metrics.get('max_drawdown', 1))
            if max_drawdown < 0.05:  # <5%
                performance_score += 30
            elif max_drawdown < 0.10:  # 5-10%
                performance_score += 20
            elif max_drawdown < 0.15:  # 10-15%
                performance_score += 10
            
            assessment['overall_score'] = min(100, performance_score)
            
            # Rating classification
            if performance_score >= 80:
                assessment['rating'] = 'Excellent'
            elif performance_score >= 60:
                assessment['rating'] = 'Good'
            elif performance_score >= 40:
                assessment['rating'] = 'Fair'
            else:
                assessment['rating'] = 'Poor'
            
            # Risk level assessment
            volatility = metrics.get('volatility', 0)
            if volatility < 0.10:
                assessment['risk_level'] = 'Conservative'
            elif volatility < 0.20:
                assessment['risk_level'] = 'Moderate'
            elif volatility < 0.30:
                assessment['risk_level'] = 'Aggressive'
            else:
                assessment['risk_level'] = 'High Risk'
            
            # Key strengths and weaknesses
            strengths = []
            weaknesses = []
            
            if sharpe_ratio > 1.5:
                strengths.append('Excellent risk-adjusted returns')
            if max_drawdown < 0.10:
                strengths.append('Low maximum drawdown')
            if metrics.get('win_rate', 0) > 0.55:
                strengths.append('High win rate')
            if metrics.get('stability_of_timeseries', 0) > 0.8:
                strengths.append('Consistent performance')
            
            if sharpe_ratio < 0.5:
                weaknesses.append('Poor risk-adjusted returns')
            if max_drawdown > 0.20:
                weaknesses.append('High maximum drawdown')
            if volatility > 0.25:
                weaknesses.append('High volatility')
            if metrics.get('tail_ratio', 1) < 0.8:
                weaknesses.append('Poor tail risk management')
            
            assessment['key_strengths'] = strengths
            assessment['key_weaknesses'] = weaknesses
            
            # Investment recommendation
            if performance_score >= 70 and max_drawdown < 0.15:
                assessment['recommendation'] = 'Strong Buy'
            elif performance_score >= 50 and max_drawdown < 0.20:
                assessment['recommendation'] = 'Buy'
            elif performance_score >= 30:
                assessment['recommendation'] = 'Hold'
            else:
                assessment['recommendation'] = 'Avoid'
            
            return assessment
            
        except Exception as e:
            self.logger.warning(f"Overall assessment calculation failed: {str(e)}")
            return {'assessment_error': str(e)}
    
    def _validate_returns(self, returns: pd.Series) -> None:
        """Validate return series."""
        if not isinstance(returns, pd.Series):
            raise ValidationError("Returns must be a pandas Series")
        
        if len(returns) < 10:
            raise ValidationError("Returns series must have at least 10 observations")
        
        if returns.isna().all():
            raise ValidationError("Returns series contains only NaN values")
        
        # Check for extreme values that might indicate data errors
        if (abs(returns) > 1.0).any():  # >100% daily return is suspicious
            self.logger.warning("Extreme returns detected (>100% single period)")
    
    def _convert_to_daily(self, returns: pd.Series, period: str) -> pd.Series:
        """Convert returns to daily frequency if needed."""
        if period.lower() == 'daily':
            return returns
        elif period.lower() == 'weekly':
            # Convert weekly to daily (approximate)
            return returns / 7
        elif period.lower() == 'monthly':
            # Convert monthly to daily (approximate)
            return returns / 21
        else:
            self.logger.warning(f"Unknown period '{period}', treating as daily")
            return returns
    
    def _annualize_to_daily(self, annual_rate: float) -> float:
        """Convert annual rate to daily rate."""
        return annual_rate / 252
    
    def calculate_factor_attribution(self, 
                                    returns: pd.Series,
                                    factor_returns: Dict[str, pd.Series]) -> Dict[str, Any]:
        """
        Calculate factor attribution analysis.
        
        Args:
            returns: Portfolio returns
            factor_returns: Dictionary of factor returns (e.g., {'market': series, 'value': series})
            
        Returns:
            Factor attribution results
        """
        try:
            attribution = {}
            
            # Combine all factor data
            factor_df = pd.DataFrame(factor_returns)
            aligned_data = pd.DataFrame({
                'portfolio': returns,
                **factor_returns
            }).dropna()
            
            if len(aligned_data) < 30:  # Need sufficient data
                return {'factor_attribution_error': 'Insufficient data for factor analysis'}
            
            # Multiple regression analysis
            from sklearn.linear_model import LinearRegression
            
            X = aligned_data[list(factor_returns.keys())]
            y = aligned_data['portfolio']
            
            model = LinearRegression().fit(X, y)
            
            attribution['factor_loadings'] = dict(zip(factor_returns.keys(), model.coef_))
            attribution['alpha'] = model.intercept_
            attribution['r_squared'] = model.score(X, y)
            
            # Factor contributions
            factor_contributions = {}
            for factor_name in factor_returns.keys():
                factor_contrib = attribution['factor_loadings'][factor_name] * aligned_data[factor_name].mean() * 252
                factor_contributions[factor_name] = factor_contrib
            
            attribution['factor_contributions'] = factor_contributions
            attribution['explained_return'] = sum(factor_contributions.values())
            attribution['unexplained_return'] = attribution['alpha'] * 252
            
            return attribution
            
        except Exception as e:
            self.logger.error(f"Factor attribution analysis failed: {str(e)}")
            return {'factor_attribution_error': str(e)}
    
    def generate_risk_report(self, metrics: Dict[str, Any]) -> str:
        """Generate a formatted risk analysis report."""
        try:
            report = []
            report.append("=" * 60)
            report.append("EMPYRICAL RISK ANALYSIS REPORT")
            report.append("=" * 60)
            
            # Performance Summary
            report.append("\n📊 PERFORMANCE SUMMARY")
            report.append("-" * 25)
            report.append(f"Total Return: {metrics.get('total_return', 0):.2%}")
            report.append(f"Annualized Return: {metrics.get('annualized_return', 0):.2%}")
            report.append(f"Volatility: {metrics.get('volatility', 0):.2%}")
            report.append(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
            
            # Risk Metrics
            report.append("\n⚠️  RISK METRICS")
            report.append("-" * 15)
            report.append(f"Maximum Drawdown: {metrics.get('max_drawdown', 0):.2%}")
            report.append(f"VaR (95%): {metrics.get('var_95', 0):.2%}")
            report.append(f"CVaR (95%): {metrics.get('cvar_95', 0):.2%}")
            report.append(f"Downside Deviation: {metrics.get('downside_deviation', 0):.2%}")
            
            # Overall Assessment
            assessment = metrics.get('overall_assessment', {})
            if assessment:
                report.append("\n🏆 OVERALL ASSESSMENT")
                report.append("-" * 20)
                report.append(f"Rating: {assessment.get('rating', 'N/A')}")
                report.append(f"Score: {assessment.get('overall_score', 0):.0f}/100")
                report.append(f"Risk Level: {assessment.get('risk_level', 'N/A')}")
                report.append(f"Recommendation: {assessment.get('recommendation', 'N/A')}")
                
                strengths = assessment.get('key_strengths', [])
                if strengths:
                    report.append(f"Strengths: {', '.join(strengths)}")
                
                weaknesses = assessment.get('key_weaknesses', [])
                if weaknesses:
                    report.append(f"Areas for Improvement: {', '.join(weaknesses)}")
            
            report.append("\n" + "=" * 60)
            
            return "\n".join(report)
            
        except Exception as e:
            return f"Report generation failed: {str(e)}"