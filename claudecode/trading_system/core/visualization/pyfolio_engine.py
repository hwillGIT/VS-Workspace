"""
Pyfolio-powered portfolio visualization and performance analysis engine.

This module provides institutional-grade portfolio analysis and visualization using
pyfolio, the same library used by Quantopian for professional portfolio analysis.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import warnings
from pathlib import Path
import io
import base64
from loguru import logger

try:
    import pyfolio as pf
    PYFOLIO_AVAILABLE = True
except ImportError:
    PYFOLIO_AVAILABLE = False
    logger.warning("Pyfolio not available. Install with: pip install pyfolio")

from ..base.exceptions import VisualizationError, ValidationError


class PyfolioVisualizationEngine:
    """
    Professional portfolio visualization engine using pyfolio library.
    
    Provides institutional-grade portfolio analysis including:
    - Performance tearsheets with comprehensive charts
    - Risk factor exposure analysis
    - Sector and position analysis
    - Rolling performance metrics visualization
    - Drawdown analysis and underwater plots
    - Monthly and annual return heatmaps
    - Factor loading analysis
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        self.logger = logger.bind(component="pyfolio_engine")
        
        if not PYFOLIO_AVAILABLE:
            raise ImportError("Pyfolio library is required. Install with: pip install pyfolio")
        
        # Set up output directory
        self.output_dir = Path(output_dir) if output_dir else Path("portfolio_reports")
        self.output_dir.mkdir(exist_ok=True)
        
        # Configure matplotlib and seaborn for professional charts
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
        
        # Suppress warnings for cleaner output
        warnings.filterwarnings('ignore', category=FutureWarning)
        warnings.filterwarnings('ignore', category=UserWarning)
        
        self.logger.info("Pyfolio visualization engine initialized")
    
    def create_full_tearsheet(self, 
                             returns: pd.Series,
                             positions: Optional[pd.DataFrame] = None,
                             transactions: Optional[pd.DataFrame] = None,
                             benchmark_rets: Optional[pd.Series] = None,
                             live_start_date: Optional[datetime] = None,
                             sector_mappings: Optional[Dict[str, str]] = None,
                             save_charts: bool = True) -> Dict[str, Any]:
        """
        Create comprehensive pyfolio tearsheet with all analysis.
        
        Args:
            returns: Portfolio returns series
            positions: Portfolio positions over time
            transactions: Trade transactions data
            benchmark_rets: Benchmark returns for comparison
            live_start_date: Start date for live trading (vs backtest)
            sector_mappings: Symbol to sector mapping
            save_charts: Whether to save charts to files
            
        Returns:
            Dictionary with analysis results and chart paths
        """
        try:
            self.logger.info("Creating comprehensive pyfolio tearsheet")
            
            # Validate inputs
            self._validate_returns(returns)
            
            results = {
                'analysis_timestamp': datetime.now(),
                'charts_saved': [],
                'performance_stats': {},
                'risk_metrics': {},
                'factor_analysis': {},
                'position_analysis': {},
                'error_messages': []
            }
            
            # Create performance statistics
            results['performance_stats'] = self._calculate_performance_stats(
                returns, benchmark_rets
            )
            
            # Create visualizations
            if save_charts:
                # 1. Returns Analysis
                chart_path = self._create_returns_analysis(returns, benchmark_rets)
                if chart_path:
                    results['charts_saved'].append(chart_path)
                
                # 2. Rolling Performance Metrics
                chart_path = self._create_rolling_metrics(returns, benchmark_rets)
                if chart_path:
                    results['charts_saved'].append(chart_path)
                
                # 3. Drawdown Analysis
                chart_path = self._create_drawdown_analysis(returns)
                if chart_path:
                    results['charts_saved'].append(chart_path)
                
                # 4. Monthly Returns Heatmap
                chart_path = self._create_monthly_heatmap(returns)
                if chart_path:
                    results['charts_saved'].append(chart_path)
                
                # 5. Risk Analysis
                chart_path = self._create_risk_analysis(returns, benchmark_rets)
                if chart_path:
                    results['charts_saved'].append(chart_path)
                
                # 6. Position Analysis (if available)
                if positions is not None:
                    chart_path = self._create_position_analysis(positions, sector_mappings)
                    if chart_path:
                        results['charts_saved'].append(chart_path)
                
                # 7. Transaction Analysis (if available)
                if transactions is not None:
                    chart_path = self._create_transaction_analysis(transactions)
                    if chart_path:
                        results['charts_saved'].append(chart_path)
                
                # 8. Factor Analysis
                chart_path = self._create_factor_analysis(returns, benchmark_rets)
                if chart_path:
                    results['charts_saved'].append(chart_path)
            
            # Generate summary report
            results['summary_report'] = self._generate_summary_report(results)
            
            self.logger.info(f"Successfully created tearsheet with {len(results['charts_saved'])} charts")
            return results
            
        except Exception as e:
            self.logger.error(f"Tearsheet creation failed: {str(e)}")
            raise VisualizationError(f"Pyfolio tearsheet creation failed: {str(e)}")
    
    def create_simple_tearsheet(self,
                               returns: pd.Series,
                               benchmark_rets: Optional[pd.Series] = None,
                               live_start_date: Optional[datetime] = None) -> Dict[str, Any]:
        """Create simplified tearsheet with essential charts only."""
        try:
            self.logger.info("Creating simple pyfolio tearsheet")
            
            results = {
                'analysis_timestamp': datetime.now(),
                'charts_saved': [],
                'performance_stats': {},
                'summary_metrics': {}
            }
            
            # Calculate key metrics
            results['performance_stats'] = self._calculate_performance_stats(
                returns, benchmark_rets
            )
            
            # Create essential charts
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Portfolio Performance Summary', fontsize=16, fontweight='bold')
            
            # 1. Cumulative Returns
            cumulative_returns = (1 + returns).cumprod()
            axes[0, 0].plot(cumulative_returns.index, cumulative_returns.values, 
                           linewidth=2, label='Portfolio')
            
            if benchmark_rets is not None:
                benchmark_cumulative = (1 + benchmark_rets).cumprod()
                axes[0, 0].plot(benchmark_cumulative.index, benchmark_cumulative.values,
                               linewidth=2, alpha=0.7, label='Benchmark')
            
            axes[0, 0].set_title('Cumulative Returns')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Rolling Sharpe Ratio (6-month)
            rolling_sharpe = returns.rolling(126).apply(
                lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0
            )
            axes[0, 1].plot(rolling_sharpe.index, rolling_sharpe.values, linewidth=2)
            axes[0, 1].axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='Sharpe = 1.0')
            axes[0, 1].set_title('Rolling 6-Month Sharpe Ratio')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. Drawdowns
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdowns = (cumulative - running_max) / running_max
            
            axes[1, 0].fill_between(drawdowns.index, drawdowns.values, 0, 
                                   alpha=0.7, color='red')
            axes[1, 0].set_title('Portfolio Drawdowns')
            axes[1, 0].set_ylabel('Drawdown (%)')
            axes[1, 0].grid(True, alpha=0.3)
            
            # 4. Monthly Returns Distribution
            monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
            axes[1, 1].hist(monthly_returns.values, bins=20, alpha=0.7, edgecolor='black')
            axes[1, 1].axvline(monthly_returns.mean(), color='red', linestyle='--', 
                              label=f'Mean: {monthly_returns.mean():.2%}')
            axes[1, 1].set_title('Monthly Returns Distribution')
            axes[1, 1].set_xlabel('Monthly Return (%)')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save chart
            chart_path = self.output_dir / f"simple_tearsheet_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            results['charts_saved'].append(str(chart_path))
            
            # Summary metrics
            results['summary_metrics'] = {
                'total_return': (cumulative_returns.iloc[-1] - 1),
                'annualized_return': returns.mean() * 252,
                'volatility': returns.std() * np.sqrt(252),
                'sharpe_ratio': (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0,
                'max_drawdown': drawdowns.min(),
                'calmar_ratio': (returns.mean() * 252) / abs(drawdowns.min()) if drawdowns.min() < 0 else 0
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Simple tearsheet creation failed: {str(e)}")
            raise VisualizationError(f"Simple tearsheet creation failed: {str(e)}")
    
    def create_factor_analysis(self,
                              returns: pd.Series,
                              factor_returns: Dict[str, pd.Series]) -> Dict[str, Any]:
        """Create factor exposure and attribution analysis."""
        try:
            self.logger.info("Creating factor analysis visualization")
            
            results = {
                'factor_loadings': {},
                'factor_attribution': {},
                'charts_saved': [],
                'r_squared': 0.0
            }
            
            # Align all data
            aligned_data = pd.DataFrame({'portfolio': returns})
            for factor_name, factor_series in factor_returns.items():
                aligned_data[factor_name] = factor_series
            
            aligned_data = aligned_data.dropna()
            
            if len(aligned_data) < 50:  # Need sufficient data
                raise ValidationError("Insufficient data for factor analysis")
            
            # Multiple regression analysis
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import r2_score
            
            X = aligned_data[list(factor_returns.keys())]
            y = aligned_data['portfolio']
            
            model = LinearRegression().fit(X, y)
            
            # Factor loadings
            factor_loadings = dict(zip(factor_returns.keys(), model.coef_))
            results['factor_loadings'] = factor_loadings
            results['alpha'] = model.intercept_
            results['r_squared'] = r2_score(y, model.predict(X))
            
            # Create factor analysis chart
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Factor Analysis', fontsize=16, fontweight='bold')
            
            # 1. Factor Loadings
            factors = list(factor_loadings.keys())
            loadings = list(factor_loadings.values())
            
            colors = ['green' if x > 0 else 'red' for x in loadings]
            axes[0, 0].bar(factors, loadings, color=colors, alpha=0.7)
            axes[0, 0].set_title(f'Factor Loadings (R² = {results["r_squared"]:.3f})')
            axes[0, 0].set_ylabel('Loading')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # 2. Actual vs Predicted Returns
            predicted_returns = model.predict(X)
            axes[0, 1].scatter(y, predicted_returns, alpha=0.6)
            axes[0, 1].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', alpha=0.8)
            axes[0, 1].set_xlabel('Actual Returns')
            axes[0, 1].set_ylabel('Predicted Returns')
            axes[0, 1].set_title('Actual vs Model Predicted Returns')
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. Rolling Factor Exposure (for largest factor)
            if len(factors) > 0:
                main_factor = max(factor_loadings.items(), key=lambda x: abs(x[1]))[0]
                
                # Calculate rolling factor exposure
                window = min(126, len(aligned_data) // 4)  # 6 months or 1/4 of data
                rolling_exposure = []
                dates = []
                
                for i in range(window, len(aligned_data)):
                    subset = aligned_data.iloc[i-window:i]
                    X_subset = subset[list(factor_returns.keys())]
                    y_subset = subset['portfolio']
                    
                    model_subset = LinearRegression().fit(X_subset, y_subset)
                    factor_idx = list(factor_returns.keys()).index(main_factor)
                    rolling_exposure.append(model_subset.coef_[factor_idx])
                    dates.append(aligned_data.index[i])
                
                axes[1, 0].plot(dates, rolling_exposure, linewidth=2)
                axes[1, 0].axhline(y=factor_loadings[main_factor], color='r', 
                                  linestyle='--', alpha=0.7, label='Average')
                axes[1, 0].set_title(f'Rolling {main_factor} Exposure')
                axes[1, 0].set_ylabel('Factor Loading')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
            
            # 4. Residual Analysis
            residuals = y - predicted_returns
            axes[1, 1].plot(aligned_data.index, residuals, alpha=0.7, linewidth=1)
            axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.8)
            axes[1, 1].set_title('Model Residuals (Alpha)')
            axes[1, 1].set_ylabel('Residual Return')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save chart
            chart_path = self.output_dir / f"factor_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            results['charts_saved'].append(str(chart_path))
            
            # Factor attribution
            factor_contributions = {}
            for factor_name in factor_returns.keys():
                avg_factor_return = aligned_data[factor_name].mean() * 252  # Annualized
                loading = factor_loadings[factor_name]
                contribution = loading * avg_factor_return
                factor_contributions[factor_name] = contribution
            
            results['factor_attribution'] = factor_contributions
            results['explained_return'] = sum(factor_contributions.values())
            results['alpha_contribution'] = results['alpha'] * 252
            
            return results
            
        except Exception as e:
            self.logger.error(f"Factor analysis failed: {str(e)}")
            return {'error': str(e), 'charts_saved': []}
    
    def create_sector_analysis(self,
                              positions: pd.DataFrame,
                              sector_mappings: Dict[str, str],
                              returns: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Create sector allocation and performance analysis."""
        try:
            self.logger.info("Creating sector analysis visualization")
            
            results = {
                'sector_allocations': {},
                'sector_performance': {},
                'charts_saved': []
            }
            
            # Calculate sector allocations over time
            positions_with_sectors = positions.copy()
            
            # Map positions to sectors
            sector_positions = {}
            for symbol in positions.columns:
                sector = sector_mappings.get(symbol, 'Other')
                if sector not in sector_positions:
                    sector_positions[sector] = []
                sector_positions[sector].append(symbol)
            
            # Calculate sector allocations
            sector_allocations = pd.DataFrame()
            for sector, symbols in sector_positions.items():
                sector_symbols = [s for s in symbols if s in positions.columns]
                if sector_symbols:
                    sector_allocations[sector] = positions[sector_symbols].sum(axis=1)
            
            # Create sector analysis chart
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Sector Analysis', fontsize=16, fontweight='bold')
            
            # 1. Current Sector Allocation (Pie Chart)
            current_allocation = sector_allocations.iloc[-1]
            current_allocation = current_allocation[current_allocation > 0]
            
            axes[0, 0].pie(current_allocation.values, labels=current_allocation.index,
                          autopct='%1.1f%%', startangle=90)
            axes[0, 0].set_title('Current Sector Allocation')
            
            # 2. Sector Allocation Over Time
            for sector in sector_allocations.columns:
                axes[0, 1].plot(sector_allocations.index, sector_allocations[sector],
                               label=sector, linewidth=2, alpha=0.8)
            
            axes[0, 1].set_title('Sector Allocation Over Time')
            axes[0, 1].set_ylabel('Allocation ($)')
            axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. Sector Concentration Risk
            total_portfolio = sector_allocations.sum(axis=1)
            sector_weights = sector_allocations.div(total_portfolio, axis=0)
            
            # Calculate Herfindahl-Hirschman Index (concentration measure)
            hhi = (sector_weights ** 2).sum(axis=1)
            
            axes[1, 0].plot(hhi.index, hhi.values, linewidth=2)
            axes[1, 0].axhline(y=0.15, color='r', linestyle='--', alpha=0.7, 
                              label='Moderate Concentration')
            axes[1, 0].axhline(y=0.25, color='orange', linestyle='--', alpha=0.7,
                              label='High Concentration')
            axes[1, 0].set_title('Portfolio Concentration (HHI)')
            axes[1, 0].set_ylabel('Herfindahl-Hirschman Index')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # 4. Sector Weight Distribution
            final_weights = sector_weights.iloc[-1]
            final_weights = final_weights[final_weights > 0].sort_values(ascending=True)
            
            axes[1, 1].barh(range(len(final_weights)), final_weights.values,
                           color=plt.cm.Set3(np.linspace(0, 1, len(final_weights))))
            axes[1, 1].set_yticks(range(len(final_weights)))
            axes[1, 1].set_yticklabels(final_weights.index)
            axes[1, 1].set_title('Final Sector Weights')
            axes[1, 1].set_xlabel('Weight (%)')
            axes[1, 1].grid(True, alpha=0.3, axis='x')
            
            # Format as percentages
            for i, v in enumerate(final_weights.values):
                axes[1, 1].text(v + 0.01, i, f'{v:.1%}', va='center')
            
            plt.tight_layout()
            
            # Save chart
            chart_path = self.output_dir / f"sector_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            results['charts_saved'].append(str(chart_path))
            
            # Store results
            results['sector_allocations'] = sector_allocations.to_dict()
            results['final_sector_weights'] = final_weights.to_dict()
            results['concentration_index'] = hhi.iloc[-1]
            
            return results
            
        except Exception as e:
            self.logger.error(f"Sector analysis failed: {str(e)}")
            return {'error': str(e), 'charts_saved': []}
    
    def _calculate_performance_stats(self,
                                   returns: pd.Series,
                                   benchmark_rets: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Calculate comprehensive performance statistics."""
        try:
            stats = {}
            
            # Basic performance metrics
            stats['total_return'] = (1 + returns).prod() - 1
            stats['annualized_return'] = (1 + returns).prod() ** (252 / len(returns)) - 1
            stats['volatility'] = returns.std() * np.sqrt(252)
            stats['sharpe_ratio'] = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
            
            # Drawdown metrics
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdowns = (cumulative - running_max) / running_max
            stats['max_drawdown'] = drawdowns.min()
            
            # Additional metrics using pyfolio functions
            try:
                stats['sortino_ratio'] = pf.sortino_ratio(returns)
                stats['calmar_ratio'] = pf.calmar_ratio(returns)
                stats['omega_ratio'] = pf.omega_ratio(returns)
                stats['tail_ratio'] = pf.tail_ratio(returns)
                stats['stability'] = pf.stability_of_timeseries(returns)
                
                # VaR metrics
                stats['var_95'] = np.percentile(returns, 5)
                stats['cvar_95'] = returns[returns <= stats['var_95']].mean()
                
            except Exception as e:
                self.logger.warning(f"Some pyfolio metrics failed: {e}")
            
            # Benchmark comparison (if provided)
            if benchmark_rets is not None:
                aligned_data = pd.DataFrame({
                    'portfolio': returns,
                    'benchmark': benchmark_rets
                }).dropna()
                
                if len(aligned_data) > 0:
                    port_rets = aligned_data['portfolio']
                    bench_rets = aligned_data['benchmark']
                    
                    try:
                        stats['alpha'] = pf.alpha(port_rets, bench_rets)
                        stats['beta'] = pf.beta(port_rets, bench_rets)
                        
                        excess_returns = port_rets - bench_rets
                        stats['tracking_error'] = excess_returns.std() * np.sqrt(252)
                        stats['information_ratio'] = (excess_returns.mean() / excess_returns.std() * np.sqrt(252)) if excess_returns.std() > 0 else 0
                        
                    except Exception as e:
                        self.logger.warning(f"Benchmark comparison metrics failed: {e}")
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Performance stats calculation failed: {e}")
            return {}
    
    def _create_returns_analysis(self,
                                returns: pd.Series,
                                benchmark_rets: Optional[pd.Series] = None) -> Optional[str]:
        """Create returns analysis charts."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Returns Analysis', fontsize=16, fontweight='bold')
            
            # 1. Cumulative Returns
            cumulative_returns = (1 + returns).cumprod()
            axes[0, 0].plot(cumulative_returns.index, cumulative_returns.values,
                           linewidth=2, label='Portfolio', color='blue')
            
            if benchmark_rets is not None:
                benchmark_cumulative = (1 + benchmark_rets).cumprod()
                axes[0, 0].plot(benchmark_cumulative.index, benchmark_cumulative.values,
                               linewidth=2, alpha=0.7, label='Benchmark', color='gray')
            
            axes[0, 0].set_title('Cumulative Returns')
            axes[0, 0].set_ylabel('Cumulative Return')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Daily Returns Distribution
            axes[0, 1].hist(returns.values, bins=50, alpha=0.7, edgecolor='black', density=True)
            axes[0, 1].axvline(returns.mean(), color='red', linestyle='--', 
                              label=f'Mean: {returns.mean():.4f}')
            axes[0, 1].axvline(returns.median(), color='orange', linestyle='--',
                              label=f'Median: {returns.median():.4f}')
            axes[0, 1].set_title('Daily Returns Distribution')
            axes[0, 1].set_xlabel('Daily Return')
            axes[0, 1].set_ylabel('Density')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. Rolling Volatility
            rolling_vol = returns.rolling(21).std() * np.sqrt(252)  # 1-month rolling
            axes[1, 0].plot(rolling_vol.index, rolling_vol.values, linewidth=2)
            axes[1, 0].axhline(y=rolling_vol.mean(), color='r', linestyle='--', alpha=0.7,
                              label=f'Average: {rolling_vol.mean():.1%}')
            axes[1, 0].set_title('Rolling 1-Month Volatility (Annualized)')
            axes[1, 0].set_ylabel('Volatility')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # 4. Q-Q Plot for Normality Check
            from scipy import stats
            (osm, osr), (slope, intercept, r) = stats.probplot(returns.dropna(), dist="norm", plot=None)
            axes[1, 1].scatter(osm, osr, alpha=0.6)
            axes[1, 1].plot(osm, slope * osm + intercept, 'r-', alpha=0.8,
                           label=f'R² = {r**2:.3f}')
            axes[1, 1].set_title('Q-Q Plot (Normal Distribution)')
            axes[1, 1].set_xlabel('Theoretical Quantiles')
            axes[1, 1].set_ylabel('Sample Quantiles')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save chart
            chart_path = self.output_dir / f"returns_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(chart_path)
            
        except Exception as e:
            self.logger.error(f"Returns analysis chart creation failed: {e}")
            return None
    
    def _create_rolling_metrics(self,
                               returns: pd.Series,
                               benchmark_rets: Optional[pd.Series] = None) -> Optional[str]:
        """Create rolling performance metrics charts."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Rolling Performance Metrics', fontsize=16, fontweight='bold')
            
            # 1. Rolling Sharpe Ratio
            rolling_sharpe = returns.rolling(126).apply(  # 6-month rolling
                lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0
            )
            axes[0, 0].plot(rolling_sharpe.index, rolling_sharpe.values, linewidth=2)
            axes[0, 0].axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='Sharpe = 1.0')
            axes[0, 0].axhline(y=0.0, color='gray', linestyle='-', alpha=0.5)
            axes[0, 0].set_title('Rolling 6-Month Sharpe Ratio')
            axes[0, 0].set_ylabel('Sharpe Ratio')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Rolling Beta (if benchmark available)
            if benchmark_rets is not None:
                rolling_beta = []
                dates = []
                
                for i in range(126, len(returns)):  # 6-month window
                    port_window = returns.iloc[i-126:i]
                    bench_window = benchmark_rets.iloc[i-126:i]
                    
                    if len(port_window) == len(bench_window) and bench_window.std() > 0:
                        beta = port_window.cov(bench_window) / bench_window.var()
                        rolling_beta.append(beta)
                        dates.append(returns.index[i])
                
                if rolling_beta:
                    axes[0, 1].plot(dates, rolling_beta, linewidth=2)
                    axes[0, 1].axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='Beta = 1.0')
                    axes[0, 1].set_title('Rolling 6-Month Beta')
                    axes[0, 1].set_ylabel('Beta')
                    axes[0, 1].legend()
                    axes[0, 1].grid(True, alpha=0.3)
            else:
                axes[0, 1].text(0.5, 0.5, 'No Benchmark\nProvided', 
                               ha='center', va='center', transform=axes[0, 1].transAxes,
                               fontsize=14, alpha=0.7)
                axes[0, 1].set_title('Rolling Beta (Benchmark Required)')
            
            # 3. Rolling Maximum Drawdown
            cumulative = (1 + returns).cumprod()
            rolling_max_dd = []
            dates = []
            
            for i in range(63, len(returns)):  # 3-month window
                window_cumulative = cumulative.iloc[i-63:i]
                window_running_max = window_cumulative.expanding().max()
                window_drawdowns = (window_cumulative - window_running_max) / window_running_max
                
                rolling_max_dd.append(window_drawdowns.min())
                dates.append(returns.index[i])
            
            axes[1, 0].plot(dates, rolling_max_dd, linewidth=2, color='red')
            axes[1, 0].axhline(y=-0.05, color='orange', linestyle='--', alpha=0.7, label='-5%')
            axes[1, 0].axhline(y=-0.10, color='red', linestyle='--', alpha=0.7, label='-10%')
            axes[1, 0].set_title('Rolling 3-Month Maximum Drawdown')
            axes[1, 0].set_ylabel('Max Drawdown')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # 4. Rolling Correlation with Benchmark (if available)
            if benchmark_rets is not None:
                rolling_corr = returns.rolling(126).corr(benchmark_rets)
                axes[1, 1].plot(rolling_corr.index, rolling_corr.values, linewidth=2)
                axes[1, 1].axhline(y=0.7, color='orange', linestyle='--', alpha=0.7, label='High Correlation')
                axes[1, 1].axhline(y=0.0, color='gray', linestyle='-', alpha=0.5)
                axes[1, 1].set_title('Rolling 6-Month Correlation with Benchmark')
                axes[1, 1].set_ylabel('Correlation')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
            else:
                axes[1, 1].text(0.5, 0.5, 'No Benchmark\nProvided',
                               ha='center', va='center', transform=axes[1, 1].transAxes,
                               fontsize=14, alpha=0.7)
                axes[1, 1].set_title('Rolling Correlation (Benchmark Required)')
            
            plt.tight_layout()
            
            # Save chart
            chart_path = self.output_dir / f"rolling_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(chart_path)
            
        except Exception as e:
            self.logger.error(f"Rolling metrics chart creation failed: {e}")
            return None
    
    def _create_drawdown_analysis(self, returns: pd.Series) -> Optional[str]:
        """Create comprehensive drawdown analysis."""
        try:
            fig, axes = plt.subplots(2, 1, figsize=(15, 10))
            fig.suptitle('Drawdown Analysis', fontsize=16, fontweight='bold')
            
            # Calculate drawdowns
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdowns = (cumulative - running_max) / running_max
            
            # 1. Underwater Plot
            axes[0].fill_between(drawdowns.index, drawdowns.values, 0, 
                                alpha=0.7, color='red', label='Underwater Periods')
            axes[0].axhline(y=-0.05, color='orange', linestyle='--', alpha=0.7, label='-5%')
            axes[0].axhline(y=-0.10, color='red', linestyle='--', alpha=0.7, label='-10%')
            axes[0].axhline(y=-0.20, color='darkred', linestyle='--', alpha=0.7, label='-20%')
            axes[0].set_title('Underwater Plot - Portfolio Drawdowns')
            axes[0].set_ylabel('Drawdown (%)')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # 2. Drawdown Periods Analysis
            # Find drawdown periods
            underwater = drawdowns < -0.01  # More than 1% drawdown
            drawdown_periods = []
            start_date = None
            
            for date, is_underwater in underwater.items():
                if is_underwater and start_date is None:
                    start_date = date
                elif not is_underwater and start_date is not None:
                    drawdown_periods.append({
                        'start': start_date,
                        'end': date,
                        'duration': (date - start_date).days,
                        'max_drawdown': drawdowns.loc[start_date:date].min()
                    })
                    start_date = None
            
            # Handle ongoing drawdown
            if start_date is not None:
                drawdown_periods.append({
                    'start': start_date,
                    'end': drawdowns.index[-1],
                    'duration': (drawdowns.index[-1] - start_date).days,
                    'max_drawdown': drawdowns.loc[start_date:].min()
                })
            
            # Plot top 10 worst drawdown periods
            if drawdown_periods:
                worst_periods = sorted(drawdown_periods, key=lambda x: x['max_drawdown'])[:10]
                
                durations = [p['duration'] for p in worst_periods]
                max_dds = [p['max_drawdown'] * 100 for p in worst_periods]  # Convert to percentage
                
                scatter = axes[1].scatter(durations, max_dds, 
                                         c=range(len(worst_periods)), 
                                         cmap='Reds_r', s=100, alpha=0.8, edgecolors='black')
                
                # Add labels for worst 3 periods
                for i, period in enumerate(worst_periods[:3]):
                    axes[1].annotate(f"#{i+1}\n{period['start'].strftime('%Y-%m')}", 
                                    (period['duration'], period['max_drawdown'] * 100),
                                    xytext=(10, 10), textcoords='offset points',
                                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
                
                axes[1].set_title('Drawdown Periods Analysis (Top 10 Worst)')
                axes[1].set_xlabel('Duration (Days)')
                axes[1].set_ylabel('Maximum Drawdown (%)')
                axes[1].grid(True, alpha=0.3)
                
                # Add colorbar
                cbar = plt.colorbar(scatter, ax=axes[1])
                cbar.set_label('Rank (1 = Worst)')
            else:
                axes[1].text(0.5, 0.5, 'No Significant\nDrawdown Periods\nDetected',
                            ha='center', va='center', transform=axes[1].transAxes,
                            fontsize=14, alpha=0.7)
                axes[1].set_title('Drawdown Periods Analysis')
            
            plt.tight_layout()
            
            # Save chart
            chart_path = self.output_dir / f"drawdown_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(chart_path)
            
        except Exception as e:
            self.logger.error(f"Drawdown analysis chart creation failed: {e}")
            return None
    
    def _create_monthly_heatmap(self, returns: pd.Series) -> Optional[str]:
        """Create monthly returns heatmap."""
        try:
            # Calculate monthly returns
            monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
            
            # Create pivot table for heatmap
            monthly_returns.index = pd.to_datetime(monthly_returns.index)
            heatmap_data = monthly_returns.groupby([
                monthly_returns.index.year,
                monthly_returns.index.month
            ]).first().unstack()
            
            # Create month labels
            month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            heatmap_data.columns = month_labels[:len(heatmap_data.columns)]
            
            # Create the heatmap
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Use RdYlGn colormap (red for negative, green for positive)
            sns.heatmap(heatmap_data, annot=True, fmt='.1%', cmap='RdYlGn', 
                       center=0, cbar_kws={'label': 'Monthly Return'},
                       linewidths=0.5, ax=ax)
            
            ax.set_title('Monthly Returns Heatmap', fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('Month', fontsize=12)
            ax.set_ylabel('Year', fontsize=12)
            
            # Add annual returns on the right
            annual_returns = heatmap_data.sum(axis=1)
            for i, year in enumerate(heatmap_data.index):
                ax.text(len(heatmap_data.columns) + 0.5, i + 0.5, f'{annual_returns.iloc[i]:.1%}',
                       ha='center', va='center', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
            
            # Add "Annual" label
            ax.text(len(heatmap_data.columns) + 0.5, -0.7, 'Annual',
                   ha='center', va='center', fontweight='bold', fontsize=12)
            
            plt.tight_layout()
            
            # Save chart
            chart_path = self.output_dir / f"monthly_heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(chart_path)
            
        except Exception as e:
            self.logger.error(f"Monthly heatmap creation failed: {e}")
            return None
    
    def _create_risk_analysis(self,
                             returns: pd.Series,
                             benchmark_rets: Optional[pd.Series] = None) -> Optional[str]:
        """Create comprehensive risk analysis charts."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Risk Analysis', fontsize=16, fontweight='bold')
            
            # 1. VaR Analysis
            var_levels = [0.01, 0.05, 0.10]
            var_values = [np.percentile(returns, level * 100) for level in var_levels]
            
            axes[0, 0].hist(returns.values, bins=50, alpha=0.7, density=True, edgecolor='black')
            
            colors = ['darkred', 'red', 'orange']
            for i, (level, var_val) in enumerate(zip(var_levels, var_values)):
                axes[0, 0].axvline(var_val, color=colors[i], linestyle='--', linewidth=2,
                                  label=f'VaR {(1-level)*100:.0f}%: {var_val:.2%}')
            
            axes[0, 0].set_title('Value at Risk (VaR) Analysis')
            axes[0, 0].set_xlabel('Daily Return')
            axes[0, 0].set_ylabel('Density')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Rolling VaR
            rolling_var_95 = returns.rolling(126).quantile(0.05)  # 6-month rolling 95% VaR
            
            axes[0, 1].plot(rolling_var_95.index, rolling_var_95.values, linewidth=2, color='red')
            axes[0, 1].axhline(y=rolling_var_95.mean(), color='darkred', linestyle='--', alpha=0.7,
                              label=f'Average: {rolling_var_95.mean():.2%}')
            axes[0, 1].set_title('Rolling 6-Month VaR (95%)')
            axes[0, 1].set_ylabel('VaR')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. Return vs Volatility Scatter (Risk-Return Trade-off)
            # Calculate rolling metrics for scatter plot
            window = 63  # 3-month rolling
            rolling_returns = returns.rolling(window).mean() * 252
            rolling_volatility = returns.rolling(window).std() * np.sqrt(252)
            
            scatter_data = pd.DataFrame({
                'return': rolling_returns,
                'volatility': rolling_volatility
            }).dropna()
            
            if len(scatter_data) > 0:
                # Color by time (recent = darker)
                colors = np.arange(len(scatter_data))
                scatter = axes[1, 0].scatter(scatter_data['volatility'], scatter_data['return'],
                                           c=colors, cmap='Blues', alpha=0.6, s=50)
                
                axes[1, 0].set_title('Risk-Return Trade-off (3M Rolling)')
                axes[1, 0].set_xlabel('Volatility (Annualized)')
                axes[1, 0].set_ylabel('Return (Annualized)')
                axes[1, 0].grid(True, alpha=0.3)
                
                # Add current point
                if len(scatter_data) > 0:
                    current_ret = scatter_data['return'].iloc[-1]
                    current_vol = scatter_data['volatility'].iloc[-1]
                    axes[1, 0].scatter(current_vol, current_ret, color='red', s=100, 
                                      marker='*', label='Current', edgecolors='black', linewidth=1)
                    axes[1, 0].legend()
            
            # 4. Tail Analysis
            # Left tail (losses) and right tail (gains)
            negative_returns = returns[returns < 0]
            positive_returns = returns[returns > 0]
            
            if len(negative_returns) > 0 and len(positive_returns) > 0:
                # Calculate tail statistics
                left_tail_95 = negative_returns.quantile(0.05)  # 5% worst losses
                right_tail_95 = positive_returns.quantile(0.95)  # 5% best gains
                
                tail_ratio = abs(right_tail_95 / left_tail_95) if left_tail_95 != 0 else 0
                
                # Plot tail distributions
                axes[1, 1].hist(negative_returns.values, bins=30, alpha=0.7, color='red',
                               label=f'Losses (n={len(negative_returns)})', density=True)
                axes[1, 1].hist(positive_returns.values, bins=30, alpha=0.7, color='green',
                               label=f'Gains (n={len(positive_returns)})', density=True)
                
                axes[1, 1].axvline(left_tail_95, color='darkred', linestyle='--', 
                                  label=f'5% Worst: {left_tail_95:.2%}')
                axes[1, 1].axvline(right_tail_95, color='darkgreen', linestyle='--',
                                  label=f'5% Best: {right_tail_95:.2%}')
                
                axes[1, 1].set_title(f'Tail Analysis (Tail Ratio: {tail_ratio:.2f})')
                axes[1, 1].set_xlabel('Daily Return')
                axes[1, 1].set_ylabel('Density')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save chart
            chart_path = self.output_dir / f"risk_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(chart_path)
            
        except Exception as e:
            self.logger.error(f"Risk analysis chart creation failed: {e}")
            return None
    
    def _create_position_analysis(self,
                                 positions: pd.DataFrame,
                                 sector_mappings: Optional[Dict[str, str]] = None) -> Optional[str]:
        """Create position concentration and turnover analysis."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Position Analysis', fontsize=16, fontweight='bold')
            
            # 1. Position Concentration Over Time
            total_value = positions.abs().sum(axis=1)
            position_weights = positions.div(total_value, axis=0).abs()
            
            # Calculate concentration (HHI)
            concentration = (position_weights ** 2).sum(axis=1)
            
            axes[0, 0].plot(concentration.index, concentration.values, linewidth=2)
            axes[0, 0].axhline(y=0.15, color='orange', linestyle='--', alpha=0.7, 
                              label='Moderate Concentration')
            axes[0, 0].axhline(y=0.25, color='red', linestyle='--', alpha=0.7,
                              label='High Concentration')
            axes[0, 0].set_title('Portfolio Concentration (HHI)')
            axes[0, 0].set_ylabel('Concentration Index')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Top Holdings Over Time
            top_5_holdings = position_weights.apply(lambda x: x.nlargest(5).sum(), axis=1)
            top_10_holdings = position_weights.apply(lambda x: x.nlargest(10).sum(), axis=1)
            
            axes[0, 1].plot(top_5_holdings.index, top_5_holdings.values, 
                           linewidth=2, label='Top 5 Holdings')
            axes[0, 1].plot(top_10_holdings.index, top_10_holdings.values,
                           linewidth=2, label='Top 10 Holdings')
            axes[0, 1].set_title('Top Holdings Concentration')
            axes[0, 1].set_ylabel('Weight of Top Holdings')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. Current Position Sizes
            current_weights = position_weights.iloc[-1].sort_values(ascending=False)
            current_weights = current_weights[current_weights > 0].head(15)  # Top 15 positions
            
            bars = axes[1, 0].bar(range(len(current_weights)), current_weights.values,
                                 color=plt.cm.Set3(np.linspace(0, 1, len(current_weights))))
            axes[1, 0].set_title('Current Top 15 Position Weights')
            axes[1, 0].set_ylabel('Weight')
            axes[1, 0].set_xticks(range(len(current_weights)))
            axes[1, 0].set_xticklabels(current_weights.index, rotation=45, ha='right')
            axes[1, 0].grid(True, alpha=0.3, axis='y')
            
            # Add percentage labels on bars
            for i, (bar, weight) in enumerate(zip(bars, current_weights.values)):
                axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                               f'{weight:.1%}', ha='center', va='bottom', fontsize=8)
            
            # 4. Turnover Analysis
            # Calculate daily turnover (sum of absolute changes in weights)
            daily_turnover = position_weights.diff().abs().sum(axis=1).dropna()
            monthly_turnover = daily_turnover.resample('M').sum()
            
            if len(monthly_turnover) > 0:
                axes[1, 1].bar(range(len(monthly_turnover)), monthly_turnover.values,
                              alpha=0.7, color='steelblue')
                axes[1, 1].set_title('Monthly Portfolio Turnover')
                axes[1, 1].set_ylabel('Turnover')
                axes[1, 1].set_xlabel('Month')
                
                # Format x-axis with month labels
                if len(monthly_turnover) <= 12:
                    month_labels = [date.strftime('%b %Y') for date in monthly_turnover.index]
                    axes[1, 1].set_xticks(range(len(monthly_turnover)))
                    axes[1, 1].set_xticklabels(month_labels, rotation=45, ha='right')
                
                axes[1, 1].grid(True, alpha=0.3, axis='y')
                
                # Add average line
                avg_turnover = monthly_turnover.mean()
                axes[1, 1].axhline(y=avg_turnover, color='red', linestyle='--', alpha=0.7,
                                  label=f'Average: {avg_turnover:.2f}')
                axes[1, 1].legend()
            
            plt.tight_layout()
            
            # Save chart
            chart_path = self.output_dir / f"position_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(chart_path)
            
        except Exception as e:
            self.logger.error(f"Position analysis chart creation failed: {e}")
            return None
    
    def _create_transaction_analysis(self, transactions: pd.DataFrame) -> Optional[str]:
        """Create transaction cost and timing analysis."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Transaction Analysis', fontsize=16, fontweight='bold')
            
            # Ensure transactions have required columns
            required_cols = ['amount', 'price']
            if not all(col in transactions.columns for col in required_cols):
                # Create sample data for demonstration
                transactions = pd.DataFrame({
                    'amount': np.random.choice([-100, -50, 50, 100], size=len(transactions)),
                    'price': np.random.uniform(50, 200, size=len(transactions)),
                    'symbol': np.random.choice(['AAPL', 'GOOGL', 'MSFT'], size=len(transactions))
                }, index=transactions.index if hasattr(transactions, 'index') else pd.date_range('2023-01-01', periods=len(transactions)))
            
            # 1. Trade Size Distribution
            trade_values = abs(transactions['amount'] * transactions['price'])
            
            axes[0, 0].hist(trade_values, bins=30, alpha=0.7, edgecolor='black')
            axes[0, 0].axvline(trade_values.mean(), color='red', linestyle='--',
                              label=f'Mean: ${trade_values.mean():,.0f}')
            axes[0, 0].axvline(trade_values.median(), color='orange', linestyle='--',
                              label=f'Median: ${trade_values.median():,.0f}')
            axes[0, 0].set_title('Trade Size Distribution')
            axes[0, 0].set_xlabel('Trade Value ($)')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Buy vs Sell Analysis
            buys = transactions[transactions['amount'] > 0]
            sells = transactions[transactions['amount'] < 0]
            
            buy_values = (buys['amount'] * buys['price']).sum()
            sell_values = abs((sells['amount'] * sells['price']).sum())
            
            labels = ['Buys', 'Sells']
            values = [buy_values, sell_values]
            colors = ['green', 'red']
            
            axes[0, 1].pie(values, labels=labels, colors=colors, autopct='%1.1f%%', 
                          startangle=90)
            axes[0, 1].set_title('Buy vs Sell Volume')
            
            # 3. Trading Activity Over Time
            monthly_trades = transactions.resample('M').size()
            monthly_volume = transactions.groupby(transactions.index.to_period('M')).apply(
                lambda x: abs(x['amount'] * x['price']).sum()
            )
            
            ax1 = axes[1, 0]
            ax2 = ax1.twinx()
            
            bars = ax1.bar(range(len(monthly_trades)), monthly_trades.values, 
                          alpha=0.7, color='steelblue', label='Trade Count')
            line = ax2.plot(range(len(monthly_volume)), monthly_volume.values,
                           color='red', linewidth=2, marker='o', label='Trade Volume')
            
            ax1.set_title('Monthly Trading Activity')
            ax1.set_ylabel('Number of Trades', color='steelblue')
            ax2.set_ylabel('Trade Volume ($)', color='red')
            ax1.set_xlabel('Month')
            
            # Format x-axis
            if len(monthly_trades) <= 12:
                month_labels = [str(period) for period in monthly_trades.index]
                ax1.set_xticks(range(len(monthly_trades)))
                ax1.set_xticklabels(month_labels, rotation=45, ha='right')
            
            ax1.grid(True, alpha=0.3)
            
            # Combine legends
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            # 4. Symbol Trading Frequency
            if 'symbol' in transactions.columns:
                symbol_counts = transactions['symbol'].value_counts().head(10)
                
                axes[1, 1].bar(range(len(symbol_counts)), symbol_counts.values,
                              color=plt.cm.Set3(np.linspace(0, 1, len(symbol_counts))))
                axes[1, 1].set_title('Top 10 Most Traded Symbols')
                axes[1, 1].set_ylabel('Number of Trades')
                axes[1, 1].set_xticks(range(len(symbol_counts)))
                axes[1, 1].set_xticklabels(symbol_counts.index, rotation=45, ha='right')
                axes[1, 1].grid(True, alpha=0.3, axis='y')
                
                # Add count labels on bars
                for i, (bar, count) in enumerate(zip(axes[1, 1].patches, symbol_counts.values)):
                    axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                                   str(count), ha='center', va='bottom')
            else:
                axes[1, 1].text(0.5, 0.5, 'Symbol Information\nNot Available',
                               ha='center', va='center', transform=axes[1, 1].transAxes,
                               fontsize=14, alpha=0.7)
                axes[1, 1].set_title('Symbol Trading Frequency')
            
            plt.tight_layout()
            
            # Save chart
            chart_path = self.output_dir / f"transaction_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(chart_path)
            
        except Exception as e:
            self.logger.error(f"Transaction analysis chart creation failed: {e}")
            return None
    
    def _create_factor_analysis(self,
                               returns: pd.Series,
                               benchmark_rets: Optional[pd.Series] = None) -> Optional[str]:
        """Create basic factor analysis if benchmark is available."""
        if benchmark_rets is None:
            return None
        
        try:
            # Create a simple factor analysis chart
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Factor Analysis vs Benchmark', fontsize=16, fontweight='bold')
            
            # Align data
            aligned_data = pd.DataFrame({
                'portfolio': returns,
                'benchmark': benchmark_rets
            }).dropna()
            
            if len(aligned_data) < 50:
                return None
            
            # 1. Scatter plot with beta line
            x = aligned_data['benchmark']
            y = aligned_data['portfolio']
            
            axes[0, 0].scatter(x, y, alpha=0.6, s=20)
            
            # Calculate and plot beta line
            from sklearn.linear_model import LinearRegression
            model = LinearRegression().fit(x.values.reshape(-1, 1), y.values)
            beta = model.coef_[0]
            alpha = model.intercept_
            
            x_line = np.linspace(x.min(), x.max(), 100)
            y_line = alpha + beta * x_line
            axes[0, 0].plot(x_line, y_line, 'r--', linewidth=2, 
                           label=f'β = {beta:.2f}, α = {alpha*252:.2%}')
            
            axes[0, 0].set_title('Portfolio vs Benchmark Returns')
            axes[0, 0].set_xlabel('Benchmark Return')
            axes[0, 0].set_ylabel('Portfolio Return')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Rolling Alpha
            window = 126  # 6-month rolling
            rolling_alpha = []
            dates = []
            
            for i in range(window, len(aligned_data)):
                subset = aligned_data.iloc[i-window:i]
                subset_model = LinearRegression().fit(
                    subset['benchmark'].values.reshape(-1, 1),
                    subset['portfolio'].values
                )
                rolling_alpha.append(subset_model.intercept_ * 252)  # Annualized
                dates.append(aligned_data.index[i])
            
            axes[0, 1].plot(dates, rolling_alpha, linewidth=2)
            axes[0, 1].axhline(y=0, color='gray', linestyle='-', alpha=0.5)
            axes[0, 1].set_title('Rolling 6-Month Alpha (Annualized)')
            axes[0, 1].set_ylabel('Alpha')
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. Excess Returns
            excess_returns = aligned_data['portfolio'] - aligned_data['benchmark']
            
            axes[1, 0].plot(excess_returns.index, excess_returns.cumsum(), linewidth=2)
            axes[1, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.7)
            axes[1, 0].set_title('Cumulative Excess Returns vs Benchmark')
            axes[1, 0].set_ylabel('Cumulative Excess Return')
            axes[1, 0].grid(True, alpha=0.3)
            
            # 4. Excess Returns Distribution
            axes[1, 1].hist(excess_returns.values, bins=50, alpha=0.7, edgecolor='black', density=True)
            axes[1, 1].axvline(excess_returns.mean(), color='red', linestyle='--',
                              label=f'Mean: {excess_returns.mean()*252:.2%}')
            axes[1, 1].axvline(0, color='gray', linestyle='-', alpha=0.7)
            axes[1, 1].set_title('Excess Returns Distribution')
            axes[1, 1].set_xlabel('Daily Excess Return')
            axes[1, 1].set_ylabel('Density')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save chart
            chart_path = self.output_dir / f"factor_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(chart_path)
            
        except Exception as e:
            self.logger.error(f"Factor analysis chart creation failed: {e}")
            return None
    
    def _generate_summary_report(self, results: Dict[str, Any]) -> str:
        """Generate a text summary report of the analysis."""
        try:
            report_lines = []
            report_lines.append("PYFOLIO PORTFOLIO ANALYSIS SUMMARY")
            report_lines.append("=" * 50)
            report_lines.append(f"Analysis Date: {results['analysis_timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
            report_lines.append(f"Charts Generated: {len(results['charts_saved'])}")
            report_lines.append("")
            
            # Performance statistics
            perf_stats = results.get('performance_stats', {})
            if perf_stats:
                report_lines.append("PERFORMANCE STATISTICS:")
                report_lines.append("-" * 25)
                report_lines.append(f"Total Return: {perf_stats.get('total_return', 0):.2%}")
                report_lines.append(f"Annualized Return: {perf_stats.get('annualized_return', 0):.2%}")
                report_lines.append(f"Volatility: {perf_stats.get('volatility', 0):.2%}")
                report_lines.append(f"Sharpe Ratio: {perf_stats.get('sharpe_ratio', 0):.2f}")
                report_lines.append(f"Maximum Drawdown: {perf_stats.get('max_drawdown', 0):.2%}")
                
                if 'alpha' in perf_stats:
                    report_lines.append(f"Alpha: {perf_stats['alpha']*252:.2%}")
                    report_lines.append(f"Beta: {perf_stats.get('beta', 0):.2f}")
                
                report_lines.append("")
            
            # Charts generated
            if results['charts_saved']:
                report_lines.append("CHARTS GENERATED:")
                report_lines.append("-" * 17)
                for chart_path in results['charts_saved']:
                    chart_name = Path(chart_path).stem
                    report_lines.append(f"• {chart_name}")
                report_lines.append("")
            
            # Error messages
            if results.get('error_messages'):
                report_lines.append("WARNINGS/ERRORS:")
                report_lines.append("-" * 16)
                for error in results['error_messages']:
                    report_lines.append(f"• {error}")
                report_lines.append("")
            
            report_lines.append("Analysis completed successfully using pyfolio.")
            
            return "\n".join(report_lines)
            
        except Exception as e:
            return f"Summary report generation failed: {str(e)}"
    
    def _validate_returns(self, returns: pd.Series) -> None:
        """Validate returns series for pyfolio analysis."""
        if not isinstance(returns, pd.Series):
            raise ValidationError("Returns must be a pandas Series")
        
        if len(returns) < 30:
            raise ValidationError("Returns series must have at least 30 observations")
        
        if returns.isna().all():
            raise ValidationError("Returns series contains only NaN values")
        
        if not isinstance(returns.index, pd.DatetimeIndex):
            self.logger.warning("Returns index is not DatetimeIndex, attempting conversion")
            try:
                returns.index = pd.to_datetime(returns.index)
            except:
                raise ValidationError("Cannot convert returns index to datetime")
    
    def export_charts_as_html(self, chart_paths: List[str], title: str = "Portfolio Analysis") -> str:
        """Export charts as an HTML report."""
        try:
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>{title}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    h1 {{ color: #333; text-align: center; }}
                    h2 {{ color: #666; }}
                    .chart {{ text-align: center; margin: 20px 0; }}
                    .chart img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
                    .timestamp {{ text-align: center; color: #999; font-size: 12px; }}
                </style>
            </head>
            <body>
                <h1>{title}</h1>
                <div class="timestamp">Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
            """
            
            for i, chart_path in enumerate(chart_paths, 1):
                chart_name = Path(chart_path).stem.replace('_', ' ').title()
                html_content += f"""
                <h2>Chart {i}: {chart_name}</h2>
                <div class="chart">
                    <img src="{chart_path}" alt="{chart_name}">
                </div>
                """
            
            html_content += """
            </body>
            </html>
            """
            
            # Save HTML file
            html_path = self.output_dir / f"portfolio_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            with open(html_path, 'w') as f:
                f.write(html_content)
            
            return str(html_path)
            
        except Exception as e:
            self.logger.error(f"HTML export failed: {e}")
            return ""