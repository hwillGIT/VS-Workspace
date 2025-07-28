"""
Main backtesting engine using Backtrader framework.
"""

import backtrader as bt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import asyncio
from loguru import logger

from ..core.base.config import config
from ..core.base.exceptions import BacktestError
from ..core.apis.market_data import MarketDataAPI
from .strategy_adapter import StrategyAdapter
from .performance_analyzer import PerformanceAnalyzer


class TradingSystemStrategy(bt.Strategy):
    """
    Backtrader strategy that wraps our trading system agents.
    """
    
    params = (
        ('agent_strategy', None),
        ('rebalance_freq', 'daily'),
        ('min_position_size', 100),
        ('max_position_size', 10000),
        ('commission', 0.001),
    )
    
    def __init__(self):
        self.logger = logger.bind(strategy="backtest")
        self.strategy_adapter = self.params.agent_strategy
        self.rebalance_days = {'daily': 1, 'weekly': 7, 'monthly': 30}
        self.last_rebalance = None
        self.position_tracker = {}
        
        if not self.strategy_adapter:
            raise BacktestError("Strategy adapter is required")
    
    def next(self):
        """Execute strategy on each bar."""
        try:
            current_date = self.datas[0].datetime.date(0)
            
            # Check if it's time to rebalance
            if self.should_rebalance(current_date):
                self.rebalance_portfolio(current_date)
                self.last_rebalance = current_date
                
        except Exception as e:
            self.logger.error(f"Strategy execution error: {str(e)}")
    
    def should_rebalance(self, current_date: datetime) -> bool:
        """Check if portfolio should be rebalanced."""
        if self.last_rebalance is None:
            return True
        
        days_since_rebalance = (current_date - self.last_rebalance).days
        rebalance_frequency = self.rebalance_days.get(self.params.rebalance_freq, 1)
        
        return days_since_rebalance >= rebalance_frequency
    
    def rebalance_portfolio(self, current_date: datetime):
        """Rebalance portfolio based on strategy signals."""
        try:
            # Get current market data
            market_data = self.get_current_market_data()
            
            # Get strategy recommendations
            recommendations = self.strategy_adapter.get_recommendations(
                market_data, current_date
            )
            
            if not recommendations:
                return
            
            # Execute trades based on recommendations
            for rec in recommendations:
                self.execute_trade(rec)
                
        except Exception as e:
            self.logger.error(f"Rebalancing error: {str(e)}")
    
    def get_current_market_data(self) -> Dict[str, Any]:
        """Get current market data for all tracked instruments."""
        market_data = {}
        
        for i, data in enumerate(self.datas):
            symbol = data._name
            market_data[symbol] = {
                'open': data.open[0],
                'high': data.high[0],
                'low': data.low[0],
                'close': data.close[0],
                'volume': data.volume[0],
                'datetime': data.datetime.datetime(0)
            }
        
        return market_data
    
    def execute_trade(self, recommendation: Dict[str, Any]):
        """Execute individual trade based on recommendation."""
        try:
            symbol = recommendation.get('symbol')
            action = recommendation.get('action', '').upper()
            target_weight = recommendation.get('position_size', 0)
            
            if not symbol or action not in ['BUY', 'SELL', 'HOLD']:
                return
            
            # Find the data feed for this symbol
            data_feed = None
            for data in self.datas:
                if data._name == symbol:
                    data_feed = data
                    break
            
            if not data_feed:
                self.logger.warning(f"No data feed found for symbol {symbol}")
                return
            
            current_position = self.getposition(data_feed).size
            current_value = self.broker.getvalue()
            target_value = current_value * target_weight
            current_price = data_feed.close[0]
            target_size = int(target_value / current_price) if current_price > 0 else 0
            
            # Calculate the trade size needed
            trade_size = target_size - current_position
            
            # Execute the trade
            if abs(trade_size) >= self.params.min_position_size:
                if trade_size > 0:
                    self.buy(data=data_feed, size=trade_size)
                    self.logger.info(f"BUY {trade_size} shares of {symbol} at {current_price}")
                elif trade_size < 0:
                    self.sell(data=data_feed, size=abs(trade_size))
                    self.logger.info(f"SELL {abs(trade_size)} shares of {symbol} at {current_price}")
            
        except Exception as e:
            self.logger.error(f"Trade execution error: {str(e)}")


class BacktestEngine:
    """
    Main backtesting engine that coordinates strategy testing.
    """
    
    def __init__(self):
        self.logger = logger.bind(service="backtest_engine")
        self.market_api = MarketDataAPI()
        self.cerebro = None
        self.results = None
        
    async def run_backtest(self, 
                          strategy_config: Dict[str, Any],
                          start_date: datetime,
                          end_date: datetime,
                          symbols: List[str],
                          initial_cash: float = 100000,
                          commission: float = 0.001) -> Dict[str, Any]:
        """
        Run comprehensive backtest for a trading strategy.
        
        Args:
            strategy_config: Strategy configuration and agent settings
            start_date: Backtest start date
            end_date: Backtest end date
            symbols: List of symbols to trade
            initial_cash: Initial portfolio cash
            commission: Commission rate
            
        Returns:
            Backtest results and performance metrics
        """
        try:
            self.logger.info(f"Starting backtest from {start_date} to {end_date}")
            self.logger.info(f"Symbols: {symbols}, Initial cash: ${initial_cash:,.2f}")
            
            # Initialize Cerebro engine
            self.cerebro = bt.Cerebro()
            
            # Set initial conditions
            self.cerebro.broker.setcash(initial_cash)
            self.cerebro.broker.setcommission(commission=commission)
            
            # Add data feeds
            await self._add_data_feeds(symbols, start_date, end_date)
            
            # Create and add strategy
            strategy_adapter = StrategyAdapter(strategy_config)
            self.cerebro.addstrategy(
                TradingSystemStrategy,
                agent_strategy=strategy_adapter,
                commission=commission
            )
            
            # Add analyzers
            self._add_analyzers()
            
            # Run backtest
            self.logger.info("🚀 Running backtest...")
            start_time = datetime.now()
            self.results = self.cerebro.run()
            end_time = datetime.now()
            
            execution_time = (end_time - start_time).total_seconds()
            self.logger.info(f"✅ Backtest completed in {execution_time:.2f} seconds")
            
            # Generate performance report
            performance_analyzer = PerformanceAnalyzer(self.results[0])
            performance_report = performance_analyzer.generate_report()
            
            # Compile final results
            final_results = {
                'strategy_config': strategy_config,
                'backtest_period': {
                    'start_date': start_date.isoformat(),
                    'end_date': end_date.isoformat(),
                    'days': (end_date - start_date).days
                },
                'symbols': symbols,
                'initial_cash': initial_cash,
                'final_value': self.cerebro.broker.getvalue(),
                'total_return': (self.cerebro.broker.getvalue() - initial_cash) / initial_cash,
                'execution_time_seconds': execution_time,
                'performance_metrics': performance_report,
                'trades': self._extract_trade_history(),
                'equity_curve': self._extract_equity_curve()
            }
            
            self.logger.info(f"📊 Final portfolio value: ${final_results['final_value']:,.2f}")
            self.logger.info(f"📈 Total return: {final_results['total_return']:.2%}")
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"Backtest failed: {str(e)}")
            raise BacktestError(f"Backtest execution failed: {str(e)}")
    
    def generate_pyfolio_tearsheet(self, result: Dict[str, Any], 
                                 save_charts: bool = True,
                                 output_dir: str = "backtest_results") -> Dict[str, Any]:
        """Generate pyfolio tearsheet from backtest results."""
        try:
            from ..core.visualization.pyfolio_engine import PyfolioVisualizationEngine
            
            # Extract returns data from backtest result
            if 'equity_curve' not in result:
                raise ValueError("Equity curve not found in backtest result")
            
            equity_curve = result['equity_curve']
            
            # Convert equity curve to returns
            if isinstance(equity_curve, dict):
                import pandas as pd
                equity_series = pd.Series(equity_curve)
                returns = equity_series.pct_change().dropna()
            else:
                returns = equity_curve.pct_change().dropna()
            
            # Initialize pyfolio engine
            pyfolio_engine = PyfolioVisualizationEngine(output_dir=output_dir)
            
            # Generate full tearsheet
            tearsheet_result = pyfolio_engine.create_full_tearsheet(
                returns=returns,
                save_charts=save_charts
            )
            
            # Add to result
            result['pyfolio_analysis'] = tearsheet_result
            
            self.logger.info(f"✅ Generated pyfolio tearsheet in {output_dir}")
            
            return tearsheet_result
            
        except Exception as e:
            self.logger.error(f"Failed to generate pyfolio tearsheet: {str(e)}")
            return {'error': str(e)}
    
    async def _add_data_feeds(self, symbols: List[str], start_date: datetime, end_date: datetime):
        """Add data feeds for all symbols."""
        try:
            # Fetch data for all symbols
            data_dict = await self.market_api.get_multiple_symbols(
                symbols, start_date, end_date, interval="1d"
            )
            
            for symbol, data in data_dict.items():
                if data.empty:
                    self.logger.warning(f"No data available for {symbol}")
                    continue
                
                # Convert to backtrader format
                bt_data = self._convert_to_bt_format(data, symbol)
                self.cerebro.adddata(bt_data, name=symbol)
                
            self.logger.info(f"Added {len(data_dict)} data feeds to backtest")
            
        except Exception as e:
            raise BacktestError(f"Failed to add data feeds: {str(e)}")
    
    def _convert_to_bt_format(self, data: pd.DataFrame, symbol: str) -> bt.feeds.PandasData:
        """Convert pandas DataFrame to Backtrader data format."""
        try:
            # Ensure required columns exist
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in data.columns:
                    raise BacktestError(f"Required column '{col}' missing for {symbol}")
            
            # Create backtrader data feed
            bt_data = bt.feeds.PandasData(
                dataname=data,
                datetime=None,  # Use index as datetime
                open='open',
                high='high',
                low='low',
                close='close',
                volume='volume',
                openinterest=None
            )
            
            return bt_data
            
        except Exception as e:
            raise BacktestError(f"Data conversion failed for {symbol}: {str(e)}")
    
    def _add_analyzers(self):
        """Add performance analyzers to the backtest."""
        # Basic analyzers
        self.cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        self.cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        self.cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        self.cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        
        # Advanced analyzers
        self.cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn')  # System Quality Number
        self.cerebro.addanalyzer(bt.analyzers.VWR, _name='vwr')  # Variability-Weighted Return
        self.cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='time_return')
        
        # Position analyzer
        self.cerebro.addanalyzer(bt.analyzers.PositionsValue, _name='positions')
    
    def _extract_trade_history(self) -> List[Dict[str, Any]]:
        """Extract trade history from backtest results."""
        try:
            if not self.results or not self.results[0].analyzers.trades:
                return []
            
            trade_analyzer = self.results[0].analyzers.trades.get_analysis()
            
            trades = []
            if 'trades' in trade_analyzer:
                for trade in trade_analyzer['trades']:
                    trades.append({
                        'symbol': getattr(trade, 'ref', 'Unknown'),
                        'entry_date': getattr(trade, 'dtopen', None),
                        'exit_date': getattr(trade, 'dtclose', None),
                        'entry_price': getattr(trade, 'price', 0),
                        'exit_price': getattr(trade, 'pnl', 0),
                        'size': getattr(trade, 'size', 0),
                        'pnl': getattr(trade, 'pnl', 0),
                        'pnl_percent': getattr(trade, 'pnlcomm', 0)
                    })
            
            return trades
            
        except Exception as e:
            self.logger.error(f"Failed to extract trade history: {str(e)}")
            return []
    
    def _extract_equity_curve(self) -> List[Dict[str, Any]]:
        """Extract equity curve from backtest results."""
        try:
            if not self.results or not self.results[0].analyzers.time_return:
                return []
            
            time_return = self.results[0].analyzers.time_return.get_analysis()
            
            equity_curve = []
            for date, return_value in time_return.items():
                equity_curve.append({
                    'date': date.isoformat() if hasattr(date, 'isoformat') else str(date),
                    'portfolio_value': return_value,
                    'return': return_value
                })
            
            return equity_curve
            
        except Exception as e:
            self.logger.error(f"Failed to extract equity curve: {str(e)}")
            return []
    
    def plot_results(self, save_path: Optional[str] = None):
        """Plot backtest results."""
        try:
            if not self.cerebro:
                raise BacktestError("No backtest results to plot")
            
            import matplotlib.pyplot as plt
            
            # Plot using backtrader's built-in plotting
            fig = self.cerebro.plot(style='candlestick', barup='green', bardown='red')[0][0]
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Results plot saved to {save_path}")
            else:
                plt.show()
                
        except Exception as e:
            self.logger.error(f"Failed to plot results: {str(e)}")


class MultiStrategyBacktest:
    """
    Run backtests for multiple strategies and compare performance.
    """
    
    def __init__(self):
        self.logger = logger.bind(service="multi_strategy_backtest")
        self.backtest_engine = BacktestEngine()
        
    async def compare_strategies(self,
                               strategy_configs: List[Dict[str, Any]],
                               start_date: datetime,
                               end_date: datetime,
                               symbols: List[str],
                               initial_cash: float = 100000) -> Dict[str, Any]:
        """
        Compare multiple strategies side by side.
        """
        try:
            self.logger.info(f"Comparing {len(strategy_configs)} strategies")
            
            results = {}
            for i, config in enumerate(strategy_configs):
                strategy_name = config.get('name', f'Strategy_{i+1}')
                self.logger.info(f"Running backtest for {strategy_name}")
                
                strategy_result = await self.backtest_engine.run_backtest(
                    config, start_date, end_date, symbols, initial_cash
                )
                
                results[strategy_name] = strategy_result
            
            # Generate comparison report
            comparison_report = self._generate_comparison_report(results)
            
            return {
                'individual_results': results,
                'comparison_report': comparison_report,
                'best_strategy': comparison_report.get('best_strategy'),
                'summary_metrics': comparison_report.get('summary_metrics')
            }
            
        except Exception as e:
            self.logger.error(f"Strategy comparison failed: {str(e)}")
            raise BacktestError(f"Multi-strategy backtest failed: {str(e)}")
    
    def _generate_comparison_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comparison report for multiple strategies."""
        try:
            comparison_metrics = {}
            
            for strategy_name, result in results.items():
                metrics = result.get('performance_metrics', {})
                comparison_metrics[strategy_name] = {
                    'total_return': result.get('total_return', 0),
                    'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                    'max_drawdown': metrics.get('max_drawdown', 0),
                    'win_rate': metrics.get('win_rate', 0),
                    'profit_factor': metrics.get('profit_factor', 0),
                    'final_value': result.get('final_value', 0)
                }
            
            # Find best strategy by Sharpe ratio
            best_strategy = max(
                comparison_metrics.items(),
                key=lambda x: x[1].get('sharpe_ratio', 0)
            )[0]
            
            return {
                'metrics_by_strategy': comparison_metrics,
                'best_strategy': best_strategy,
                'summary_metrics': self._calculate_summary_metrics(comparison_metrics)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate comparison report: {str(e)}")
            return {}
    
    def _calculate_summary_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate summary statistics across all strategies."""
        try:
            all_returns = [m.get('total_return', 0) for m in metrics.values()]
            all_sharpe = [m.get('sharpe_ratio', 0) for m in metrics.values()]
            all_drawdowns = [m.get('max_drawdown', 0) for m in metrics.values()]
            
            return {
                'avg_return': np.mean(all_returns),
                'std_return': np.std(all_returns),
                'avg_sharpe': np.mean(all_sharpe),
                'avg_max_drawdown': np.mean(all_drawdowns),
                'best_return': max(all_returns),
                'worst_return': min(all_returns),
                'best_sharpe': max(all_sharpe),
                'worst_drawdown': max(all_drawdowns)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to calculate summary metrics: {str(e)}")
            return {}