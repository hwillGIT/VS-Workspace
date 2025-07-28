"""
Backtesting module for the trading system.
"""

from .backtest_engine import BacktestEngine
from .strategy_adapter import StrategyAdapter
from .performance_analyzer import PerformanceAnalyzer

__all__ = ["BacktestEngine", "StrategyAdapter", "PerformanceAnalyzer"]