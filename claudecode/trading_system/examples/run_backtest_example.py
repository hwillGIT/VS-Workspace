"""
Example script demonstrating how to run backtests with the trading system.
"""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add the trading system to Python path
sys.path.append(str(Path(__file__).parent.parent))

from backtest.backtest_engine import BacktestEngine, MultiStrategyBacktest


async def run_momentum_backtest():
    """Example: Run a momentum strategy backtest."""
    print("🚀 Running Momentum Strategy Backtest")
    print("=" * 50)
    
    # Initialize backtest engine
    backtest_engine = BacktestEngine()
    
    # Define strategy configuration
    momentum_config = {
        'type': 'momentum',
        'name': 'Momentum Strategy - 20 Day Lookback',
        'agent_config': {
            'lookback_period': 20,
            'threshold': 0.02,
            'min_volume': 1000000,
            'max_positions': 8,
            'momentum_windows': [5, 10, 20],
            'volume_confirmation': True
        }
    }
    
    # Define backtest parameters
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # 1 year backtest
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX']
    initial_cash = 100000
    
    print(f"📅 Period: {start_date.date()} to {end_date.date()}")
    print(f"💰 Initial Cash: ${initial_cash:,}")
    print(f"📊 Symbols: {', '.join(symbols)}")
    
    try:
        # Run backtest
        results = await backtest_engine.run_backtest(
            strategy_config=momentum_config,
            start_date=start_date,
            end_date=end_date,
            symbols=symbols,
            initial_cash=initial_cash,
            commission=0.001
        )
        
        # Display results
        print(f"\n📈 BACKTEST RESULTS")
        print("=" * 30)
        print(f"Final Portfolio Value: ${results['final_value']:,.2f}")
        print(f"Total Return: {results['total_return']:.2%}")
        print(f"Execution Time: {results['execution_time_seconds']:.2f} seconds")
        
        # Performance metrics
        perf_metrics = results.get('performance_metrics', {})
        basic_metrics = perf_metrics.get('basic_metrics', {})
        risk_metrics = perf_metrics.get('risk_metrics', {})
        trade_analysis = perf_metrics.get('trade_analysis', {})
        
        if basic_metrics:
            print(f"\n📊 PERFORMANCE METRICS")
            print("=" * 25)
            print(f"Annualized Return: {basic_metrics.get('annualized_return', 0):.2%}")
            print(f"Sharpe Ratio: {basic_metrics.get('sharpe_ratio', 0):.2f}")
            print(f"System Quality Number: {basic_metrics.get('sqn', 0):.2f}")
        
        if risk_metrics:
            print(f"\n📉 RISK METRICS")
            print("=" * 15)
            print(f"Maximum Drawdown: {risk_metrics.get('max_drawdown', 0):.2%}")
            print(f"Volatility: {risk_metrics.get('volatility', 0):.2%}")
            print(f"Sortino Ratio: {risk_metrics.get('sortino_ratio', 0):.2f}")
            print(f"VaR (95%): {risk_metrics.get('var_95', 0):.2%}")
        
        if trade_analysis:
            print(f"\n💼 TRADE ANALYSIS")
            print("=" * 17)
            print(f"Total Trades: {trade_analysis.get('total_trades', 0)}")
            print(f"Win Rate: {trade_analysis.get('win_rate', 0):.2%}")
            print(f"Profit Factor: {trade_analysis.get('profit_factor', 0):.2f}")
            print(f"Average Win: ${trade_analysis.get('avg_win', 0):.2f}")
            print(f"Average Loss: ${trade_analysis.get('avg_loss', 0):.2f}")
        
        # Overall assessment
        assessment = perf_metrics.get('overall_assessment', {})
        if assessment:
            print(f"\n🏆 OVERALL ASSESSMENT")
            print("=" * 22)
            print(f"Rating: {assessment.get('overall_rating', 0):.1f}/100")
            print(f"Recommendation: {assessment.get('recommendation', 'N/A')}")
            print(f"Risk Level: {assessment.get('risk_level', 'N/A')}")
            print(f"Consistency: {assessment.get('consistency', 'N/A')}")
            
            strengths = assessment.get('key_strengths', [])
            if strengths:
                print(f"Key Strengths: {', '.join(strengths)}")
            
            weaknesses = assessment.get('key_weaknesses', [])
            if weaknesses:
                print(f"Areas for Improvement: {', '.join(weaknesses)}")
        
        return results
        
    except Exception as e:
        print(f"❌ Backtest failed: {str(e)}")
        return None


async def compare_multiple_strategies():
    """Example: Compare multiple strategies."""
    print("\n\n🔄 Comparing Multiple Strategies")
    print("=" * 40)
    
    # Initialize multi-strategy backtest
    multi_backtest = MultiStrategyBacktest()
    
    # Define strategies to compare
    strategies = [
        {
            'type': 'momentum',
            'name': 'Momentum - Aggressive',
            'agent_config': {
                'lookback_period': 10,
                'threshold': 0.015,
                'max_positions': 5
            }
        },
        {
            'type': 'momentum',
            'name': 'Momentum - Conservative',
            'agent_config': {
                'lookback_period': 30,
                'threshold': 0.03,
                'max_positions': 10
            }
        },
        {
            'type': 'stat_arb',
            'name': 'Statistical Arbitrage',
            'agent_config': {
                'z_score_entry': 2.0,
                'z_score_exit': 0.5,
                'max_pairs': 3
            }
        },
        {
            'type': 'multi_agent',
            'name': 'Multi-Agent Ensemble',
            'agent_config': {
                'agent_weights': {
                    'momentum': 0.5,
                    'stat_arb': 0.3,
                    'event_driven': 0.2
                },
                'consensus_threshold': 0.6
            }
        }
    ]
    
    # Backtest parameters
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)  # 6 months for comparison
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
    initial_cash = 50000
    
    print(f"📅 Comparison Period: {start_date.date()} to {end_date.date()}")
    print(f"📊 Strategies: {len(strategies)}")
    
    try:
        comparison_results = await multi_backtest.compare_strategies(
            strategy_configs=strategies,
            start_date=start_date,
            end_date=end_date,
            symbols=symbols,
            initial_cash=initial_cash
        )
        
        print(f"\n📊 STRATEGY COMPARISON RESULTS")
        print("=" * 35)
        
        # Display individual strategy results
        individual_results = comparison_results.get('individual_results', {})
        
        print(f"{'Strategy':<25} {'Return':<10} {'Sharpe':<8} {'Max DD':<8} {'Rating':<8}")
        print("-" * 65)
        
        for strategy_name, result in individual_results.items():
            total_return = result.get('total_return', 0)
            performance_metrics = result.get('performance_metrics', {})
            basic_metrics = performance_metrics.get('basic_metrics', {})
            risk_metrics = performance_metrics.get('risk_metrics', {})
            assessment = performance_metrics.get('overall_assessment', {})
            
            sharpe = basic_metrics.get('sharpe_ratio', 0)
            max_dd = risk_metrics.get('max_drawdown', 0)
            rating = assessment.get('overall_rating', 0)
            
            print(f"{strategy_name:<25} {total_return:<10.2%} {sharpe:<8.2f} {max_dd:<8.2%} {rating:<8.1f}")
        
        # Best strategy
        best_strategy = comparison_results.get('best_strategy')
        if best_strategy:
            print(f"\n🏆 Best Performing Strategy: {best_strategy}")
        
        # Summary metrics
        summary_metrics = comparison_results.get('summary_metrics', {})
        if summary_metrics:
            print(f"\n📈 SUMMARY STATISTICS")
            print("=" * 22)
            print(f"Average Return: {summary_metrics.get('avg_return', 0):.2%}")
            print(f"Best Return: {summary_metrics.get('best_return', 0):.2%}")
            print(f"Worst Return: {summary_metrics.get('worst_return', 0):.2%}")
            print(f"Average Sharpe: {summary_metrics.get('avg_sharpe', 0):.2f}")
            print(f"Best Sharpe: {summary_metrics.get('best_sharpe', 0):.2f}")
        
        return comparison_results
        
    except Exception as e:
        print(f"❌ Strategy comparison failed: {str(e)}")
        return None


async def main():
    """Main function to run backtest examples."""
    print("🎯 Multi-Agent Trading System - Backtesting Examples")
    print("=" * 60)
    
    # Run momentum strategy backtest
    momentum_results = await run_momentum_backtest()
    
    # Compare multiple strategies
    comparison_results = await compare_multiple_strategies()
    
    print(f"\n✅ Backtesting examples completed!")
    print("💡 Tip: Modify strategy configurations to experiment with different parameters")
    print("📊 Tip: Add more symbols or extend the time period for more comprehensive testing")
    print("🎯 Tip: Use the results to optimize your trading strategies")


if __name__ == "__main__":
    asyncio.run(main())