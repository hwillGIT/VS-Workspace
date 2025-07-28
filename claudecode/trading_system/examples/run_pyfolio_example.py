"""
Example demonstrating pyfolio-enhanced backtesting and portfolio visualization.
"""

import asyncio
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Add the trading system to Python path
sys.path.append(str(Path(__file__).parent.parent))

from backtest.backtest_engine import BacktestEngine
from core.visualization.pyfolio_engine import PyfolioVisualizationEngine


async def run_pyfolio_enhanced_backtest():
    """Run backtest with pyfolio-enhanced visualization."""
    print("📊 Pyfolio-Enhanced Backtesting Analysis")
    print("=" * 50)
    
    try:
        # Initialize backtest engine
        backtest_engine = BacktestEngine()
        
        # Define strategy for visualization
        strategy_config = {
            'type': 'momentum',
            'name': 'Multi-Asset Momentum Strategy',
            'agent_config': {
                'lookback_period': 30,
                'threshold': 0.02,
                'max_positions': 6,
                'volume_confirmation': True,
                'risk_management': {
                    'stop_loss': 0.08,
                    'take_profit': 0.15,
                    'position_sizing': 'kelly'
                }
            }
        }
        
        # Backtest parameters
        end_date = datetime.now()
        start_date = end_date - timedelta(days=500)  # ~1.5 years
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM']
        initial_cash = 100000
        
        print(f"📅 Backtest Period: {start_date.date()} to {end_date.date()}")
        print(f"💰 Initial Cash: ${initial_cash:,}")
        print(f"📊 Universe: {', '.join(symbols)}")
        print(f"🎯 Strategy: {strategy_config['name']}")
        
        # Run backtest
        print(f"\n🚀 Running backtest...")
        result = await backtest_engine.run_backtest(
            strategy_config=strategy_config,
            start_date=start_date,
            end_date=end_date,
            symbols=symbols,
            initial_cash=initial_cash,
            commission=0.001
        )
        
        # Display basic results
        print(f"\n✅ BACKTEST RESULTS:")
        print(f"   Final Value: ${result['final_value']:,.2f}")
        print(f"   Total Return: {result['total_return']:.2%}")
        print(f"   Execution Time: {result['execution_time_seconds']:.2f}s")
        
        # Generate pyfolio tearsheet
        print(f"\n📈 Generating Pyfolio Tearsheet...")
        output_dir = "pyfolio_analysis"
        tearsheet_result = backtest_engine.generate_pyfolio_tearsheet(
            result=result,
            save_charts=True,
            output_dir=output_dir
        )
        
        if 'error' not in tearsheet_result:
            print(f"✅ Pyfolio analysis completed successfully!")
            
            # Display key pyfolio insights
            pyfolio_insights = tearsheet_result.get('insights', {})
            if pyfolio_insights:
                print(f"\n📊 PYFOLIO INSIGHTS:")
                
                # Performance insights
                performance = pyfolio_insights.get('performance', {})
                if performance:
                    print(f"   📈 Performance Analysis:")
                    print(f"      Best Month: {performance.get('best_month', 'N/A')}")
                    print(f"      Worst Month: {performance.get('worst_month', 'N/A')}")
                    print(f"      Winning Months: {performance.get('winning_months', 0):.0f}%")
                    print(f"      Max Monthly Gain: {performance.get('max_monthly_return', 0):.2%}")
                    print(f"      Max Monthly Loss: {performance.get('max_monthly_loss', 0):.2%}")
                
                # Risk insights
                risk = pyfolio_insights.get('risk', {})
                if risk:
                    print(f"   ⚠️  Risk Analysis:")
                    print(f"      Longest Drawdown: {risk.get('longest_drawdown_days', 0):.0f} days")
                    print(f"      Recovery Time: {risk.get('avg_recovery_time', 0):.0f} days")
                    print(f"      Drawdown Frequency: {risk.get('drawdown_frequency', 0):.1f} per year")
                    print(f"      Worst 5 Drawdowns: {len(risk.get('worst_drawdowns', []))} periods")
                
                # Chart analysis
                charts = pyfolio_insights.get('charts', {})
                if charts:
                    print(f"   📊 Charts Generated:")
                    chart_types = list(charts.keys())
                    print(f"      Chart Types: {', '.join(chart_types[:5])}")
                    if len(chart_types) > 5:
                        print(f"      + {len(chart_types) - 5} more charts")
            
            # File outputs
            files_created = tearsheet_result.get('files_created', [])
            if files_created:
                print(f"\n📁 FILES CREATED:")
                for file_path in files_created[:8]:  # Show first 8 files
                    print(f"   • {file_path}")
                if len(files_created) > 8:
                    print(f"   ... and {len(files_created) - 8} more files")
            
            # HTML report
            html_report = tearsheet_result.get('html_report')
            if html_report:
                print(f"\n🌐 HTML REPORT:")
                print(f"   Location: {html_report}")
                print(f"   Open this file in your browser for interactive analysis")
        
        else:
            print(f"❌ Pyfolio analysis failed: {tearsheet_result['error']}")
        
        return result, tearsheet_result
        
    except Exception as e:
        print(f"❌ Pyfolio backtest example failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None


async def demonstrate_pyfolio_chart_types():
    """Demonstrate different pyfolio chart types with mock data."""
    print(f"\n\n🎨 Pyfolio Chart Types Demonstration")
    print("=" * 45)
    
    try:
        # Create mock portfolio returns
        print(f"📊 Creating mock portfolio data for demonstration...")
        
        # Generate realistic portfolio returns
        np.random.seed(42)
        n_days = 365
        dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D')
        
        # Multi-regime returns simulation
        returns_data = []
        for i, date in enumerate(dates):
            # Base trend
            base_return = 0.0008  # ~20% annual
            
            # Market regime effects
            if 50 < i < 120:  # Bull market
                base_return = 0.0015
                volatility = 0.012
            elif 200 < i < 280:  # Bear market
                base_return = -0.001
                volatility = 0.025
            else:  # Normal market
                volatility = 0.015
            
            # Generate daily return
            daily_return = np.random.normal(base_return, volatility)
            
            # Add occasional extreme events
            if np.random.random() < 0.02:
                extreme_return = np.random.choice([-1, 1]) * np.random.uniform(0.03, 0.07)
                daily_return += extreme_return
            
            returns_data.append(daily_return)
        
        returns = pd.Series(returns_data, index=dates)
        
        print(f"   Generated {len(returns)} days of returns")
        print(f"   Period: {returns.index[0].date()} to {returns.index[-1].date()}")
        print(f"   Total Return: {(1 + returns).prod() - 1:.2%}")
        print(f"   Annualized Return: {returns.mean() * 252:.2%}")
        print(f"   Volatility: {returns.std() * np.sqrt(252):.2%}")
        
        # Initialize pyfolio engine
        pyfolio_engine = PyfolioVisualizationEngine(output_dir="pyfolio_demo")
        
        # Generate different types of tearsheets
        print(f"\n🔍 Generating Multiple Tearsheet Types...")
        
        # 1. Full tearsheet
        print(f"\n1️⃣ Full Tearsheet (comprehensive analysis):")
        full_result = pyfolio_engine.create_full_tearsheet(
            returns=returns,
            save_charts=True
        )
        
        if 'error' not in full_result:
            charts_created = len(full_result.get('charts', {}))
            print(f"   ✅ Full tearsheet: {charts_created} charts created")
            
            insights = full_result.get('insights', {})
            if insights:
                perf = insights.get('performance', {})
                if perf:
                    print(f"   📈 Best month: {perf.get('max_monthly_return', 0):.2%}")
                    print(f"   📉 Worst month: {perf.get('max_monthly_loss', 0):.2%}")
        
        # 2. Simple tearsheet
        print(f"\n2️⃣ Simple Tearsheet (essential charts only):")
        simple_result = pyfolio_engine.create_simple_tearsheet(
            returns=returns,
            save_charts=True
        )
        
        if 'error' not in simple_result:
            charts_created = len(simple_result.get('charts', {}))
            print(f"   ✅ Simple tearsheet: {charts_created} charts created")
        
        # 3. Risk analysis
        print(f"\n3️⃣ Risk Analysis Focus:")
        risk_result = pyfolio_engine.create_risk_analysis(
            returns=returns,
            save_charts=True
        )
        
        if 'error' not in risk_result:
            charts_created = len(risk_result.get('charts', {}))
            print(f"   ✅ Risk analysis: {charts_created} charts created")
            
            insights = risk_result.get('insights', {})
            if insights:
                risk_info = insights.get('risk', {})
                if risk_info:
                    print(f"   ⚠️  Max drawdown duration: {risk_info.get('longest_drawdown_days', 0):.0f} days")
                    print(f"   📊 Drawdown frequency: {risk_info.get('drawdown_frequency', 0):.1f}/year")
        
        # 4. Rolling analysis
        print(f"\n4️⃣ Rolling Metrics Analysis:")
        rolling_result = pyfolio_engine.create_rolling_analysis(
            returns=returns,
            save_charts=True
        )
        
        if 'error' not in rolling_result:
            charts_created = len(rolling_result.get('charts', {}))
            print(f"   ✅ Rolling analysis: {charts_created} charts created")
        
        # Summary of all files created
        all_files = []
        for result in [full_result, simple_result, risk_result, rolling_result]:
            if 'files_created' in result:
                all_files.extend(result['files_created'])
        
        print(f"\n📁 TOTAL FILES CREATED: {len(all_files)}")
        print(f"   Chart types include:")
        print(f"   • Cumulative returns plots")
        print(f"   • Rolling Sharpe ratio analysis")
        print(f"   • Drawdown periods visualization")
        print(f"   • Monthly returns heatmap")
        print(f"   • Distribution analysis")
        print(f"   • Risk metrics over time")
        print(f"   • Performance attribution charts")
        print(f"   • And many more professional visualizations")
        
        # Best practices and insights
        print(f"\n💡 PYFOLIO BEST PRACTICES:")
        print(f"   ✅ Use for institutional-quality reporting")
        print(f"   ✅ Generate tearsheets after every backtest")
        print(f"   ✅ Share HTML reports with stakeholders")
        print(f"   ✅ Monitor rolling metrics for regime changes")
        print(f"   ✅ Analyze drawdown patterns for risk management")
        print(f"   ✅ Use monthly heatmaps for seasonal analysis")
        
        return {
            'full_tearsheet': full_result,
            'simple_tearsheet': simple_result,
            'risk_analysis': risk_result,
            'rolling_analysis': rolling_result
        }
        
    except Exception as e:
        print(f"❌ Pyfolio demonstration failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


async def main():
    """Main demonstration function."""
    print("🚀 Pyfolio-Enhanced Trading System Analysis")
    print("=" * 55)
    print("Professional portfolio visualization using pyfolio")
    print("(the same library used by Quantopian for tearsheets)")
    
    # Run pyfolio-enhanced backtest
    backtest_result, tearsheet_result = await run_pyfolio_enhanced_backtest()
    
    # Demonstrate chart types
    chart_demo_results = await demonstrate_pyfolio_chart_types()
    
    print(f"\n\n🎉 ANALYSIS COMPLETE!")
    print("=" * 30)
    print("🎯 What you've experienced:")
    print("   ✅ Professional-grade portfolio visualization")
    print("   ✅ Institutional-quality tearsheets")
    print("   ✅ Comprehensive risk analysis charts")
    print("   ✅ Rolling performance metrics")
    print("   ✅ Drawdown period visualization")
    print("   ✅ Monthly performance heatmaps")
    print("   ✅ Distribution and factor analysis")
    print("   ✅ HTML report generation")
    
    print(f"\n🏆 Pyfolio Integration Benefits:")
    print("   • Same library used by professional hedge funds")
    print("   • 20+ chart types for comprehensive analysis")
    print("   • Interactive HTML reports for stakeholders")
    print("   • Automated performance attribution")
    print("   • Professional-grade risk visualization")
    print("   • Factor exposure and style analysis")
    print("   • Seamless integration with backtesting")
    
    print(f"\n🚀 Ready for Production:")
    print("   • Generate tearsheets automatically after backtests")
    print("   • Share professional reports with investors")
    print("   • Monitor performance with rolling analytics")
    print("   • Identify risk patterns with drawdown analysis")
    print("   • Perform factor attribution for strategy explanation")
    
    # File locations
    print(f"\n📁 Output Locations:")
    print("   • Backtest pyfolio charts: pyfolio_analysis/")
    print("   • Demo charts: pyfolio_demo/")
    print("   • Open the HTML files in your browser for interactive analysis")
    
    if backtest_result:
        print(f"\n📊 Backtest Summary:")
        print(f"   • Final Value: ${backtest_result['final_value']:,.2f}")
        print(f"   • Total Return: {backtest_result['total_return']:.2%}")
        print(f"   • Pyfolio Enhanced: {'✅' if tearsheet_result and 'error' not in tearsheet_result else '❌'}")


if __name__ == "__main__":
    asyncio.run(main())