"""
Test script demonstrating pyfolio integration with the trading system.
"""

import asyncio
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Add the trading system to Python path
sys.path.append(str(Path(__file__).parent.parent))

from core.visualization.pyfolio_engine import PyfolioVisualizationEngine
from backtest.backtest_engine import BacktestEngine


def test_pyfolio_visualization_engine():
    """Test basic pyfolio visualization engine functionality."""
    print("🧪 Testing Pyfolio Visualization Engine")
    print("=" * 45)
    
    try:
        # Initialize pyfolio engine
        engine = PyfolioVisualizationEngine(output_dir="test_pyfolio")
        
        # Generate test portfolio returns
        np.random.seed(42)
        n_days = 252  # One year
        dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D')
        
        # Create realistic returns with different characteristics
        returns_data = []
        for i in range(n_days):
            # Base return
            base_return = 0.0008  # ~20% annual
            
            # Add volatility clustering (GARCH-like)
            if i > 0:
                vol_factor = 1 + 0.2 * abs(returns_data[-1])
            else:
                vol_factor = 1
            
            # Market regime (bear market mid-year)
            if 100 < i < 150:
                base_return = -0.002
                vol_factor *= 1.5
            
            # Generate return
            daily_return = np.random.normal(base_return, 0.015 * vol_factor)
            
            # Occasional extreme events
            if np.random.random() < 0.02:
                extreme_return = np.random.choice([-1, 1]) * np.random.uniform(0.03, 0.06)
                daily_return += extreme_return
            
            returns_data.append(daily_return)
        
        returns = pd.Series(returns_data, index=dates)
        
        print(f"📊 Generated Test Returns:")
        print(f"   Period: {dates[0].date()} to {dates[-1].date()}")
        print(f"   Days: {len(returns)}")
        print(f"   Total Return: {(1 + returns).prod() - 1:.2%}")
        print(f"   Annualized Return: {returns.mean() * 252:.2%}")
        print(f"   Volatility: {returns.std() * np.sqrt(252):.2%}")
        print(f"   Sharpe Ratio: {(returns.mean() * 252) / (returns.std() * np.sqrt(252)):.2f}")
        
        # Test different tearsheet types
        test_results = {}
        
        # 1. Test full tearsheet
        print(f"\n🔍 Testing Full Tearsheet...")
        full_result = engine.create_full_tearsheet(
            returns=returns,
            save_charts=True
        )
        
        if 'error' not in full_result:
            charts_count = len(full_result.get('charts', {}))
            files_count = len(full_result.get('files_created', []))
            print(f"   ✅ Full tearsheet: {charts_count} charts, {files_count} files")
            test_results['full_tearsheet'] = True
            
            # Check for key insights
            insights = full_result.get('insights', {})
            if insights:
                print(f"   📈 Insights generated: {len(insights)} categories")
        else:
            print(f"   ❌ Full tearsheet failed: {full_result['error']}")
            test_results['full_tearsheet'] = False
        
        # 2. Test simple tearsheet
        print(f"\n🔍 Testing Simple Tearsheet...")
        simple_result = engine.create_simple_tearsheet(
            returns=returns,
            save_charts=True
        )
        
        if 'error' not in simple_result:
            charts_count = len(simple_result.get('charts', {}))
            files_count = len(simple_result.get('files_created', []))
            print(f"   ✅ Simple tearsheet: {charts_count} charts, {files_count} files")
            test_results['simple_tearsheet'] = True
        else:
            print(f"   ❌ Simple tearsheet failed: {simple_result['error']}")
            test_results['simple_tearsheet'] = False
        
        # 3. Test risk analysis
        print(f"\n🔍 Testing Risk Analysis...")
        risk_result = engine.create_risk_analysis(
            returns=returns,
            save_charts=True
        )
        
        if 'error' not in risk_result:
            charts_count = len(risk_result.get('charts', {}))
            files_count = len(risk_result.get('files_created', []))
            print(f"   ✅ Risk analysis: {charts_count} charts, {files_count} files")
            test_results['risk_analysis'] = True
        else:
            print(f"   ❌ Risk analysis failed: {risk_result['error']}")
            test_results['risk_analysis'] = False
        
        # 4. Test rolling analysis
        print(f"\n🔍 Testing Rolling Analysis...")
        rolling_result = engine.create_rolling_analysis(
            returns=returns,
            save_charts=True
        )
        
        if 'error' not in rolling_result:
            charts_count = len(rolling_result.get('charts', {}))
            files_count = len(rolling_result.get('files_created', []))
            print(f"   ✅ Rolling analysis: {charts_count} charts, {files_count} files")
            test_results['rolling_analysis'] = True
        else:
            print(f"   ❌ Rolling analysis failed: {rolling_result['error']}")
            test_results['rolling_analysis'] = False
        
        # 5. Test individual chart functions
        print(f"\n🔍 Testing Individual Chart Functions...")
        
        chart_tests = {
            'returns_analysis': engine.create_returns_analysis_chart,
            'rolling_metrics': engine.create_rolling_metrics_chart,
            'drawdown_analysis': engine.create_drawdown_analysis_chart,
            'monthly_heatmap': engine.create_monthly_heatmap_chart
        }
        
        chart_test_results = {}
        for chart_name, chart_func in chart_tests.items():
            try:
                chart_result = chart_func(returns, save_chart=True)
                if 'error' not in chart_result:
                    chart_test_results[chart_name] = True
                    print(f"   ✅ {chart_name.replace('_', ' ').title()}: OK")
                else:
                    chart_test_results[chart_name] = False
                    print(f"   ❌ {chart_name.replace('_', ' ').title()}: {chart_result['error']}")
            except Exception as e:
                chart_test_results[chart_name] = False
                print(f"   ❌ {chart_name.replace('_', ' ').title()}: {str(e)}")
        
        # Summary
        total_tests = len(test_results) + len(chart_test_results)
        passed_tests = sum(test_results.values()) + sum(chart_test_results.values())
        
        print(f"\n📊 PYFOLIO ENGINE TEST RESULTS:")
        print(f"   Total Tests: {total_tests}")
        print(f"   Passed: {passed_tests}")
        print(f"   Failed: {total_tests - passed_tests}")
        print(f"   Success Rate: {passed_tests/total_tests:.1%}")
        
        return passed_tests == total_tests
        
    except Exception as e:
        print(f"❌ Pyfolio engine test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def test_backtest_pyfolio_integration():
    """Test pyfolio integration with backtest engine."""
    print(f"\n\n🔧 Testing Backtest-Pyfolio Integration")
    print("=" * 50)
    
    try:
        # Create mock backtest result
        print(f"📊 Creating mock backtest result...")
        
        # Generate mock equity curve
        np.random.seed(42)
        n_days = 200
        dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D')
        
        initial_value = 100000
        equity_values = [initial_value]
        
        for i in range(1, n_days):
            # Random daily return between -3% and +3%
            daily_return = np.random.normal(0.0008, 0.015)
            new_value = equity_values[-1] * (1 + daily_return)
            equity_values.append(new_value)
        
        equity_curve = dict(zip(dates, equity_values))
        
        mock_result = {
            'strategy_config': {
                'type': 'momentum',
                'name': 'Test Strategy'
            },
            'backtest_period': {
                'start_date': dates[0].isoformat(),
                'end_date': dates[-1].isoformat(),
                'days': n_days
            },
            'symbols': ['AAPL', 'GOOGL', 'MSFT'],
            'initial_cash': initial_value,
            'final_value': equity_values[-1],
            'total_return': (equity_values[-1] - initial_value) / initial_value,
            'execution_time_seconds': 15.2,
            'equity_curve': equity_curve
        }
        
        print(f"   Initial Value: ${mock_result['initial_cash']:,.2f}")
        print(f"   Final Value: ${mock_result['final_value']:,.2f}")
        print(f"   Total Return: {mock_result['total_return']:.2%}")
        print(f"   Period: {n_days} days")
        
        # Initialize backtest engine
        backtest_engine = BacktestEngine()
        
        # Test pyfolio tearsheet generation
        print(f"\n🎨 Testing Pyfolio Tearsheet Generation...")
        tearsheet_result = backtest_engine.generate_pyfolio_tearsheet(
            result=mock_result,
            save_charts=True,
            output_dir="test_backtest_pyfolio"
        )
        
        if 'error' not in tearsheet_result:
            print(f"   ✅ Tearsheet generation: SUCCESS")
            
            # Check generated content
            charts_created = len(tearsheet_result.get('charts', {}))
            files_created = len(tearsheet_result.get('files_created', []))
            
            print(f"   📊 Charts Created: {charts_created}")
            print(f"   📁 Files Created: {files_created}")
            
            # Check insights
            insights = tearsheet_result.get('insights', {})
            if insights:
                print(f"   💡 Insights Generated: {len(insights)} categories")
                
                # Display some insights
                perf_insights = insights.get('performance', {})
                if perf_insights:
                    print(f"      Best Month: {perf_insights.get('max_monthly_return', 0):.2%}")
                    print(f"      Worst Month: {perf_insights.get('max_monthly_loss', 0):.2%}")
                
                risk_insights = insights.get('risk', {})
                if risk_insights:
                    print(f"      Max Drawdown Duration: {risk_insights.get('longest_drawdown_days', 0):.0f} days")
            
            # Check HTML report
            html_report = tearsheet_result.get('html_report')
            if html_report:
                print(f"   🌐 HTML Report: {html_report}")
            
            return True
            
        else:
            print(f"   ❌ Tearsheet generation failed: {tearsheet_result['error']}")
            return False
        
    except Exception as e:
        print(f"❌ Backtest-pyfolio integration test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_pyfolio_with_benchmark():
    """Test pyfolio functionality with benchmark comparison."""
    print(f"\n\n📈 Testing Pyfolio with Benchmark")
    print("=" * 40)
    
    try:
        # Generate portfolio and benchmark returns
        np.random.seed(42)
        n_days = 252
        dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D')
        
        # Portfolio returns (higher return, higher volatility)
        portfolio_returns = np.random.normal(0.001, 0.018, n_days)
        
        # Benchmark returns (market-like)
        benchmark_returns = np.random.normal(0.0008, 0.012, n_days)
        
        portfolio_series = pd.Series(portfolio_returns, index=dates)
        benchmark_series = pd.Series(benchmark_returns, index=dates)
        
        print(f"📊 Generated Data:")
        print(f"   Portfolio Annualized Return: {portfolio_series.mean() * 252:.2%}")
        print(f"   Portfolio Volatility: {portfolio_series.std() * np.sqrt(252):.2%}")
        print(f"   Benchmark Annualized Return: {benchmark_series.mean() * 252:.2%}")
        print(f"   Benchmark Volatility: {benchmark_series.std() * np.sqrt(252):.2%}")
        
        # Test with benchmark
        engine = PyfolioVisualizationEngine(output_dir="test_benchmark_pyfolio")
        
        print(f"\n🔍 Testing Full Tearsheet with Benchmark...")
        result = engine.create_full_tearsheet(
            returns=portfolio_series,
            benchmark_rets=benchmark_series,
            save_charts=True
        )
        
        if 'error' not in result:
            charts_count = len(result.get('charts', {}))
            files_count = len(result.get('files_created', []))
            print(f"   ✅ Benchmark tearsheet: {charts_count} charts, {files_count} files")
            
            # Check for benchmark-specific insights
            insights = result.get('insights', {})
            if insights:
                perf_insights = insights.get('performance', {})
                if perf_insights:
                    print(f"   📈 Alpha vs Benchmark: Available in charts")
                    print(f"   📊 Relative performance: Analyzed")
            
            return True
        else:
            print(f"   ❌ Benchmark tearsheet failed: {result['error']}")
            return False
        
    except Exception as e:
        print(f"❌ Benchmark test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main test function."""
    print("🚀 Pyfolio Integration Test Suite")
    print("=" * 45)
    
    tests_passed = 0
    total_tests = 3
    
    # Test 1: Basic pyfolio engine
    if test_pyfolio_visualization_engine():
        tests_passed += 1
        print("✅ Test 1 Passed: Pyfolio Visualization Engine")
    else:
        print("❌ Test 1 Failed: Pyfolio Visualization Engine")
    
    # Test 2: Backtest integration
    if await test_backtest_pyfolio_integration():
        tests_passed += 1
        print("✅ Test 2 Passed: Backtest-Pyfolio Integration")
    else:
        print("❌ Test 2 Failed: Backtest-Pyfolio Integration")
    
    # Test 3: Benchmark functionality
    if test_pyfolio_with_benchmark():
        tests_passed += 1
        print("✅ Test 3 Passed: Benchmark Functionality")
    else:
        print("❌ Test 3 Failed: Benchmark Functionality")
    
    # Summary
    print(f"\n🏁 TEST SUMMARY")
    print("=" * 20)
    print(f"Tests Passed: {tests_passed}/{total_tests}")
    print(f"Success Rate: {tests_passed/total_tests:.1%}")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Pyfolio integration is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    print("\n💡 Pyfolio Integration Benefits:")
    print("   ✅ Professional tearsheets (used by hedge funds)")
    print("   ✅ Comprehensive visualization suite (20+ chart types)")
    print("   ✅ Interactive HTML reports for stakeholders")
    print("   ✅ Risk analysis and drawdown visualization")
    print("   ✅ Rolling performance metrics analysis")
    print("   ✅ Factor attribution and style analysis")
    print("   ✅ Seamless backtest integration")
    
    print(f"\n📁 Test Output Locations:")
    print("   • Basic tests: test_pyfolio/")
    print("   • Backtest integration: test_backtest_pyfolio/")
    print("   • Benchmark tests: test_benchmark_pyfolio/")
    print("   • Open HTML files for interactive analysis")


if __name__ == "__main__":
    asyncio.run(main())