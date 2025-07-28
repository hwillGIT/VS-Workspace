"""
Test script demonstrating empyrical integration with the trading system.
"""

import asyncio
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Add the trading system to Python path
sys.path.append(str(Path(__file__).parent.parent))

from core.risk.empyrical_engine import EmpyricalRiskEngine
from backtest.performance_analyzer import PerformanceAnalyzer


def test_empyrical_risk_engine():
    """Test basic empyrical risk engine functionality."""
    print("🧪 Testing Empyrical Risk Engine")
    print("=" * 40)
    
    try:
        # Initialize empyrical engine
        engine = EmpyricalRiskEngine()
        
        # Generate realistic portfolio returns
        np.random.seed(42)
        num_days = 500
        
        # Simulate realistic portfolio returns with different regimes
        returns_data = []
        dates = pd.date_range(start='2023-01-01', periods=num_days, freq='D')
        
        for i in range(num_days):
            # Base return with trend
            base_return = 0.0008  # ~20% annual
            
            # Volatility clustering
            if i > 0:
                vol_factor = 1 + 0.3 * abs(returns_data[-1])  # GARCH-like effect
            else:
                vol_factor = 1
            
            # Regime switching (bear market mid-period)
            if 150 < i < 250:
                base_return = -0.002  # Bear market
                vol_factor *= 1.5
            
            # Generate return
            daily_return = np.random.normal(base_return, 0.015 * vol_factor)
            
            # Add occasional extreme events
            if np.random.random() < 0.02:  # 2% chance of extreme event
                extreme_magnitude = np.random.choice([-1, 1]) * np.random.uniform(0.03, 0.08)
                daily_return += extreme_magnitude
            
            returns_data.append(daily_return)
        
        returns_series = pd.Series(returns_data, index=dates)
        
        print(f"📊 Generated Portfolio Returns:")
        print(f"   Period: {dates[0].date()} to {dates[-1].date()}")
        print(f"   Total Days: {len(returns_series)}")
        print(f"   Total Return: {(1 + returns_series).prod() - 1:.2%}")
        print(f"   Annualized Return: {returns_series.mean() * 252:.2%}")
        print(f"   Volatility: {returns_series.std() * np.sqrt(252):.2%}")
        
        # Calculate comprehensive metrics
        print(f"\n🔍 Running Empyrical Analysis...")
        metrics = engine.calculate_comprehensive_metrics(
            returns=returns_series,
            benchmark_returns=None,
            risk_free_rate=0.02,
            period='daily'
        )
        
        # Display results
        print(f"\n✅ EMPYRICAL ANALYSIS RESULTS:")
        print(f"   Total Return: {metrics.get('total_return', 0):.2%}")
        print(f"   Annualized Return: {metrics.get('annualized_return', 0):.2%}")
        print(f"   Volatility: {metrics.get('volatility', 0):.2%}")
        print(f"   Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"   Sortino Ratio: {metrics.get('sortino_ratio', 0):.2f}")
        print(f"   Calmar Ratio: {metrics.get('calmar_ratio', 0):.2f}")
        
        print(f"\n📉 RISK METRICS:")
        print(f"   Maximum Drawdown: {metrics.get('max_drawdown', 0):.2%}")
        print(f"   VaR (95%): {metrics.get('var_95', 0):.2%}")
        print(f"   CVaR (95%): {metrics.get('cvar_95', 0):.2%}")
        print(f"   Downside Deviation: {metrics.get('downside_deviation', 0):.2%}")
        print(f"   Tail Ratio: {metrics.get('tail_ratio', 0):.2f}")
        
        print(f"\n📈 PERFORMANCE CHARACTERISTICS:")
        print(f"   Win Rate: {metrics.get('win_rate', 0):.1%}")
        print(f"   Profit Factor: {metrics.get('profit_factor', 0):.2f}")
        print(f"   Stability: {metrics.get('stability_of_timeseries', 0):.2f}")
        print(f"   Skewness: {metrics.get('skewness', 0):.2f}")
        print(f"   Kurtosis: {metrics.get('kurtosis', 0):.2f}")
        
        # Overall assessment
        assessment = metrics.get('overall_assessment', {})
        if assessment:
            print(f"\n🏆 OVERALL ASSESSMENT:")
            print(f"   Rating: {assessment.get('rating', 'N/A')}")
            print(f"   Score: {assessment.get('overall_score', 0):.0f}/100")
            print(f"   Risk Level: {assessment.get('risk_level', 'N/A')}")
            print(f"   Recommendation: {assessment.get('recommendation', 'N/A')}")
            
            strengths = assessment.get('key_strengths', [])
            if strengths:
                print(f"   Key Strengths: {', '.join(strengths)}")
            
            weaknesses = assessment.get('key_weaknesses', [])
            if weaknesses:
                print(f"   Areas for Improvement: {', '.join(weaknesses)}")
        
        # Rolling metrics
        rolling_metrics = metrics.get('rolling_metrics_series', {})
        if rolling_metrics:
            print(f"\n📊 ROLLING METRICS SUMMARY:")
            if 'sharpe' in rolling_metrics:
                rolling_sharpe = rolling_metrics['sharpe']
                print(f"   Rolling Sharpe (1Y): {rolling_sharpe.mean():.2f} ± {rolling_sharpe.std():.2f}")
                print(f"   Best Period: {rolling_sharpe.max():.2f}")
                print(f"   Worst Period: {rolling_sharpe.min():.2f}")
        
        # Risk report
        risk_report = metrics.get('risk_report', '')
        if risk_report:
            print(f"\n📋 DETAILED RISK REPORT:")
            print(risk_report)
        
        return True
        
    except Exception as e:
        print(f"❌ Empyrical test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_performance_analyzer_enhancement():
    """Test enhanced performance analyzer with empyrical."""
    print("\n\n🔧 Testing Enhanced Performance Analyzer")
    print("=" * 45)
    
    try:
        # Create mock strategy result
        class MockStrategyResult:
            def __init__(self):
                self.analyzers = MockAnalyzers()
        
        class MockAnalyzers:
            def __init__(self):
                self.returns = MockReturnsAnalyzer()
                self.sharpe = MockSharpeAnalyzer()
                self.drawdown = MockDrawdownAnalyzer()
        
        class MockReturnsAnalyzer:
            def get_analysis(self):
                return {
                    'rtot': 0.25,  # 25% total return
                    'rnorm': 0.20,  # 20% annualized
                    'ravg': 0.0008  # Average daily return
                }
        
        class MockSharpeAnalyzer:
            def get_analysis(self):
                return {'sharperatio': 1.45}
        
        class MockDrawdownAnalyzer:
            def get_analysis(self):
                return {
                    'max': {
                        'drawdown': -0.12,
                        'len': 45,
                        'moneydown': -12000
                    }
                }
        
        # Initialize performance analyzer
        mock_result = MockStrategyResult()
        analyzer = PerformanceAnalyzer(mock_result)
        
        print(f"📊 Running Enhanced Performance Analysis...")
        
        # Generate performance report
        report = analyzer.generate_report()
        
        # Display results
        print(f"\n✅ PERFORMANCE ANALYSIS RESULTS:")
        
        # Basic metrics
        basic_metrics = report.get('basic_metrics', {})
        print(f"   Total Return: {basic_metrics.get('total_return', 0):.2%}")
        print(f"   Annualized Return: {basic_metrics.get('annualized_return', 0):.2%}")
        print(f"   Sharpe Ratio: {basic_metrics.get('sharpe_ratio', 0):.2f}")
        
        # Risk metrics
        risk_metrics = report.get('risk_metrics', {})
        print(f"   Max Drawdown: {risk_metrics.get('max_drawdown', 0):.2%}")
        print(f"   Volatility: {risk_metrics.get('volatility', 0):.2%}")
        
        # Empyrical metrics (if available)
        empyrical_metrics = report.get('empyrical_metrics', {})
        if empyrical_metrics:
            print(f"\n📈 EMPYRICAL ENHANCED METRICS:")
            print(f"   Enhanced Sharpe: {empyrical_metrics.get('sharpe_ratio', 0):.2f}")
            print(f"   Sortino Ratio: {empyrical_metrics.get('sortino_ratio', 0):.2f}")
            print(f"   Calmar Ratio: {empyrical_metrics.get('calmar_ratio', 0):.2f}")
            print(f"   VaR (95%): {empyrical_metrics.get('var_95', 0):.2%}")
            print(f"   CVaR (95%): {empyrical_metrics.get('cvar_95', 0):.2%}")
            print(f"   Tail Ratio: {empyrical_metrics.get('tail_ratio', 0):.2f}")
            print(f"   Stability: {empyrical_metrics.get('stability_of_timeseries', 0):.2f}")
            
            # Empyrical assessment
            empyrical_analysis = empyrical_metrics.get('empyrical_analysis', {})
            assessment = empyrical_analysis.get('overall_assessment', {})
            if assessment:
                print(f"   Empyrical Rating: {assessment.get('rating', 'N/A')}")
                print(f"   Risk Level: {assessment.get('risk_level', 'N/A')}")
        
        # Overall assessment
        overall_assessment = report.get('overall_assessment', {})
        if overall_assessment:
            print(f"\n🏆 OVERALL ASSESSMENT:")
            print(f"   Combined Rating: {overall_assessment.get('overall_rating', 0):.1f}/100")
            print(f"   Recommendation: {overall_assessment.get('recommendation', 'N/A')}")
            
            if 'empyrical_rating' in overall_assessment:
                print(f"   Empyrical Rating: {overall_assessment['empyrical_rating']}")
            
            strengths = overall_assessment.get('key_strengths', [])
            if strengths:
                print(f"   Combined Strengths: {', '.join(strengths[:3])}")  # Show top 3
            
            weaknesses = overall_assessment.get('key_weaknesses', [])
            if weaknesses:
                print(f"   Areas for Improvement: {', '.join(weaknesses[:3])}")  # Show top 3
        
        return True
        
    except Exception as e:
        print(f"❌ Performance analyzer test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def test_risk_modeling_agent_integration():
    """Test risk modeling agent with empyrical integration."""
    print("\n\n🤖 Testing Risk Modeling Agent Integration")
    print("=" * 50)
    
    try:
        from agents.risk_management.risk_modeling_agent import RiskModelingAgent
        
        # Initialize risk modeling agent
        risk_agent = RiskModelingAgent()
        
        # Create mock inputs
        mock_recommendations = [
            {
                "symbol": "AAPL",
                "action": "BUY",
                "position_size": 0.30,
                "confidence": 0.85,
                "expected_return": 0.12
            },
            {
                "symbol": "GOOGL", 
                "action": "BUY",
                "position_size": 0.25,
                "confidence": 0.75,
                "expected_return": 0.10
            },
            {
                "symbol": "MSFT",
                "action": "BUY", 
                "position_size": 0.20,
                "confidence": 0.80,
                "expected_return": 0.11
            }
        ]
        
        mock_portfolio_data = {
            "current_value": 100000,
            "cash": 25000,
            "positions": {
                "AAPL": {"quantity": 100, "value": 15000},
                "GOOGL": {"quantity": 50, "value": 7500}
            }
        }
        
        mock_market_data = {
            "current_date": datetime.now(),
            "market_volatility": 0.18,
            "risk_free_rate": 0.02
        }
        
        mock_risk_constraints = {
            "max_portfolio_var": 0.05,
            "max_single_position": 0.40,
            "max_sector_exposure": 0.60,
            "max_drawdown": 0.15
        }
        
        inputs = {
            "recommendations": mock_recommendations,
            "portfolio_data": mock_portfolio_data,
            "market_data": mock_market_data,
            "risk_constraints": mock_risk_constraints
        }
        
        print(f"📊 Running Risk Modeling Analysis:")
        print(f"   Recommendations: {len(mock_recommendations)}")
        print(f"   Portfolio Value: ${mock_portfolio_data['current_value']:,}")
        print(f"   Empyrical Available: {risk_agent.empyrical_available}")
        
        # Execute risk modeling
        result = await risk_agent.execute(inputs)
        
        print(f"\n✅ RISK MODELING RESULTS:")
        print(f"   Agent: {result.agent_name}")
        print(f"   Risk Models Used: {result.metadata['risk_models_used']}")
        print(f"   Consensus Achieved: {result.metadata['consensus_achieved']}")
        print(f"   Recommendations Processed: {result.metadata['recommendations_processed']}")
        print(f"   Recommendations Modified: {result.metadata['recommendations_modified']}")
        
        # Risk estimates
        risk_estimates = result.data.get('risk_estimates', {})
        print(f"\n📉 RISK ESTIMATES:")
        for model_name, estimates in risk_estimates.items():
            if isinstance(estimates, dict):
                print(f"   {model_name.replace('_', ' ').title()}:")
                print(f"      VaR (95%): {estimates.get('var_95', 0):.2%}")
                print(f"      CVaR (95%): {estimates.get('cvar_95', 0):.2%}")
                print(f"      Volatility: {estimates.get('volatility', 0):.2%}")
                print(f"      Max Drawdown: {estimates.get('max_drawdown', 0):.2%}")
                
                # Show empyrical-specific metrics
                if model_name == 'empyrical_analysis' and 'empyrical_analysis' in estimates:
                    emp_analysis = estimates['empyrical_analysis']
                    assessment = emp_analysis.get('overall_assessment', {})
                    if assessment:
                        print(f"      Empyrical Rating: {assessment.get('rating', 'N/A')}")
                        print(f"      Risk Level: {assessment.get('risk_level', 'N/A')}")
        
        # Consensus validation
        consensus = result.data.get('consensus_validation', {})
        if consensus:
            print(f"\n🤝 CONSENSUS VALIDATION:")
            print(f"   Valid: {consensus.get('is_valid', False)}")
            print(f"   Confidence: {consensus.get('consensus_confidence', 0):.1%}")
            print(f"   Models in Agreement: {consensus.get('validation_details', {}).get('total_models', 0)}")
            
            outliers = consensus.get('outlier_models', [])
            if outliers:
                print(f"   Outlier Models: {', '.join(outliers)}")
        
        # Risk-validated recommendations
        validated_recs = result.data.get('risk_validated_recommendations', [])
        if validated_recs:
            print(f"\n✅ RISK-VALIDATED RECOMMENDATIONS:")
            for i, rec in enumerate(validated_recs[:3], 1):  # Show first 3
                print(f"   {i}. {rec.get('symbol', 'N/A')} - {rec.get('action', 'N/A')}")
                print(f"      Position Size: {rec.get('position_size', 0):.1%}")
                print(f"      Risk Score: {rec.get('risk_score', 0):.2f}")
                print(f"      Risk-Adjusted: {rec.get('risk_adjusted', False)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Risk modeling agent test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main test function."""
    print("🚀 Empyrical Integration Test Suite")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 3
    
    # Test 1: Basic empyrical engine
    if test_empyrical_risk_engine():
        tests_passed += 1
        print("✅ Test 1 Passed: Empyrical Risk Engine")
    else:
        print("❌ Test 1 Failed: Empyrical Risk Engine")
    
    # Test 2: Enhanced performance analyzer
    if test_performance_analyzer_enhancement():
        tests_passed += 1
        print("✅ Test 2 Passed: Enhanced Performance Analyzer")
    else:
        print("❌ Test 2 Failed: Enhanced Performance Analyzer")
    
    # Test 3: Risk modeling agent integration
    if await test_risk_modeling_agent_integration():
        tests_passed += 1
        print("✅ Test 3 Passed: Risk Modeling Agent Integration")
    else:
        print("❌ Test 3 Failed: Risk Modeling Agent Integration")
    
    # Summary
    print(f"\n🏁 TEST SUMMARY")
    print("=" * 20)
    print(f"Tests Passed: {tests_passed}/{total_tests}")
    print(f"Success Rate: {tests_passed/total_tests:.1%}")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Empyrical integration is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    print("\n💡 Empyrical Integration Benefits:")
    print("   ✅ Professional-grade risk metrics (used by Quantopian)")
    print("   ✅ Comprehensive performance analysis (20+ metrics)")
    print("   ✅ Institutional-quality risk assessment")
    print("   ✅ Rolling metrics for time-varying analysis")
    print("   ✅ Factor attribution and tail risk analysis")
    print("   ✅ Automated performance scoring and rating")


if __name__ == "__main__":
    asyncio.run(main())