"""
Test script for QuantLib integration with Options Strategy Agent.
"""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add the trading system to Python path
sys.path.append(str(Path(__file__).parent.parent))

from core.pricing.quantlib_engine import QuantLibPricingEngine, OptionType, PricingModel, OptionsStrategyPricer
from agents.strategies.options.options_agent import OptionsAgent


def test_quantlib_pricing_engine():
    """Test basic QuantLib pricing functionality."""
    print("🧪 Testing QuantLib Pricing Engine")
    print("=" * 40)
    
    try:
        # Initialize QuantLib engine
        engine = QuantLibPricingEngine()
        
        # Test vanilla option pricing
        spot_price = 100.0
        strike = 105.0
        risk_free_rate = 0.05
        dividend_yield = 0.0
        volatility = 0.20
        maturity_date = datetime.now() + timedelta(days=30)
        
        print(f"📊 Pricing Call Option:")
        print(f"   Spot: ${spot_price:.2f}")
        print(f"   Strike: ${strike:.2f}")
        print(f"   Volatility: {volatility:.1%}")
        print(f"   Time to Expiry: 30 days")
        
        # Price call option
        call_result = engine.price_vanilla_option(
            spot_price=spot_price,
            strike=strike,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
            volatility=volatility,
            maturity_date=maturity_date,
            option_type=OptionType.CALL,
            model=PricingModel.BLACK_SCHOLES
        )
        
        print(f"\n✅ Call Option Results:")
        print(f"   Price: ${call_result['price']:.4f}")
        print(f"   Delta: {call_result['delta']:.4f}")
        print(f"   Gamma: {call_result['gamma']:.4f}")
        print(f"   Theta: ${call_result['theta']:.4f}")
        print(f"   Vega: {call_result['vega']:.4f}")
        print(f"   Rho: {call_result['rho']:.4f}")
        
        # Price put option
        put_result = engine.price_vanilla_option(
            spot_price=spot_price,
            strike=strike,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
            volatility=volatility,
            maturity_date=maturity_date,
            option_type=OptionType.PUT,
            model=PricingModel.BLACK_SCHOLES
        )
        
        print(f"\n✅ Put Option Results:")
        print(f"   Price: ${put_result['price']:.4f}")
        print(f"   Delta: {put_result['delta']:.4f}")
        print(f"   Gamma: {put_result['gamma']:.4f}")
        print(f"   Theta: ${put_result['theta']:.4f}")
        print(f"   Vega: {put_result['vega']:.4f}")
        print(f"   Rho: {put_result['rho']:.4f}")
        
        # Test American option pricing
        print(f"\n🇺🇸 Testing American Option Pricing:")
        american_result = engine.price_american_option(
            spot_price=spot_price,
            strike=strike,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
            volatility=volatility,
            maturity_date=maturity_date,
            option_type=OptionType.PUT
        )
        
        print(f"   American Put Price: ${american_result['price']:.4f}")
        print(f"   vs European Put: ${put_result['price']:.4f}")
        print(f"   Early Exercise Premium: ${american_result['price'] - put_result['price']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ QuantLib pricing test failed: {str(e)}")
        return False


def test_options_strategy_pricing():
    """Test complex options strategy pricing."""
    print("\n\n🔧 Testing Options Strategy Pricing")
    print("=" * 40)
    
    try:
        # Initialize engines
        engine = QuantLibPricingEngine()
        strategy_pricer = OptionsStrategyPricer(engine)
        
        # Test iron condor pricing
        spot_price = 100.0
        strikes = (90.0, 95.0, 105.0, 110.0)  # put_low, put_high, call_low, call_high
        risk_free_rate = 0.05
        dividend_yield = 0.0
        volatility = 0.25
        maturity_date = datetime.now() + timedelta(days=45)
        
        print(f"🦅 Iron Condor Strategy:")
        print(f"   Underlying: ${spot_price:.2f}")
        print(f"   Strikes: {strikes}")
        print(f"   Volatility: {volatility:.1%}")
        print(f"   Days to Expiry: 45")
        
        iron_condor_result = strategy_pricer.price_iron_condor(
            spot_price=spot_price,
            strikes=strikes,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
            volatility=volatility,
            maturity_date=maturity_date
        )
        
        print(f"\n✅ Iron Condor Results:")
        print(f"   Strategy Price: ${iron_condor_result['strategy_price']:.4f}")
        print(f"   Max Profit: ${iron_condor_result['max_profit']:.4f}")
        print(f"   Max Loss: ${iron_condor_result['max_loss']:.4f}")
        print(f"   Lower Breakeven: ${iron_condor_result['breakeven_lower']:.4f}")
        print(f"   Upper Breakeven: ${iron_condor_result['breakeven_upper']:.4f}")
        
        # Display Greeks
        greeks = iron_condor_result['greeks']
        print(f"\n📊 Strategy Greeks:")
        print(f"   Delta: {greeks['delta']:.4f}")
        print(f"   Gamma: {greeks['gamma']:.4f}")
        print(f"   Theta: ${greeks['theta']:.4f}")
        print(f"   Vega: {greeks['vega']:.4f}")
        
        # Test butterfly spread
        print(f"\n🦋 Butterfly Spread Strategy:")
        butterfly_result = strategy_pricer.price_butterfly_spread(
            spot_price=spot_price,
            center_strike=100.0,
            wing_width=5.0,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
            volatility=volatility,
            maturity_date=maturity_date,
            option_type=OptionType.CALL
        )
        
        print(f"   Strategy Price: ${butterfly_result['strategy_price']:.4f}")
        print(f"   Max Profit: ${butterfly_result['max_profit']:.4f}")
        print(f"   Max Loss: ${butterfly_result['max_loss']:.4f}")
        print(f"   Lower Breakeven: ${butterfly_result['breakeven_lower']:.4f}")
        print(f"   Upper Breakeven: ${butterfly_result['breakeven_upper']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Strategy pricing test failed: {str(e)}")
        return False


async def test_options_agent_integration():
    """Test Options Agent with QuantLib integration."""
    print("\n\n🤖 Testing Options Agent Integration")
    print("=" * 40)
    
    try:
        # Initialize Options Agent
        options_agent = OptionsAgent()
        
        # Create mock options chain data
        options_chain = [
            {
                "underlying_symbol": "AAPL",
                "underlying_price": 150.0,
                "strike": 145.0,
                "option_type": "call",
                "expiration_date": "2024-02-16",
                "implied_volatility": 0.25,
                "bid_price": 7.50,
                "ask_price": 7.70,
                "volume": 1000,
                "open_interest": 5000,
                "risk_free_rate": 0.05
            },
            {
                "underlying_symbol": "AAPL",
                "underlying_price": 150.0,
                "strike": 155.0,
                "option_type": "call",
                "expiration_date": "2024-02-16",
                "implied_volatility": 0.28,
                "bid_price": 4.20,
                "ask_price": 4.40,
                "volume": 800,
                "open_interest": 3000,
                "risk_free_rate": 0.05
            },
            {
                "underlying_symbol": "AAPL",
                "underlying_price": 150.0,
                "strike": 145.0,
                "option_type": "put",
                "expiration_date": "2024-02-16",
                "implied_volatility": 0.26,
                "bid_price": 2.80,
                "ask_price": 3.00,
                "volume": 600,
                "open_interest": 2000,
                "risk_free_rate": 0.05
            },
            {
                "underlying_symbol": "AAPL",
                "underlying_price": 150.0,
                "strike": 155.0,
                "option_type": "put",
                "expiration_date": "2024-02-16",
                "implied_volatility": 0.24,
                "bid_price": 6.50,
                "ask_price": 6.80,
                "volume": 400,
                "open_interest": 1500,
                "risk_free_rate": 0.05
            }
        ]
        
        # Market view and risk constraints
        market_view = {
            "direction": "neutral",
            "volatility_expectation": "decreasing",
            "confidence": 0.7
        }
        
        risk_constraints = {
            "max_loss_per_trade": 1000,
            "max_position_size": 10
        }
        
        # Execute options agent
        inputs = {
            "options_chain": options_chain,
            "symbols": ["AAPL"],
            "market_view": market_view,
            "risk_constraints": risk_constraints
        }
        
        print(f"📊 Running Options Agent Analysis:")
        print(f"   Symbol: AAPL")
        print(f"   Underlying Price: $150.00")
        print(f"   Market View: {market_view['direction']}")
        print(f"   Options in Chain: {len(options_chain)}")
        
        result = await options_agent.execute(inputs)
        
        print(f"\n✅ Options Agent Results:")
        print(f"   Agent: {result.agent_name}")
        print(f"   Symbols Analyzed: {result.metadata['symbols_analyzed']}")
        print(f"   Strategies Generated: {result.metadata['strategies_generated']}")
        print(f"   Recommendations: {result.metadata['recommendations_count']}")
        
        # Display recommendations
        recommendations = result.data.get("recommendations", [])
        for i, rec in enumerate(recommendations[:2], 1):  # Show first 2 recommendations
            print(f"\n📋 Recommendation {i}:")
            print(f"   Strategy: {rec['strategy_name']}")
            print(f"   Expected Profit: ${rec.get('expected_profit', 0):.2f}")
            print(f"   Maximum Loss: ${rec.get('maximum_loss', 0):.2f}")
            print(f"   Profit Probability: {rec.get('profit_probability', 0):.1%}")
            print(f"   Confidence: {rec.get('confidence', 0):.1%}")
            print(f"   Risk Level: {rec.get('risk_level', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Options agent integration test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main test function."""
    print("🚀 QuantLib Integration Test Suite")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 3
    
    # Test 1: Basic QuantLib pricing
    if test_quantlib_pricing_engine():
        tests_passed += 1
        print("✅ Test 1 Passed: QuantLib Pricing Engine")
    else:
        print("❌ Test 1 Failed: QuantLib Pricing Engine")
    
    # Test 2: Strategy pricing
    if test_options_strategy_pricing():
        tests_passed += 1
        print("✅ Test 2 Passed: Options Strategy Pricing")
    else:
        print("❌ Test 2 Failed: Options Strategy Pricing")
    
    # Test 3: Agent integration
    if await test_options_agent_integration():
        tests_passed += 1
        print("✅ Test 3 Passed: Options Agent Integration")
    else:
        print("❌ Test 3 Failed: Options Agent Integration")
    
    # Summary
    print(f"\n🏁 TEST SUMMARY")
    print("=" * 20)
    print(f"Tests Passed: {tests_passed}/{total_tests}")
    print(f"Success Rate: {tests_passed/total_tests:.1%}")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! QuantLib integration is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    print("\n💡 Next Steps:")
    print("- Run the backtest examples with QuantLib-enhanced options strategies")
    print("- Test with real market data")
    print("- Explore advanced pricing models (Heston, Monte Carlo)")


if __name__ == "__main__":
    asyncio.run(main())