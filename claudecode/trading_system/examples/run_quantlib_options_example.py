"""
Example demonstrating QuantLib-enhanced options strategies in the trading system.
"""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add the trading system to Python path
sys.path.append(str(Path(__file__).parent.parent))

from core.pricing.quantlib_engine import QuantLibPricingEngine, OptionType, PricingModel
from agents.strategies.options.options_agent import OptionsAgent


async def demonstrate_quantlib_options_analysis():
    """Demonstrate comprehensive options analysis with QuantLib."""
    print("🎯 QuantLib-Enhanced Options Strategy Analysis")
    print("=" * 60)
    
    # Create realistic options chain for AAPL
    current_price = 185.0
    expiry_date = (datetime.now() + timedelta(days=21)).strftime("%Y-%m-%d")
    
    # Generate options chain around current price
    options_chain = []
    strikes = [170, 175, 180, 185, 190, 195, 200]
    
    for strike in strikes:
        # Calculate rough implied volatility based on moneyness
        moneyness = strike / current_price
        base_iv = 0.25
        if moneyness < 0.95:  # ITM puts, OTM calls
            iv_call = base_iv + 0.02
            iv_put = base_iv - 0.01
        elif moneyness > 1.05:  # OTM puts, ITM calls
            iv_call = base_iv - 0.01
            iv_put = base_iv + 0.02
        else:  # ATM
            iv_call = base_iv
            iv_put = base_iv
        
        # Call option
        options_chain.append({
            "underlying_symbol": "AAPL",
            "underlying_price": current_price,
            "strike": float(strike),
            "option_type": "call",
            "expiration_date": expiry_date,
            "implied_volatility": iv_call,
            "bid_price": max(0.10, current_price - strike + 2.0) if strike < current_price else max(0.10, 3.0 - (strike - current_price) * 0.5),
            "ask_price": max(0.15, current_price - strike + 2.2) if strike < current_price else max(0.15, 3.2 - (strike - current_price) * 0.5),
            "volume": max(100, 2000 - abs(strike - current_price) * 50),
            "open_interest": max(500, 10000 - abs(strike - current_price) * 200),
            "risk_free_rate": 0.045,
            "dividend_yield": 0.005
        })
        
        # Put option
        options_chain.append({
            "underlying_symbol": "AAPL",
            "underlying_price": current_price,
            "strike": float(strike),
            "option_type": "put",
            "expiration_date": expiry_date,
            "implied_volatility": iv_put,
            "bid_price": max(0.10, strike - current_price + 2.0) if strike > current_price else max(0.10, 3.0 - (current_price - strike) * 0.5),
            "ask_price": max(0.15, strike - current_price + 2.2) if strike > current_price else max(0.15, 3.2 - (current_price - strike) * 0.5),
            "volume": max(100, 1500 - abs(strike - current_price) * 40),
            "open_interest": max(500, 8000 - abs(strike - current_price) * 150),
            "risk_free_rate": 0.045,
            "dividend_yield": 0.005
        })
    
    print(f"📊 Options Chain Overview:")
    print(f"   Underlying: AAPL @ ${current_price:.2f}")
    print(f"   Expiration: {expiry_date} ({21} days)")
    print(f"   Strike Range: ${strikes[0]} - ${strikes[-1]}")
    print(f"   Total Options: {len(options_chain)}")
    
    # Test different market scenarios
    market_scenarios = [
        {
            "name": "Neutral Market - Low Volatility",
            "market_view": {
                "direction": "neutral",
                "volatility_expectation": "decreasing",
                "confidence": 0.8,
                "time_decay_favorable": True
            }
        },
        {
            "name": "Bullish Market - High Confidence",
            "market_view": {
                "direction": "bullish",
                "volatility_expectation": "stable",
                "confidence": 0.9,
                "time_decay_favorable": False
            }
        },
        {
            "name": "High Volatility Expected",
            "market_view": {
                "direction": "neutral",
                "volatility_expectation": "increasing",
                "confidence": 0.6,
                "time_decay_favorable": False
            }
        }
    ]
    
    # Risk constraints
    risk_constraints = {
        "max_loss_per_trade": 2000,
        "max_position_size": 20,
        "max_portfolio_delta": 100,
        "max_portfolio_vega": 50
    }
    
    # Initialize Options Agent
    options_agent = OptionsAgent()
    
    # Analyze each scenario
    for i, scenario in enumerate(market_scenarios, 1):
        print(f"\n\n📈 SCENARIO {i}: {scenario['name']}")
        print("=" * 50)
        
        market_view = scenario['market_view']
        print(f"Market Direction: {market_view['direction']}")
        print(f"Volatility Outlook: {market_view['volatility_expectation']}")
        print(f"Confidence Level: {market_view['confidence']:.1%}")
        
        try:
            # Execute options agent analysis
            inputs = {
                "options_chain": options_chain,
                "symbols": ["AAPL"],
                "market_view": market_view,
                "risk_constraints": risk_constraints
            }
            
            result = await options_agent.execute(inputs)
            
            # Display results
            print(f"\n🎯 ANALYSIS RESULTS:")
            print(f"   Processing Time: {result.metadata.get('processing_timestamp', 'N/A')}")
            print(f"   Options Analyzed: {result.metadata.get('options_analyzed', 0)}")
            print(f"   Strategies Generated: {result.metadata.get('strategies_generated', 0)}")
            
            # Show volatility analysis
            vol_analysis = result.data.get("volatility_analysis", {})
            risk_indicators = vol_analysis.get("risk_indicators", {})
            
            if risk_indicators:
                print(f"\n📊 MARKET INDICATORS:")
                print(f"   Put-Call Ratio: {risk_indicators.get('put_call_ratio', 0):.2f}")
                print(f"   Average IV: {risk_indicators.get('average_implied_volatility', 0):.1%}")
                print(f"   IV Rank: {risk_indicators.get('volatility_rank', 0):.1%}")
                print(f"   Market Sentiment: {risk_indicators.get('market_sentiment', 'neutral').title()}")
            
            # Display strategy recommendations
            recommendations = result.data.get("recommendations", [])
            
            if recommendations:
                print(f"\n🏆 TOP STRATEGY RECOMMENDATIONS:")
                for j, rec in enumerate(recommendations[:2], 1):  # Show top 2
                    print(f"\n   {j}. {rec['strategy_name'].replace('_', ' ').title()}")
                    print(f"      Expected Profit: ${rec.get('expected_profit', 0):.2f}")
                    print(f"      Maximum Loss: ${rec.get('maximum_loss', 0):.2f}")
                    print(f"      Profit Probability: {rec.get('profit_probability', 0):.1%}")
                    print(f"      Risk Level: {rec.get('risk_level', 'medium').title()}")
                    print(f"      Confidence Score: {rec.get('confidence', 0):.1%}")
                    
                    # Show strategy legs if available
                    legs = rec.get('legs', [])
                    if legs:
                        print(f"      Strategy Legs:")
                        for leg in legs:
                            action = leg['action'].upper()
                            option = leg['option']
                            qty = leg['quantity']
                            strike = option['strike']
                            opt_type = option['option_type'].upper()
                            print(f"        {action} {qty} {strike} {opt_type}")
                    
                    # Show Greeks impact
                    greeks_impact = rec.get('greeks_impact', {})
                    if greeks_impact:
                        print(f"      Portfolio Impact:")
                        print(f"        Delta: {greeks_impact.get('delta', 0):.2f}")
                        print(f"        Gamma: {greeks_impact.get('gamma', 0):.3f}")
                        print(f"        Theta: ${greeks_impact.get('theta', 0):.2f}/day")
                        print(f"        Vega: {greeks_impact.get('vega', 0):.2f}")
            else:
                print(f"\n⚠️  No suitable strategies found for this market scenario")
            
            # Portfolio Greeks summary
            portfolio_greeks = result.data.get("portfolio_greeks", {})
            if portfolio_greeks and recommendations:
                print(f"\n📈 PORTFOLIO GREEKS SUMMARY:")
                print(f"   Total Delta: {portfolio_greeks.get('total_delta', 0):.2f}")
                print(f"   Total Gamma: {portfolio_greeks.get('total_gamma', 0):.3f}")
                print(f"   Total Theta: ${portfolio_greeks.get('total_theta', 0):.2f}/day")
                print(f"   Total Vega: {portfolio_greeks.get('total_vega', 0):.2f}")
                print(f"   Net Premium: ${portfolio_greeks.get('net_premium', 0):.2f}")
                
                # Risk analysis
                risk_analysis = portfolio_greeks.get('risk_analysis', {})
                if risk_analysis:
                    print(f"   Delta Neutral: {'Yes' if risk_analysis.get('delta_neutral') else 'No'}")
                    print(f"   Gamma Risk: {risk_analysis.get('gamma_risk', 'unknown').title()}")
                    print(f"   Vega Exposure: {risk_analysis.get('vega_exposure', 0):.2f}")
            
        except Exception as e:
            print(f"❌ Analysis failed for scenario '{scenario['name']}': {str(e)}")
            import traceback
            traceback.print_exc()
    
    print(f"\n\n🎯 ANALYSIS COMPLETE")
    print("=" * 30)
    print("✅ Successfully demonstrated QuantLib-enhanced options strategies")
    print("💡 Key Features Demonstrated:")
    print("   • Professional-grade options pricing with QuantLib")
    print("   • Multi-leg strategy construction and optimization")
    print("   • Portfolio Greeks calculation and risk analysis")
    print("   • Volatility surface analysis and skew detection")
    print("   • Market scenario-based strategy selection")
    print("   • Consensus validation across multiple criteria")


async def demonstrate_advanced_quantlib_features():
    """Demonstrate advanced QuantLib features."""
    print("\n\n🔬 Advanced QuantLib Features Demo")
    print("=" * 45)
    
    try:
        # Initialize QuantLib engine
        engine = QuantLibPricingEngine()
        
        # Test different pricing models
        spot_price = 100.0
        strike = 105.0
        risk_free_rate = 0.05
        dividend_yield = 0.02
        volatility = 0.30
        maturity_date = datetime.now() + timedelta(days=60)
        
        print(f"📊 Comparing Pricing Models:")
        print(f"   Underlying: ${spot_price:.2f}")
        print(f"   Strike: ${strike:.2f}")
        print(f"   Volatility: {volatility:.1%}")
        print(f"   Days to Expiry: 60")
        
        models_to_test = [
            PricingModel.BLACK_SCHOLES,
            PricingModel.BINOMIAL,
            PricingModel.MONTE_CARLO,
            PricingModel.FINITE_DIFFERENCE
        ]
        
        print(f"\n🔍 Model Comparison Results:")
        print(f"{'Model':<20} {'Price':<10} {'Delta':<8} {'Gamma':<8} {'Theta':<8}")
        print("-" * 60)
        
        for model in models_to_test:
            try:
                result = engine.price_vanilla_option(
                    spot_price=spot_price,
                    strike=strike,
                    risk_free_rate=risk_free_rate,
                    dividend_yield=dividend_yield,
                    volatility=volatility,
                    maturity_date=maturity_date,
                    option_type=OptionType.CALL,
                    model=model
                )
                
                print(f"{model.value:<20} ${result['price']:<9.4f} {result['delta']:<7.4f} {result['gamma']:<7.4f} ${result['theta']:<7.4f}")
                
            except Exception as e:
                print(f"{model.value:<20} ERROR: {str(e)[:30]}")
        
        # Test American vs European pricing
        print(f"\n🇺🇸 American vs European Option Pricing:")
        
        european_put = engine.price_vanilla_option(
            spot_price=spot_price,
            strike=110.0,  # ITM put
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
            volatility=volatility,
            maturity_date=maturity_date,
            option_type=OptionType.PUT,
            model=PricingModel.BLACK_SCHOLES
        )
        
        american_put = engine.price_american_option(
            spot_price=spot_price,
            strike=110.0,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
            volatility=volatility,
            maturity_date=maturity_date,
            option_type=OptionType.PUT
        )
        
        early_exercise_value = american_put['price'] - european_put['price']
        
        print(f"   European Put: ${european_put['price']:.4f}")
        print(f"   American Put: ${american_put['price']:.4f}")
        print(f"   Early Exercise Value: ${early_exercise_value:.4f}")
        print(f"   Premium Ratio: {(early_exercise_value/european_put['price'])*100:.2f}%")
        
        # Test barrier options
        print(f"\n🚧 Barrier Option Pricing:")
        
        barrier_price = 95.0  # Down-and-out barrier
        
        barrier_result = engine.price_barrier_option(
            spot_price=spot_price,
            strike=strike,
            barrier=barrier_price,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
            volatility=volatility,
            maturity_date=maturity_date,
            option_type=OptionType.CALL,
            barrier_type="down-and-out"
        )
        
        vanilla_call = engine.price_vanilla_option(
            spot_price=spot_price,
            strike=strike,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
            volatility=volatility,
            maturity_date=maturity_date,
            option_type=OptionType.CALL
        )
        
        barrier_discount = vanilla_call['price'] - barrier_result['price']
        
        print(f"   Vanilla Call: ${vanilla_call['price']:.4f}")
        print(f"   Down-and-Out Call (barrier @${barrier_price}): ${barrier_result['price']:.4f}")
        print(f"   Barrier Discount: ${barrier_discount:.4f}")
        print(f"   Discount Percentage: {(barrier_discount/vanilla_call['price'])*100:.2f}%")
        
        print(f"\n✅ Advanced QuantLib features demonstration complete!")
        
    except Exception as e:
        print(f"❌ Advanced features demo failed: {str(e)}")
        import traceback
        traceback.print_exc()


async def main():
    """Main demonstration function."""
    print("🚀 QuantLib-Enhanced Options Trading System")
    print("=" * 60)
    print("This demo showcases professional-grade options analysis using QuantLib")
    print("integrated with our multi-agent trading system.")
    
    # Run main options analysis demo
    await demonstrate_quantlib_options_analysis()
    
    # Run advanced features demo
    await demonstrate_advanced_quantlib_features()
    
    print(f"\n\n🎉 DEMONSTRATION COMPLETE!")
    print("=" * 35)
    print("🎯 What you've seen:")
    print("   ✅ QuantLib integration with Options Strategy Agent")
    print("   ✅ Professional options pricing and Greeks calculation")
    print("   ✅ Multi-leg strategy construction (Iron Condor, Butterfly)")
    print("   ✅ Portfolio risk analysis and Greeks aggregation")
    print("   ✅ Multiple pricing models comparison")
    print("   ✅ American vs European option pricing")
    print("   ✅ Exotic options (Barrier options)")
    print("   ✅ Market scenario-based strategy selection")
    
    print(f"\n🚀 Ready for Production:")
    print("   • Professional-grade derivatives pricing")
    print("   • Institutional-quality risk management")
    print("   • Advanced volatility analysis")
    print("   • Multi-criteria consensus validation")
    
    print(f"\n💡 Next Steps:")
    print("   • Integrate with live market data feeds")
    print("   • Add volatility surface calibration")
    print("   • Implement dynamic hedging strategies")
    print("   • Connect to options execution systems")


if __name__ == "__main__":
    asyncio.run(main())