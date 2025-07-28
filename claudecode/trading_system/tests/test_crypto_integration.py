"""
Test script for CCXT cryptocurrency integration.
"""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add the trading system to Python path
sys.path.append(str(Path(__file__).parent.parent))

from core.apis.market_data import MarketDataAPI
from core.base.config import config


async def test_crypto_integration():
    """Test cryptocurrency data integration."""
    print("🔄 Testing CCXT Cryptocurrency Integration")
    print("=" * 50)
    
    try:
        # Initialize market data API
        market_api = MarketDataAPI()
        print("✅ MarketDataAPI initialized successfully")
        
        # Test crypto symbol detection
        crypto_symbols = ["BTC/USDT", "ETH/USD", "BNB/BUSD"]
        traditional_symbols = ["AAPL", "GOOGL", "MSFT"]
        
        print("\n📊 Testing Symbol Detection:")
        for symbol in crypto_symbols + traditional_symbols:
            is_crypto = market_api.is_crypto_symbol(symbol)
            symbol_type = "Crypto" if is_crypto else "Traditional"
            print(f"  {symbol}: {symbol_type}")
        
        # Test crypto markets (without API keys - will fail gracefully)
        print("\n🏪 Testing Crypto Markets (Demo Mode):")
        try:
            markets = await market_api.get_crypto_markets("binance")
            print(f"  Found {len(markets)} markets on Binance")
            if markets:
                print(f"  Sample markets: {[m['symbol'] for m in markets[:5]]}")
        except Exception as e:
            print(f"  ⚠️  Market data unavailable (expected without API keys): {str(e)[:100]}...")
        
        # Test historical data (demo mode)
        print("\n📈 Testing Historical Data (Demo Mode):")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        test_symbol = "BTC/USDT"
        try:
            data = await market_api.get_crypto_data(
                test_symbol, start_date, end_date, "1d", "binance"
            )
            print(f"  ✅ Retrieved {len(data)} days of data for {test_symbol}")
            print(f"  Columns: {list(data.columns)}")
        except Exception as e:
            print(f"  ⚠️  Historical data unavailable (expected without API keys): {str(e)[:100]}...")
        
        # Test auto-detection
        print("\n🔍 Testing Auto-Detection:")
        for symbol in ["BTC/USDT", "AAPL"]:
            try:
                data = await market_api.get_data_auto_detect(
                    symbol, start_date, end_date, "1d"
                )
                print(f"  ✅ Auto-detected and retrieved data for {symbol}: {len(data)} records")
            except Exception as e:
                print(f"  ⚠️  Auto-detection failed for {symbol}: {str(e)[:100]}...")
        
        print("\n🎉 CCXT Integration Test Complete!")
        print("Note: Some tests may fail without API keys configured - this is expected")
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(test_crypto_integration())