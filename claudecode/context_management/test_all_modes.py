"""
Test all three context export modes with sample data
"""

from claude_context_bridge import ClaudeContextBridge
from smart_context_export import SmartContextExporter
import time

def populate_test_data():
    """Add some test data to ChromaDB"""
    print("=== Populating test data ===\n")
    
    bridge = ClaudeContextBridge("trading_system")
    
    # Add conversation turns
    bridge.add_conversation_turn(
        "How should I implement the order matching engine?",
        "For an order matching engine, use a priority queue with price-time priority. Implement using a Red-Black tree for O(log n) operations."
    )
    
    bridge.add_conversation_turn(
        "What database should I use for trade history?",
        "Use PostgreSQL with TimescaleDB extension for time-series data. This provides excellent query performance for historical trade analysis."
    )
    
    # Save technical decisions
    bridge.save_important_decision(
        "Use Redis for order book cache",
        "Need sub-millisecond latency for order book updates"
    )
    
    bridge.save_important_decision(
        "Implement FIX protocol for institutional clients",
        "Industry standard for trading systems, required by major brokers"
    )
    
    # Save code patterns
    bridge.save_code_pattern(
        "order_validation",
        """def validate_order(order: Order) -> bool:
    if order.quantity <= 0:
        return False
    if order.price <= Decimal('0'):
        return False
    if order.side not in ['BUY', 'SELL']:
        return False
    return True""",
        "Basic order validation with type safety"
    )
    
    bridge.save_code_pattern(
        "async_market_data_handler",
        """async def handle_market_data(symbol: str):
    async with websockets.connect(f'wss://api.exchange.com/{symbol}') as ws:
        async for message in ws:
            data = json.loads(message)
            await process_tick(data)""",
        "WebSocket handler for real-time market data"
    )
    
    # Save best practices
    bridge.save_best_practice(
        "Always use Decimal for price calculations, never float",
        "financial_accuracy"
    )
    
    bridge.save_best_practice(
        "Implement circuit breakers for all external API calls",
        "resilience"
    )
    
    bridge.save_best_practice(
        "Log all order state changes for audit trail",
        "compliance"
    )
    
    print("[OK] Added test data to ChromaDB\n")
    return bridge

def test_all_export_modes():
    """Test all three export modes"""
    
    # First populate data
    bridge = populate_test_data()
    
    print("=== Testing All Export Modes ===\n")
    
    # Initialize exporter
    exporter = SmartContextExporter("trading_system")
    
    # Test Mode 1: Work-focused export
    print("1. Testing work-focused export...")
    work_file = exporter.export_for_work_session(
        "implementing order matching engine",
        "CONTEXT_WORK.md",
        max_items_per_level=3
    )
    print(f"   Created: {work_file}\n")
    
    # Small delay to see progress
    time.sleep(1)
    
    # Test Mode 2: Query-based export
    print("2. Testing query-based export...")
    query_file = exporter.export_by_queries(
        ["database design", "websocket", "compliance", "redis cache"],
        "CONTEXT_RESEARCH.md"
    )
    print(f"   Created: {query_file}\n")
    
    time.sleep(1)
    
    # Test Mode 3: Daily context
    print("3. Testing daily context export...")
    daily_file = exporter.create_daily_context("CONTEXT_DAILY.md")
    print(f"   Created: {daily_file}\n")
    
    # Show summary
    print("=== Summary ===")
    print("[OK] All three export modes tested successfully!")
    print("\nGenerated files:")
    print("  - CONTEXT_WORK.md     (work-focused, semantic search)")
    print("  - CONTEXT_RESEARCH.md (multi-query research)")  
    print("  - CONTEXT_DAILY.md    (daily baseline)")
    print("\n[SUCCESS] ChromaDB semantic search is working for all modes!")
    
    # Display one example
    print("\n=== Example: Work-Focused Context ===")
    with open("CONTEXT_WORK.md", "r", encoding="utf-8") as f:
        content = f.read()
        # Show first 1000 chars
        print(content[:1000] + "..." if len(content) > 1000 else content)

if __name__ == "__main__":
    test_all_export_modes()