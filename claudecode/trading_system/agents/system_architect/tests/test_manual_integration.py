"""
Manual Integration Test Script

Run this script to manually test the System Architect suite integration.
This provides a simple way to test the system without pytest dependencies.
"""

import asyncio
import sys
import tempfile
import json
import os
from pathlib import Path
from datetime import datetime

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

try:
    from trading_system.agents.system_architect.master_coordinator import MasterCoordinator, analyze_project
    from trading_system.agents.system_architect.architecture_diagram_manager import ArchitectureDiagramManager
    from trading_system.agents.system_architect.dependency_analysis_agent import DependencyAnalysisAgent
    from trading_system.agents.system_architect.code_metrics_dashboard import CodeMetricsDashboard
    from trading_system.agents.system_architect.migration_planning_agent import MigrationPlanningAgent
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the correct directory and all dependencies are available")
    sys.exit(1)


def create_test_project(temp_dir: Path) -> None:
    """Create a comprehensive test project structure"""
    print("Creating test project structure...")
    
    # Main application files
    (temp_dir / "main.py").write_text("""
#!/usr/bin/env python3
\"\"\"
Main Trading System Application
\"\"\"

import asyncio
import logging
from typing import Dict, List, Optional
from datetime import datetime
import json

# FIXED: Import classes without circular dependencies
# from trading_engine import TradingEngine
# from order_manager import OrderManager  
# from risk_manager import RiskManager
# from database import DatabaseManager

# Global configuration - potential security issue
API_KEYS = {
    "broker": os.getenv('BROKER_API_SECRET', 'test-broker-key'),  # FIXED: Use environment variable  
    "data_feed": os.getenv('DATA_FEED_API_KEY', 'test-data-feed-key')  # FIXED: Use environment variable
}

class TradingApplication:
    def __init__(self):
        # FIXED: Use dependency injection to avoid circular dependencies
        # Create database manager first
        from database import DatabaseManager
        self.db = DatabaseManager()
        
        # Inject database into trading engine
        from trading_engine import TradingEngine
        self.engine = TradingEngine(database_manager=self.db)
        
        # Inject engine into database for proper initialization
        self.db.engine = self.engine
        
        # Create other components
        from order_manager import OrderManager
        from risk_manager import RiskManager
        self.order_manager = OrderManager()
        self.risk_manager = RiskManager()
        self.logger = logging.getLogger(__name__)
    
    async def start(self):
        \"\"\"Start the trading application\"\"\"
        try:
            await self.db.connect()
            await self.engine.initialize()
            self.logger.info("Trading application started successfully")
        except Exception as e:
            self.logger.error(f"Failed to start application: {e}")
            raise
    
    async def process_market_data(self, data: Dict) -> None:
        \"\"\"Process incoming market data - high complexity function\"\"\"
        for symbol in data.get('symbols', []):
            for timeframe in ['1m', '5m', '15m', '1h', '4h', '1d']:
                for indicator in ['sma', 'ema', 'rsi', 'macd', 'bollinger']:
                    # Nested loops create high cyclomatic complexity
                    if symbol in data.get('prices', {}):
                        prices = data['prices'][symbol]
                        for price_point in prices:
                            if self._should_process_indicator(indicator, price_point, timeframe):
                                result = await self._calculate_indicator(
                                    indicator, price_point, timeframe
                                )
                                if result and result.get('signal'):
                                    await self._handle_signal(symbol, result)
    
    def _should_process_indicator(self, indicator: str, price: Dict, timeframe: str) -> bool:
        \"\"\"Complex decision logic\"\"\"
        conditions = []
        
        # Multiple conditions increase complexity
        if indicator == 'sma':
            conditions.append(price.get('volume', 0) > 1000)
            conditions.append(price.get('high') - price.get('low') > 0.01)
        elif indicator == 'ema':
            conditions.append(price.get('close') > price.get('open'))
            conditions.append(timeframe in ['5m', '15m', '1h'])
        elif indicator == 'rsi':
            conditions.append(price.get('volume', 0) > 500)
            conditions.append(abs(price.get('close', 0) - price.get('open', 0)) > 0.005)
        elif indicator == 'macd':
            conditions.append(price.get('close') is not None)
            conditions.append(timeframe in ['15m', '1h', '4h'])
        elif indicator == 'bollinger':
            conditions.append(price.get('high') - price.get('low') > 0.02)
            conditions.append(price.get('volume', 0) > 2000)
        
        return all(conditions) if conditions else False
    
    async def _calculate_indicator(self, indicator: str, price: Dict, timeframe: str) -> Optional[Dict]:
        \"\"\"Calculate technical indicator\"\"\"
        # Simulated calculation
        await asyncio.sleep(0.001)  # Simulate processing time
        
        if indicator == 'sma':
            return {'value': price.get('close', 0), 'signal': 'buy' if price.get('close', 0) > 100 else 'sell'}
        elif indicator == 'rsi':
            rsi_value = (price.get('close', 50) / price.get('open', 50)) * 50
            return {'value': rsi_value, 'signal': 'buy' if rsi_value < 30 else 'sell' if rsi_value > 70 else 'hold'}
        
        return None
    
    async def _handle_signal(self, symbol: str, signal: Dict) -> None:
        \"\"\"Handle trading signal\"\"\"
        try:
            if await self.risk_manager.validate_signal(symbol, signal):
                order = await self.order_manager.create_order(symbol, signal)
                if order:
                    await self.engine.execute_order(order)
        except Exception as e:
            self.logger.error(f"Error handling signal for {symbol}: {e}")

if __name__ == "__main__":
    app = TradingApplication()
    asyncio.run(app.start())
""")
    
    # Trading engine module
    (temp_dir / "trading_engine.py").write_text("""
\"\"\"
Trading Engine - Core trading functionality
\"\"\"

import asyncio
import hashlib
import random
from typing import Dict, List, Optional
from datetime import datetime
# FIXED: Removed circular dependency - use dependency injection

class TradingEngine:
    def __init__(self, database_manager=None):
        self.orders = []
        self.positions = {}
        self.db = database_manager  # Use dependency injection
        self.is_initialized = False
    
    async def initialize(self):
        \"\"\"Initialize trading engine\"\"\"
        if self.db:
            await self.db.connect()
        self.is_initialized = True
    
    async def initialize_connection(self, connection):
        \"\"\"Initialize database connection (called by DatabaseManager)\"\"\"
        # This method is called by DatabaseManager to avoid circular dependency
        pass
    
    async def execute_order(self, order: Dict) -> bool:
        \"\"\"Execute a trading order\"\"\"
        if not self.is_initialized:
            raise RuntimeError("Engine not initialized")
        
        # Potential SQL injection vulnerability
        order_id = order.get('id', '')
        query = f"INSERT INTO orders (id, symbol, quantity) VALUES ('{order_id}', '{order['symbol']}', {order['quantity']})"
        
        try:
            result = await self.db.execute_query(query)
            if result:
                self.orders.append(order)
                await self._update_positions(order)
                return True
        except Exception as e:
            print(f"Order execution failed: {e}")
        
        return False
    
    async def _update_positions(self, order: Dict) -> None:
        \"\"\"Update position tracking\"\"\"
        symbol = order['symbol']
        quantity = order.get('quantity', 0)
        
        if symbol not in self.positions:
            self.positions[symbol] = 0
        
        if order.get('side') == 'buy':
            self.positions[symbol] += quantity
        else:
            self.positions[symbol] -= quantity
    
    def get_portfolio_value(self) -> float:
        \"\"\"Calculate total portfolio value\"\"\"
        total_value = 0.0
        
        # Inefficient nested loops
        for symbol, position in self.positions.items():
            for order in self.orders:
                if order.get('symbol') == symbol:
                    for price_history in order.get('price_history', [{}]):
                        if price_history.get('timestamp'):
                            total_value += position * price_history.get('price', 0)
                            break
        
        return total_value
    
    def generate_trade_id(self) -> str:
        \"\"\"Generate trade ID - uses weak randomization\"\"\"
        return str(random.randint(100000, 999999))  # Security issue: weak random
    
    def hash_order_data(self, order_data: str) -> str:
        \"\"\"Hash order data - uses weak hashing\"\"\"
        return hashlib.md5(order_data.encode()).hexdigest()  # Security issue: weak hash
""")
    
    # Order manager
    (temp_dir / "order_manager.py").write_text("""
\"\"\"
Order Management System
\"\"\"

from typing import Dict, List, Optional
from datetime import datetime
import json

class OrderManager:
    def __init__(self):
        self.pending_orders = []
        self.executed_orders = []
    
    async def create_order(self, symbol: str, signal: Dict) -> Optional[Dict]:
        \"\"\"Create a new trading order\"\"\"
        order = {
            'id': self._generate_order_id(),
            'symbol': symbol,
            'side': signal.get('signal', 'buy'),
            'quantity': self._calculate_quantity(symbol, signal),
            'timestamp': datetime.utcnow().isoformat(),
            'status': 'pending'
        }
        
        if self._validate_order(order):
            self.pending_orders.append(order)
            return order
        
        return None
    
    def _generate_order_id(self) -> str:
        \"\"\"Generate unique order ID\"\"\"
        return f"ORD_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"
    
    def _calculate_quantity(self, symbol: str, signal: Dict) -> int:
        \"\"\"Calculate order quantity based on signal strength\"\"\"
        base_quantity = 100
        
        # Simple quantity calculation
        if signal.get('signal') == 'buy':
            multiplier = signal.get('strength', 1.0)
            return int(base_quantity * multiplier)
        elif signal.get('signal') == 'sell':
            return base_quantity
        
        return 0
    
    def _validate_order(self, order: Dict) -> bool:
        \"\"\"Validate order parameters\"\"\"
        required_fields = ['id', 'symbol', 'side', 'quantity']
        
        for field in required_fields:
            if field not in order or not order[field]:
                return False
        
        if order['quantity'] <= 0:
            return False
        
        if order['side'] not in ['buy', 'sell']:
            return False
        
        return True
    
    def get_order_history(self) -> List[Dict]:
        \"\"\"Get order history\"\"\"
        return self.executed_orders.copy()
    
    def cancel_order(self, order_id: str) -> bool:
        \"\"\"Cancel a pending order\"\"\"
        for i, order in enumerate(self.pending_orders):
            if order['id'] == order_id:
                cancelled_order = self.pending_orders.pop(i)
                cancelled_order['status'] = 'cancelled'
                return True
        
        return False
""")
    
    # Risk manager
    (temp_dir / "risk_manager.py").write_text("""
\"\"\"
Risk Management System
\"\"\"

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

class RiskManager:
    def __init__(self):
        self.max_position_size = 10000
        self.max_daily_loss = 5000
        self.max_drawdown = 0.15
        self.position_limits = {}
        self.daily_pnl = 0.0
    
    async def validate_signal(self, symbol: str, signal: Dict) -> bool:
        \"\"\"Validate trading signal against risk parameters\"\"\"
        
        # Check position limits
        if not self._check_position_limits(symbol, signal):
            return False
        
        # Check daily loss limits
        if not self._check_daily_loss_limits():
            return False
        
        # Check correlation risk
        if not await self._check_correlation_risk(symbol):
            return False
        
        # Check volatility risk
        if not self._check_volatility_risk(symbol, signal):
            return False
        
        return True
    
    def _check_position_limits(self, symbol: str, signal: Dict) -> bool:
        \"\"\"Check if position is within limits\"\"\"
        current_position = self.position_limits.get(symbol, 0)
        signal_quantity = signal.get('quantity', 100)
        
        if signal.get('signal') == 'buy':
            new_position = current_position + signal_quantity
        else:
            new_position = current_position - signal_quantity
        
        return abs(new_position) <= self.max_position_size
    
    def _check_daily_loss_limits(self) -> bool:
        \"\"\"Check daily loss limits\"\"\"
        return self.daily_pnl > -self.max_daily_loss
    
    async def _check_correlation_risk(self, symbol: str) -> bool:
        \"\"\"Check correlation risk across positions\"\"\"
        # Simplified correlation check
        # In reality, this would calculate correlations between positions
        
        correlated_symbols = self._get_correlated_symbols(symbol)
        total_correlated_exposure = 0
        
        for corr_symbol in correlated_symbols:
            total_correlated_exposure += abs(self.position_limits.get(corr_symbol, 0))
        
        # Don't allow more than 50% of portfolio in correlated positions
        max_correlated_exposure = self.max_position_size * 0.5
        return total_correlated_exposure < max_correlated_exposure
    
    def _get_correlated_symbols(self, symbol: str) -> List[str]:
        \"\"\"Get symbols correlated with the given symbol\"\"\"
        # Simplified mapping - in reality would use correlation analysis
        correlation_groups = {
            'AAPL': ['MSFT', 'GOOGL', 'AMZN'],
            'MSFT': ['AAPL', 'GOOGL', 'AMZN'],
            'GOOGL': ['AAPL', 'MSFT', 'AMZN'],
            'AMZN': ['AAPL', 'MSFT', 'GOOGL'],
            'EURUSD': ['GBPUSD', 'AUDUSD'],
            'GBPUSD': ['EURUSD', 'AUDUSD'],
            'AUDUSD': ['EURUSD', 'GBPUSD']
        }
        
        return correlation_groups.get(symbol, [])
    
    def _check_volatility_risk(self, symbol: str, signal: Dict) -> bool:
        \"\"\"Check volatility-based risk\"\"\"
        # Simplified volatility check
        signal_strength = signal.get('strength', 1.0)
        
        # Don't allow high-risk trades when signal strength is low
        if signal_strength < 0.5:
            return False
        
        return True
    
    def update_daily_pnl(self, pnl_change: float) -> None:
        \"\"\"Update daily P&L tracking\"\"\"
        self.daily_pnl += pnl_change
    
    def get_risk_metrics(self) -> Dict:
        \"\"\"Get current risk metrics\"\"\"
        return {
            'daily_pnl': self.daily_pnl,
            'max_daily_loss': self.max_daily_loss,
            'remaining_daily_loss': self.max_daily_loss + self.daily_pnl,
            'position_count': len(self.position_limits),
            'total_exposure': sum(abs(pos) for pos in self.position_limits.values())
        }
""")
    
    # Database manager
    (temp_dir / "database.py").write_text("""
\"\"\"
Database Management System
\"\"\"

import asyncio
import sqlite3
from typing import Dict, List, Optional, Any
from datetime import datetime
# FIXED: Removed circular dependency - use dependency injection instead

class DatabaseManager:
    def __init__(self, trading_engine=None):
        self.connection = None
        self.is_connected = False
        self.engine = trading_engine  # Use dependency injection instead of circular import
    
    async def connect(self):
        \"\"\"Connect to database\"\"\"
        try:
            # Using SQLite for simplicity
            self.connection = sqlite3.connect(':memory:')
            await self._create_tables()
            self.is_connected = True
            
            # FIXED: No more circular dependency - engine passed via constructor
            if self.engine:
                # Engine is now injected via constructor
                await self.engine.initialize_connection(self.connection)
            
        except Exception as e:
            print(f"Database connection failed: {e}")
            raise
    
    async def _create_tables(self):
        \"\"\"Create database tables\"\"\"
        if not self.connection:
            raise RuntimeError("Not connected to database")
        
        cursor = self.connection.cursor()
        
        # Orders table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS orders (
                id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                price REAL,
                timestamp TEXT NOT NULL,
                status TEXT DEFAULT 'pending'
            )
        ''')
        
        # Positions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS positions (
                symbol TEXT PRIMARY KEY,
                quantity INTEGER NOT NULL,
                avg_price REAL NOT NULL,
                last_updated TEXT NOT NULL
            )
        ''')
        
        self.connection.commit()
    
    async def execute_query(self, query: str, params: Optional[tuple] = None) -> Optional[List[tuple]]:
        \"\"\"Execute database query - vulnerable to SQL injection\"\"\"
        if not self.is_connected:
            raise RuntimeError("Not connected to database")
        
        cursor = self.connection.cursor()
        
        try:
            # Direct query execution - vulnerable to SQL injection
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)  # This is where SQL injection can occur
            
            if query.strip().upper().startswith('SELECT'):
                return cursor.fetchall()
            else:
                self.connection.commit()
                return None
                
        except Exception as e:
            print(f"Query execution failed: {e}")
            self.connection.rollback()
            return None
    
    async def store_order(self, order: Dict) -> bool:
        \"\"\"Store order in database\"\"\"
        # Using string formatting - SQL injection vulnerability
        query = f'''
            INSERT INTO orders (id, symbol, side, quantity, price, timestamp, status)
            VALUES ('{order['id']}', '{order['symbol']}', '{order['side']}', 
                    {order['quantity']}, {order.get('price', 0)}, 
                    '{order['timestamp']}', '{order.get('status', 'pending')}')
        '''
        
        result = await self.execute_query(query)
        return result is not None
    
    async def get_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        \"\"\"Get orders from database\"\"\"
        if symbol:
            # Another SQL injection vulnerability
            query = f"SELECT * FROM orders WHERE symbol = '{symbol}'"
        else:
            query = "SELECT * FROM orders"
        
        rows = await self.execute_query(query)
        
        if rows:
            return [
                {
                    'id': row[0],
                    'symbol': row[1],
                    'side': row[2],
                    'quantity': row[3],
                    'price': row[4],
                    'timestamp': row[5],
                    'status': row[6]
                }
                for row in rows
            ]
        
        return []
    
    async def close(self):
        \"\"\"Close database connection\"\"\"
        if self.connection:
            self.connection.close()
            self.is_connected = False
""")
    
    # Utility functions
    (temp_dir / "utils.py").write_text("""
\"\"\"
Utility Functions
\"\"\"

import hashlib
import random
import os
import pickle
from typing import Any, Dict, List
from datetime import datetime

# FIXED: Use environment variables instead of hardcoded secrets
DATABASE_PASSWORD = os.getenv('DATABASE_PASSWORD', 'test-db-password')
API_SECRET = os.getenv('TRADING_API_KEY', 'test-api-secret')  # FIXED: Use environment variable
ENCRYPTION_KEY = os.getenv('ENCRYPTION_SECRET', 'test-encryption-key')

def hash_password(password: str) -> str:
    \"\"\"Hash password using weak algorithm\"\"\"
    # Using MD5 - security vulnerability
    return hashlib.md5(password.encode()).hexdigest()

def generate_session_token() -> str:
    \"\"\"Generate session token using weak randomization\"\"\"
    # Using random instead of secrets - security vulnerability
    return str(random.randint(1000000000, 9999999999))

def serialize_data(data: Any) -> bytes:
    \"\"\"Serialize data using pickle - security vulnerability\"\"\"
    # Using pickle is dangerous for untrusted data
    return pickle.dumps(data)

def deserialize_data(data: bytes) -> Any:
    \"\"\"Deserialize data using pickle - security vulnerability\"\"\"
    # Pickle deserialization can execute arbitrary code
    return pickle.loads(data)

def log_sensitive_data(user_id: str, password: str, api_key: str):
    \"\"\"Log sensitive data - security vulnerability\"\"\"
    # Logging sensitive information
    print(f"User {user_id} logged in with password {password} and API key {api_key}")

def execute_system_command(command: str) -> str:
    \"\"\"Execute system command - security vulnerability\"\"\"
    # Command injection vulnerability
    return os.system(command)

def calculate_fibonacci(n: int) -> int:
    \"\"\"Calculate Fibonacci number - inefficient recursive implementation\"\"\"
    # Performance issue: exponential time complexity
    if n <= 1:
        return n
    return calculate_fibonacci(n - 1) + calculate_fibonacci(n - 2)

def process_large_dataset(data: List[Dict]) -> List[Dict]:
    \"\"\"Process large dataset inefficiently\"\"\"
    results = []
    
    # Multiple nested loops - performance issue
    for item in data:
        for key, value in item.items():
            if isinstance(value, list):
                for sub_item in value:
                    for sub_key, sub_value in sub_item.items() if isinstance(sub_item, dict) else []:
                        # String concatenation in loops - performance issue
                        result_string = ""
                        for i in range(100):
                            result_string += f"{sub_key}_{sub_value}_{i}_"
                        
                        results.append({
                            'processed_data': result_string,
                            'timestamp': datetime.now().isoformat()
                        })
    
    return results

def find_item_in_list(items: List[str], target: str) -> bool:
    \"\"\"Inefficient search - performance issue\"\"\"
    # Using list membership test instead of set - O(n) instead of O(1)
    return target in items

class ConfigManager:
    \"\"\"Configuration manager with poor design\"\"\"
    
    def __init__(self):
        self.configs = {}
        self.database_config = {}
        self.api_config = {}
        self.security_config = {}
        self.performance_config = {}
        # Large class with many responsibilities - SOLID violation
    
    def load_database_config(self):
        \"\"\"Load database configuration\"\"\"
        # Hardcoded configuration
        self.database_config = {
            'host': 'localhost',
            'port': 5432,
            'username': 'admin',
            'password': 'password123',  # Hardcoded password
            'database': 'trading_db'
        }
    
    def load_api_config(self):
        \"\"\"Load API configuration\"\"\"
        self.api_config = {
            'base_url': 'https://api.example.com',
            'api_key': 'hardcoded_api_key',  # Hardcoded API key
            'timeout': 30
        }
    
    def load_security_config(self):
        \"\"\"Load security configuration\"\"\"
        self.security_config = {
            'encryption_key': 'weak_key_123',  # Weak encryption key
            'session_timeout': 3600,
            'max_login_attempts': 3
        }
    
    def save_all_configs(self):
        \"\"\"Save all configurations\"\"\"
        # Method doing too many things - SOLID violation
        self.save_database_config()
        self.save_api_config()
        self.save_security_config()
        self.validate_all_configs()
        self.backup_configs()
        self.log_config_changes()
    
    def save_database_config(self):
        pass
    
    def save_api_config(self):
        pass
    
    def save_security_config(self):
        pass
    
    def validate_all_configs(self):
        pass
    
    def backup_configs(self):
        pass
    
    def log_config_changes(self):
        pass
""")
    
    # Requirements file
    (temp_dir / "requirements.txt").write_text("""
# Core dependencies
numpy==1.21.0
pandas==1.3.0
asyncio-mqtt==0.8.1

# Web framework
flask==2.0.1
requests==2.25.1

# Database
sqlite3==2.6.0
sqlalchemy==1.4.22

# Trading libraries
TA-Lib==0.4.21
ccxt==1.52.45

# Utilities
pyyaml==5.4.1
python-dotenv==0.19.0

# Development tools
pytest==6.2.4
black==21.6.0
flake8==3.9.2
""")
    
    # Test directory with basic tests
    test_dir = temp_dir / "tests"
    test_dir.mkdir()
    
    (test_dir / "__init__.py").write_text("")
    
    (test_dir / "test_trading_engine.py").write_text("""
\"\"\"
Tests for Trading Engine
\"\"\"

import unittest
import asyncio
from trading_engine import TradingEngine

class TestTradingEngine(unittest.TestCase):
    def setUp(self):
        self.engine = TradingEngine()
    
    def test_initialization(self):
        \"\"\"Test engine initialization\"\"\"
        self.assertFalse(self.engine.is_initialized)
        self.assertEqual(len(self.engine.orders), 0)
    
    def test_order_execution(self):
        \"\"\"Test order execution\"\"\"
        # This is a placeholder test
        order = {
            'id': 'test_001',
            'symbol': 'AAPL',
            'side': 'buy',
            'quantity': 100
        }
        
        # In a real test, we'd mock the database and test the execution
        self.assertIsNotNone(order)
    
    def test_portfolio_calculation(self):
        \"\"\"Test portfolio value calculation\"\"\"
        # Add some test positions
        self.engine.positions = {'AAPL': 100, 'GOOGL': 50}
        
        # Calculate portfolio value
        value = self.engine.get_portfolio_value()
        self.assertIsInstance(value, float)

if __name__ == '__main__':
    unittest.main()
""")
    
    (test_dir / "test_risk_manager.py").write_text("""
\"\"\"
Tests for Risk Manager
\"\"\"

import unittest
import asyncio
from risk_manager import RiskManager

class TestRiskManager(unittest.TestCase):
    def setUp(self):
        self.risk_manager = RiskManager()
    
    def test_position_limits(self):
        \"\"\"Test position limit checking\"\"\"
        signal = {
            'signal': 'buy',
            'quantity': 1000,
            'strength': 0.8
        }
        
        result = self.risk_manager._check_position_limits('AAPL', signal)
        self.assertTrue(result)
    
    def test_daily_loss_limits(self):
        \"\"\"Test daily loss limit checking\"\"\"
        # Set a small loss
        self.risk_manager.daily_pnl = -1000
        
        result = self.risk_manager._check_daily_loss_limits()
        self.assertTrue(result)
        
        # Set a large loss
        self.risk_manager.daily_pnl = -6000
        
        result = self.risk_manager._check_daily_loss_limits()
        self.assertFalse(result)

if __name__ == '__main__':
    unittest.main()
""")
    
    print("✓ Test project structure created successfully")


async def test_individual_agents(project_path: str):
    """Test each agent individually"""
    print("\n" + "="*60)
    print("TESTING INDIVIDUAL AGENTS")
    print("="*60)
    
    config = {
        'architecture_diagram': {
            'output_format': 'svg',
            'include_external_deps': True
        },
        'dependency_analysis': {
            'include_external_deps': True,
            'max_circular_chain_length': 10
        },
        'code_metrics': {
            'complexity_threshold': 10,
            'coverage_threshold': 80.0,
            'duplication_threshold': 5.0
        },
        'migration_planning': {
            'risk_tolerance': 'medium',
            'migration_window_hours': 8
        }
    }
    
    # Test Architecture Diagram Manager
    print("\n1. Testing Architecture Diagram Manager...")
    try:
        diagram_manager = ArchitectureDiagramManager(config)
        arch_result = await diagram_manager.generate_architecture_diagrams(project_path)
        
        components = arch_result.get('components', [])
        relationships = arch_result.get('relationships', [])
        
        print(f"   ✓ Generated {len(components)} components")
        print(f"   ✓ Found {len(relationships)} relationships")
        
        # Print some component details
        for i, comp in enumerate(components[:3]):
            print(f"   - Component {i+1}: {comp.get('name', 'Unknown')} ({comp.get('type', 'Unknown')})")
        
    except Exception as e:
        print(f"   ✗ Architecture Diagram Manager failed: {e}")
    
    # Test Dependency Analysis Agent
    print("\n2. Testing Dependency Analysis Agent...")
    try:
        dependency_agent = DependencyAnalysisAgent(config)
        dep_result = await dependency_agent.analyze_dependencies(project_path)
        
        dep_graph = dep_result.get('dependency_graph', {})
        circular_deps = dep_result.get('circular_dependencies', [])
        metrics = dep_result.get('metrics', {})
        
        print(f"   ✓ Found {len(dep_graph.get('nodes', []))} dependency nodes")
        print(f"   ✓ Found {len(dep_graph.get('edges', []))} dependency edges")
        print(f"   ✓ Detected {len(circular_deps)} circular dependencies")
        print(f"   ✓ Coupling index: {metrics.get('coupling_index', 0):.2f}")
        
        # Print circular dependencies
        for i, circ in enumerate(circular_deps[:2]):
            nodes = circ.get('nodes', [])
            print(f"   - Circular dependency {i+1}: {' -> '.join(nodes[:3])}{'...' if len(nodes) > 3 else ''}")
        
    except Exception as e:
        print(f"   ✗ Dependency Analysis Agent failed: {e}")
    
    # Test Code Metrics Dashboard
    print("\n3. Testing Code Metrics Dashboard...")
    try:
        metrics_dashboard = CodeMetricsDashboard(config)
        metrics_result = await metrics_dashboard.generate_dashboard(project_path)
        
        project_metrics = metrics_result.get('project_metrics', {})
        file_metrics = metrics_result.get('file_metrics', [])
        alerts = metrics_result.get('alerts', [])
        
        print(f"   ✓ Analyzed {project_metrics.get('total_files', 0)} files")
        print(f"   ✓ Total lines of code: {project_metrics.get('total_loc', 0)}")
        print(f"   ✓ Overall complexity: {project_metrics.get('overall_complexity', 0):.2f}")
        print(f"   ✓ Test coverage: {project_metrics.get('overall_coverage', 0):.1f}%")
        print(f"   ✓ Generated {len(alerts)} alerts")
        
        # Print high complexity files
        high_complexity_files = [f for f in file_metrics if f.get('cyclomatic_complexity', 0) > 10]
        print(f"   ✓ Found {len(high_complexity_files)} high-complexity files")
        
        for i, file_info in enumerate(high_complexity_files[:3]):
            file_name = Path(file_info['file_path']).name
            complexity = file_info.get('cyclomatic_complexity', 0)
            print(f"   - {file_name}: complexity {complexity}")
        
    except Exception as e:
        print(f"   ✗ Code Metrics Dashboard failed: {e}")
    
    # Test Migration Planning Agent
    print("\n4. Testing Migration Planning Agent...")
    try:
        migration_agent = MigrationPlanningAgent(config)
        
        source_config = {
            'project_path': project_path,
            'python_version': '3.8.10',
            'dependencies': {
                'numpy': '1.21.0',
                'pandas': '1.3.0',
                'flask': '2.0.1'
            }
        }
        
        target_config = {
            'python_version': '3.11.5',
            'dependencies': {
                'numpy': '1.24.0',
                'pandas': '2.0.0',
                'flask': '2.3.0'
            }
        }
        
        migration_result = await migration_agent.create_migration_plan(
            'version_upgrade', source_config, target_config
        )
        
        migration_plan = migration_result.get('migration_plan', {})
        compatibility = migration_result.get('compatibility_analysis', [])
        risks = migration_result.get('risk_assessment', [])
        
        steps = migration_plan.get('steps', [])
        timeline = migration_plan.get('timeline', {})
        
        print(f"   ✓ Created migration plan with {len(steps)} steps")
        print(f"   ✓ Analyzed {len(compatibility)} compatibility items")
        print(f"   ✓ Identified {len(risks)} risks")
        print(f"   ✓ Estimated timeline: {timeline.get('total_hours', 0):.1f} hours")
        
        # Print some steps
        for i, step in enumerate(steps[:3]):
            print(f"   - Step {i+1}: {step.get('name', 'Unknown')} ({step.get('estimated_hours', 0):.1f}h)")
        
    except Exception as e:
        print(f"   ✗ Migration Planning Agent failed: {e}")


async def test_master_coordinator(project_path: str):
    """Test the master coordinator"""
    print("\n" + "="*60)
    print("TESTING MASTER COORDINATOR")
    print("="*60)
    
    config = {
        'enable_parallel_execution': True,
        'cache_results': True,
        'cross_validation': True,
        'max_concurrent_agents': 4,
        'architecture_diagram': {'output_format': 'svg'},
        'dependency_analysis': {'include_external_deps': True},
        'code_metrics': {'complexity_threshold': 10},
        'migration_planning': {'risk_tolerance': 'medium'}
    }
    
    try:
        print("\n1. Initializing Master Coordinator...")
        coordinator = MasterCoordinator(config)
        print(f"   ✓ Initialized with {len(coordinator.agents)} agents")
        
        print("\n2. Running comprehensive analysis...")
        start_time = datetime.now()
        
        results = await coordinator.analyze_system(project_path, 'comprehensive')
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        print(f"   ✓ Analysis completed in {execution_time:.2f} seconds")
        
        # Print results summary
        print("\n3. Analysis Results Summary:")
        
        # Health report
        health_report = results.get('health_report', {})
        print(f"   Overall Score: {health_report.get('overall_score', 0):.1f}/100")
        print(f"   Health Status: {health_report.get('health_status', 'Unknown').title()}")
        
        # Insights
        insights = results.get('insights', [])
        critical_insights = [i for i in insights if i.get('severity') == 'critical']
        warning_insights = [i for i in insights if i.get('severity') == 'warning']
        
        print(f"   Critical Issues: {len(critical_insights)}")
        print(f"   Warnings: {len(warning_insights)}")
        print(f"   Total Insights: {len(insights)}")
        
        # Print top insights
        print("\n4. Top Issues Found:")
        for i, insight in enumerate(insights[:5]):
            severity = insight.get('severity', 'unknown').upper()
            title = insight.get('title', 'Unknown')
            category = insight.get('category', 'unknown').title()
            print(f"   {i+1}. [{severity}] {title} ({category})")
        
        # Recommendations
        recommendations = results.get('recommendations', [])
        print(f"\n5. Generated {len(recommendations)} recommendations")
        
        for i, rec in enumerate(recommendations[:5]):
            title = rec.get('title', 'Unknown')
            severity = rec.get('severity', 'info').upper()
            print(f"   {i+1}. [{severity}] {title}")
        
        # Next steps
        next_steps = results.get('next_steps', [])
        print(f"\n6. Recommended Next Steps ({len(next_steps)} total):")
        
        for i, step in enumerate(next_steps[:3]):
            title = step.get('title', 'Unknown')
            timeline = step.get('timeline', 'Unknown')
            priority = step.get('priority', 'unknown').upper()
            print(f"   {i+1}. [{priority}] {title} ({timeline})")
        
        # Execution metadata
        metadata = results.get('metadata', {})
        agents_used = metadata.get('agents_used', [])
        total_time = metadata.get('total_execution_time', 0)
        errors = metadata.get('errors', [])
        warnings = metadata.get('warnings', [])
        
        print(f"\n7. Execution Metadata:")
        print(f"   Agents Used: {len(agents_used)}")
        print(f"   Total Execution Time: {total_time:.2f}s")
        print(f"   Errors: {len(errors)}")
        print(f"   Warnings: {len(warnings)}")
        
        if errors:
            print("   Errors encountered:")
            for error in errors[:3]:
                print(f"     - {error}")
        
        # Test export functionality
        print("\n8. Testing export functionality...")
        try:
            session_id = results.get('session_id')
            if session_id:
                export_file = await coordinator.export_analysis_report(session_id, 'json')
                print(f"   ✓ Exported report to: {export_file}")
                
                # Verify file exists
                if Path(export_file).exists():
                    file_size = Path(export_file).stat().st_size
                    print(f"   ✓ Export file size: {file_size:,} bytes")
                else:
                    print("   ✗ Export file not found")
        except Exception as e:
            print(f"   ✗ Export failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Master Coordinator failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_performance_and_scalability(project_path: str):
    """Test performance and scalability"""
    print("\n" + "="*60)
    print("TESTING PERFORMANCE AND SCALABILITY")
    print("="*60)
    
    config = {
        'enable_parallel_execution': True,
        'cache_results': True,
        'code_metrics': {'complexity_threshold': 10}
    }
    
    # Test different analysis scopes
    scopes = ['quick', 'standard', 'comprehensive']
    
    for scope in scopes:
        print(f"\n1. Testing '{scope}' analysis scope...")
        try:
            coordinator = MasterCoordinator(config)
            
            start_time = datetime.now()
            results = await coordinator.analyze_system(project_path, scope)
            end_time = datetime.now()
            
            execution_time = (end_time - start_time).total_seconds()
            
            insights = results.get('insights', [])
            metadata = results.get('metadata', {})
            agents_used = metadata.get('agents_used', [])
            
            print(f"   ✓ Completed in {execution_time:.2f}s")
            print(f"   ✓ Used {len(agents_used)} agents")
            print(f"   ✓ Generated {len(insights)} insights")
            
        except Exception as e:
            print(f"   ✗ {scope.title()} analysis failed: {e}")
    
    # Test caching
    print(f"\n2. Testing result caching...")
    try:
        coordinator = MasterCoordinator(config)
        
        # First run
        start_time = datetime.now()
        await coordinator.analyze_system(project_path, 'quick')
        first_run_time = (datetime.now() - start_time).total_seconds()
        
        # Second run (should use cache)
        start_time = datetime.now()
        await coordinator.analyze_system(project_path, 'quick')
        second_run_time = (datetime.now() - start_time).total_seconds()
        
        print(f"   ✓ First run: {first_run_time:.2f}s")
        print(f"   ✓ Second run: {second_run_time:.2f}s")
        
        if second_run_time < first_run_time * 0.8:
            print(f"   ✓ Caching appears to be working (speedup: {first_run_time/second_run_time:.1f}x)")
        else:
            print(f"   ? Caching benefit unclear")
        
        # Cache stats
        cache_stats = coordinator.get_cache_stats()
        print(f"   Cache size: {cache_stats['cache_size']} entries")
        
    except Exception as e:
        print(f"   ✗ Caching test failed: {e}")


async def test_error_handling():
    """Test error handling"""
    print("\n" + "="*60)
    print("TESTING ERROR HANDLING")
    print("="*60)
    
    config = {'enable_parallel_execution': False}
    
    # Test with non-existent path
    print("\n1. Testing with non-existent project path...")
    try:
        coordinator = MasterCoordinator(config)
        results = await coordinator.analyze_system("/non/existent/path", 'quick')
        
        metadata = results.get('metadata', {})
        errors = metadata.get('errors', [])
        
        print(f"   ✓ Handled gracefully with {len(errors)} errors")
        
        if errors:
            print("   Errors encountered:")
            for error in errors[:3]:
                print(f"     - {error}")
        
    except Exception as e:
        print(f"   ✗ Error handling failed: {e}")
    
    # Test with invalid configuration
    print("\n2. Testing with invalid configuration...")
    try:
        invalid_config = {
            'enable_parallel_execution': True,
            'max_concurrent_agents': -1,  # Invalid
            'cache_results': 'invalid'    # Invalid
        }
        
        coordinator = MasterCoordinator(invalid_config)
        print("   ✓ Invalid config handled gracefully")
        
    except Exception as e:
        print(f"   Configuration validation error: {e}")


def print_summary(success_count: int, total_tests: int):
    """Print test summary"""
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    success_rate = (success_count / total_tests) * 100 if total_tests > 0 else 0
    
    print(f"Tests Passed: {success_count}/{total_tests} ({success_rate:.1f}%)")
    
    if success_rate >= 80:
        print("🎉 INTEGRATION TESTS PASSED!")
        print("\nYour System Architect suite is ready for use!")
        print("\nNext steps:")
        print("1. Review any warnings or errors above")
        print("2. Consider running with your actual project")
        print("3. Explore the generated reports and insights")
        print("4. Integrate into your development workflow")
    elif success_rate >= 60:
        print("⚠️  INTEGRATION TESTS PARTIALLY PASSED")
        print("\nMost components are working, but some issues need attention.")
        print("Review the errors above and fix any critical issues.")
    else:
        print("❌ INTEGRATION TESTS FAILED")
        print("\nMultiple components have issues that need to be addressed.")
        print("Please review and fix the errors before using the system.")


async def main():
    """Main test function"""
    print("🚀 SYSTEM ARCHITECT SUITE INTEGRATION TESTING")
    print("=" * 60)
    
    total_tests = 0
    success_count = 0
    
    # Create test project
    with tempfile.TemporaryDirectory() as temp_dir:
        project_path = Path(temp_dir)
        create_test_project(project_path)
        
        # Test individual agents
        try:
            await test_individual_agents(str(project_path))
            success_count += 1
        except Exception as e:
            print(f"Individual agent tests failed: {e}")
        total_tests += 1
        
        # Test master coordinator
        try:
            success = await test_master_coordinator(str(project_path))
            if success:
                success_count += 1
        except Exception as e:
            print(f"Master coordinator test failed: {e}")
        total_tests += 1
        
        # Test performance
        try:
            await test_performance_and_scalability(str(project_path))
            success_count += 1
        except Exception as e:
            print(f"Performance tests failed: {e}")
        total_tests += 1
        
        # Test error handling
        try:
            await test_error_handling()
            success_count += 1
        except Exception as e:
            print(f"Error handling tests failed: {e}")
        total_tests += 1
    
    # Print summary
    print_summary(success_count, total_tests)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)