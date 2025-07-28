# Trading System - Agent API Documentation

## 📚 **API Reference Overview**

This document provides comprehensive API documentation for all agents in the trading system. Each agent exposes a consistent interface for integration and automation.

## 🏗️ **Base Agent Architecture**

### **BaseAgent Class**

All agents inherit from the `BaseAgent` base class, providing consistent interface and functionality.

```python
from trading_system.core.base.agent import BaseAgent

class BaseAgent:
    """Base class for all trading system agents"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize base agent
        
        Args:
            name: Agent identifier
            config: Agent configuration dictionary
        """
        self.name = name
        self.config = config
        self.logger = logging.getLogger(name)
        self.is_active = False
    
    async def start(self) -> bool:
        """Start the agent"""
        pass
    
    async def stop(self) -> bool:
        """Stop the agent"""
        pass
    
    async def process(self, data: Any) -> Any:
        """Process input data and return results"""
        pass
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status"""
        pass
```

## 🤖 **System Architect Suite APIs**

### **Master Coordinator API**

The central orchestrator for all architecture analysis agents.

```python
from trading_system.agents.system_architect.master_coordinator import MasterCoordinator

class MasterCoordinator(BaseAgent):
    """Master coordinator for system architecture analysis"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize master coordinator
        
        Args:
            config: Configuration dictionary with agent settings
        """
    
    async def analyze_system(self, 
                           project_path: str, 
                           analysis_scope: str = "comprehensive") -> Dict[str, Any]:
        """
        Perform comprehensive system analysis
        
        Args:
            project_path: Path to the project to analyze
            analysis_scope: Analysis depth ('quick', 'standard', 'comprehensive', 'deep')
            
        Returns:
            Dict containing complete analysis results:
            {
                'session_id': str,
                'analysis_scope': str,
                'project_path': str,
                'timestamp': str,
                'raw_results': Dict[str, Any],
                'insights': List[Dict[str, Any]],
                'health_report': Dict[str, Any],
                'recommendations': List[Dict[str, Any]],
                'next_steps': List[Dict[str, Any]],
                'metadata': Dict[str, Any]
            }
        """
    
    def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """
        Get status of an analysis session
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            Dict with session status information
        """
    
    async def export_analysis_report(self, 
                                   session_id: str, 
                                   format_type: str = 'json') -> str:
        """
        Export analysis report in specified format
        
        Args:
            session_id: Session to export
            format_type: Export format ('json', 'html', 'csv')
            
        Returns:
            str: Path to exported file
        """

# Usage Example
config = {
    'enable_parallel_execution': True,
    'cache_results': True,
    'cross_validation': True
}

coordinator = MasterCoordinator(config)
results = await coordinator.analyze_system("/path/to/project", "comprehensive")
```

### **Architecture Diagram Manager API**

Generates visual architecture diagrams and component analysis.

```python
from trading_system.agents.system_architect.architecture_diagram_manager import ArchitectureDiagramManager

class ArchitectureDiagramManager(BaseAgent):
    """Manages architecture diagram generation and analysis"""
    
    async def generate_architecture_diagrams(self, project_path: str) -> Dict[str, Any]:
        """
        Generate comprehensive architecture diagrams
        
        Args:
            project_path: Path to project for analysis
            
        Returns:
            Dict containing:
            {
                'components': List[Dict] - Discovered components
                'relationships': List[Dict] - Component relationships
                'diagrams': Dict[str, str] - Generated diagram paths
                'metrics': Dict[str, float] - Architecture metrics
            }
        """
    
    async def analyze_component_structure(self, component_path: str) -> Dict[str, Any]:
        """
        Analyze individual component structure
        
        Args:
            component_path: Path to component
            
        Returns:
            Dict with component analysis results
        """
    
    def generate_mermaid_diagram(self, components: List[Dict], 
                                relationships: List[Dict]) -> str:
        """
        Generate Mermaid diagram syntax
        
        Args:
            components: List of component definitions
            relationships: List of relationship definitions
            
        Returns:
            str: Mermaid diagram syntax
        """

# Usage Example
diagram_manager = ArchitectureDiagramManager(config)
diagrams = await diagram_manager.generate_architecture_diagrams("/path/to/project")
```

### **Dependency Analysis Agent API**

Analyzes code dependencies and detects circular dependencies.

```python
from trading_system.agents.system_architect.dependency_analysis_agent import DependencyAnalysisAgent

class DependencyAnalysisAgent(BaseAgent):
    """Analyzes project dependencies and relationships"""
    
    async def analyze_dependencies(self, project_path: str) -> Dict[str, Any]:
        """
        Comprehensive dependency analysis
        
        Args:
            project_path: Path to project
            
        Returns:
            Dict containing:
            {
                'dependency_graph': Dict - NetworkX graph representation
                'circular_dependencies': List[Dict] - Detected cycles
                'metrics': Dict - Coupling and stability metrics
                'violations': List[Dict] - Architecture violations
                'recommendations': List[str] - Improvement suggestions
            }
        """
    
    async def detect_circular_dependencies(self, dependency_graph: Dict) -> List[Dict]:
        """
        Detect circular dependencies in the graph
        
        Args:
            dependency_graph: Graph representation of dependencies
            
        Returns:
            List of circular dependency chains with severity assessment
        """
    
    def calculate_coupling_metrics(self, dependency_graph: Dict) -> Dict[str, float]:
        """
        Calculate coupling and stability metrics
        
        Args:
            dependency_graph: Dependency graph
            
        Returns:
            Dict with coupling index, stability index, and other metrics
        """

# Usage Example
dep_agent = DependencyAnalysisAgent(config)
dependencies = await dep_agent.analyze_dependencies("/path/to/project")
```

### **Code Metrics Dashboard API**

Provides comprehensive code quality metrics and analysis.

```python
from trading_system.agents.system_architect.code_metrics_dashboard import CodeMetricsDashboard

class CodeMetricsDashboard(BaseAgent):
    """Generates comprehensive code quality metrics"""
    
    async def generate_dashboard(self, project_path: str) -> Dict[str, Any]:
        """
        Generate complete metrics dashboard
        
        Args:
            project_path: Path to project
            
        Returns:
            Dict containing:
            {
                'project_metrics': Dict - Overall project metrics
                'file_metrics': List[Dict] - Per-file detailed metrics
                'quality_gates': Dict - Quality gate results
                'trends': Dict - Historical trend data
                'alerts': List[Dict] - Quality alerts and warnings
            }
        """
    
    async def analyze_code_complexity(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze complexity metrics for a single file
        
        Args:
            file_path: Path to source file
            
        Returns:
            Dict with complexity metrics (cyclomatic, cognitive, halstead)
        """
    
    async def calculate_maintainability_index(self, file_path: str) -> float:
        """
        Calculate maintainability index for a file
        
        Args:
            file_path: Path to source file
            
        Returns:
            float: Maintainability index (0-100)
        """
    
    def generate_quality_report(self, metrics: Dict[str, Any]) -> str:
        """
        Generate HTML quality report
        
        Args:
            metrics: Computed metrics dictionary
            
        Returns:
            str: HTML report content
        """

# Usage Example
metrics_dashboard = CodeMetricsDashboard(config)
metrics = await metrics_dashboard.generate_dashboard("/path/to/project")
```

### **Security Audit Agent API**

Performs comprehensive security analysis and vulnerability detection.

```python
from trading_system.agents.system_architect.security_audit_agent import SecurityAuditAgent

class SecurityAuditAgent(BaseAgent):
    """Performs security analysis and vulnerability detection"""
    
    async def audit_security(self, project_path: str) -> Dict[str, Any]:
        """
        Comprehensive security audit
        
        Args:
            project_path: Path to project
            
        Returns:
            Dict containing:
            {
                'vulnerabilities': List[Dict] - Detected vulnerabilities
                'security_score': float - Overall security score (0-100)
                'compliance': Dict - Compliance with security standards
                'recommendations': List[Dict] - Security improvements
                'risk_assessment': Dict - Risk analysis
            }
        """
    
    async def scan_for_vulnerabilities(self, file_path: str) -> List[Dict]:
        """
        Scan individual file for security vulnerabilities
        
        Args:
            file_path: Path to source file
            
        Returns:
            List of vulnerability dictionaries with details
        """
    
    def check_compliance(self, project_path: str) -> Dict[str, Any]:
        """
        Check compliance with security standards (OWASP, CWE)
        
        Args:
            project_path: Path to project
            
        Returns:
            Dict with compliance status and recommendations
        """

# Usage Example
security_agent = SecurityAuditAgent(config)
security_results = await security_agent.audit_security("/path/to/project")
```

### **Migration Planning Agent API**

Plans and manages system migrations and upgrades.

```python
from trading_system.agents.system_architect.migration_planning_agent import MigrationPlanningAgent

class MigrationPlanningAgent(BaseAgent):
    """Plans and manages system migrations"""
    
    async def create_migration_plan(self, 
                                  migration_type: str,
                                  source_config: Dict[str, Any], 
                                  target_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create comprehensive migration plan
        
        Args:
            migration_type: Type of migration ('version_upgrade', 'framework_change', etc.)
            source_config: Current system configuration
            target_config: Target system configuration
            
        Returns:
            Dict containing:
            {
                'migration_plan': Dict - Detailed migration steps
                'compatibility_analysis': List[Dict] - Compatibility assessment
                'risk_assessment': List[Dict] - Risk analysis
                'timeline': Dict - Time and resource estimates
                'rollback_plan': Dict - Rollback procedures
            }
        """
    
    async def analyze_compatibility(self, 
                                  source_version: str, 
                                  target_version: str) -> Dict[str, Any]:
        """
        Analyze compatibility between versions
        
        Args:
            source_version: Current version
            target_version: Target version
            
        Returns:
            Dict with compatibility analysis and breaking changes
        """
    
    def estimate_migration_effort(self, migration_plan: Dict) -> Dict[str, float]:
        """
        Estimate migration effort and resources
        
        Args:
            migration_plan: Migration plan dictionary
            
        Returns:
            Dict with time, resource, and cost estimates
        """

# Usage Example
migration_agent = MigrationPlanningAgent(config)
plan = await migration_agent.create_migration_plan(
    'version_upgrade',
    {'python_version': '3.8'},
    {'python_version': '3.11'}
)
```

## 📊 **Trading System Core Agents APIs**

### **Data Universe Agent API**

Manages market data ingestion, processing, and distribution.

```python
from trading_system.agents.data_universe.data_universe_agent import DataUniverseAgent

class DataUniverseAgent(BaseAgent):
    """Manages market data and universe construction"""
    
    async def ingest_market_data(self, 
                               data_source: str, 
                               symbols: List[str]) -> Dict[str, Any]:
        """
        Ingest market data from specified source
        
        Args:
            data_source: Data provider identifier ('bloomberg', 'reuters', etc.)
            symbols: List of symbols to ingest
            
        Returns:
            Dict with ingestion status and data summary
        """
    
    async def get_universe_data(self, 
                              universe_name: str, 
                              start_date: str, 
                              end_date: str) -> pd.DataFrame:
        """
        Retrieve universe data for specified period
        
        Args:
            universe_name: Universe identifier
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with market data
        """
    
    def validate_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate data quality and completeness
        
        Args:
            data: Market data DataFrame
            
        Returns:
            Dict with quality metrics and validation results
        """

# Usage Example
data_agent = DataUniverseAgent(config)
data = await data_agent.get_universe_data("SP500", "2024-01-01", "2024-12-31")
```

### **Technical Analysis Agent API**

Generates technical indicators and trading signals.

```python
from trading_system.agents.feature_engineering.technical_analysis_agent import TechnicalAnalysisAgent

class TechnicalAnalysisAgent(BaseAgent):
    """Technical analysis and indicator generation"""
    
    async def calculate_indicators(self, 
                                 data: pd.DataFrame, 
                                 indicators: List[str]) -> pd.DataFrame:
        """
        Calculate technical indicators
        
        Args:
            data: Price data DataFrame
            indicators: List of indicator names
            
        Returns:
            DataFrame with calculated indicators
        """
    
    async def generate_signals(self, 
                             data: pd.DataFrame, 
                             strategy: str) -> pd.DataFrame:
        """
        Generate trading signals based on technical analysis
        
        Args:
            data: Data with indicators
            strategy: Signal generation strategy
            
        Returns:
            DataFrame with buy/sell/hold signals
        """
    
    def backtest_strategy(self, 
                         signals: pd.DataFrame, 
                         prices: pd.DataFrame) -> Dict[str, float]:
        """
        Backtest trading strategy
        
        Args:
            signals: Trading signals
            prices: Historical prices
            
        Returns:
            Dict with backtest results (returns, sharpe, drawdown, etc.)
        """

# Usage Example
ta_agent = TechnicalAnalysisAgent(config)
indicators = await ta_agent.calculate_indicators(data, ['SMA_20', 'RSI', 'MACD'])
```

### **ML Ensemble Agent API**

Machine learning model orchestration and ensemble predictions.

```python
from trading_system.agents.ml_ensemble.ml_ensemble_agent import MLEnsembleAgent

class MLEnsembleAgent(BaseAgent):
    """Machine learning ensemble model management"""
    
    async def train_models(self, 
                          training_data: pd.DataFrame, 
                          target_column: str) -> Dict[str, Any]:
        """
        Train ensemble of ML models
        
        Args:
            training_data: Training dataset
            target_column: Target variable name
            
        Returns:
            Dict with training results and model performance
        """
    
    async def predict(self, 
                     features: pd.DataFrame, 
                     model_ensemble: str = 'default') -> pd.DataFrame:
        """
        Generate ensemble predictions
        
        Args:
            features: Feature data for prediction
            model_ensemble: Ensemble configuration to use
            
        Returns:
            DataFrame with predictions and confidence intervals
        """
    
    def evaluate_models(self, 
                       test_data: pd.DataFrame, 
                       target_column: str) -> Dict[str, Dict[str, float]]:
        """
        Evaluate model performance
        
        Args:
            test_data: Test dataset
            target_column: Target variable name
            
        Returns:
            Dict with performance metrics for each model
        """
    
    async def retrain_models(self, 
                           new_data: pd.DataFrame, 
                           incremental: bool = True) -> Dict[str, Any]:
        """
        Retrain models with new data
        
        Args:
            new_data: New training data
            incremental: Whether to use incremental learning
            
        Returns:
            Dict with retraining results
        """

# Usage Example
ml_agent = MLEnsembleAgent(config)
predictions = await ml_agent.predict(features_df)
```

### **Risk Management Agent API**

Comprehensive risk assessment and management.

```python
from trading_system.agents.risk_management.risk_modeling_agent import RiskModelingAgent

class RiskModelingAgent(BaseAgent):
    """Risk modeling and management"""
    
    async def calculate_portfolio_risk(self, 
                                     portfolio: Dict[str, float], 
                                     market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate comprehensive portfolio risk metrics
        
        Args:
            portfolio: Portfolio positions {symbol: weight}
            market_data: Historical market data
            
        Returns:
            Dict with risk metrics (VaR, CVaR, volatility, correlations, etc.)
        """
    
    async def stress_test(self, 
                         portfolio: Dict[str, float], 
                         scenarios: List[Dict]) -> Dict[str, Any]:
        """
        Perform stress testing on portfolio
        
        Args:
            portfolio: Portfolio positions
            scenarios: List of stress test scenarios
            
        Returns:
            Dict with stress test results
        """
    
    def calculate_var(self, 
                     returns: pd.Series, 
                     confidence_level: float = 0.95) -> float:
        """
        Calculate Value at Risk
        
        Args:
            returns: Return series
            confidence_level: Confidence level (0.95 = 5% VaR)
            
        Returns:
            float: VaR value
        """
    
    async def monitor_real_time_risk(self, 
                                   portfolio: Dict[str, float]) -> Dict[str, Any]:
        """
        Real-time risk monitoring
        
        Args:
            portfolio: Current portfolio positions
            
        Returns:
            Dict with real-time risk assessment
        """

# Usage Example
risk_agent = RiskModelingAgent(config)
risk_metrics = await risk_agent.calculate_portfolio_risk(portfolio, market_data)
```

### **Options Trading Agent API**

Specialized options trading strategies and analytics.

```python
from trading_system.agents.strategies.options.options_agent import OptionsAgent

class OptionsAgent(BaseAgent):
    """Options trading strategies and analytics"""
    
    async def price_option(self, 
                          option_params: Dict[str, Any], 
                          market_data: Dict[str, float]) -> Dict[str, float]:
        """
        Price options using various models
        
        Args:
            option_params: Option contract parameters
            market_data: Current market data
            
        Returns:
            Dict with option prices from different models
        """
    
    async def calculate_greeks(self, 
                             option_params: Dict[str, Any], 
                             market_data: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate option Greeks
        
        Args:
            option_params: Option contract parameters
            market_data: Current market data
            
        Returns:
            Dict with Greeks (delta, gamma, theta, vega, rho)
        """
    
    async def find_arbitrage_opportunities(self, 
                                         options_chain: pd.DataFrame) -> List[Dict]:
        """
        Identify arbitrage opportunities in options chain
        
        Args:
            options_chain: Options chain data
            
        Returns:
            List of arbitrage opportunities with details
        """
    
    def optimize_options_strategy(self, 
                                 market_view: Dict[str, Any], 
                                 constraints: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize options strategy based on market view
        
        Args:
            market_view: Expected market conditions
            constraints: Strategy constraints (capital, risk, etc.)
            
        Returns:
            Dict with optimized strategy details
        """

# Usage Example
options_agent = OptionsAgent(config)
greeks = await options_agent.calculate_greeks(option_params, market_data)
```

### **Recommendation Agent API**

Generates trade recommendations and portfolio advice.

```python
from trading_system.agents.output.recommendation_agent import RecommendationAgent

class RecommendationAgent(BaseAgent):
    """Trade recommendations and portfolio optimization"""
    
    async def generate_trade_recommendations(self, 
                                           market_analysis: Dict[str, Any], 
                                           portfolio: Dict[str, float]) -> List[Dict]:
        """
        Generate trade recommendations
        
        Args:
            market_analysis: Output from analysis agents
            portfolio: Current portfolio positions
            
        Returns:
            List of trade recommendations with rationale
        """
    
    async def optimize_portfolio(self, 
                               expected_returns: pd.Series, 
                               risk_model: Dict[str, Any], 
                               constraints: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize portfolio allocation
        
        Args:
            expected_returns: Expected returns for assets
            risk_model: Risk model parameters
            constraints: Portfolio constraints
            
        Returns:
            Dict with optimal portfolio weights and metrics
        """
    
    def calculate_performance_attribution(self, 
                                        portfolio_returns: pd.Series, 
                                        benchmark_returns: pd.Series) -> Dict[str, float]:
        """
        Calculate performance attribution
        
        Args:
            portfolio_returns: Portfolio return series
            benchmark_returns: Benchmark return series
            
        Returns:
            Dict with attribution analysis
        """

# Usage Example
rec_agent = RecommendationAgent(config)
recommendations = await rec_agent.generate_trade_recommendations(analysis, portfolio)
```

## 🔧 **Configuration Management**

### **Agent Configuration Structure**

```python
# Standard configuration structure for all agents
AGENT_CONFIG = {
    'agent_name': {
        'enabled': bool,              # Whether agent is active
        'update_frequency': str,      # Update frequency ('1min', '5min', '1hour', etc.)
        'data_sources': List[str],    # Data sources to use
        'parameters': Dict[str, Any], # Agent-specific parameters
        'risk_limits': Dict[str, float], # Risk management limits
        'logging': {
            'level': str,             # Log level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
            'file_path': str,         # Log file path
            'max_size': int           # Max log file size in MB
        },
        'monitoring': {
            'health_checks': bool,    # Enable health monitoring
            'performance_tracking': bool, # Track performance metrics
            'alert_thresholds': Dict[str, float] # Alert thresholds
        }
    }
}

# Example configuration
TRADING_SYSTEM_CONFIG = {
    'data_universe_agent': {
        'enabled': True,
        'update_frequency': '1min',
        'data_sources': ['bloomberg', 'reuters'],
        'parameters': {
            'universe_size': 500,
            'data_validation': True,
            'cache_duration': 3600
        }
    },
    'ml_ensemble_agent': {
        'enabled': True,
        'update_frequency': '15min',
        'parameters': {
            'model_types': ['random_forest', 'xgboost', 'neural_network'],
            'ensemble_method': 'weighted_average',
            'retrain_frequency': 'daily'
        }
    },
    'system_architect': {
        'enabled': True,
        'update_frequency': '1hour',
        'parameters': {
            'analysis_scope': 'standard',
            'export_formats': ['json', 'html'],
            'quality_gates': {
                'health_score_threshold': 75,
                'security_score_threshold': 80
            }
        }
    }
}
```

## 📝 **Error Handling and Exceptions**

### **Standard Exception Classes**

```python
# Custom exception hierarchy
class TradingSystemException(Exception):
    """Base exception for trading system"""
    pass

class AgentException(TradingSystemException):
    """Base exception for agent-related errors"""
    pass

class DataException(AgentException):
    """Data-related exceptions"""
    pass

class ModelException(AgentException):
    """Model-related exceptions"""
    pass

class RiskException(AgentException):
    """Risk management exceptions"""
    pass

class ConfigurationException(TradingSystemException):
    """Configuration-related exceptions"""
    pass

# Usage in agents
async def process_data(self, data):
    try:
        # Processing logic
        pass
    except DataException as e:
        self.logger.error(f"Data processing failed: {e}")
        raise
    except Exception as e:
        self.logger.error(f"Unexpected error: {e}")
        raise AgentException(f"Agent {self.name} processing failed") from e
```

## 🔍 **Logging and Monitoring**

### **Structured Logging**

```python
import structlog

# Configure structured logging for all agents
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

# Usage in agents
class MyAgent(BaseAgent):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.logger = structlog.get_logger(name)
    
    async def process(self, data):
        self.logger.info("Processing started", 
                        data_size=len(data), 
                        agent=self.name)
        
        try:
            result = await self._internal_process(data)
            self.logger.info("Processing completed", 
                           result_count=len(result),
                           execution_time=processing_time)
            return result
        except Exception as e:
            self.logger.error("Processing failed", 
                            error=str(e),
                            error_type=type(e).__name__)
            raise
```

## 🧪 **Testing Framework**

### **Agent Testing Base Class**

```python
import pytest
from unittest.mock import AsyncMock, MagicMock

class BaseAgentTest:
    """Base test class for all agents"""
    
    @pytest.fixture
    def mock_config(self):
        return {
            'parameter1': 'value1',
            'parameter2': 42,
            'logging': {'level': 'DEBUG'}
        }
    
    @pytest.fixture
    def mock_data(self):
        # Return test data appropriate for the agent
        pass
    
    @pytest.fixture
    async def agent(self, mock_config):
        # Create agent instance with mock config
        pass
    
    async def test_agent_initialization(self, agent):
        assert agent.name is not None
        assert agent.config is not None
        assert not agent.is_active
    
    async def test_agent_start_stop(self, agent):
        assert await agent.start()
        assert agent.is_active
        assert await agent.stop()
        assert not agent.is_active
    
    async def test_agent_process(self, agent, mock_data):
        # Test the main processing logic
        pass

# Example usage
class TestDataUniverseAgent(BaseAgentTest):
    @pytest.fixture
    async def agent(self, mock_config):
        return DataUniverseAgent("test_data_agent", mock_config)
    
    @pytest.fixture
    def mock_data(self):
        return pd.DataFrame({
            'symbol': ['AAPL', 'GOOGL', 'MSFT'],
            'price': [150.0, 2800.0, 300.0],
            'timestamp': pd.to_datetime(['2024-01-01', '2024-01-01', '2024-01-01'])
        })
    
    async def test_data_ingestion(self, agent, mock_data):
        result = await agent.ingest_market_data('test_source', ['AAPL', 'GOOGL'])
        assert result['status'] == 'success'
        assert len(result['symbols']) == 2
```

## 📈 **Performance Monitoring**

### **Agent Performance Metrics**

```python
import time
from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class PerformanceMetrics:
    """Performance metrics for agent monitoring"""
    agent_name: str
    operation: str
    start_time: float
    end_time: float
    success: bool
    data_size: int = 0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    
    @property
    def execution_time(self) -> float:
        return self.end_time - self.start_time

class PerformanceMonitor:
    """Monitor agent performance and health"""
    
    def __init__(self):
        self.metrics: List[PerformanceMetrics] = []
    
    def record_operation(self, 
                        agent_name: str, 
                        operation: str, 
                        execution_time: float,
                        success: bool,
                        **kwargs):
        """Record operation performance"""
        metric = PerformanceMetrics(
            agent_name=agent_name,
            operation=operation,
            start_time=time.time() - execution_time,
            end_time=time.time(),
            success=success,
            **kwargs
        )
        self.metrics.append(metric)
    
    def get_agent_stats(self, agent_name: str) -> Dict[str, Any]:
        """Get performance statistics for an agent"""
        agent_metrics = [m for m in self.metrics if m.agent_name == agent_name]
        
        if not agent_metrics:
            return {}
        
        return {
            'total_operations': len(agent_metrics),
            'success_rate': sum(m.success for m in agent_metrics) / len(agent_metrics),
            'average_execution_time': sum(m.execution_time for m in agent_metrics) / len(agent_metrics),
            'max_execution_time': max(m.execution_time for m in agent_metrics),
            'min_execution_time': min(m.execution_time for m in agent_metrics)
        }

# Usage in agents
class MonitoredAgent(BaseAgent):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.performance_monitor = PerformanceMonitor()
    
    async def process(self, data):
        start_time = time.time()
        success = False
        
        try:
            result = await self._internal_process(data)
            success = True
            return result
        finally:
            execution_time = time.time() - start_time
            self.performance_monitor.record_operation(
                agent_name=self.name,
                operation='process',
                execution_time=execution_time,
                success=success,
                data_size=len(data) if hasattr(data, '__len__') else 0
            )
```

---

## 🎯 **Quick Reference**

### **Common Usage Patterns**

```python
# 1. Initialize and start an agent
agent = DataUniverseAgent("market_data", config)
await agent.start()

# 2. Process data
result = await agent.process(input_data)

# 3. Get agent status
status = agent.get_status()

# 4. Stop agent
await agent.stop()

# 5. Run System Architect analysis
coordinator = MasterCoordinator(config)
analysis = await coordinator.analyze_system("/path/to/project")

# 6. Generate recommendations
rec_agent = RecommendationAgent(config)
recommendations = await rec_agent.generate_trade_recommendations(analysis, portfolio)
```

### **Configuration Templates**

```python
# Minimal configuration
MINIMAL_CONFIG = {
    'enabled': True,
    'parameters': {}
}

# Production configuration
PRODUCTION_CONFIG = {
    'enabled': True,
    'update_frequency': '1min',
    'parameters': {...},
    'logging': {'level': 'INFO'},
    'monitoring': {'health_checks': True},
    'risk_limits': {...}
}
```

---

**Last Updated:** July 27, 2025  
**Version:** 2.0.0  
**Status:** Production Ready ✅