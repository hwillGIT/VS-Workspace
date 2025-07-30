# ADR-003: Singleton Pattern for Global API Key Management

## Status
Proposed - Pending Implementation (2025-07-30)

## Context and Problem Statement

Following the decision in ADR-002 to implement global API key management with automatic failover through the MinimalMultiKeyManager, we need to determine the appropriate design pattern for ensuring single instance coordination across the application.

### Core Requirements for Single Instance Management
1. **Global State Coordination**: All system components must share the same provider health status, rate limit tracking, and failover decisions
2. **Resource Management**: API connections, Redis state persistence, and background health monitoring threads require coordinated lifecycle management
3. **Thread Safety**: High-frequency trading operations require concurrent access to API key retrieval without data races
4. **Memory Efficiency**: Single instance should minimize memory overhead for health monitoring and state caching
5. **Initialization Control**: Complex initialization sequence involving provider discovery, state recovery, and health check setup

### Current Architecture Context
The ClaudeCode trading system operates with:
- **11 independent trading agents** accessing APIs concurrently
- **Multiple API providers per service** requiring coordinated health tracking
- **Real-time market data streams** demanding low-latency API key resolution
- **State persistence requirements** for provider health across system restarts
- **Background monitoring threads** for continuous provider health assessment

### Decision Drivers
1. **Zero Code Changes**: Existing agents must continue working with current `os.getenv()` patterns
2. **Performance Requirements**: Sub-100ms API key resolution in high-frequency scenarios
3. **Reliability**: Failover decisions must be consistent across all system components
4. **Resource Efficiency**: Single background monitoring process for all providers
5. **Testing Complexity**: Pattern must support effective unit and integration testing

## Problem Statement

**How should we implement the MinimalMultiKeyManager to ensure single instance coordination while maintaining thread safety, testability, and performance requirements?**

### Specific Challenges
1. **Instance Coordination**: Multiple components attempting to create manager instances simultaneously
2. **Initialization Race Conditions**: Complex startup sequence with provider discovery and state recovery
3. **Thread Safety**: Concurrent API key requests from multiple trading agents
4. **Testing Isolation**: Unit tests need ability to reset/mock singleton state
5. **Memory Management**: Background threads and persistent connections lifecycle
6. **Error Recovery**: Handling initialization failures and state corruption gracefully

## Considered Options

### Option 1: Thread-Safe Singleton with Lazy Loading (CHOSEN)
**Description**: Classic singleton with thread-safe lazy initialization using `__new__` method

```python
class MinimalMultiKeyManager:
    _instance = None
    _lock = threading.Lock()
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    self._setup_providers()
                    self._initialize_state_persistence()
                    self._start_background_monitoring()
                    MinimalMultiKeyManager._initialized = True
```

**Pros:**
- **Thread Safety**: Double-checked locking prevents race conditions
- **Lazy Loading**: Instance created only when first needed
- **Memory Efficient**: Single instance across entire application
- **Performance**: Fast access after initialization
- **Simple Implementation**: Well-understood pattern

**Cons:**
- **Testing Challenges**: Global state complicates unit tests
- **Initialization Complexity**: Complex startup sequence in constructor
- **Hidden Dependencies**: Singleton access masks component dependencies
- **Memory Leaks**: Instance persists for application lifetime

### Option 2: Dependency Injection with Container
**Description**: Use dependency injection framework to manage single instance

```python
# Using dependency injection container
@singleton
class MinimalMultiKeyManager:
    def __init__(self, config_manager: ConfigManager, 
                 state_persistence: StateManager):
        self._config = config_manager
        self._state = state_persistence
        self._setup_providers()

# Usage in agents
class TradingAgent:
    def __init__(self, api_manager: MinimalMultiKeyManager):
        self._api_manager = api_manager
```

**Pros:**
- **Explicit Dependencies**: Clear component relationships
- **Testability**: Easy to inject mocks for testing
- **Lifecycle Management**: Container handles instance creation/destruction
- **Configuration Flexibility**: Different instances for testing/production

**Cons:**
- **Breaking Changes**: Requires modifying all existing agents
- **Framework Dependency**: Additional dependency injection library
- **Complex Configuration**: Container setup and wiring complexity
- **No Environment Variable Interception**: Cannot transparently replace `os.getenv()`

### Option 3: Module-Level Singleton
**Description**: Use Python module-level variables for singleton behavior

```python
# api_manager.py
_manager_instance = None

def get_api_manager():
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = MinimalMultiKeyManager()
    return _manager_instance

class MinimalMultiKeyManager:
    def __init__(self):
        # Normal initialization
        pass
```

**Pros:**
- **Simple Implementation**: Straightforward module-level pattern
- **No Class Complexity**: Regular class without singleton machinery
- **Easy Testing**: Can reset module variable in tests
- **Python Idiomatic**: Follows Python conventions

**Cons:**
- **Thread Safety Issues**: No built-in thread safety
- **Global State**: Still maintains global state problems
- **Import Dependencies**: Requires consistent import patterns
- **No Lazy Loading**: Instance may be created unnecessarily

### Option 4: Factory Pattern with Registry
**Description**: Factory creates and manages instances in internal registry

```python
class APIManagerFactory:
    _instances = {}
    _lock = threading.Lock()
    
    @classmethod
    def get_manager(cls, config_name='default'):
        if config_name not in cls._instances:
            with cls._lock:
                if config_name not in cls._instances:
                    cls._instances[config_name] = MinimalMultiKeyManager(config_name)
        return cls._instances[config_name]

# Usage
manager = APIManagerFactory.get_manager()
```

**Pros:**
- **Multiple Configurations**: Support for different manager configurations
- **Factory Benefits**: Centralized creation logic
- **Registry Pattern**: Clean instance management
- **Testing Support**: Different instances for different test scenarios

**Cons:**
- **Additional Complexity**: Factory layer adds indirection
- **Memory Overhead**: Multiple instances possible
- **Configuration Management**: Complex configuration selection logic
- **Inconsistent Access**: Different ways to access same functionality

### Option 5: Context Manager Pattern
**Description**: Use context manager to control instance lifecycle

```python
class APIManagerContext:
    def __enter__(self):
        if not hasattr(self, '_manager'):
            self._manager = MinimalMultiKeyManager()
        return self._manager
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Cleanup if needed
        pass

# Usage
with APIManagerContext() as manager:
    api_key = manager.get_api_key('market_data')
```

**Pros:**
- **Explicit Lifecycle**: Clear instance creation and cleanup
- **Resource Management**: Automatic cleanup when done
- **Testing Friendly**: Scoped instance per test
- **Exception Safety**: Cleanup guaranteed even on errors

**Cons:**
- **Usage Complexity**: Requires context manager in all usage
- **Performance Overhead**: Context setup/teardown cost
- **Global State Still**: Same global state issues within context
- **Breaking Changes**: Significant changes to current usage patterns

## Decision Outcome

**Selected Option 1: Thread-Safe Singleton with Lazy Loading**

### Rationale

The thread-safe singleton pattern best aligns with our requirements for the following reasons:

1. **Zero Breaking Changes**: Maintains current `os.getenv()` usage patterns through transparent environment variable interception
2. **Performance Optimized**: Single instance with fast access after initialization meets high-frequency trading requirements
3. **Resource Efficiency**: Single background monitoring process and shared state across all components
4. **Thread Safety**: Double-checked locking ensures safe concurrent access from all trading agents
5. **Implementation Simplicity**: Well-understood pattern with clear implementation path

### Key Implementation Details

#### Thread-Safe Singleton Implementation
```python
import threading
from typing import Optional, Dict, Any

class MinimalMultiKeyManager:
    """
    Thread-safe singleton for global API key management with automatic failover.
    
    Features:
    - Double-checked locking for thread safety
    - Lazy initialization with complex startup sequence
    - Global state coordination across all system components
    - Background health monitoring with shared resources
    """
    
    _instance: Optional['MinimalMultiKeyManager'] = None
    _lock: threading.Lock = threading.Lock()
    _initialized: bool = False
    
    def __new__(cls) -> 'MinimalMultiKeyManager':
        """Thread-safe singleton instance creation."""
        if cls._instance is None:
            with cls._lock:
                # Double-checked locking pattern
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize manager only once with thread safety."""
        if not MinimalMultiKeyManager._initialized:
            with MinimalMultiKeyManager._lock:
                # Double-checked initialization
                if not MinimalMultiKeyManager._initialized:
                    self._initialize_manager()
                    MinimalMultiKeyManager._initialized = True
    
    def _initialize_manager(self):
        """Complex initialization sequence for manager setup."""
        # Provider discovery and configuration
        self._providers: Dict[str, Dict] = {}
        self._health_status: Dict[str, Dict] = {}
        self._usage_tracking: Dict[str, Dict] = {}
        
        # State persistence setup
        self._state_manager = self._setup_state_persistence()
        
        # Background monitoring threads
        self._monitoring_threads: Dict[str, threading.Thread] = {}
        self._shutdown_event = threading.Event()
        
        # Initialize core components
        self._discover_providers()
        self._recover_state()
        self._start_health_monitoring()
        
        # Environment variable interception setup
        self._setup_env_var_interception()
    
    @classmethod
    def instance(cls) -> 'MinimalMultiKeyManager':
        """Get singleton instance (alternative access method)."""
        return cls()
    
    @classmethod
    def reset_instance(cls):
        """Reset singleton instance (for testing purposes only)."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance._shutdown()
            cls._instance = None
            cls._initialized = False
```

#### Initialization Sequence Management
```python
def _initialize_manager(self):
    """Comprehensive initialization with error handling."""
    try:
        # Phase 1: Basic setup
        self._setup_data_structures()
        
        # Phase 2: External dependencies
        self._state_manager = self._setup_state_persistence()
        
        # Phase 3: Provider discovery
        self._discover_providers()
        
        # Phase 4: State recovery
        self._recover_previous_state()
        
        # Phase 5: Background services
        self._start_health_monitoring()
        
        # Phase 6: Integration setup
        self._setup_env_var_interception()
        
        logging.info("MinimalMultiKeyManager initialized successfully")
        
    except Exception as e:
        logging.error(f"Failed to initialize MinimalMultiKeyManager: {e}")
        # Ensure clean state on initialization failure
        self._cleanup_partial_initialization()
        raise
```

#### Thread Safety for API Operations
```python
def get_api_key(self, service: str) -> str:
    """Thread-safe API key retrieval with failover."""
    with self._lock:
        try:
            # Select best provider based on current health
            provider = self._select_best_provider(service)
            
            # Update usage tracking
            self._track_api_usage(provider, service)
            
            # Return API key for selected provider
            return self._get_provider_api_key(provider, service)
            
        except Exception as e:
            logging.error(f"API key retrieval failed for {service}: {e}")
            # Trigger failover on error
            return self._handle_failover(service, e)
```

## Positive Consequences

### Immediate Benefits
1. **Simplified Integration**: No changes required to existing trading agents
2. **Performance Optimized**: Single instance with fast concurrent access
3. **Resource Efficiency**: Shared background monitoring and connections
4. **Thread Safety**: Safe concurrent access from all system components
5. **Memory Efficient**: Single instance minimizes memory footprint

### Long-term Benefits
1. **Maintenance Simplicity**: Single point of control for API management
2. **Debugging Efficiency**: Centralized logging and state inspection
3. **Operational Excellence**: Consistent behavior across all components
4. **Scalability**: Easy to extend with additional providers and features

## Negative Consequences

### Testing Challenges
1. **Global State**: Unit tests require careful state management
2. **Test Isolation**: Tests may interfere with each other
3. **Mock Complexity**: Difficult to mock singleton behavior
4. **Integration Testing**: Complex setup for realistic testing scenarios

### Architectural Concerns
1. **Hidden Dependencies**: Components don't explicitly declare API manager dependency
2. **Singleton Anti-Pattern**: Global state considered anti-pattern in some contexts
3. **Lifecycle Management**: Instance persists for entire application lifetime
4. **Flexibility Limitations**: Difficult to change implementation later

### Debugging Complexity
1. **Transparent Behavior**: Environment variable interception may hide issues
2. **State Inspection**: Complex internal state difficult to debug
3. **Initialization Errors**: Complex startup sequence failure diagnosis
4. **Concurrency Issues**: Thread safety bugs difficult to reproduce

## Testing Strategy

### Unit Testing Approach
```python
import unittest
from unittest.mock import patch, MagicMock

class TestMinimalMultiKeyManager(unittest.TestCase):
    """Unit tests for singleton API key manager."""
    
    def setUp(self):
        """Reset singleton state before each test."""
        # Reset singleton instance
        MinimalMultiKeyManager.reset_instance()
        
        # Mock external dependencies
        self.mock_redis = MagicMock()
        self.mock_config = MagicMock()
    
    def tearDown(self):
        """Clean up after each test."""
        # Ensure singleton is reset
        MinimalMultiKeyManager.reset_instance()
    
    @patch('trading_system.core.api.minimal_multi_key_manager.redis.Redis')
    def test_singleton_instance_creation(self, mock_redis):
        """Test singleton instance creation and reuse."""
        # First call creates instance
        manager1 = MinimalMultiKeyManager()
        
        # Second call returns same instance
        manager2 = MinimalMultiKeyManager()
        
        self.assertIs(manager1, manager2)
        self.assertTrue(MinimalMultiKeyManager._initialized)
    
    @patch('threading.Thread')
    def test_thread_safety(self, mock_thread):
        """Test concurrent instance creation is thread-safe."""
        import concurrent.futures
        
        def create_instance():
            return MinimalMultiKeyManager()
        
        # Create multiple instances concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(create_instance) for _ in range(10)]
            instances = [future.result() for future in futures]
        
        # All instances should be the same object
        first_instance = instances[0]
        for instance in instances[1:]:
            self.assertIs(instance, first_instance)
```

### Integration Testing Strategy
```python
class TestAPIManagerIntegration(unittest.TestCase):
    """Integration tests for API manager with real providers."""
    
    def setUp(self):
        """Setup test environment with test API keys."""
        MinimalMultiKeyManager.reset_instance()
        
        # Use test environment variables
        self.test_env = {
            'ALPHA_VANTAGE_API_KEY_TEST': 'test_key_1',
            'YAHOO_FINANCE_API_KEY_TEST': 'test_key_2',
        }
        
    @patch.dict('os.environ', test_env)
    def test_provider_failover_behavior(self):
        """Test automatic failover between providers."""
        manager = MinimalMultiKeyManager()
        
        # Mock provider failure
        with patch.object(manager, '_check_provider_health') as mock_health:
            mock_health.side_effect = [False, True]  # First fails, second succeeds
            
            api_key = manager.get_api_key('market_data')
            
            # Should return backup provider key
            self.assertEqual(api_key, 'test_key_2')
```

### Test Isolation Techniques
```python
class SingletonTestCase(unittest.TestCase):
    """Base test case with singleton reset capabilities."""
    
    def setUp(self):
        """Reset singleton state before each test."""
        self._reset_singleton_state()
        
    def tearDown(self):
        """Clean singleton state after each test."""
        self._reset_singleton_state()
        
    def _reset_singleton_state(self):
        """Comprehensive singleton state reset."""
        # Reset class-level state
        MinimalMultiKeyManager.reset_instance()
        
        # Reset any module-level patches
        if hasattr(MinimalMultiKeyManager, '_original_getenv'):
            import os
            os.getenv = MinimalMultiKeyManager._original_getenv
            
        # Clear any cached data
        import sys
        if 'trading_system.core.api.minimal_multi_key_manager' in sys.modules:
            module = sys.modules['trading_system.core.api.minimal_multi_key_manager']
            if hasattr(module, '_cached_data'):
                module._cached_data.clear()
```

### Mock Strategies for Testing
```python
def create_test_manager_factory():
    """Factory for creating test instances of API manager."""
    
    class TestMinimalMultiKeyManager(MinimalMultiKeyManager):
        """Test version with dependency injection."""
        
        def __init__(self, mock_providers=None, mock_state_manager=None):
            # Skip singleton initialization
            self._providers = mock_providers or {}
            self._state_manager = mock_state_manager or MagicMock()
            self._health_status = {}
            self._usage_tracking = {}
            
        @classmethod
        def reset_instance(cls):
            """Reset without affecting production singleton."""
            pass
    
    return TestMinimalMultiKeyManager
```

## Implementation Plan

### Phase 1: Core Singleton Implementation (Week 1)
- [ ] Implement thread-safe singleton with double-checked locking
- [ ] Add comprehensive initialization sequence
- [ ] Create basic provider management structure
- [ ] Implement instance reset capability for testing
- [ ] Add logging and error handling

### Phase 2: Testing Infrastructure (Week 1-2)
- [ ] Develop singleton testing framework
- [ ] Create test isolation mechanisms
- [ ] Implement mock strategies for dependencies
- [ ] Add thread safety test suite
- [ ] Create integration testing framework

### Phase 3: Production Integration (Week 2-3)
- [ ] Integrate with existing SecureConfigManager
- [ ] Implement environment variable interception
- [ ] Add provider discovery and health monitoring
- [ ] Implement state persistence integration
- [ ] Performance testing and optimization

### Phase 4: Advanced Features (Week 3-4)
- [ ] Add comprehensive error recovery
- [ ] Implement graceful shutdown procedures
- [ ] Add monitoring and metrics collection
- [ ] Create debugging and introspection tools
- [ ] Documentation and examples

## Alternative Implementations Considered

### Metaclass-Based Singleton
```python
class SingletonMeta(type):
    _instances = {}
    _lock = threading.Lock()
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class MinimalMultiKeyManager(metaclass=SingletonMeta):
    pass
```

**Rejected because**: More complex than necessary and harder to test.

### Borg Pattern (Shared State)
```python
class MinimalMultiKeyManager:
    _shared_state = {}
    
    def __init__(self):
        self.__dict__ = self._shared_state
```

**Rejected because**: Less control over initialization and thread safety concerns.

### Decorator-Based Singleton
```python
def singleton(cls):
    instances = {}
    lock = threading.Lock()
    
    def get_instance(*args, **kwargs):
        if cls not in instances:
            with lock:
                if cls not in instances:
                    instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    
    return get_instance

@singleton
class MinimalMultiKeyManager:
    pass
```

**Rejected because**: Decorator approach adds complexity and reduces clarity.

## Future Considerations

### Migration Path from Singleton
If singleton pattern becomes problematic, migration options include:

1. **Dependency Injection Migration**
   - Gradual introduction of DI container
   - Backward compatibility layer during transition
   - Automated refactoring tools for agent updates

2. **Service Locator Pattern**
   - Registry-based service location
   - Interface-based provider access
   - Easier testing and mocking capabilities

3. **Context-Based Management**
   - Application context manages instances
   - Scoped instance lifecycle
   - Better resource management

### Performance Monitoring
Monitor key metrics to validate singleton decision:
- Instance creation overhead
- Memory usage patterns
- Thread contention measurements
- API key retrieval latency
- Failover performance impact

### Security Enhancements
- Regular security reviews of singleton implementation
- Access pattern monitoring and anomaly detection
- Instance state encryption for sensitive data
- Audit trail for singleton access patterns

## Success Criteria

### Technical Success Metrics
- **Thread Safety**: Zero race conditions in concurrent testing
- **Performance**: <10ms overhead for API key retrieval
- **Memory Efficiency**: <50MB total memory usage for manager
- **Reliability**: 99.99% successful instance creation rate

### Quality Metrics
- **Test Coverage**: >95% code coverage for singleton implementation
- **Test Isolation**: Zero test interdependencies
- **Documentation**: Complete testing and usage documentation
- **Code Review**: Approved by senior architects and security team

### Operational Metrics
- **Initialization Success**: 100% successful startup rate
- **Error Recovery**: <30 seconds recovery from initialization failures
- **Monitoring Coverage**: Complete visibility into singleton state
- **Debugging Capability**: Full introspection of manager state

## Related ADRs
- **ADR-002**: Global API Key Management with Automatic Failover (Parent decision)
- **ADR-004**: [Planned] Comprehensive Error Handling Strategy
- **ADR-005**: [Planned] Real-time Monitoring and Alerting Framework
- **ADR-006**: [Planned] Security Architecture Enhancement

## References and Standards
- [Gang of Four Singleton Pattern](https://refactoring.guru/design-patterns/singleton)
- [Python Threading Best Practices](https://docs.python.org/3/library/threading.html)
- [Double-Checked Locking Pattern](https://en.wikipedia.org/wiki/Double-checked_locking)
- [Python Memory Model and Thread Safety](https://docs.python.org/3/faq/library.html#what-kinds-of-global-value-mutation-are-thread-safe)
- [Testing Singleton Patterns](https://martinfowler.com/articles/injection.html)
- [Financial Software Architecture Guidelines](https://www.iso.org/standard/74539.html)

---
**Author**: System Architect Agent  
**Date**: 2025-07-30  
**Reviewers**: [To be assigned]  
**Implementation Status**: Proposed  
**Last Updated**: 2025-07-30  
**Security Review**: Pending  
**Performance Review**: Pending  
**Related Implementation**: MinimalMultiKeyManager (to be implemented)