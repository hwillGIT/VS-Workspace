# ADR-002: Global API Key Management with Automatic Failover

## Status
Proposed - Pending Implementation (2025-07-30)

## Context and Problem Statement

The ClaudeCode trading system operates across multiple financial data providers and trading platforms, each requiring API keys for access. The current architecture faces several critical challenges in API key management:

### Core Problems
1. **Rate Limiting and Service Failures**: Financial APIs frequently impose strict rate limits and experience outages, causing system-wide failures when primary providers become unavailable
2. **Scattered API Key Management**: API keys are managed individually by each agent/component, leading to inconsistent handling and lack of coordination
3. **No Failover Mechanism**: When a primary API provider fails or reaches rate limits, the system cannot automatically switch to backup providers
4. **Reliability Issues**: Trading systems require 99.9%+ uptime, but single points of failure in API providers create unacceptable downtime
5. **Lack of Usage Coordination**: Multiple system components may simultaneously exhaust rate limits without awareness of each other's usage
6. **Security Concerns**: API keys scattered throughout codebase increase exposure risk and make rotation difficult

### Business Impact
- **Trading Opportunity Loss**: System downtime during market hours results in missed profitable trades
- **Risk Management Failures**: Inability to access real-time data compromises risk monitoring capabilities
- **Operational Overhead**: Manual intervention required when API providers fail
- **Compliance Issues**: Financial regulations require robust system availability and monitoring

### Current Architecture Limitations
The existing `SecureConfigManager` in `config/security_config.py` provides basic API key retrieval but lacks:
- Automatic failover between providers
- Rate limit awareness and management
- Global coordination of API usage
- State persistence across system restarts
- Dynamic provider health monitoring

## Decision Drivers and Constraints

### Technical Drivers
- **High Availability**: Trading systems must maintain >99.9% uptime
- **Automatic Recovery**: System must handle provider failures without manual intervention
- **Rate Limit Management**: Intelligent distribution of API calls across available providers
- **Global Coordination**: All system components must share provider status and usage information
- **Security**: Maintain or improve current security posture for API key management

### Business Constraints
- **Minimal Disruption**: Implementation must not require changes to existing agent code
- **Backward Compatibility**: Current API key access patterns must continue to work
- **Cost Control**: Avoid unnecessary API usage charges from provider switching
- **Regulatory Compliance**: Maintain audit trails and secure key management

### Technical Constraints
- **Python Ecosystem**: Must integrate seamlessly with existing Python infrastructure
- **Environment Variable Compatibility**: Must work with current `os.getenv()` usage patterns
- **Performance**: Cannot introduce latency in high-frequency trading operations
- **State Management**: Must handle system restarts and maintain provider health state

## Considered Options

### Option 1: Per-Agent Failover Management
**Description**: Each trading agent implements its own failover logic

**Pros:**
- Distributed responsibility
- Agent-specific optimization possible
- No global coordination complexity

**Cons:**
- Code duplication across agents
- Inconsistent failover behavior
- No global rate limit awareness
- Higher maintenance overhead
- Potential for simultaneous provider exhaustion

### Option 2: Dependency Injection Pattern
**Description**: Inject API provider interfaces into each component

**Pros:**
- Clean separation of concerns
- Testable architecture
- Flexible provider switching

**Cons:**
- Requires significant refactoring of existing code
- Complex configuration management
- No automatic environment variable interception
- Breaking changes to existing agents

### Option 3: Configuration-Based Provider Management
**Description**: YAML/JSON configuration files defining provider hierarchies

**Pros:**
- Easy to modify provider priorities
- Clear configuration structure
- No code changes required

**Cons:**
- Static configuration (no dynamic health checking)
- No automatic rate limit management
- Requires configuration deployment for changes
- Limited real-time adaptation

### Option 4: Global Singleton with Environment Variable Interception (CHOSEN)
**Description**: Singleton-based global manager that intercepts `os.getenv()` calls

**Pros:**
- Zero code changes required in existing agents
- Transparent failover implementation
- Global coordination and state management
- Automatic rate limit and health monitoring
- Maintains current security patterns

**Cons:**
- Higher implementation complexity
- Global state management challenges
- Potential for singleton anti-pattern issues
- Monkey patching concerns

### Option 5: Microservice-Based API Gateway
**Description**: Separate service handling all API provider interactions

**Pros:**
- Complete separation of concerns
- Scalable architecture
- Language agnostic

**Cons:**
- Significant architectural change required
- Network latency introduction
- Additional infrastructure complexity
- Operational overhead

## Decision Outcome and Rationale

**Selected Option 4: Global Singleton with Environment Variable Interception**

### Key Components to Implement

#### 1. MinimalMultiKeyManager (`core/api/minimal_multi_key_manager.py`)
```python
# Core singleton manager with automatic failover
- Global state management for all API providers
- Automatic failover logic with health checking
- Rate limit tracking and enforcement
- State persistence for system restarts
- Environment variable interception via os.getenv() replacement
```

#### 2. Provider Health Monitoring
```python
# Real-time provider health assessment
- API response time monitoring
- Error rate tracking
- Rate limit status monitoring
- Automatic provider ranking updates
```

#### 3. State Persistence Layer
```python
# Persistent storage of provider health and usage
- Redis-backed state storage
- Graceful degradation when Redis unavailable
- Automatic state recovery on system restart
```

#### 4. Environment Variable Interception
```python
# Transparent integration with existing code
- Monkey patching of os.getenv() for API key requests
- Intelligent provider selection based on current health
- Fallback to original environment variables when needed
```

### Implementation Strategy
1. **Phase 1**: Implement core MinimalMultiKeyManager with basic failover
2. **Phase 2**: Add state persistence and health monitoring
3. **Phase 3**: Implement environment variable interception
4. **Phase 4**: Add advanced features (rate limiting, monitoring dashboards)

### Integration Points
- **Existing SecureConfigManager**: Enhanced to work with new multi-key manager
- **All Trading Agents**: Transparent integration via environment variable interception
- **Configuration System**: Extended to support multiple providers per service
- **Monitoring System**: New metrics for provider health and failover events

## Positive Consequences

### Immediate Benefits
1. **Improved System Reliability**
   - Automatic failover eliminates single points of failure
   - System continues operating even when primary providers fail
   - Reduced manual intervention requirements

2. **Enhanced Rate Limit Management**
   - Intelligent distribution of API calls across providers
   - Automatic throttling when approaching limits
   - Prevention of simultaneous rate limit exhaustion

3. **Zero-Code-Change Integration**
   - Existing agents continue working without modification
   - Current `os.getenv()` patterns maintained
   - Backward compatibility preserved

4. **Global Coordination**
   - All system components share provider health information
   - Coordinated failover decisions across the entire system
   - Centralized monitoring and alerting

### Long-term Benefits
1. **Operational Excellence**
   - Reduced system downtime
   - Automated recovery from provider issues
   - Comprehensive monitoring and alerting

2. **Cost Optimization**
   - Efficient use of API quotas across providers
   - Reduced unnecessary API calls during failures
   - Better provider cost management

3. **Scalability**
   - Easy addition of new API providers
   - Dynamic provider priority adjustment
   - Horizontal scaling of API capacity

4. **Security Enhancement**
   - Centralized API key management
   - Easier key rotation procedures
   - Improved audit trail for API usage

## Negative Consequences

### Short-term Challenges
1. **Implementation Complexity**
   - Singleton pattern requires careful thread-safety implementation
   - Environment variable interception adds complexity
   - State management across system components

2. **Testing Challenges**
   - Global state makes unit testing more complex
   - Need to test failover scenarios thoroughly
   - Integration testing with multiple providers required

3. **Debugging Complexity**
   - Transparent failover may mask underlying provider issues
   - More complex troubleshooting when issues occur
   - Need for comprehensive logging and monitoring

### Long-term Considerations
1. **Maintenance Overhead**
   - Global singleton requires careful maintenance
   - Provider interface changes need coordination
   - State persistence layer adds operational complexity

2. **Performance Monitoring**
   - Need to monitor impact on API call latency
   - Memory usage of global state management
   - Potential contention issues in high-frequency scenarios

3. **Architectural Debt**
   - Singleton pattern may limit future architectural flexibility
   - Monkey patching concerns for long-term maintainability
   - Global state management anti-patterns

## Implementation Details

### Core Architecture

#### MinimalMultiKeyManager Design
```python
class MinimalMultiKeyManager:
    """
    Global singleton managing API keys with automatic failover
    
    Features:
    - Thread-safe singleton implementation
    - Provider health monitoring and ranking
    - Automatic failover logic
    - State persistence via Redis
    - Rate limit tracking and enforcement
    """
    
    def __init__(self):
        self._providers = {}  # service -> [provider_configs]
        self._health_status = {}  # provider -> health_metrics
        self._usage_tracking = {}  # provider -> usage_stats
        self._state_persistence = StateManager()
    
    def get_api_key(self, service: str) -> str:
        """Get API key with automatic failover"""
        # 1. Get available providers for service
        # 2. Select best provider based on health/usage
        # 3. Return API key for selected provider
        # 4. Update usage tracking
    
    def _monitor_provider_health(self, provider: str):
        """Monitor provider health in background"""
        # Health check implementation
    
    def _handle_provider_failure(self, provider: str, service: str):
        """Handle provider failure and trigger failover"""
        # Failover logic implementation
```

#### Environment Variable Interception
```python
# Patch os.getenv to intercept API key requests
original_getenv = os.getenv

def patched_getenv(key: str, default=None):
    """Patched getenv with intelligent API key failover"""
    if key in API_KEY_PATTERNS:
        service = extract_service_from_key(key)
        return MinimalMultiKeyManager.instance().get_api_key(service)
    return original_getenv(key, default)

os.getenv = patched_getenv
```

#### State Persistence Design
```python
class StateManager:
    """Handles persistent state for provider health and usage"""
    
    def __init__(self):
        self.redis_client = redis.Redis(...)
        self.fallback_storage = {}
    
    def save_provider_health(self, provider: str, health_data: dict):
        """Save provider health data"""
        # Redis implementation with fallback
    
    def load_provider_health(self) -> dict:
        """Load provider health data"""
        # Recovery implementation
```

### Configuration Extensions

#### Enhanced Environment Variables
```bash
# Primary providers
ALPHA_VANTAGE_API_KEY=primary_key_here
YAHOO_FINANCE_API_KEY=backup_key_here

# Backup providers (automatic discovery)
ALPHA_VANTAGE_API_KEY_BACKUP_1=backup1_key_here
ALPHA_VANTAGE_API_KEY_BACKUP_2=backup2_key_here

# Provider priority and health settings
API_PROVIDER_PRIORITIES='{"market_data": ["alpha_vantage", "yahoo_finance", "quandl"]}'
API_HEALTH_CHECK_INTERVAL=60
API_FAILOVER_THRESHOLD=0.95
```

#### Configuration Schema
```yaml
# Enhanced config.yaml with multi-provider support
api_management:
  global_failover: true
  health_monitoring: true
  state_persistence: true
  
  providers:
    market_data:
      primary: alpha_vantage
      backups: [yahoo_finance, quandl]
      failover_threshold: 0.95
      health_check_interval: 60
      
    crypto:
      primary: binance
      backups: [coinbase, kraken]
      failover_threshold: 0.90
      health_check_interval: 30
      
  rate_limiting:
    intelligent_throttling: true
    cross_provider_coordination: true
    usage_analytics: true
```

### Security Enhancements

#### Secure Key Storage
```python
class SecureKeyVault:
    """Enhanced security for multi-provider key management"""
    
    def __init__(self):
        self.encryption_key = get_config_manager().get_encryption_key()
        self.key_store = {}
    
    def store_key(self, provider: str, service: str, api_key: str):
        """Store encrypted API key"""
        encrypted_key = self._encrypt(api_key)
        self.key_store[f"{service}:{provider}"] = encrypted_key
    
    def retrieve_key(self, provider: str, service: str) -> str:
        """Retrieve and decrypt API key"""
        encrypted_key = self.key_store.get(f"{service}:{provider}")
        return self._decrypt(encrypted_key) if encrypted_key else None
```

#### Audit Trail Implementation
```python
class APIUsageAuditor:
    """Comprehensive audit trail for API key usage"""
    
    def log_api_call(self, provider: str, service: str, endpoint: str, status: str):
        """Log API call for compliance and monitoring"""
        audit_record = {
            'timestamp': datetime.utcnow(),
            'provider': provider,
            'service': service,
            'endpoint': endpoint,
            'status': status,
            'user_id': get_current_user(),
            'session_id': get_session_id()
        }
        self._write_audit_log(audit_record)
```

## Monitoring and Alerting

### Key Metrics
- **Provider Health Score**: Real-time health rating for each provider
- **Failover Events**: Count and frequency of automatic failovers
- **Rate Limit Utilization**: Percentage of rate limits used per provider
- **API Response Times**: Latency monitoring across all providers
- **Error Rates**: Failed API calls per provider
- **Cost Tracking**: API usage costs per provider

### Alert Conditions
- Provider health score drops below threshold
- Automatic failover event occurs
- Rate limit exceeds 80% utilization
- Multiple provider failures detected
- Unusual API usage patterns detected

### Dashboard Components
- Real-time provider health status
- API usage analytics and trends
- Failover history and patterns
- Cost analysis and optimization recommendations
- Security audit trail and compliance reports

## Testing Strategy

### Unit Testing
- MinimalMultiKeyManager singleton behavior
- Provider health monitoring algorithms
- Failover decision logic
- State persistence operations
- Security key encryption/decryption

### Integration Testing
- Multi-provider failover scenarios
- Environment variable interception
- State persistence across system restarts
- Real provider API health checks
- Performance under load conditions

### Chaos Engineering
- Simulated provider outages
- Network partition scenarios
- Rate limit exhaustion testing
- Concurrent access stress testing
- State corruption recovery testing

## Security Implications and Mitigation Strategies

### Security Risks
1. **Centralized Key Storage**: All API keys managed by single component
2. **Monkey Patching**: Environment variable interception could be exploited
3. **State Persistence**: Provider health data stored in Redis
4. **Network Communication**: Health checks expose provider endpoints

### Mitigation Strategies
1. **Enhanced Encryption**
   - All API keys encrypted at rest using AES-256
   - Separate encryption keys for different provider types
   - Regular key rotation procedures implemented

2. **Access Control**
   - Role-based access to API key management
   - Audit trail for all key access operations
   - Secure communication channels for health checks

3. **Runtime Security**
   - Input validation for all API key operations
   - Secure defaults for configuration settings
   - Regular security scanning of the global manager

4. **Monitoring and Detection**
   - Anomaly detection for unusual API usage patterns
   - Real-time alerts for security violations
   - Comprehensive logging of all security events

## Follow-up Actions

### Phase 1: Core Implementation (Weeks 1-2)
- [ ] Implement MinimalMultiKeyManager singleton
- [ ] Basic provider health monitoring
- [ ] Simple failover logic
- [ ] Unit tests for core functionality
- [ ] Integration with existing SecureConfigManager

### Phase 2: Advanced Features (Weeks 3-4)
- [ ] State persistence via Redis
- [ ] Environment variable interception
- [ ] Rate limit tracking and management
- [ ] Enhanced security features
- [ ] Integration testing with real providers

### Phase 3: Monitoring and Operations (Weeks 5-6)
- [ ] Comprehensive logging and metrics
- [ ] Monitoring dashboard implementation
- [ ] Alert system integration
- [ ] Performance optimization
- [ ] Documentation and runbooks

### Phase 4: Production Deployment (Weeks 7-8)
- [ ] Staging environment testing
- [ ] Production deployment planning
- [ ] Rollback procedures
- [ ] Team training and knowledge transfer
- [ ] Post-deployment monitoring and optimization

## Success Criteria

### Technical Metrics
- **System Uptime**: >99.9% availability during market hours
- **Failover Speed**: <30 seconds average failover time
- **API Success Rate**: >99.5% successful API calls across all providers
- **Zero Breaking Changes**: All existing agents continue functioning without modification

### Quality Metrics
- **Test Coverage**: >95% code coverage for all manager components
- **Security Compliance**: Pass all security audits and penetration tests
- **Performance Impact**: <100ms additional latency for API key retrieval
- **Error Rate**: <0.1% errors in provider selection and failover

### Business Metrics
- **Reduced Downtime**: 90% reduction in API-related system outages
- **Operational Efficiency**: 50% reduction in manual intervention for API issues
- **Cost Optimization**: 20% improvement in API quota utilization
- **Incident Response**: 75% faster resolution of API-related incidents

## Related ADRs
- ADR-001: Functional Programming Integration for Enhanced Reliability
- ADR-003: [Planned] Comprehensive Error Handling Strategy
- ADR-004: [Planned] Real-time Monitoring and Alerting Framework
- ADR-005: [Planned] Security Architecture Enhancement

## References
- [Singleton Pattern Best Practices](https://refactoring.guru/design-patterns/singleton)
- [API Rate Limiting Strategies](https://cloud.google.com/architecture/rate-limiting-strategies-techniques)
- [Financial System Reliability Requirements](https://www.federalreserve.gov/paymentsystems/regcc-interp.htm)
- [Environment Variable Security](https://owasp.org/www-community/vulnerabilities/Storing_passwords_in_a_recoverable_format)
- [ClaudeCode Security Practices](../SECURITY_PRACTICES.md)
- [Trading System Configuration Guide](../config/README.md)

---
**Author**: System Architect Agent  
**Date**: 2025-07-30  
**Reviewers**: [To be assigned]  
**Implementation Status**: Proposed  
**Last Updated**: 2025-07-30  
**Security Review**: Pending  
**Performance Review**: Pending