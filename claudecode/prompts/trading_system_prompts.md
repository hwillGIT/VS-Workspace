# Trading System Specific Prompts

Custom prompts designed for this multi-agent trading system project.

## Code Analysis Prompts

### Trading Agent Analysis
```
Analyze this trading agent code for:
1. Strategy logic correctness
2. Risk management implementation
3. Performance optimization opportunities
4. Integration with other agents
5. Compliance with trading regulations

Consider the multi-agent architecture where this agent interacts with:
- Risk Management Agent
- ML Ensemble Agent
- Technical Analysis Agent
- Signal Synthesis Agent
- Data Universe Agent

[PASTE YOUR AGENT CODE HERE]
```

### System Architecture Review
```
Review this trading system architecture for:
1. Scalability and performance
2. Fault tolerance and reliability
3. Security and compliance
4. Maintainability and modularity
5. Integration patterns between components

This is part of a comprehensive trading system with 11 core agents:
- Data Universe, Technical Analysis, ML Ensemble, Risk Modeling
- Options, Momentum, Statistical Arbitrage, Cross-Asset, Event-Driven
- Signal Synthesis, Recommendation, plus System Architect suite

[PASTE YOUR ARCHITECTURE CODE HERE]
```

## Strategy Development Prompts

### New Strategy Creation
```
Help me design a new trading strategy agent that:
1. Follows the existing agent pattern in this codebase
2. Integrates with the signal synthesis framework
3. Implements proper risk controls
4. Includes comprehensive backtesting capabilities
5. Supports both crypto and traditional assets

Strategy requirements:
[DESCRIBE YOUR STRATEGY REQUIREMENTS]

Reference the existing agent structure at: trading_system/agents/strategies/
```

### Risk Assessment
```
Evaluate the risk characteristics of this trading strategy:
1. Maximum drawdown potential
2. Correlation with existing strategies
3. Market regime sensitivity
4. Liquidity requirements
5. Regulatory compliance

Consider this operates within a multi-strategy framework with portfolio-level risk management.

[PASTE YOUR STRATEGY CODE HERE]
```

## Testing and Validation Prompts

### Test Case Generation
```
Generate comprehensive test cases for this trading component:
1. Unit tests for core functionality
2. Integration tests with other agents
3. Scenario-based backtests
4. Stress testing scenarios
5. Performance benchmarks

Follow the existing test patterns in: trading_system/tests/

Component details:
[DESCRIBE YOUR COMPONENT]
```

### Performance Analysis
```
Analyze the performance characteristics of this trading system:
1. Strategy performance metrics
2. Risk-adjusted returns
3. Transaction cost analysis
4. Slippage and market impact
5. Comparison with benchmarks

This system trades across multiple asset classes with the following agents:
[LIST RELEVANT AGENTS]

Performance data:
[PASTE YOUR PERFORMANCE DATA]
```

## Documentation Prompts

### Technical Documentation
```
Generate technical documentation for this trading system component:
1. Purpose and functionality
2. Input/output specifications
3. Configuration parameters
4. Integration requirements
5. Troubleshooting guide

Target audience: Quantitative developers and portfolio managers
Documentation should follow the existing patterns in the codebase.

Component:
[DESCRIBE YOUR COMPONENT]
```

### API Documentation
```
Create API documentation for this trading system endpoint:
1. Endpoint description and purpose
2. Request/response schemas
3. Authentication requirements
4. Rate limiting and error handling
5. Example usage and code samples

This API is part of a larger trading system with real-time market data integration.

API details:
[PASTE YOUR API CODE]
```

## Deployment and Operations Prompts

### Production Readiness Review
```
Assess this trading system component for production deployment:
1. Error handling and logging
2. Configuration management
3. Monitoring and alerting
4. Disaster recovery capabilities
5. Security considerations

This will operate in a live trading environment with:
- Real-time market data feeds
- Multiple broker connections
- Regulatory compliance requirements
- High availability needs

Component:
[PASTE YOUR COMPONENT CODE]
```

### Troubleshooting Guide
```
Create a troubleshooting guide for this trading system issue:
1. Common symptoms and causes
2. Diagnostic steps and tools
3. Resolution procedures
4. Prevention strategies
5. Escalation criteria

Context: Multi-agent trading system with real-time processing requirements

Issue description:
[DESCRIBE THE ISSUE]
```

---

*These prompts are specifically designed for the trading system project structure and can be customized for your specific needs.*