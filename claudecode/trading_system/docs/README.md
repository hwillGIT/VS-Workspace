# ClaudeCode Trading System Documentation

This directory contains comprehensive documentation for the ClaudeCode Trading System architecture, design decisions, and implementation details.

## Documentation Structure

### Architecture Decision Records (ADRs)
- [ADR Index](./adrs/README.md) - Track all architectural decisions
- [ADR-001: Functional Programming Integration](./adrs/ADR-001-functional-programming-integration.md) - Latest architectural decision

### Architecture Documentation
- [System Architecture Overview](../COMPLETE_SYSTEM_OVERVIEW.md)
- [Implementation Summary](../IMPLEMENTATION_SUMMARY.md)
- [System Status](../SYSTEM_STATUS.md)
- [Coding Standards](../CODING_STANDARDS.md)

### Agent Documentation
- [System Architect Suite](../agents/system_architect/README.md) - Comprehensive architecture tools
- [Agent Implementation Guides](../agents/) - Individual agent documentation

### API Documentation
- [Core APIs](../core/apis/) - Market data, options, fundamental data APIs
- [Agent APIs](../agents/) - Individual agent interfaces

### Examples and Tutorials
- [Examples Directory](../examples/) - Working code examples
- [Quickstart Guide](../scripts/quickstart.py) - Getting started

## Key Recent Changes

### 2025-07-30: Functional Programming Integration
- Added comprehensive functional programming utilities at ClaudeCode global level
- Enhanced mathematical operations with functional patterns
- Implemented functional signal processing for synthesis agent
- Maintained full backward compatibility

See [ADR-001](./adrs/ADR-001-functional-programming-integration.md) for complete details.

## Navigation

### For Developers
- Start with [System Architecture Overview](../COMPLETE_SYSTEM_OVERVIEW.md)
- Review [Coding Standards](../CODING_STANDARDS.md)
- Check [Implementation Summary](../IMPLEMENTATION_SUMMARY.md)

### For Architects
- Review [Architecture Decision Records](./adrs/README.md)
- Explore [System Architect Tools](../agents/system_architect/README.md)
- Check [Dependency Analysis](../agents/system_architect/dependency_analysis_agent.py)

### For System Integrators
- Check [API Documentation](../core/apis/)
- Review [Configuration](../config/)
- See [Integration Examples](../examples/)

## Contributing to Documentation

When making significant architectural changes:
1. Create an ADR following the [template](./adrs/README.md)
2. Update relevant documentation
3. Add examples if applicable
4. Update this README if structure changes

## Tools Available

The system includes automated documentation tools:
- **ADR Manager**: Create and manage architecture decisions
- **Documentation Agent**: Generate comprehensive documentation
- **Architecture Diagram Manager**: Create and maintain diagrams
- **Dependency Analysis**: Track system dependencies

See [System Architect Suite](../agents/system_architect/README.md) for details.