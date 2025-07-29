# Circular Dependency Prevention Guide

## 🎯 GLOBAL DESIGN RULE
**Zero Tolerance for Circular Dependencies** - All code must follow strict dependency hierarchy with no circular references between modules.

## 📋 Table of Contents
1. [Core Principles](#core-principles)
2. [Design Patterns to Prevent Cycles](#design-patterns)
3. [Detection and Analysis Tools](#detection-tools)
4. [Refactoring Strategies](#refactoring-strategies)
5. [Team Processes](#team-processes)
6. [Implementation Guidelines](#implementation-guidelines)

---

## 🏗️ Core Principles

### 1. Dependency Direction Rule
**Dependencies must flow in ONE direction only**
```
Higher Level → Lower Level → Foundation Level
     ↓              ↓              ↓
   Never ←    Never ←    Never ← 
```

### 2. Layer Architecture
```
┌─────────────────┐
│   UI/API Layer  │ ← Can import from Service & Core
├─────────────────┤
│  Service Layer  │ ← Can import from Core only
├─────────────────┤
│   Core Layer    │ ← No dependencies on upper layers
└─────────────────┘
```

### 3. Module Responsibility
- **Single Responsibility**: Each module has ONE clear purpose
- **Interface Segregation**: Large interfaces split into focused contracts
- **Dependency Inversion**: Depend on abstractions, not concretions

---

## 🎨 Design Patterns to Prevent Cycles

### 1. Dependency Injection Pattern
**Problem**: Module A needs Module B, Module B needs Module A
```python
# ❌ BAD - Circular dependency
# user_service.py
from order_service import OrderService
class UserService:
    def __init__(self):
        self.order_service = OrderService()

# order_service.py  
from user_service import UserService
class OrderService:
    def __init__(self):
        self.user_service = UserService()  # CIRCULAR!
```

**Solution**: Use dependency injection
```python
# ✅ GOOD - Dependency injection
# user_service.py
from abc import ABC, abstractmethod

class OrderServiceInterface(ABC):
    @abstractmethod
    def get_orders(self, user_id: str): pass

class UserService:
    def __init__(self, order_service: OrderServiceInterface):
        self.order_service = order_service

# order_service.py
from user_service import OrderServiceInterface

class OrderService(OrderServiceInterface):
    def __init__(self):
        pass  # No circular dependency
    
    def get_orders(self, user_id: str):
        # Implementation here
        pass

# main.py - Wire dependencies
from user_service import UserService
from order_service import OrderService

order_service = OrderService()
user_service = UserService(order_service)
```

### 2. Event-Driven Pattern
**Problem**: Module A needs to notify Module B, Module B needs to notify Module A
```python
# ❌ BAD - Circular notifications
# payment_processor.py
from order_manager import OrderManager
class PaymentProcessor:
    def __init__(self):
        self.order_manager = OrderManager()
    
    def process_payment(self):
        self.order_manager.mark_paid()  # Direct dependency

# order_manager.py
from payment_processor import PaymentProcessor
class OrderManager:
    def __init__(self):
        self.payment_processor = PaymentProcessor()  # CIRCULAR!
```

**Solution**: Use event system
```python
# ✅ GOOD - Event-driven decoupling
# event_bus.py
from typing import Dict, List, Callable
from collections import defaultdict

class EventBus:
    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
    
    def subscribe(self, event_type: str, handler: Callable):
        self._subscribers[event_type].append(handler)
    
    def publish(self, event_type: str, data):
        for handler in self._subscribers[event_type]:
            handler(data)

# payment_processor.py
class PaymentProcessor:
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
    
    def process_payment(self, payment_data):
        # Process payment logic
        self.event_bus.publish('payment_completed', payment_data)

# order_manager.py  
class OrderManager:
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        event_bus.subscribe('payment_completed', self.handle_payment_completed)
    
    def handle_payment_completed(self, payment_data):
        # Mark order as paid
        pass
```

### 3. Interface Extraction Pattern
**Problem**: Two modules need each other's functionality
```python
# ❌ BAD - Direct circular dependency
# database_manager.py
from cache_manager import CacheManager
class DatabaseManager:
    def __init__(self):
        self.cache = CacheManager()

# cache_manager.py
from database_manager import DatabaseManager
class CacheManager:
    def __init__(self):
        self.db = DatabaseManager()  # CIRCULAR!
```

**Solution**: Extract common interface
```python
# ✅ GOOD - Interface-based design
# interfaces.py
from abc import ABC, abstractmethod

class DataStore(ABC):
    @abstractmethod
    def get(self, key: str): pass
    
    @abstractmethod
    def set(self, key: str, value): pass

# database_manager.py
from interfaces import DataStore

class DatabaseManager(DataStore):
    def __init__(self, cache: DataStore = None):
        self.cache = cache
    
    def get(self, key: str):
        if self.cache:
            cached = self.cache.get(key)
            if cached:
                return cached
        return self._get_from_db(key)

# cache_manager.py
from interfaces import DataStore

class CacheManager(DataStore):
    def __init__(self, fallback: DataStore = None):
        self.fallback = fallback
        self._cache = {}
    
    def get(self, key: str):
        if key in self._cache:
            return self._cache[key]
        elif self.fallback:
            return self.fallback.get(key)
```

---

## 🔍 Detection and Analysis Tools

### 1. Automated Detection Script
```bash
# Run circular dependency analysis
python dependency_analyzer.py /path/to/project --export analysis.json --visualize deps.png
```

### 2. Pre-commit Hook
```bash
#!/bin/bash
# .git/hooks/pre-commit
echo "Checking for circular dependencies..."
python tools/dependency_analyzer.py . --quick-check
if [ $? -ne 0 ]; then
    echo "❌ Circular dependencies detected! Commit rejected."
    exit 1
fi
echo "✅ No circular dependencies found"
```

### 3. CI/CD Integration
```yaml
# .github/workflows/dependency-check.yml
name: Dependency Analysis
on: [push, pull_request]
jobs:
  check-dependencies:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Check Circular Dependencies
        run: |
          python dependency_analyzer.py . --export deps.json
          # Fail if critical cycles found
          python -c "
          import json
          with open('deps.json') as f:
              data = json.load(f)
          critical = data['summary']['critical_cycles']
          if critical > 0:
              print(f'❌ {critical} critical circular dependencies found!')
              exit(1)
          print('✅ No critical circular dependencies')
          "
```

---

## 🔧 Refactoring Strategies

### 1. **Extract Interface** (Most Common)
- Create abstract base class or protocol
- Move shared functionality to interface
- Implement interface in both modules

### 2. **Extract Shared Module**
- Move common code to new module
- Both original modules depend on new module
- New module has no dependencies on original modules

### 3. **Dependency Injection**
- Pass dependencies as constructor/method parameters
- Use factory pattern or DI container
- Invert control flow

### 4. **Event-Driven Architecture**
- Replace direct calls with event publishing
- Use message queues or event buses
- Implement observer pattern

### 5. **Merge Modules** (Last Resort)
- If modules are too tightly coupled
- Consider if they should be one module
- Maintain single responsibility principle

---

## 👥 Team Processes

### 1. Code Review Checklist
- [ ] No new circular dependencies introduced
- [ ] Dependencies flow in correct direction (up → down)
- [ ] Interfaces used instead of concrete classes where appropriate
- [ ] Event-driven patterns used for cross-cutting concerns

### 2. Architecture Review Process
1. **Design Phase**: Review module dependencies before coding
2. **Implementation Phase**: Run dependency analysis during development
3. **Review Phase**: Team reviews dependency graph changes
4. **Approval Phase**: Architect approves significant dependency changes

### 3. Refactoring Workflow
1. **Detection**: Automated tools find circular dependencies
2. **Assessment**: Team assesses impact and priority
3. **Planning**: Choose refactoring strategy
4. **Implementation**: Implement refactoring with tests
5. **Validation**: Verify cycle is broken and functionality preserved

---

## 📝 Implementation Guidelines

### 1. New Module Creation
```python
# Template for new modules
"""
Module: module_name.py
Purpose: Single clear purpose
Dependencies: List all dependencies and justify each
Interface: Public API this module exposes
"""

# 1. Import only what you need
from typing import Protocol  # Standard library first
from third_party import SomeLib  # Third party second
from project_core import CoreInterface  # Project imports last

# 2. Define clear interfaces
class MyModuleInterface(Protocol):
    def do_something(self) -> str: ...

# 3. Implement with minimal dependencies
class MyModule(MyModuleInterface):
    def __init__(self, dependency: CoreInterface):
        self._dependency = dependency  # Inject, don't import
    
    def do_something(self) -> str:
        return self._dependency.get_data()
```

### 2. Dependency Rules by Layer
```python
# ✅ ALLOWED DEPENDENCIES
# UI Layer
from services.user_service import UserService     # ✅ UI → Service
from core.models import User                      # ✅ UI → Core

# Service Layer  
from core.repositories import UserRepository     # ✅ Service → Core
from core.models import User                     # ✅ Service → Core

# Core Layer
from typing import Protocol                      # ✅ Core → Standard Library
# NO imports from Service or UI layers           # ✅ No upward dependencies

# ❌ FORBIDDEN DEPENDENCIES
# Core Layer
from services.user_service import UserService   # ❌ Core → Service (UPWARD!)
from ui.user_controller import UserController   # ❌ Core → UI (UPWARD!)
```

### 3. Testing Circular Dependencies
```python
# test_no_circular_dependencies.py
import pytest
from dependency_analyzer import DependencyAnalyzer
from pathlib import Path

def test_no_circular_dependencies():
    """Ensure no circular dependencies exist in codebase"""
    analyzer = DependencyAnalyzer(Path('.'))
    results = analyzer.analyze_project()
    
    critical_cycles = results['summary']['critical_cycles']
    high_cycles = results['summary']['high_priority_cycles']
    
    assert critical_cycles == 0, f"Found {critical_cycles} critical circular dependencies"
    assert high_cycles == 0, f"Found {high_cycles} high priority circular dependencies"

def test_dependency_direction():
    """Test that dependencies flow in correct direction"""
    analyzer = DependencyAnalyzer(Path('.'))
    results = analyzer.analyze_project()
    
    # Define layer hierarchy (lower number = lower level)
    layers = {
        'core': 1,
        'services': 2, 
        'api': 3,
        'ui': 3
    }
    
    # Check that dependencies only flow downward
    violations = []
    for module, imports in analyzer.module_imports.items():
        module_layer = get_module_layer(module, layers)
        for imported in imports:
            imported_layer = get_module_layer(imported, layers)
            if imported_layer > module_layer:
                violations.append(f"{module} → {imported} (upward dependency)")
    
    assert not violations, f"Upward dependencies found: {violations}"

def get_module_layer(module_name: str, layers: dict) -> int:
    """Determine which layer a module belongs to"""
    for layer, level in layers.items():
        if layer in module_name.lower():
            return level
    return 999  # Unknown layer, treat as highest level
```

---

## 🚀 Quick Start Commands

### Run Analysis
```bash
# Analyze current project
python architecture_intelligence/dependency_analyzer.py . 

# Export detailed report
python architecture_intelligence/dependency_analyzer.py . --export circular_deps.json

# Create visual graph
python architecture_intelligence/dependency_analyzer.py . --visualize dependency_graph.png
```

### Add to Git Hooks
```bash
# Copy pre-commit hook
cp tools/pre-commit-dependency-check .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

### Integrate with CI/CD
```bash
# Add to your build process
python dependency_analyzer.py . --quick-check || exit 1
```

---

## 📊 Success Metrics

### Project Health Indicators
- **Zero critical circular dependencies**
- **Zero high-priority circular dependencies**  
- **Dependency graph density < 0.3**
- **Average module dependencies < 5**
- **Clear layer separation (no upward dependencies)**

### Team Process Metrics
- **100% of PRs pass dependency analysis**
- **Dependency violations caught in code review**
- **Time to resolve circular dependencies < 1 week**
- **Zero circular dependencies introduced in new features**

---

## 🎯 Remember: Prevention > Detection > Remediation

1. **🛡️ PREVENT** with good architecture and design patterns
2. **🔍 DETECT** early with automated tools and processes  
3. **🔧 REMEDIATE** quickly with proven refactoring strategies
4. **📚 EDUCATE** team on principles and best practices

**The best circular dependency is the one that never gets created!**