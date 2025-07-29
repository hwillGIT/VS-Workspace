# Universal Coding Rules

## 📋 Rules to Prevent Repeated Mistakes

These rules prevent common issues that appear repeatedly across projects.

### **CR-1: Cross-Platform Compatibility**
```python
# ❌ NEVER - Causes UnicodeEncodeError on Windows
print("✓ Success!")
print("❌ Failed!")
print("🚀 Started!")

# ✅ ALWAYS - Works everywhere
print("[OK] Success!")
print("[ERROR] Failed!")
print("[START] Started!")
```

**Why:** Windows command prompt uses cp1252 encoding by default, which can't display Unicode symbols.

### **CR-2: Environment File Protection**
```gitignore
# ✅ ALWAYS include in .gitignore
.env
.env.*
!.env.example
*.key
*.pem
secrets.json
```

**Why:** Prevents accidental API key exposure in git history.

### **CR-3: Type Hints Everywhere**
```python
# ❌ NEVER
def process_data(data):
    return data.upper()

# ✅ ALWAYS  
def process_data(data: str) -> str:
    return data.upper()
```

**Why:** Catches errors early, improves maintainability, enables better IDE support.

### **CR-4: Error Handling with Context**
```python
# ❌ NEVER - Silent failures
try:
    result = risky_operation()
except:
    pass

# ✅ ALWAYS - Clear error context
try:
    result = risky_operation()
except SpecificError as e:
    logger.error(f"Failed to process {item_id}: {e}")
    raise ProcessingError(f"Could not process item {item_id}") from e
```

**Why:** Makes debugging possible, provides actionable error messages.

### **CR-5: Path Handling**
```python
# ❌ NEVER - Platform specific
file_path = "data\\file.txt"  # Windows only
file_path = "data/file.txt"   # Unix only

# ✅ ALWAYS - Cross-platform
from pathlib import Path
file_path = Path("data") / "file.txt"
```

**Why:** Works on all operating systems, handles path edge cases.

### **CR-6: Logging Setup**
```python
# ❌ NEVER - No context
print("Starting process")

# ✅ ALWAYS - Structured logging
import logging
logger = logging.getLogger(__name__)
logger.info("Starting process", extra={"process_id": process_id})
```

**Why:** Enables proper debugging, audit trails, and monitoring.

### **CR-7: Magic Numbers and Strings**
```python
# ❌ NEVER - Magic values
if user.role == 3:
    allow_access()

if status == "PROC":
    continue_processing()

# ✅ ALWAYS - Named constants
class UserRole:
    ADMIN = 3
    USER = 1

class ProcessStatus:
    PROCESSING = "PROC"
    COMPLETED = "COMP"

if user.role == UserRole.ADMIN:
    allow_access()

if status == ProcessStatus.PROCESSING:
    continue_processing()
```

**Why:** Makes code self-documenting, prevents typos, enables refactoring.

### **CR-8: Input Validation**
```python
# ❌ NEVER - Trust user input
def create_user(email):
    # Direct use without validation
    return User(email=email)

# ✅ ALWAYS - Validate everything
def create_user(email: str) -> User:
    if not email or "@" not in email:
        raise ValueError("Invalid email format")
    if len(email) > 254:  # RFC 5321 limit
        raise ValueError("Email too long")
    return User(email=email.strip().lower())
```

**Why:** Prevents security vulnerabilities, data corruption, and crashes.

### **CR-9: Resource Management**
```python
# ❌ NEVER - Manual resource management
file = open("data.txt")
data = file.read()
file.close()  # Easy to forget!

# ✅ ALWAYS - Context managers
with open("data.txt") as file:
    data = file.read()
# Automatically closed
```

**Why:** Prevents resource leaks, ensures cleanup even during exceptions.

### **CR-10: Secret Management**
```python
# ❌ NEVER - Hardcoded secrets
API_KEY = "sk-ant-api03-abc123..."

# ❌ NEVER - Default secrets in production
if not os.getenv("API_KEY"):
    API_KEY = "default-key-123"

# ✅ ALWAYS - Environment variables with validation
import os
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("API_KEY environment variable required")
```

**Why:** Prevents credential exposure, enables proper secret rotation.

### **CR-11: Zero Circular Dependencies**
```python
# ❌ NEVER - Circular imports
# user_service.py
from order_service import OrderService
class UserService:
    def __init__(self):
        self.order_service = OrderService()

# order_service.py  
from user_service import UserService  # CIRCULAR!
class OrderService:
    def __init__(self):
        self.user_service = UserService()

# ✅ ALWAYS - Dependency injection or interfaces
# user_service.py
from abc import ABC, abstractmethod

class OrderServiceInterface(ABC):
    @abstractmethod
    def get_orders(self, user_id: str): pass

class UserService:
    def __init__(self, order_service: OrderServiceInterface):
        self.order_service = order_service

# order_service.py
class OrderService(OrderServiceInterface):
    def get_orders(self, user_id: str):
        # Implementation here
        pass
```

**Why:** Enables unit testing, prevents import errors, improves maintainability. **GLOBAL RULE: Zero tolerance for circular dependencies.**

## 🔄 Enforcement Strategy

### **In Code Reviews:**
- Check for Unicode symbols in print statements
- Verify .env files are in .gitignore
- Ensure type hints are present
- Look for proper error handling
- **Verify no circular dependencies introduced**

### **In CI/CD:**
- Add linting rules for these patterns
- Automated checks for secret patterns
- Cross-platform testing
- **Circular dependency analysis (run `python team_framework/circular_dependency/analyzer.py .`)**

### **In Documentation:**
- Reference these rules in project setup
- Include in coding standards
- Add to onboarding checklists

## 📚 Quick Reference

**Before writing any code, ask:**
1. Will this work on Windows? (Unicode, paths)
2. Are secrets properly handled?
3. Do I have type hints?
4. Is error handling meaningful?
5. Are resources properly managed?
6. **Will this create circular dependencies?**

---

*These rules prevent 90% of repeated debugging sessions.*