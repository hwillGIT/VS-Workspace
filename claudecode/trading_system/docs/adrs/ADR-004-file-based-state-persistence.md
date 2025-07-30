# ADR-004: File-Based State Persistence for API Key Management

## Status
Proposed - Pending Implementation (2025-07-30)

## Context and Problem Statement

Following the decisions in ADR-002 (Global API Key Management) and ADR-003 (Singleton Pattern), we need to implement persistent state management for API key health metrics, failover decisions, and usage tracking across system restarts.

### Core Requirements for State Persistence
1. **Health Metrics Persistence**: API provider health scores, failure counts, and recovery timestamps must survive restarts
2. **Failover State Recovery**: Active provider selections and backup preferences need to be restored after downtime
3. **Usage Analytics Continuity**: Rate limiting counters, usage patterns, and performance metrics should persist
4. **Configuration Drift Detection**: Track and persist provider configuration changes over time
5. **Atomic State Updates**: Prevent state corruption during concurrent writes or system crashes

### Current Architecture Context
The ClaudeCode trading system operates with:
- **High-frequency API calls** requiring immediate state access (<10ms response time)
- **Multiple concurrent agents** reading/writing state simultaneously
- **System restarts** for deployments, maintenance, and crash recovery
- **Critical failover decisions** that cannot reset to default state
- **Financial data integrity** requirements with audit trail needs

### Decision Drivers
1. **Performance Requirements**: Sub-10ms state read/write operations for trading speed
2. **Reliability**: Zero data loss during system crashes or unexpected shutdowns
3. **Portability**: State files should work across different deployment environments
4. **Security**: Sensitive metrics must be protected but accessible to the system
5. **Simplicity**: Minimal external dependencies for reduced deployment complexity
6. **Auditability**: Complete history of state changes for compliance and debugging

## Problem Statement

**How should we implement persistent state storage for the MinimalMultiKeyManager to maintain API provider health, failover state, and usage metrics across system restarts while meeting performance, security, and reliability requirements?**

### Specific Challenges
1. **State Corruption Prevention**: Concurrent access from multiple agents could corrupt state files
2. **Performance vs. Safety**: Trade-off between write frequency and data consistency
3. **File Location Security**: Balance between accessibility and security for state files
4. **Schema Evolution**: Handle state format changes without breaking existing deployments
5. **Recovery Strategies**: Graceful handling of corrupted, missing, or incompatible state files
6. **Cross-Platform Compatibility**: Consistent behavior across Windows, Linux, and macOS

## Considered Options

### Option 1: JSON Files in User Home Directory (CHOSEN)
**Description**: Store state as JSON files in `~/.claudecode/` directory with atomic write operations

```python
# State storage structure
~/.claudecode/
├── api_key_manager_state.json          # Main state file
├── api_key_manager_state.json.backup   # Automatic backup
├── provider_health_history.json        # Historical health data
├── usage_analytics.json                # Usage tracking data
└── config_version.json                 # Schema version tracking

# Example state file format
{
    "version": "1.0.0",
    "timestamp": "2025-07-30T15:30:00Z",
    "providers": {
        "alpha_vantage": {
            "health_score": 0.95,
            "last_failure": null,
            "failure_count": 0,
            "success_count": 1247,
            "avg_response_time": 150,
            "rate_limit_remaining": 450,
            "last_updated": "2025-07-30T15:29:45Z"
        },
        "yahoo_finance": {
            "health_score": 0.78,
            "last_failure": "2025-07-30T14:15:22Z",
            "failure_count": 3,
            "success_count": 892,
            "avg_response_time": 320,
            "rate_limit_remaining": 1200,
            "last_updated": "2025-07-30T15:29:50Z"
        }
    },
    "failover_state": {
        "active_provider": "alpha_vantage",
        "backup_providers": ["yahoo_finance", "quandl"],
        "last_failover": "2025-07-29T09:22:15Z",
        "failover_count": 2
    }
}
```

**Pros:**
- **Simple Implementation**: No external database dependencies
- **User-Specific**: Isolated state per user account
- **Cross-Platform**: Works consistently across all operating systems
- **File Permissions**: Leverages OS-level security for access control
- **Backup Integration**: Easy integration with existing backup systems
- **Debugging Friendly**: Human-readable JSON format for troubleshooting

**Cons:**
- **Concurrent Access**: Potential race conditions without proper locking
- **Performance Limitations**: JSON parsing overhead for large state files
- **Network Deployment**: Challenges with shared state in containerized environments
- **Disk I/O Dependency**: Performance tied to disk speed and availability

### Option 2: SQLite Database with WAL Mode
**Description**: Use embedded SQLite database with Write-Ahead Logging for ACID transactions

```python
# Database schema
CREATE TABLE provider_health (
    provider_name TEXT PRIMARY KEY,
    health_score REAL NOT NULL,
    last_failure TIMESTAMP,
    failure_count INTEGER DEFAULT 0,
    success_count INTEGER DEFAULT 0,
    avg_response_time INTEGER,
    rate_limit_remaining INTEGER,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE failover_state (
    id INTEGER PRIMARY KEY,
    active_provider TEXT NOT NULL,
    backup_providers TEXT, -- JSON array
    last_failover TIMESTAMP,
    failover_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE usage_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    provider_name TEXT NOT NULL,
    service_type TEXT NOT NULL,
    request_count INTEGER,
    error_count INTEGER,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Pros:**
- **ACID Compliance**: Guaranteed data consistency and atomic updates
- **Performance**: Optimized for concurrent read/write operations
- **Query Flexibility**: SQL queries for complex analytics and reporting
- **WAL Mode**: Better concurrent access handling than JSON files
- **Transaction Support**: Rollback capability for failed operations

**Cons:**
- **Complexity**: Additional dependency and schema management
- **File Size**: Database overhead larger than JSON for small datasets
- **Migration Complexity**: Schema updates require database migrations
- **Debugging Difficulty**: Binary format harder to inspect manually

### Option 3: Redis-Based Persistence
**Description**: Use Redis with persistence enabled for high-performance state management

```python
# Redis key structure
claudecode:api_manager:providers:alpha_vantage:health_score
claudecode:api_manager:providers:alpha_vantage:failure_count
claudecode:api_manager:failover:active_provider
claudecode:api_manager:usage:hourly:{provider}:{timestamp}

# Redis configuration
save 900 1      # Save if at least 1 key changed in 900 seconds
save 300 10     # Save if at least 10 keys changed in 300 seconds
save 60 10000   # Save if at least 10000 keys changed in 60 seconds
```

**Pros:**
- **High Performance**: Sub-millisecond read/write operations
- **Concurrent Access**: Excellent support for multiple clients
- **Data Structures**: Rich data types beyond simple key-value
- **Persistence Options**: Multiple persistence strategies available
- **Atomic Operations**: Built-in atomic increments and transactions

**Cons:**
- **External Dependency**: Requires Redis server installation and management
- **Memory Usage**: Higher memory consumption than file-based approaches
- **Network Dependency**: Additional failure point with network calls
- **Deployment Complexity**: Container orchestration and service discovery needed

### Option 4: Cloud-Based Storage (AWS S3/Azure Blob)
**Description**: Store state files in cloud object storage with versioning

```python
# Cloud storage structure
s3://claudecode-state-bucket/
├── users/{user_id}/api_manager_state.json
├── users/{user_id}/versions/state_v{timestamp}.json
└── global/aggregated_metrics.json

# State file with cloud metadata
{
    "version": "1.0.0",
    "cloud_metadata": {
        "etag": "d41d8cd98f00b204e9800998ecf8427e",
        "version_id": "v1.2.3",
        "last_modified": "2025-07-30T15:30:00Z"
    },
    "state": { /* ... provider state ... */ }
}
```

**Pros:**
- **Scalability**: Unlimited storage capacity
- **Durability**: 99.999999999% (11 9's) data durability
- **Versioning**: Built-in state history and rollback capability
- **Global Access**: Available from multiple deployment regions
- **Backup Integration**: Automatic replication and disaster recovery

**Cons:**
- **Network Dependency**: Requires internet connectivity for all operations
- **Latency**: Higher latency than local storage options
- **Cost**: Ongoing storage and API call costs
- **Complexity**: Authentication, credential management, and service configuration

### Option 5: Registry-Based Storage (Windows) / Config Files (Linux/Mac)
**Description**: Use platform-specific configuration storage mechanisms

```python
# Windows Registry structure
HKEY_CURRENT_USER\Software\ClaudeCode\APIManager\
├── Providers\
│   ├── AlphaVantage\
│   │   ├── HealthScore (REG_DWORD)
│   │   ├── FailureCount (REG_DWORD)
│   │   └── LastUpdated (REG_SZ)
│   └── YahooFinance\
│       ├── HealthScore (REG_DWORD)
│       └── FailureCount (REG_DWORD)
└── FailoverState\
    ├── ActiveProvider (REG_SZ)
    └── BackupProviders (REG_MULTI_SZ)

# Linux/Mac: ~/.config/claudecode/api_manager.conf
[providers.alpha_vantage]
health_score = 0.95
failure_count = 0
last_updated = 2025-07-30T15:30:00Z

[failover]
active_provider = alpha_vantage
backup_providers = yahoo_finance,quandl
```

**Pros:**
- **Platform Native**: Uses OS-recommended storage mechanisms
- **System Integration**: Better integration with OS backup and restore
- **Access Control**: Leverages OS user permissions and security
- **Performance**: Optimized for local system access patterns

**Cons:**
- **Platform Specific**: Different implementations for each OS
- **Limited Structure**: Registry limitations on data types and nesting
- **Portability**: Difficult to migrate state between systems
- **Debugging Complexity**: Platform-specific tools required for inspection

## Decision Outcome

**Selected Option 1: JSON Files in User Home Directory**

### Rationale

The JSON file approach best aligns with our requirements for the following reasons:

1. **Deployment Simplicity**: No external dependencies or service management required
2. **Cross-Platform Consistency**: Identical behavior across Windows, Linux, and macOS environments
3. **Performance Adequacy**: File I/O performance sufficient for API key management use case
4. **Security Through OS**: Leverages operating system file permissions for access control
5. **Debugging Accessibility**: Human-readable format enables easy troubleshooting and inspection
6. **Backup Integration**: Standard file-based backups work without special configuration

### Key Implementation Details

#### File Structure and Organization
```python
import os
import json
import threading
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timezone

class FileBasedStateManager:
    """
    Manages persistent state for API key management using JSON files.
    
    Features:
    - Atomic write operations to prevent corruption
    - Automatic backup creation and rotation
    - Schema versioning for backward compatibility
    - Thread-safe concurrent access
    - File locking for multi-process safety
    """
    
    def __init__(self):
        self.state_dir = Path.home() / '.claudecode'
        self.state_file = self.state_dir / 'api_key_manager_state.json'
        self.backup_file = self.state_dir / 'api_key_manager_state.json.backup'
        self.lock_file = self.state_dir / 'api_key_manager_state.lock'
        
        # Thread safety
        self._file_lock = threading.RLock()
        
        # Ensure directory exists with proper permissions
        self._ensure_state_directory()
        
        # Current schema version
        self.schema_version = "1.0.0"
    
    def _ensure_state_directory(self):
        """Create state directory with secure permissions."""
        self.state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        
        # Verify permissions (Unix-like systems)
        if os.name != 'nt':  # Not Windows
            current_perms = oct(self.state_dir.stat().st_mode)[-3:]
            if current_perms != '700':
                self.state_dir.chmod(0o700)
```

#### Atomic Write Operations
```python
def save_state(self, state_data: Dict[str, Any]) -> bool:
    """
    Atomically save state to disk with backup creation.
    
    Args:
        state_data: Dictionary containing complete state to save
        
    Returns:
        bool: True if save successful, False otherwise
    """
    with self._file_lock:
        try:
            # Add metadata to state
            timestamped_state = {
                'version': self.schema_version,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'data': state_data
            }
            
            # Create backup of existing state
            if self.state_file.exists():
                shutil.copy2(self.state_file, self.backup_file)
            
            # Atomic write using temporary file and rename
            with tempfile.NamedTemporaryFile(
                mode='w', 
                dir=self.state_dir, 
                suffix='.tmp',
                delete=False
            ) as temp_file:
                json.dump(timestamped_state, temp_file, indent=2, sort_keys=True)
                temp_file.flush()
                os.fsync(temp_file.fileno())  # Force write to disk
                temp_path = temp_file.name
            
            # Atomic rename (platform-specific implementation)
            if os.name == 'nt':  # Windows
                # Windows doesn't support atomic rename over existing file
                if self.state_file.exists():
                    backup_temp = self.state_file.with_suffix('.old')
                    self.state_file.rename(backup_temp)
                    try:
                        Path(temp_path).rename(self.state_file)
                        backup_temp.unlink()  # Remove old file
                    except Exception:
                        # Restore original file on failure
                        backup_temp.rename(self.state_file)
                        raise
                else:
                    Path(temp_path).rename(self.state_file)
            else:  # Unix-like systems
                Path(temp_path).rename(self.state_file)
            
            # Set secure permissions
            if os.name != 'nt':
                self.state_file.chmod(0o600)
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to save state: {e}")
            # Clean up temporary file if it exists
            try:
                if 'temp_path' in locals():
                    Path(temp_path).unlink(missing_ok=True)
            except Exception:
                pass
            return False
```

#### State Loading with Recovery
```python
def load_state(self) -> Optional[Dict[str, Any]]:
    """
    Load state from disk with automatic recovery from backup.
    
    Returns:
        Optional[Dict]: Loaded state data or None if no valid state found
    """
    with self._file_lock:
        # Try loading main state file
        state_data = self._load_state_file(self.state_file)
        
        if state_data is not None:
            # Validate schema version
            if self._validate_schema_version(state_data):
                return state_data.get('data', {})
            else:
                logging.warning("State file schema version mismatch, attempting backup")
        
        # Try loading backup file
        if self.backup_file.exists():
            logging.info("Loading state from backup file")
            backup_data = self._load_state_file(self.backup_file)
            
            if backup_data is not None and self._validate_schema_version(backup_data):
                # Restore backup as main file
                shutil.copy2(self.backup_file, self.state_file)
                return backup_data.get('data', {})
        
        # No valid state found
        logging.info("No valid state file found, starting with empty state")
        return None

def _load_state_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
    """
    Load and parse JSON state file with error handling.
    
    Args:
        file_path: Path to JSON file to load
        
    Returns:
        Optional[Dict]: Parsed JSON data or None if load failed
    """
    try:
        if not file_path.exists():
            return None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Validate basic structure
        if not isinstance(data, dict):
            logging.error(f"Invalid state file format: {file_path}")
            return None
        
        required_fields = ['version', 'timestamp', 'data']
        if not all(field in data for field in required_fields):
            logging.error(f"Missing required fields in state file: {file_path}")
            return None
        
        return data
        
    except json.JSONDecodeError as e:
        logging.error(f"JSON decode error in {file_path}: {e}")
        return None
    except Exception as e:
        logging.error(f"Failed to load state file {file_path}: {e}")
        return None
```

#### Schema Version Management
```python
def _validate_schema_version(self, state_data: Dict[str, Any]) -> bool:
    """
    Validate state file schema version and handle migrations.
    
    Args:
        state_data: Loaded state data with version information
        
    Returns:
        bool: True if schema is compatible, False otherwise
    """
    file_version = state_data.get('version', '0.0.0')
    current_version = self.schema_version
    
    # Parse version numbers
    file_major, file_minor, file_patch = map(int, file_version.split('.'))
    curr_major, curr_minor, curr_patch = map(int, current_version.split('.'))
    
    # Major version compatibility check
    if file_major != curr_major:
        logging.error(f"Incompatible major version: {file_version} vs {current_version}")
        return False
    
    # Minor version backward compatibility
    if file_minor > curr_minor:
        logging.warning(f"Newer minor version detected: {file_version} vs {current_version}")
        return False
    
    # Apply migrations if needed
    if (file_major, file_minor, file_patch) < (curr_major, curr_minor, curr_patch):
        return self._migrate_state_schema(state_data, file_version, current_version)
    
    return True

def _migrate_state_schema(self, state_data: Dict[str, Any], 
                         from_version: str, to_version: str) -> bool:
    """
    Migrate state data from older schema version to current version.
    
    Args:
        state_data: State data to migrate (modified in-place)
        from_version: Source schema version
        to_version: Target schema version
        
    Returns:
        bool: True if migration successful, False otherwise
    """
    try:
        logging.info(f"Migrating state schema from {from_version} to {to_version}")
        
        # Example migration: v0.9.0 -> v1.0.0
        if from_version.startswith('0.9.'):
            # Add new fields with defaults
            data = state_data.get('data', {})
            
            for provider_name, provider_data in data.get('providers', {}).items():
                # Add missing fields with defaults
                provider_data.setdefault('avg_response_time', 0)
                provider_data.setdefault('rate_limit_remaining', 1000)
                
            # Update failover state structure
            if 'failover_state' in data:
                failover = data['failover_state']
                failover.setdefault('failover_count', 0)
                failover.setdefault('last_failover', None)
        
        # Update version in state data
        state_data['version'] = to_version
        
        return True
        
    except Exception as e:
        logging.error(f"Schema migration failed: {e}")
        return False
```

#### Multi-Process File Locking
```python
import fcntl  # Unix
import msvcrt  # Windows

class CrossPlatformFileLock:
    """Cross-platform file locking for multi-process safety."""
    
    def __init__(self, lock_file_path: Path):
        self.lock_file_path = lock_file_path
        self.lock_file = None
    
    def __enter__(self):
        """Acquire file lock."""
        try:
            self.lock_file = open(self.lock_file_path, 'w')
            
            if os.name == 'nt':  # Windows
                msvcrt.locking(self.lock_file.fileno(), msvcrt.LK_NBLCK, 1)
            else:  # Unix-like
                fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            
            return self
            
        except (IOError, OSError) as e:
            if self.lock_file:
                self.lock_file.close()
                self.lock_file = None
            raise Exception(f"Could not acquire file lock: {e}")
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Release file lock."""
        if self.lock_file:
            try:
                if os.name == 'nt':  # Windows
                    msvcrt.locking(self.lock_file.fileno(), msvcrt.LK_UNLCK, 1)
                else:  # Unix-like
                    fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass  # Lock will be released when file is closed
            finally:
                self.lock_file.close()
                self.lock_file = None

# Usage in state manager
def save_state_with_lock(self, state_data: Dict[str, Any]) -> bool:
    """Save state with cross-process file locking."""
    try:
        with CrossPlatformFileLock(self.lock_file):
            return self.save_state(state_data)
    except Exception as e:
        logging.error(f"Failed to acquire lock for state save: {e}")
        return False
```

## Positive Consequences

### Immediate Benefits
1. **Simple Deployment**: No external service dependencies or complex setup requirements
2. **Cross-Platform Consistency**: Identical behavior across all supported operating systems
3. **User Isolation**: Each user account maintains separate state without conflicts
4. **Backup Integration**: Standard file backup systems automatically include state files
5. **Security Through OS**: File system permissions provide adequate security for non-sensitive state
6. **Debugging Accessibility**: Human-readable JSON format enables easy inspection and troubleshooting

### Long-term Benefits
1. **Maintenance Simplicity**: No database schema migrations or service updates required
2. **Disaster Recovery**: Simple file restoration procedures for state recovery
3. **Performance Predictability**: File I/O performance characteristics well understood
4. **Development Efficiency**: Easy testing and development without external service setup
5. **Cost Efficiency**: No ongoing service costs or licensing requirements

## Negative Consequences

### Performance Limitations
1. **JSON Parsing Overhead**: Serialization/deserialization cost for every state access
2. **File I/O Blocking**: Disk operations may block high-frequency trading operations
3. **Concurrent Access**: File locking may cause contention under high concurrent load
4. **Storage Efficiency**: JSON format less space efficient than binary alternatives

### Operational Challenges
1. **Backup Complexity**: Users responsible for backing up state files manually
2. **Multi-Instance Conflicts**: Multiple application instances may conflict over state files
3. **File Corruption Risk**: System crashes during write operations may corrupt state
4. **Cross-Network Access**: Shared state challenging in distributed deployments

### Scalability Concerns
1. **State Size Growth**: Large state files may impact performance over time
2. **Concurrent Users**: Multiple users on same system may experience file system contention
3. **Network File Systems**: Performance degradation when home directory is network-mounted
4. **Platform Limitations**: File system performance varies significantly across platforms

## Implementation Plan

### Phase 1: Core File-Based State Manager (Week 1)
- [ ] Implement `FileBasedStateManager` class with atomic write operations
- [ ] Add cross-platform file locking for multi-process safety
- [ ] Create schema versioning and migration framework
- [ ] Implement state loading with automatic backup recovery
- [ ] Add comprehensive error handling and logging

### Phase 2: Integration with API Key Manager (Week 1-2)
- [ ] Integrate state manager with `MinimalMultiKeyManager` singleton
- [ ] Implement provider health state persistence
- [ ] Add failover state tracking and recovery
- [ ] Create usage analytics persistence layer
- [ ] Add state validation and consistency checks

### Phase 3: Advanced Features and Optimization (Week 2-3)
- [ ] Implement state compression for large datasets
- [ ] Add state file rotation and cleanup policies
- [ ] Create state export/import functionality
- [ ] Implement incremental state updates for performance
- [ ] Add state corruption detection and repair

### Phase 4: Testing and Production Readiness (Week 3-4)
- [ ] Comprehensive unit tests for all state operations
- [ ] Multi-process concurrency testing
- [ ] Performance benchmarking and optimization
- [ ] Cross-platform compatibility testing
- [ ] Production deployment and monitoring setup

## File Format Specification

### Main State File Structure
```json
{
    "version": "1.0.0",
    "timestamp": "2025-07-30T15:30:00Z",
    "metadata": {
        "created_by": "MinimalMultiKeyManager",
        "system_info": {
            "platform": "Windows-10",
            "python_version": "3.11.5",
            "user": "trading_user"
        }
    },
    "data": {
        "providers": {
            "alpha_vantage": {
                "health_score": 0.95,
                "last_failure": null,
                "failure_count": 0,
                "success_count": 1247,
                "avg_response_time": 150,
                "rate_limit_remaining": 450,
                "last_updated": "2025-07-30T15:29:45Z",
                "configuration": {
                    "base_url": "https://www.alphavantage.co/query",
                    "timeout": 30,
                    "retry_count": 3
                }
            },
            "yahoo_finance": {
                "health_score": 0.78,
                "last_failure": "2025-07-30T14:15:22Z",
                "failure_count": 3,
                "success_count": 892,
                "avg_response_time": 320,
                "rate_limit_remaining": 1200,
                "last_updated": "2025-07-30T15:29:50Z",
                "configuration": {
                    "base_url": "https://query1.finance.yahoo.com/v8/finance/chart",
                    "timeout": 25,
                    "retry_count": 2
                }
            }
        },
        "failover_state": {
            "active_provider": "alpha_vantage",
            "backup_providers": ["yahoo_finance", "quandl"],
            "last_failover": "2025-07-29T09:22:15Z",
            "failover_count": 2,
            "failover_threshold": 0.5,
            "recovery_threshold": 0.8
        },
        "usage_analytics": {
            "total_requests": 15678,
            "successful_requests": 14892,
            "failed_requests": 786,
            "avg_daily_requests": 1245,
            "peak_requests_per_hour": 156,
            "last_reset": "2025-07-01T00:00:00Z"
        },
        "rate_limiting": {
            "global_limits": {
                "requests_per_minute": 300,
                "requests_per_hour": 1000,
                "requests_per_day": 25000
            },
            "current_usage": {
                "minute_count": 45,
                "hour_count": 234,
                "day_count": 1567,
                "window_start": "2025-07-30T15:29:00Z"
            }
        }
    }
}
```

### Historical Data File Structure
```json
{
    "version": "1.0.0",
    "timestamp": "2025-07-30T15:30:00Z",
    "retention_policy": {
        "max_entries": 10000,
        "max_age_days": 90,
        "compression_enabled": true
    },
    "history": [
        {
            "timestamp": "2025-07-30T15:00:00Z",
            "event_type": "health_check",
            "provider": "alpha_vantage",
            "data": {
                "response_time": 145,
                "success": true,
                "health_score": 0.95
            }
        },
        {
            "timestamp": "2025-07-30T14:45:00Z",
            "event_type": "failover",
            "from_provider": "yahoo_finance",
            "to_provider": "alpha_vantage",
            "reason": "health_score_below_threshold",
            "data": {
                "trigger_score": 0.45,
                "threshold": 0.5
            }
        }
    ]
}
```

## Security Considerations

### File Permissions and Access Control
```python
def _secure_file_permissions(self):
    """Set secure file permissions for state files."""
    
    if os.name == 'nt':  # Windows
        # Use Windows ACLs for access control
        import win32security
        import win32file
        
        # Get current user SID
        user_sid = win32security.GetTokenInformation(
            win32security.OpenProcessToken(
                win32api.GetCurrentProcess(),
                win32security.TOKEN_QUERY
            ),
            win32security.TokenUser
        )[0]
        
        # Create DACL with only current user access
        dacl = win32security.ACL()
        dacl.AddAccessAllowedAce(
            win32security.ACL_REVISION,
            win32file.FILE_ALL_ACCESS,
            user_sid
        )
        
        # Apply security descriptor
        sd = win32security.SECURITY_DESCRIPTOR()
        sd.SetSecurityDescriptorDacl(1, dacl, 0)
        
        win32security.SetFileSecurity(
            str(self.state_file),
            win32security.DACL_SECURITY_INFORMATION,
            sd
        )
    
    else:  # Unix-like systems
        # Set file permissions to user read/write only
        os.chmod(self.state_file, 0o600)
        os.chmod(self.state_dir, 0o700)
```

### Sensitive Data Handling
```python
def _sanitize_state_for_storage(self, state_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove or encrypt sensitive data before storage.
    
    Args:
        state_data: Raw state data containing potentially sensitive information
        
    Returns:
        Dict: Sanitized state data safe for persistent storage
    """
    sanitized = copy.deepcopy(state_data)
    
    # Remove sensitive fields
    sensitive_fields = [
        'api_keys',
        'authentication_tokens',
        'user_credentials',
        'private_keys'
    ]
    
    for field in sensitive_fields:
        if field in sanitized:
            del sanitized[field]
    
    # Hash sensitive identifiers
    for provider_name, provider_data in sanitized.get('providers', {}).items():
        if 'user_id' in provider_data:
            provider_data['user_id_hash'] = hashlib.sha256(
                provider_data['user_id'].encode()
            ).hexdigest()[:16]
            del provider_data['user_id']
    
    return sanitized
```

### Audit Trail Implementation
```python
def _log_state_change(self, operation: str, details: Dict[str, Any]):
    """
    Log state changes for audit trail and debugging.
    
    Args:
        operation: Type of operation (save, load, migrate, etc.)
        details: Additional context about the operation
    """
    audit_entry = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'operation': operation,
        'user': os.getenv('USER', 'unknown'),
        'process_id': os.getpid(),
        'details': details
    }
    
    # Write to audit log file
    audit_file = self.state_dir / 'audit.log'
    with open(audit_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(audit_entry) + '\n')
    
    # Also log to application logger
    logging.info(f"State audit: {operation}", extra=audit_entry)
```

## Performance Characteristics

### Expected Performance Metrics
- **State Loading**: <50ms for typical state files (<1MB)
- **State Saving**: <100ms with atomic write operations
- **Concurrent Access**: Support for up to 10 concurrent readers with file locking
- **Memory Usage**: <10MB for loaded state data in memory
- **Disk Usage**: <5MB for typical state files with history

### Performance Optimization Strategies
```python
def _optimize_state_access(self):
    """Implement performance optimizations for state access."""
    
    # 1. In-memory caching with write-through
    self._state_cache = {}
    self._cache_timestamp = None
    self._cache_ttl = 30  # seconds
    
    # 2. Lazy loading of historical data
    self._history_loaded = False
    self._history_cache = None
    
    # 3. Incremental updates instead of full rewrites
    self._pending_updates = []
    self._batch_update_timer = None
    
    # 4. Compression for large state files
    self._compression_threshold = 1024 * 1024  # 1MB
    self._compression_enabled = True

def _get_cached_state(self) -> Optional[Dict[str, Any]]:
    """Get state from cache if available and valid."""
    if (
        self._state_cache and 
        self._cache_timestamp and 
        (datetime.now() - self._cache_timestamp).seconds < self._cache_ttl
    ):
        return self._state_cache
    return None

def _batch_state_updates(self, update_data: Dict[str, Any]):
    """Batch multiple state updates for better performance."""
    self._pending_updates.append({
        'timestamp': datetime.now(timezone.utc),
        'data': update_data
    })
    
    # Schedule batch write
    if self._batch_update_timer:
        self._batch_update_timer.cancel()
    
    self._batch_update_timer = threading.Timer(
        5.0,  # Wait 5 seconds for more updates
        self._flush_pending_updates
    )
    self._batch_update_timer.start()
```

## Backup and Recovery Strategies

### Automatic Backup Creation
```python
def _create_backup_rotation(self):
    """Create rotating backups of state files."""
    
    if not self.state_file.exists():
        return
    
    # Create timestamped backup
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_name = f'api_key_manager_state_{timestamp}.json.backup'
    timestamped_backup = self.state_dir / backup_name
    
    shutil.copy2(self.state_file, timestamped_backup)
    
    # Maintain backup rotation (keep last 10 backups)
    backup_pattern = self.state_dir.glob('api_key_manager_state_*.json.backup')
    backups = sorted(backup_pattern, key=lambda p: p.stat().st_mtime, reverse=True)
    
    # Remove old backups beyond retention limit
    for old_backup in backups[10:]:
        try:
            old_backup.unlink()
            logging.info(f"Removed old backup: {old_backup}")
        except Exception as e:
            logging.warning(f"Failed to remove old backup {old_backup}: {e}")
```

### State Corruption Recovery
```python
def _recover_from_corruption(self) -> Optional[Dict[str, Any]]:
    """
    Attempt to recover state from various backup sources.
    
    Returns:
        Optional[Dict]: Recovered state data or None if recovery failed
    """
    recovery_sources = [
        ('primary_backup', self.backup_file),
        ('timestamped_backups', self._find_recent_backups()),
        ('empty_state', None)  # Last resort: start fresh
    ]
    
    for source_name, source_path in recovery_sources:
        if source_path is None:
            logging.warning("All recovery attempts failed, starting with empty state")
            return {}
        
        try:
            if isinstance(source_path, list):
                # Try each timestamped backup
                for backup_path in source_path:
                    state_data = self._load_state_file(backup_path)
                    if state_data and self._validate_state_integrity(state_data):
                        logging.info(f"State recovered from {backup_path}")
                        return state_data.get('data', {})
            else:
                # Try single backup file
                state_data = self._load_state_file(source_path)
                if state_data and self._validate_state_integrity(state_data):
                    logging.info(f"State recovered from {source_name}")
                    return state_data.get('data', {})
        
        except Exception as e:
            logging.error(f"Failed to recover from {source_name}: {e}")
            continue
    
    return None

def _validate_state_integrity(self, state_data: Dict[str, Any]) -> bool:
    """
    Validate the integrity and completeness of state data.
    
    Args:
        state_data: State data to validate
        
    Returns:
        bool: True if state data is valid and complete
    """
    try:
        # Check required top-level fields
        required_fields = ['version', 'timestamp', 'data']
        if not all(field in state_data for field in required_fields):
            return False
        
        data = state_data['data']
        
        # Validate providers section
        if 'providers' in data:
            for provider_name, provider_data in data['providers'].items():
                required_provider_fields = [
                    'health_score', 'failure_count', 'success_count'
                ]
                if not all(field in provider_data for field in required_provider_fields):
                    return False
                
                # Validate data types and ranges
                if not (0.0 <= provider_data['health_score'] <= 1.0):
                    return False
                
                if provider_data['failure_count'] < 0 or provider_data['success_count'] < 0:
                    return False
        
        # Validate failover state
        if 'failover_state' in data:
            failover = data['failover_state']
            if 'active_provider' not in failover or 'backup_providers' not in failover:
                return False
        
        return True
        
    except Exception as e:
        logging.error(f"State integrity validation failed: {e}")
        return False
```

## Success Criteria

### Technical Success Metrics
- **Data Persistence**: 100% successful state persistence across system restarts
- **Performance**: <100ms for state save operations, <50ms for load operations
- **Reliability**: Zero data loss in crash scenarios with atomic write operations
- **Concurrency**: Safe operation with up to 10 concurrent access attempts

### Quality Metrics
- **Test Coverage**: >95% code coverage for state management functionality
- **Cross-Platform**: Identical behavior verified on Windows, Linux, and macOS
- **Schema Migration**: Successful migration testing for all version transitions
- **Recovery Testing**: 100% successful recovery from simulated corruption scenarios

### Operational Metrics
- **File System Usage**: <10MB total disk usage for state files and backups
- **Backup Success**: Automatic backup creation success rate >99.9%
- **Error Recovery**: <10 seconds to recover from state file corruption
- **Audit Compliance**: Complete audit trail for all state changes

## Related ADRs
- **ADR-002**: Global API Key Management with Automatic Failover (Parent decision)
- **ADR-003**: Singleton Pattern for Global API Key Management (Dependent decision)
- **ADR-005**: [Planned] Comprehensive Error Handling Strategy
- **ADR-006**: [Planned] Security Architecture Enhancement
- **ADR-007**: [Planned] Performance Monitoring and Metrics Collection

## References and Standards
- [JSON Specification (RFC 7159)](https://tools.ietf.org/html/rfc7159)
- [POSIX File Locking Standards](https://pubs.opengroup.org/onlinepubs/9699919799/functions/fcntl.html)
- [Windows File System Security](https://docs.microsoft.com/en-us/windows/win32/fileio/file-security-and-access-rights)
- [Python pathlib Documentation](https://docs.python.org/3/library/pathlib.html)
- [Atomic File Operations Best Practices](https://lwn.net/Articles/457667/)
- [Cross-Platform File Permissions](https://docs.python.org/3/library/os.html#os.chmod)

---
**Author**: System Architect Agent  
**Date**: 2025-07-30  
**Reviewers**: [To be assigned]  
**Implementation Status**: Proposed  
**Last Updated**: 2025-07-30  
**Security Review**: Pending  
**Performance Review**: Pending  
**Related Implementation**: FileBasedStateManager (to be implemented)