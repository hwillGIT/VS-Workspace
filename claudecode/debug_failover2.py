#!/usr/bin/env python3
"""Debug failover issue - check key availability"""

import minimal_global_api_keys as api
from datetime import datetime
api.enable_global_failover()

print('=== DEBUGGING KEY AVAILABILITY ===')

# Get manager instance
manager = api._minimal_manager

# Check initial state
print('\n1. Initial key discovery:')
print(f'Keys discovered: {manager.keys}')
print(f'Active keys: {manager.active_keys}')

print('\n2. Key status before failover:')
for key_id, status in manager.key_status.items():
    if status.provider == 'anthropic':
        print(f'{key_id}: active={status.is_active}, available={status.is_available}, consecutive_errors={status.consecutive_errors}')
        if status.rate_limited_until:
            print(f'  Rate limited until: {status.rate_limited_until}')

# Record failures
print('\n3. Recording failures...')
for i in range(3):
    api.record_api_result('ANTHROPIC_API_KEY', success=False, error_type='rate_limit')
    print(f'Failure {i+1} recorded')

print('\n4. Key status after failures:')
for key_id, status in manager.key_status.items():
    if status.provider == 'anthropic':
        print(f'{key_id}: active={status.is_active}, available={status.is_available}, consecutive_errors={status.consecutive_errors}')
        if status.rate_limited_until:
            print(f'  Rate limited until: {status.rate_limited_until} (now: {datetime.now()})')

print(f'\n5. Active keys after failures: {manager.active_keys}')

# Try to get key
print('\n6. Attempting to get active key:')
active_key = manager.get_active_key('anthropic')
print(f'get_active_key result: {active_key[:15] if active_key else None}')

# Check find available key method
print('\n7. Finding available key:')
available = manager._find_available_key('anthropic')
print(f'_find_available_key result: {available}')