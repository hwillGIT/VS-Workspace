#!/usr/bin/env python3
"""Test failover with proper key management"""

import minimal_global_api_keys as api
from datetime import datetime, timedelta
api.enable_global_failover()
import os

print('=== TESTING WORKING FAILOVER ===')

# Get manager instance for direct manipulation
manager = api._minimal_manager

print('1. Initial state:')
key1 = os.getenv('ANTHROPIC_API_KEY')
print(f'Initial key: {key1[:15]}...')
print(f'Active key ID: {manager.active_keys.get("anthropic")}')

print('\n2. Manually marking first key as rate-limited:')
# Directly mark the first key as rate-limited without affecting the second
first_key_status = manager.key_status['anthropic_key_1']
first_key_status.rate_limited_until = datetime.now() + timedelta(minutes=15)
first_key_status.consecutive_errors = 3
print(f'anthropic_key_1 rate limited until: {first_key_status.rate_limited_until}')

print('\n3. Triggering failover by attempting to get key:')
# This should trigger failover to the second key
key2 = manager.get_active_key('anthropic')
print(f'After failover - Direct manager call: {key2[:15] if key2 else "None"}...')

# Test through os.getenv
key3 = os.getenv('ANTHROPIC_API_KEY')
print(f'After failover - os.getenv(): {key3[:15] if key3 else "None"}...')

print(f'Failover successful: {key1 != key3}')
print(f'New active key ID: {manager.active_keys.get("anthropic")}')

print('\n4. Key status after failover:')
for key_id, status in manager.key_status.items():
    if status.provider == 'anthropic':
        print(f'{key_id}: available={status.is_available}, consecutive_errors={status.consecutive_errors}')

print('\n5. Raw environment comparison:')
print(f'Raw ANTHROPIC_API_KEY: {api._original_getenv("ANTHROPIC_API_KEY")[:15]}...')
print(f'Raw ANTHROPIC_API_KEY_2: {api._original_getenv("ANTHROPIC_API_KEY_2")[:15]}...')
print(f'Returned key matches key 2: {key3 == api._original_getenv("ANTHROPIC_API_KEY_2")}')