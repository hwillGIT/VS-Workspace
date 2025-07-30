#!/usr/bin/env python3
"""Test proper failover behavior with realistic failure simulation"""

import minimal_global_api_keys as api
api.enable_global_failover()
import os

print('=== TESTING PROPER FAILOVER BEHAVIOR ===')

# Get initial key
initial_key = os.getenv('ANTHROPIC_API_KEY')
print(f'Initial key: {initial_key[:15]}...')

# In a real scenario, we would record failures against the specific key that was used
# for each request. To simulate this properly, we need to record failures against 
# the specific key that failed, not just call record_api_result generically.

manager = api._minimal_manager
initial_active_key_id = manager.active_keys.get('anthropic')
print(f'Initial active key ID: {initial_active_key_id}')

print('\nSimulating 3 consecutive failures on the same key:')
for i in range(3):
    # Record failure against the key that was actually used (the initial key)
    # In real usage, this would happen when an API request to the initial key fails
    manager.record_result('anthropic', success=False, error_type='rate_limit')
    
    current_active = manager.active_keys.get('anthropic')
    print(f'Failure {i+1} recorded - Active key: {current_active}')
    
    # If failover happened on the first rate limit, break
    if current_active != initial_active_key_id:
        print(f'Failover occurred after failure {i+1}')
        break

# Get key after potential failover
final_key = os.getenv('ANTHROPIC_API_KEY')
print(f'\nFinal key: {final_key[:15]}...')
print(f'Failover successful: {initial_key != final_key}')

# Show system status
status = api.get_api_status()
anthro_info = status['providers'].get('anthropic', {})
print(f'Final active key ID: {anthro_info.get("active_key")}')
print(f'Total available keys: {anthro_info.get("total_keys")}')

# Show key availability
print('\nKey availability:')
for key_id, key_status in manager.key_status.items():
    if key_status.provider == 'anthropic':
        print(f'{key_id}: available={key_status.is_available}, errors={key_status.consecutive_errors}')
        if key_status.rate_limited_until:
            print(f'  Rate limited until: {key_status.rate_limited_until}')