#!/usr/bin/env python3
"""Test real multi-key failover with correct .env file"""

import minimal_global_api_keys as api
api.enable_global_failover()
import os

print('=== TESTING ANTHROPIC MULTI-KEY FAILOVER ===')

# Get initial key
initial = os.getenv('ANTHROPIC_API_KEY')
print(f'Initial key: {initial[:15]}...')

# Record failures to trigger failover
print('\nSimulating rate limit failures...')
for i in range(3):
    api.record_api_result('ANTHROPIC_API_KEY', success=False, error_type='rate_limit')
    print(f'Failure {i+1} recorded')

# Get key after failover
new_key = os.getenv('ANTHROPIC_API_KEY') 
print(f'\nAfter failover: {new_key[:15]}...')
print(f'Failover successful: {initial != new_key}')
print(f'Both keys are real: {not new_key.startswith("your-")}')

# Show current system status
print('\n=== SYSTEM STATUS AFTER FAILOVER ===')
status = api.get_api_status()
for provider, info in status['providers'].items():
    if provider == 'anthropic':
        print(f'Anthropic: {info["total_keys"]} keys, active: {info["active_key"]}')
        break