#!/usr/bin/env python3
"""Debug failover issue step by step"""

import minimal_global_api_keys as api
api.enable_global_failover()
import os

print('=== DEBUGGING FAILOVER ISSUE ===')

# Test direct API key access first
print('\n1. Direct API key access:')
key1 = api.get_api_key('ANTHROPIC_API_KEY')
print(f'Direct get_api_key(): {key1[:15]}...')

# Test if environment interception is working
print('\n2. Environment variable access:')
key2 = os.getenv('ANTHROPIC_API_KEY')
print(f'os.getenv(): {key2[:15]}...')

print(f'Both return same key: {key1 == key2}')

# Check provider mapping
print('\n3. Provider mapping test:')
for pattern in ['ANTHROPIC_API_KEY', 'ANTHROPIC_API_KEY_2']:
    matches = any(pattern == p or pattern.startswith(p + '_') for p in api.PROVIDER_MAP)
    print(f'{pattern} matches provider map: {matches}')

# Record failures and test again
print('\n4. Recording failures...')
for i in range(3):
    api.record_api_result('ANTHROPIC_API_KEY', success=False, error_type='rate_limit')
    print(f'Failure {i+1} recorded')

# Test after failover
print('\n5. After failover:')
key3_direct = api.get_api_key('ANTHROPIC_API_KEY')
key3_env = os.getenv('ANTHROPIC_API_KEY')
print(f'Direct get_api_key(): {key3_direct[:15]}...')
print(f'os.getenv(): {key3_env[:15]}...')

# Check system status
status = api.get_api_status()
anthro_info = status['providers'].get('anthropic', {})
print(f'Active key: {anthro_info.get("active_key")}')

# Test the actual key values from env
print('\n6. Raw environment values:')
print(f'ANTHROPIC_API_KEY: {api._original_getenv("ANTHROPIC_API_KEY")[:15]}...')
print(f'ANTHROPIC_API_KEY_2: {api._original_getenv("ANTHROPIC_API_KEY_2")[:15]}...')