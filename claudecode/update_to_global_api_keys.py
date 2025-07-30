#!/usr/bin/env python3
"""
Update Script: Convert os.getenv() API key calls to use global API key manager

This script finds and updates direct os.getenv() calls for API keys to use
the global multi-key failover system.
"""

import re
import os
from pathlib import Path
from typing import List, Tuple

# API key environment variables to update
API_KEY_PATTERNS = [
    'ANTHROPIC_API_KEY',
    'OPENAI_API_KEY', 
    'GOOGLE_API_KEY',
    'OPENROUTER_API_KEY',
    'GROQ_API_KEY'
]

# Files to skip
SKIP_FILES = {
    'global_api_keys.py',
    'multi_key_manager.py',
    'update_to_global_api_keys.py',
    '__init__.py'
}

# Directories to skip
SKIP_DIRS = {
    '.git',
    '__pycache__',
    '.venv',
    'venv',
    'node_modules',
    '.pytest_cache'
}


def find_api_key_usage(file_path: Path) -> List[Tuple[int, str, str]]:
    """Find os.getenv() calls for API keys in a file."""
    findings = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for i, line in enumerate(lines):
            # Look for os.getenv calls with API keys
            for api_key in API_KEY_PATTERNS:
                # Match os.getenv('API_KEY') or os.getenv("API_KEY")
                pattern = rf'os\.getenv\s*\(\s*[\'"]({api_key}(?:_\d+)?)[\'"]'
                matches = re.findall(pattern, line)
                
                if matches:
                    for match in matches:
                        findings.append((i + 1, line.strip(), match))
                
                # Also match os.environ['API_KEY'] or os.environ.get('API_KEY')
                pattern2 = rf'os\.environ(?:\[|\.\w+\s*\()\s*[\'"]({api_key}(?:_\d+)?)[\'"]'
                matches2 = re.findall(pattern2, line)
                
                if matches2:
                    for match in matches2:
                        findings.append((i + 1, line.strip(), match))
    
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    
    return findings


def update_file(file_path: Path, dry_run: bool = True) -> int:
    """Update a file to use global API key manager."""
    findings = find_api_key_usage(file_path)
    
    if not findings:
        return 0
    
    print(f"\n{'[DRY RUN] ' if dry_run else ''}Updating: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        updated = False
        
        # Check if global_api_keys is already imported
        has_import = 'from global_api_keys import' in content or 'import global_api_keys' in content
        
        # Add import if needed
        if not has_import:
            # Find where to add import (after other imports)
            import_lines = []
            lines = content.split('\n')
            
            last_import_idx = 0
            for i, line in enumerate(lines):
                if line.startswith('import ') or line.startswith('from '):
                    last_import_idx = i
            
            # Add import after last import
            if last_import_idx > 0:
                lines.insert(last_import_idx + 1, '\n# Global API key management for multi-key failover')
                lines.insert(last_import_idx + 2, 'try:')
                lines.insert(last_import_idx + 3, '    from global_api_keys import get_api_key_sync')
                lines.insert(last_import_idx + 4, '    GLOBAL_API_KEYS_AVAILABLE = True')
                lines.insert(last_import_idx + 5, 'except ImportError:')
                lines.insert(last_import_idx + 6, '    GLOBAL_API_KEYS_AVAILABLE = False')
                
                content = '\n'.join(lines)
                updated = True
        
        # Replace os.getenv() calls
        for api_key in API_KEY_PATTERNS:
            # Replace os.getenv('API_KEY')
            pattern = rf'os\.getenv\s*\(\s*[\'"]({api_key}(?:_\d+)?)[\'"]([^)]*)\)'
            
            def replace_getenv(match):
                key = match.group(1)
                rest = match.group(2)
                if rest:
                    # Has default value
                    return f'get_api_key_sync("{key}"{rest}) if GLOBAL_API_KEYS_AVAILABLE else os.getenv("{key}"{rest})'
                else:
                    return f'get_api_key_sync("{key}") if GLOBAL_API_KEYS_AVAILABLE else os.getenv("{key}")'
            
            new_content = re.sub(pattern, replace_getenv, content)
            if new_content != content:
                content = new_content
                updated = True
        
        if updated and not dry_run:
            # Create backup
            backup_path = file_path.with_suffix(file_path.suffix + '.bak')
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(original_content)
            
            # Write updated content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"  Updated {len(findings)} API key references")
            print(f"  Backup saved to: {backup_path}")
        elif updated:
            print(f"  Would update {len(findings)} API key references")
            for line_no, line, key in findings:
                print(f"    Line {line_no}: {key}")
        
        return len(findings) if updated else 0
    
    except Exception as e:
        print(f"  Error updating file: {e}")
        return 0


def scan_directory(root_path: Path, dry_run: bool = True) -> Tuple[int, int]:
    """Scan directory for files to update."""
    total_files = 0
    total_updates = 0
    
    for path in root_path.rglob('*.py'):
        # Skip certain files and directories
        if path.name in SKIP_FILES:
            continue
        
        if any(skip_dir in path.parts for skip_dir in SKIP_DIRS):
            continue
        
        # Skip test files (optional)
        if 'test_' in path.name or '_test.py' in path.name:
            continue
        
        findings = find_api_key_usage(path)
        if findings:
            total_files += 1
            total_updates += update_file(path, dry_run)
    
    return total_files, total_updates


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Update os.getenv() API key calls to use global multi-key manager'
    )
    parser.add_argument(
        '--apply',
        action='store_true',
        help='Actually apply changes (default is dry run)'
    )
    parser.add_argument(
        '--path',
        type=Path,
        default=Path.cwd(),
        help='Root path to scan (default: current directory)'
    )
    
    args = parser.parse_args()
    
    print("Global API Key Update Script")
    print("=" * 60)
    print(f"Root path: {args.path}")
    print(f"Mode: {'APPLY CHANGES' if args.apply else 'DRY RUN'}")
    print("=" * 60)
    
    if not args.apply:
        print("\nThis is a DRY RUN. Use --apply to make actual changes.\n")
    
    # Scan for files
    print("Scanning for API key usage...")
    total_files, total_updates = scan_directory(args.path, dry_run=not args.apply)
    
    print("\n" + "=" * 60)
    print(f"Total files with API keys: {total_files}")
    print(f"Total references {'updated' if args.apply else 'to update'}: {total_updates}")
    
    if not args.apply and total_updates > 0:
        print("\nRun with --apply to make these changes.")


if __name__ == '__main__':
    main()