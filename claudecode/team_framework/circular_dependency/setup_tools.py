#!/usr/bin/env python3
"""
Setup script for circular dependency checking infrastructure
"""

import os
import shutil
import stat
from pathlib import Path

def setup_git_hooks():
    """Install pre-commit hook for dependency checking"""
    
    project_root = Path(__file__).parent.parent
    git_hooks_dir = project_root / ".git" / "hooks"
    
    if not git_hooks_dir.exists():
        print("[ERROR] No .git directory found. Are you in a git repository?")
        return False
    
    # Copy pre-commit hook
    source_hook = project_root / "tools" / "pre-commit-dependency-check.sh"
    dest_hook = git_hooks_dir / "pre-commit"
    
    if source_hook.exists():
        shutil.copy2(source_hook, dest_hook)
        
        # Make executable
        st = os.stat(dest_hook)
        os.chmod(dest_hook, st.st_mode | stat.S_IEXEC)
        
        print("[SUCCESS] Pre-commit hook installed successfully")
        return True
    else:
        print(f"[ERROR] Source hook not found: {source_hook}")
        return False

def setup_ci_workflow():
    """Create GitHub Actions workflow for dependency checking"""
    
    project_root = Path(__file__).parent.parent
    workflows_dir = project_root / ".github" / "workflows"
    workflows_dir.mkdir(parents=True, exist_ok=True)
    
    workflow_content = """name: Circular Dependency Check
on: 
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  dependency-analysis:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install networkx matplotlib
        
    - name: Run circular dependency analysis
      run: |
        python architecture_intelligence/dependency_analyzer.py . --export deps.json
        
    - name: Check for critical cycles
      run: |
        python -c "
        import json
        with open('deps.json') as f:
            data = json.load(f)
        
        critical = data['summary']['critical_cycles']
        high = data['summary']['high_priority_cycles']
        total = data['summary']['circular_cycles']
        
        print(f'📊 Dependency Analysis Results:')
        print(f'   Total cycles: {total}')
        print(f'   High priority: {high}')
        print(f'   Critical: {critical}')
        
        if critical > 0:
            print(f'❌ {critical} critical circular dependencies found!')
            print('Review the analysis and refactor before merging.')
            exit(1)
        elif high > 0:
            print(f'⚠️  {high} high priority circular dependencies found.')
            print('Consider addressing these in upcoming iterations.')
        else:
            print('✅ No critical circular dependencies found!')
        "
        
    - name: Upload analysis results
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: dependency-analysis
        path: deps.json
"""
    
    workflow_file = workflows_dir / "dependency-check.yml"
    with open(workflow_file, 'w') as f:
        f.write(workflow_content)
        
    print(f"[SUCCESS] GitHub Actions workflow created: {workflow_file}")

def create_vscode_task():
    """Create VS Code task for running dependency analysis"""
    
    project_root = Path(__file__).parent.parent
    vscode_dir = project_root / ".vscode"
    vscode_dir.mkdir(exist_ok=True)
    
    tasks_file = vscode_dir / "tasks.json"
    
    tasks_config = {
        "version": "2.0.0",
        "tasks": [
            {
                "label": "Check Circular Dependencies",
                "type": "shell",
                "command": "python",
                "args": [
                    "architecture_intelligence/dependency_analyzer.py",
                    ".",
                    "--export",
                    "dependency_analysis.json"
                ],
                "group": {
                    "kind": "test",
                    "isDefault": False
                },
                "presentation": {
                    "echo": True,
                    "reveal": "always",
                    "focus": False,
                    "panel": "shared"
                },
                "problemMatcher": [],
                "detail": "Analyze project for circular dependencies"
            },
            {
                "label": "Visualize Dependencies",
                "type": "shell", 
                "command": "python",
                "args": [
                    "architecture_intelligence/dependency_analyzer.py",
                    ".",
                    "--visualize",
                    "dependency_graph.png"
                ],
                "group": "test",
                "presentation": {
                    "echo": True,
                    "reveal": "always",
                    "focus": False,
                    "panel": "shared"
                },
                "problemMatcher": [],
                "detail": "Create visual dependency graph"
            }
        ]
    }
    
    import json
    with open(tasks_file, 'w') as f:
        json.dump(tasks_config, f, indent=2)
    
    print(f"[SUCCESS] VS Code tasks created: {tasks_file}")

def main():
    """Setup all dependency checking infrastructure"""
    
    print("SETUP: Circular Dependency Prevention Infrastructure")
    print("=" * 60)
    
    success_count = 0
    
    # Setup Git hooks
    if setup_git_hooks():
        success_count += 1
    
    # Setup CI/CD workflow
    try:
        setup_ci_workflow()
        success_count += 1
    except Exception as e:
        print(f"[ERROR] Failed to setup CI workflow: {e}")
    
    # Setup VS Code tasks
    try:
        create_vscode_task()
        success_count += 1
    except Exception as e:
        print(f"[ERROR] Failed to setup VS Code tasks: {e}")
    
    print("\n" + "=" * 60)
    print(f"[SUCCESS] Setup complete! {success_count}/3 components installed")
    
    if success_count > 0:
        print("\nWHAT'S BEEN SET UP:")
        print("   * Pre-commit git hook to prevent circular dependencies")
        print("   * GitHub Actions workflow for CI/CD checking")
        print("   * VS Code tasks for manual dependency analysis")
        
        print("\nHOW TO USE:")
        print("   * Git commits will automatically check for cycles")
        print("   * Press Ctrl+Shift+P -> 'Tasks: Run Task' -> 'Check Circular Dependencies'")
        print("   * Run manually: python architecture_intelligence/dependency_analyzer.py .")
        
        print("\nFOR MORE INFO, SEE:")
        print("   * CIRCULAR_DEPENDENCY_PREVENTION.md")
        print("   * architecture_intelligence/dependency_analyzer.py --help")
    
if __name__ == "__main__":
    main()