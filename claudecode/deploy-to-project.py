#!/usr/bin/env python3
"""
Deploy Circular Dependency Prevention System to Another Project

Usage: python deploy-to-project.py /path/to/target/project
"""

import sys
import shutil
from pathlib import Path

def deploy_to_project(target_path: Path):
    """Deploy dependency prevention system to target project"""
    
    if not target_path.exists():
        print(f"[ERROR] Target path does not exist: {target_path}")
        return False
    
    source_root = Path(__file__).parent
    
    # Files to copy
    files_to_copy = [
        ("architecture_intelligence/dependency_analyzer.py", "tools/dependency_analyzer.py"),
        ("CIRCULAR_DEPENDENCY_PREVENTION.md", "CIRCULAR_DEPENDENCY_PREVENTION.md"),
        ("CIRCULAR_DEPENDENCY_SOLUTION.md", "docs/CIRCULAR_DEPENDENCY_SOLUTION.md"),
        ("tools/setup-dependency-checking.py", "tools/setup-dependency-checking.py"),
        ("tools/pre-commit-dependency-check.sh", "tools/pre-commit-dependency-check.sh")
    ]
    
    print(f"[DEPLOY] Deploying to {target_path}")
    print("=" * 60)
    
    success_count = 0
    
    for source_file, target_file in files_to_copy:
        source_path = source_root / source_file
        target_file_path = target_path / target_file
        
        if source_path.exists():
            # Create target directory if needed
            target_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy file
            shutil.copy2(source_path, target_file_path)
            print(f"[OK] Copied {source_file} -> {target_file}")
            success_count += 1
        else:
            print(f"[SKIP] Source not found: {source_file}")
    
    # Create quick-start README
    readme_content = f"""# Circular Dependency Prevention

This project now includes circular dependency prevention tools.

## Quick Start

```bash
# Check for circular dependencies
python tools/dependency_analyzer.py .

# Export detailed analysis
python tools/dependency_analyzer.py . --export analysis.json

# Create visual graph
python tools/dependency_analyzer.py . --visualize deps.png

# Setup automation (git hooks, CI/CD)
python tools/setup-dependency-checking.py
```

## Documentation

- `CIRCULAR_DEPENDENCY_PREVENTION.md` - Complete prevention guide
- `docs/CIRCULAR_DEPENDENCY_SOLUTION.md` - Implementation overview

## Global Rule

**Zero Tolerance for Circular Dependencies** - All code must follow strict dependency hierarchy.

## Support

For questions or issues with the dependency prevention system, refer to the documentation or contact the architecture team.
"""
    
    readme_path = target_path / "DEPENDENCY_PREVENTION_README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"[OK] Created quick-start guide: DEPENDENCY_PREVENTION_README.md")
    success_count += 1
    
    print("=" * 60)
    print(f"[SUCCESS] Deployment complete! {success_count} files deployed")
    
    print(f"""
[NEXT STEPS] In the target project:

1. Read the quick-start guide:
   cat DEPENDENCY_PREVENTION_README.md

2. Run initial analysis:
   python tools/dependency_analyzer.py .

3. Setup automation:
   python tools/setup-dependency-checking.py

4. Add to team workflow:
   - Include in code review checklist
   - Add to CI/CD pipeline
   - Train team on prevention patterns
""")
    
    return True

def main():
    if len(sys.argv) != 2:
        print("Usage: python deploy-to-project.py /path/to/target/project")
        print("")
        print("Examples:")
        print("  python deploy-to-project.py ../MyOtherProject")
        print("  python deploy-to-project.py /home/user/new-project")
        print("  python deploy-to-project.py C:\\Projects\\WebApp")
        sys.exit(1)
    
    target_path = Path(sys.argv[1])
    success = deploy_to_project(target_path)
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()