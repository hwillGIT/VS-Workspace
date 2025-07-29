# Claude Team Development Framework - Windows PowerShell Installer
# Cross-platform setup script for team-ready Claude Code environment

param(
    [string]$InstallDir = "$env:USERPROFILE\.claude-team",
    [switch]$Force
)

# Configuration
$FrameworkVersion = "1.0.0"
$PythonMinVersion = [Version]"3.8"

# Colors for output
function Write-ColorOutput($ForegroundColor) {
    $fc = $host.UI.RawUI.ForegroundColor
    $host.UI.RawUI.ForegroundColor = $ForegroundColor
    if ($args) {
        Write-Output $args
    }
    $host.UI.RawUI.ForegroundColor = $fc
}

function Log-Info($Message) {
    Write-ColorOutput Blue "[INFO] $Message"
}

function Log-Success($Message) {
    Write-ColorOutput Green "[SUCCESS] $Message"
}

function Log-Warning($Message) {
    Write-ColorOutput Yellow "[WARNING] $Message"
}

function Log-Error($Message) {
    Write-ColorOutput Red "[ERROR] $Message"
}

# Banner
function Print-Banner {
    Write-ColorOutput Blue @"
╔══════════════════════════════════════════════════════════════╗
║                Claude Team Development Framework             ║
║                     Windows Installer                       ║
║                        Version $FrameworkVersion                        ║
╚══════════════════════════════════════════════════════════════╝
"@
}

# Check prerequisites
function Test-Prerequisites {
    Log-Info "Checking prerequisites..."
    
    # Check Python
    try {
        $pythonVersion = python --version 2>&1
        if ($LASTEXITCODE -ne 0) {
            throw "Python not found"
        }
        
        # Parse version
        $versionString = ($pythonVersion -split " ")[1]
        $version = [Version]$versionString
        
        if ($version -lt $PythonMinVersion) {
            Log-Error "Python $version detected, but $PythonMinVersion+ is required"
            exit 1
        }
        
        Log-Success "Python $version detected"
    }
    catch {
        Log-Error "Python 3.8+ is required but not installed"
        Log-Info "Please install Python from https://python.org and try again"
        exit 1
    }
    
    # Check pip
    try {
        python -m pip --version | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "pip not available"
        }
        Log-Success "pip is available"
    }
    catch {
        Log-Error "pip is required but not available"
        Log-Info "Please ensure pip is installed with Python"
        exit 1
    }
    
    # Check git
    try {
        git --version | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "git not found"
        }
        Log-Success "git is available"
    }
    catch {
        Log-Error "git is required but not installed"
        Log-Info "Please install git from https://git-scm.com and try again"
        exit 1
    }
}

# Create directory structure
function New-Directories {
    Log-Info "Creating directory structure..."
    
    $dirs = @(
        "$InstallDir",
        "$InstallDir\config",
        "$InstallDir\workflows",
        "$InstallDir\workflows\onboarding",
        "$InstallDir\workflows\development", 
        "$InstallDir\workflows\review",
        "$InstallDir\workflows\deployment",
        "$InstallDir\templates",
        "$InstallDir\templates\projects",
        "$InstallDir\templates\commands",
        "$InstallDir\templates\configs",
        "$InstallDir\knowledge",
        "$InstallDir\scripts",
        "$InstallDir\logs"
    )
    
    foreach ($dir in $dirs) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
    
    Log-Success "Directory structure created"
}

# Install Python dependencies
function Install-Dependencies {
    Log-Info "Installing Python dependencies..."
    
    # Create requirements file
    $requirements = @"
# Core dependencies
chromadb>=0.4.0
sentence-transformers>=2.2.0
pyyaml>=6.0
click>=8.0.0
rich>=13.0.0
pathlib2>=2.3.0

# Optional performance enhancements
numpy>=1.21.0
pandas>=1.3.0

# Development tools
pytest>=7.0.0
pytest-asyncio>=0.21.0
black>=22.0.0
"@
    
    $requirements | Out-File -FilePath "$InstallDir\requirements.txt" -Encoding UTF8
    
    # Install dependencies
    python -m pip install --user -r "$InstallDir\requirements.txt"
    
    if ($LASTEXITCODE -ne 0) {
        Log-Error "Failed to install Python dependencies"
        exit 1
    }
    
    Log-Success "Dependencies installed"
}

# Install core files
function Install-CoreFiles {
    Log-Info "Installing core framework files..."
    
    # Team config manager
    $teamConfigManager = @'
"""
Team Configuration Manager
Handles team-wide settings and project-specific configurations
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

class TeamConfigManager:
    def __init__(self):
        self.team_config_dir = Path.home() / '.claude-team' / 'config'
        self.team_config_dir.mkdir(parents=True, exist_ok=True)
        
    def get_team_config(self) -> Dict[str, Any]:
        """Load team-wide configuration"""
        config_file = self.team_config_dir / 'team.yaml'
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        return self._create_default_team_config()
    
    def _create_default_team_config(self) -> Dict[str, Any]:
        """Create default team configuration"""
        default_config = {
            'team': {
                'name': 'Development Team',
                'version': '1.0.0'
            },
            'context': {
                'shared_database': True,
                'sync_enabled': True,
                'max_context_items': 20
            },
            'workflows': {
                'default_namespace': 'dev',
                'auto_context': True
            },
            'security': {
                'filter_secrets': True,
                'access_control': 'team'
            }
        }
        
        config_file = self.team_config_dir / 'team.yaml'
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(default_config, f, default_flow_style=False)
            
        return default_config
'@
    
    $teamConfigManager | Out-File -FilePath "$InstallDir\scripts\team_config_manager.py" -Encoding UTF8
    
    # Claude team init script
    $claudeTeamInit = @'
#!/usr/bin/env python3
"""
Claude Team Initialization Script
Sets up a new project for team development with Claude Code
"""

import os
import sys
import yaml
import shutil
from pathlib import Path
from team_config_manager import TeamConfigManager

def init_project(project_path: str):
    """Initialize a project for team Claude development"""
    project_dir = Path(project_path).resolve()
    
    if not project_dir.exists():
        print(f"[ERROR] Project directory {project_dir} does not exist")
        sys.exit(1)
    
    print(f"[INFO] Initializing Claude Team framework in {project_dir}")
    
    # Create .claude-team directory
    claude_team_dir = project_dir / '.claude-team'
    claude_team_dir.mkdir(exist_ok=True)
    
    # Create project config
    config = {
        'project': {
            'name': project_dir.name,
            'type': 'software_development',
            'version': '1.0.0'
        },
        'team': {
            'shared_context': True,
            'auto_sync': True
        },
        'workflows': {
            'enabled': ['code-review', 'security-audit', 'performance-audit'],
            'custom_templates': []
        }
    }
    
    config_file = claude_team_dir / 'config.yaml'
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Create standards file
    standards_file = claude_team_dir / 'standards.md'
    with open(standards_file, 'w', encoding='utf-8') as f:
        f.write(f"""# {project_dir.name} Development Standards

## Team Claude Code Configuration

This project is configured for team development with Claude Code.

### Available Workflows
- `claude-team code-review --focus "component name"`
- `claude-team security-audit --focus "security area"`  
- `claude-team performance-audit --focus "performance concern"`

### Getting Started
1. Make sure Claude Team framework is installed
2. Navigate to project directory: `cd {project_dir}`
3. Use team workflows: `claude-team code-review --help`

### Team Standards
- All code must pass team review workflows
- Security audits required for auth/API changes
- Performance audits required for database/scaling changes
""")
    
    print(f"[SUCCESS] Project {project_dir.name} initialized for team Claude development")
    print(f"[INFO] Configuration saved to {config_file}")
    print(f"[INFO] Standards documented in {standards_file}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python claude_team_init.py <project_directory>")
        sys.exit(1)
    
    init_project(sys.argv[1])
'@
    
    $claudeTeamInit | Out-File -FilePath "$InstallDir\scripts\claude_team_init.py" -Encoding UTF8
    
    Log-Success "Core files installed"
}

# Create command wrapper
function New-CommandWrapper {
    Log-Info "Creating team command wrapper..."
    
    # PowerShell wrapper script
    $wrapperScript = @"
# Claude Team Command Wrapper - PowerShell Version
param(
    [Parameter(Position=0)]
    [string]`$Command,
    
    [Parameter(Position=1, ValueFromRemainingArguments=`$true)]
    [string[]]`$Arguments
)

`$claudeTeamDir = "`$env:USERPROFILE\.claude-team"
`$contextScript = "`$claudeTeamDir\scripts\enhanced_workflow.py"

# Check if we're in a project directory
if (Test-Path ".claude-team\config.yaml") {
    `$projectName = (Get-Item .).Name
    `$env:CLAUDE_TEAM_PROJECT = `$projectName
} else {
    Write-Warning "Not in a Claude Team project directory"
    Write-Host "[INFO] Run 'claude-team init' to set up this project"
    `$projectName = "default"
}

# Main command dispatch
switch (`$Command) {
    "init" {
        if (`$Arguments.Count -eq 0) {
            python "`$claudeTeamDir\scripts\claude_team_init.py" "."
        } else {
            python "`$claudeTeamDir\scripts\claude_team_init.py" `$Arguments[0]
        }
    }
    
    { `$_ -in @("code-review", "security-audit", "performance-audit", "debug-session", "architecture-review", "test-generation") } {
        python "`$contextScript" `$projectName `$Command @Arguments
    }
    
    { `$_ -in @("list", "--list", "-l") } {
        python "`$contextScript" `$projectName --list
    }
    
    { `$_ -in @("help", "--help", "-h") } {
        Write-Host "Claude Team Development Framework"
        Write-Host ""
        Write-Host "Usage: claude-team <command> [options]"
        Write-Host ""
        Write-Host "Commands:"
        Write-Host "  init                 Initialize current directory for team development"
        Write-Host "  code-review          Run context-aware code review"
        Write-Host "  security-audit       Run security audit with team context"
        Write-Host "  performance-audit    Run performance analysis with team context"
        Write-Host "  debug-session        Start debugging session with project context"
        Write-Host "  architecture-review  Review system architecture with team knowledge"
        Write-Host "  test-generation      Generate tests with project patterns"
        Write-Host "  list                 List all available workflows"
        Write-Host ""
        Write-Host "Examples:"
        Write-Host "  claude-team init                              # Setup current project"
        Write-Host "  claude-team code-review -focus 'auth module'  # Review with context"
        Write-Host "  claude-team security-audit -focus 'API endpoints'  # Security audit"
    }
    
    default {
        Write-Error "Unknown command: `$Command"
        Write-Host "Run 'claude-team help' for usage information"
        exit 1
    }
}
"@
    
    $wrapperScript | Out-File -FilePath "$InstallDir\scripts\claude-team.ps1" -Encoding UTF8
    
    Log-Success "Command wrapper created"
}

# Copy workflow system
function Copy-WorkflowSystem {
    Log-Info "Setting up enhanced workflow system..."
    
    # Check if we're running from the ClaudeCode directory
    if (Test-Path "context_management\enhanced_workflow.py") {
        # Copy the existing system
        Copy-Item -Path "context_management" -Destination "$InstallDir\" -Recurse -Force
        Log-Success "Copied existing context management system"
    } else {
        Log-Warning "Enhanced workflow system not found in current directory"
        Log-Info "Creating minimal workflow system..."
        
        # Create minimal workflow directory
        New-Item -ItemType Directory -Path "$InstallDir\context_management" -Force | Out-Null
        
        # Create minimal enhanced_workflow.py
        $minimalWorkflow = @'
#!/usr/bin/env python3
"""
Simplified Enhanced Workflow Manager for Team Framework
"""

import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Enhanced Context-Aware Workflow Manager")
    parser.add_argument("project", help="Project name")
    parser.add_argument("workflow", nargs="?", help="Workflow type")
    parser.add_argument("-f", "--focus", help="Focus area")
    parser.add_argument("-l", "--list", action="store_true", help="List workflows")
    
    args = parser.parse_args()
    
    if args.list:
        print("Available Workflows:")
        print("\nDEVELOPMENT:")
        print("  - code-review")
        print("  - debug-session") 
        print("  - architecture-review")
        print("\nQUALITY:")
        print("  - security-audit")
        print("  - performance-audit")
        print("  - test-generation")
        return
    
    if not args.workflow:
        parser.error("workflow argument required unless using --list")
    
    # Generate basic template
    focus_text = f" focusing on {args.focus}" if args.focus else ""
    
    output = f"""# {args.workflow.title().replace('-', ' ')} for {args.project}

## Context
Project: {args.project}
Workflow: {args.workflow}
Focus: {args.focus or 'General analysis'}

## Instructions
Please perform a {args.workflow.replace('-', ' ')}{focus_text}.

## Next Steps
1. Review the analysis
2. Implement recommended changes
3. Update team knowledge base
"""
    
    output_file = f"{args.workflow.upper()}_CONTEXT.md"
    Path(output_file).write_text(output, encoding='utf-8')
    
    print(f"[SUCCESS] Generated {output_file}")
    print("Copy this context to your Claude conversation for enhanced results")

if __name__ == "__main__":
    main()
'@
        
        $minimalWorkflow | Out-File -FilePath "$InstallDir\context_management\enhanced_workflow.py" -Encoding UTF8
        
        # Create symlink equivalent
        Copy-Item -Path "$InstallDir\context_management\enhanced_workflow.py" -Destination "$InstallDir\scripts\enhanced_workflow.py"
    }
}

# Setup PATH integration  
function Set-PathIntegration {
    Log-Info "Setting up PATH integration..."
    
    # Add to PowerShell profile
    $profilePath = $PROFILE.CurrentUserAllHosts
    $profileDir = Split-Path $profilePath -Parent
    
    if (!(Test-Path $profileDir)) {
        New-Item -ItemType Directory -Path $profileDir -Force | Out-Null
    }
    
    $pathAddition = @"

# Claude Team Framework
`$env:PATH += ";$InstallDir\scripts"

# Create claude-team function for PowerShell
function claude-team {
    param([Parameter(ValueFromRemainingArguments=`$true)]`$args)
    & powershell -ExecutionPolicy Bypass -File "$InstallDir\scripts\claude-team.ps1" @args
}
"@
    
    if (Test-Path $profilePath) {
        $content = Get-Content $profilePath -Raw
        if ($content -notmatch "Claude Team Framework") {
            Add-Content -Path $profilePath -Value $pathAddition
            Log-Success "Added to PowerShell profile: $profilePath"
        }
    } else {
        $pathAddition | Out-File -FilePath $profilePath -Encoding UTF8
        Log-Success "Created PowerShell profile: $profilePath"
    }
    
    # Add to current session
    $env:PATH += ";$InstallDir\scripts"
    
    # Create function for current session
    Invoke-Expression @"
function global:claude-team {
    param([Parameter(ValueFromRemainingArguments=`$true)]`$args)
    & powershell -ExecutionPolicy Bypass -File "$InstallDir\scripts\claude-team.ps1" @args
}
"@
}

# Create onboarding guide
function New-OnboardingGuide {
    Log-Info "Creating onboarding documentation..."
    
    $onboardingContent = @'
# Claude Team Development Framework - Quick Start

## Welcome to Team Claude Development!

You now have access to a context-aware development framework that learns from your team's collective knowledge.

## Getting Started (5-minute setup)

### 1. Initialize Your First Project
```powershell
cd C:\path\to\your\project
claude-team init
```

### 2. Try Your First Workflow
```powershell
# Context-aware code review
claude-team code-review -focus "authentication module"

# Security audit with team knowledge
claude-team security-audit -focus "API endpoints"

# Performance analysis with project history
claude-team performance-audit -focus "database queries"
```

### 3. Use the Generated Context
Each workflow creates a markdown file (e.g., `CODE_REVIEW_CONTEXT.md`) that you copy into your Claude conversation for enhanced results.

## Available Commands

- `claude-team init` - Set up current project for team development
- `claude-team code-review` - Context-aware code quality review
- `claude-team security-audit` - Security analysis with team patterns
- `claude-team performance-audit` - Performance review with benchmarks
- `claude-team debug-session` - Debugging with project history
- `claude-team architecture-review` - System design analysis
- `claude-team test-generation` - Test creation with project patterns
- `claude-team list` - Show all available workflows

## How It Works

1. **Team Knowledge**: The system accumulates knowledge from all team members
2. **Project Context**: Each project maintains its own context database
3. **Workflow Intelligence**: Different workflows use appropriate context types
4. **Continuous Learning**: The system gets smarter with each use

## Team Benefits

- **5-minute onboarding** for new developers
- **Consistent development practices** across the team
- **Accumulated team knowledge** that doesn't leave with individuals
- **Context-aware AI assistance** that knows your project

## Windows-Specific Notes

- Commands work in both PowerShell and Command Prompt
- PowerShell functions are available after restarting your shell
- All file paths use Windows-style backslashes automatically

## Next Steps

1. Restart PowerShell to load the claude-team function
2. Initialize your current project: `claude-team init`
3. Try a workflow: `claude-team code-review -help`
4. Share this framework with your team members
5. Watch your AI assistant get smarter over time!

---

**Need help?** Run `claude-team help` for command reference.
'@
    
    $onboardingContent | Out-File -FilePath "$InstallDir\TEAM_ONBOARDING.md" -Encoding UTF8
    
    Log-Success "Onboarding guide created"
}

# Main installation function
function Install-ClaudeTeamFramework {
    Print-Banner
    
    Log-Info "Starting Claude Team Development Framework installation..."
    Log-Info "This will install to: $InstallDir"
    
    # Installation steps
    Test-Prerequisites
    New-Directories
    Install-Dependencies
    Install-CoreFiles
    New-CommandWrapper
    Copy-WorkflowSystem  
    Set-PathIntegration
    New-OnboardingGuide
    
    # Success message
    Write-Host ""
    Log-Success "Installation completed successfully!"
    Write-Host ""
    Write-ColorOutput Green "┌─────────────────────────────────────────────────────────────┐"
    Write-ColorOutput Green "│                    🎉 READY TO GO! 🎉                      │"
    Write-ColorOutput Green "└─────────────────────────────────────────────────────────────┘"
    Write-Host ""
    Write-ColorOutput Blue "Quick Start:"
    Write-Host "  1. Restart PowerShell to load the claude-team function"
    Write-Host "  2. cd C:\path\to\your\project"
    Write-Host "  3. claude-team init"
    Write-Host "  4. claude-team code-review -help"
    Write-Host ""
    Write-ColorOutput Blue "Documentation:" 
    Write-Host "  $InstallDir\TEAM_ONBOARDING.md"
    Write-Host ""
    Write-ColorOutput Yellow "Note:"
    Write-Host "  You may need to restart PowerShell for the 'claude-team' function to be available."
}

# Run installation
Install-ClaudeTeamFramework