#!/bin/bash
# Claude Team Development Framework - Universal Installer
# Cross-platform setup script for team-ready Claude Code environment

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
FRAMEWORK_VERSION="1.0.0"
INSTALL_DIR="$HOME/.claude-team"
CONFIG_DIR="$INSTALL_DIR/config"
PYTHON_MIN_VERSION="3.8"

# Logging
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Banner
print_banner() {
    echo -e "${BLUE}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║                Claude Team Development Framework             ║"
    echo "║                     Universal Installer                     ║"
    echo "║                        Version $FRAMEWORK_VERSION                        ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is required but not installed"
        log_info "Please install Python 3.8+ and try again"
        exit 1
    fi
    
    # Check Python version
    python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
        log_error "Python $python_version detected, but $PYTHON_MIN_VERSION+ is required"
        exit 1
    fi
    log_success "Python $python_version detected"
    
    # Check pip
    if ! python3 -m pip --version &> /dev/null; then
        log_error "pip is required but not available"
        log_info "Please install pip and try again"
        exit 1
    fi
    log_success "pip is available"
    
    # Check git
    if ! command -v git &> /dev/null; then
        log_error "git is required but not installed"
        log_info "Please install git and try again"
        exit 1
    fi
    log_success "git is available"
}

# Create directory structure
create_directories() {
    log_info "Creating directory structure..."
    
    # Main directories
    mkdir -p "$INSTALL_DIR"/{config,workflows,templates,knowledge,scripts,logs}
    mkdir -p "$INSTALL_DIR/workflows"/{onboarding,development,review,deployment}
    mkdir -p "$INSTALL_DIR/templates"/{projects,commands,configs}
    
    log_success "Directory structure created"
}

# Install Python dependencies
install_dependencies() {
    log_info "Installing Python dependencies..."
    
    # Create requirements file
    cat > "$INSTALL_DIR/requirements.txt" << 'EOF'
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
EOF

    # Add circular dependency prevention dependencies
    cat >> "$INSTALL_DIR/requirements.txt" << 'EOF'

# Circular dependency prevention
networkx>=2.8.0
matplotlib>=3.5.0
EOF

    # Install in user mode to avoid permission issues
    python3 -m pip install --user -r "$INSTALL_DIR/requirements.txt"
    
    log_success "Dependencies installed (including circular dependency prevention)"
}

# Download and install core files
install_core_files() {
    log_info "Installing core framework files..."
    
    # Core Python modules
    cat > "$INSTALL_DIR/scripts/team_config_manager.py" << 'EOF'
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
            with open(config_file, 'r') as f:
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
        with open(config_file, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
            
        return default_config
EOF

    # Team installer script
    cat > "$INSTALL_DIR/scripts/claude_team_init.py" << 'EOF'
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
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Create standards file
    standards_file = claude_team_dir / 'standards.md'
    with open(standards_file, 'w') as f:
        f.write(f"""# {project_dir.name} Development Standards

## Team Claude Code Configuration

This project is configured for team development with Claude Code.

### Available Workflows
- `claude-team code-review --focus "component name"`
- `claude-team security-audit --focus "security area"`  
- `claude-team performance-audit --focus "performance concern"`

### Getting Started
1. Make sure Claude Team framework is installed: `curl -sSL https://team.dev/claude-setup | bash`
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
EOF

    chmod +x "$INSTALL_DIR/scripts/claude_team_init.py"
    
    # Copy circular dependency prevention system
    if [ -d "team_framework/circular_dependency" ]; then
        cp -r team_framework/circular_dependency "$INSTALL_DIR/"
        log_success "Circular dependency prevention system installed"
    else
        log_warning "Circular dependency system not found - skipping"
    fi
    
    log_success "Core files installed"
}

# Create team command wrapper
create_command_wrapper() {
    log_info "Creating team command wrapper..."
    
    # Create the main claude-team command
    cat > "$INSTALL_DIR/scripts/claude-team" << 'EOF'
#!/bin/bash
# Claude Team Command Wrapper
# Provides unified interface to team development workflows

CLAUDE_TEAM_DIR="$HOME/.claude-team"
CONTEXT_SCRIPT="$CLAUDE_TEAM_DIR/scripts/enhanced_workflow.py"

# Check if we're in a project directory
if [ -f ".claude-team/config.yaml" ]; then
    PROJECT_NAME=$(basename "$PWD")
    export CLAUDE_TEAM_PROJECT="$PROJECT_NAME"
else
    echo "[WARNING] Not in a Claude Team project directory"
    echo "[INFO] Run 'claude-team init' to set up this project"
    PROJECT_NAME="default"
fi

# Main command dispatch
case "$1" in
    "init")
        if [ -z "$2" ]; then
            python3 "$CLAUDE_TEAM_DIR/scripts/claude_team_init.py" "."
        else
            python3 "$CLAUDE_TEAM_DIR/scripts/claude_team_init.py" "$2"
        fi
        ;;
    "code-review"|"security-audit"|"performance-audit"|"debug-session"|"architecture-review"|"test-generation")
        python3 "$CONTEXT_SCRIPT" "$PROJECT_NAME" "$@"
        ;;
    "check-deps"|"analyze-deps")
        if [ -f "$CLAUDE_TEAM_DIR/circular_dependency/analyzer.py" ]; then
            python3 "$CLAUDE_TEAM_DIR/circular_dependency/analyzer.py" "${2:-.}" "${@:3}"
        else
            echo "[ERROR] Circular dependency analyzer not installed"
            exit 1
        fi
        ;;
    "list"|"--list"|"-l")
        python3 "$CONTEXT_SCRIPT" "$PROJECT_NAME" --list
        ;;
    "help"|"--help"|"-h")
        echo "Claude Team Development Framework"
        echo ""
        echo "Usage: claude-team <command> [options]"
        echo ""
        echo "Commands:"
        echo "  init                 Initialize current directory for team development"
        echo "  code-review          Run context-aware code review"
        echo "  security-audit       Run security audit with team context"
        echo "  performance-audit    Run performance analysis with team context"  
        echo "  debug-session        Start debugging session with project context"
        echo "  architecture-review  Review system architecture with team knowledge"
        echo "  test-generation      Generate tests with project patterns"
        echo "  check-deps           Analyze circular dependencies in current project"
        echo "  list                 List all available workflows"
        echo ""
        echo "Examples:"
        echo "  claude-team init                              # Setup current project"
        echo "  claude-team code-review --focus 'auth module'  # Review with context"
        echo "  claude-team security-audit --focus 'API endpoints'  # Security audit"
        echo "  claude-team check-deps --export analysis.json      # Dependency analysis"
        echo ""
        ;;
    *)
        echo "[ERROR] Unknown command: $1"
        echo "Run 'claude-team help' for usage information"
        exit 1
        ;;
esac
EOF

    chmod +x "$INSTALL_DIR/scripts/claude-team"
    
    log_success "Command wrapper created"
}

# Copy enhanced workflow system
copy_workflow_system() {
    log_info "Setting up enhanced workflow system..."
    
    # Check if we're running from the ClaudeCode directory
    if [ -f "context_management/enhanced_workflow.py" ]; then
        # Copy the existing system
        cp -r context_management "$INSTALL_DIR/"
        log_success "Copied existing context management system"
    else
        log_warning "Enhanced workflow system not found in current directory"
        log_info "Creating minimal workflow system..."
        
        # Create minimal workflow directory
        mkdir -p "$INSTALL_DIR/context_management"
        
        # Create minimal enhanced_workflow.py
        cat > "$INSTALL_DIR/context_management/enhanced_workflow.py" << 'EOF'
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
EOF

        chmod +x "$INSTALL_DIR/context_management/enhanced_workflow.py"
    fi
    
    # Create symlink for easy access
    ln -sf "$INSTALL_DIR/context_management/enhanced_workflow.py" "$INSTALL_DIR/scripts/enhanced_workflow.py"
}

# Setup PATH integration
setup_path() {
    log_info "Setting up PATH integration..."
    
    # Add to PATH in shell profiles
    SCRIPT_DIR="$INSTALL_DIR/scripts"
    
    # Bash
    if [ -f "$HOME/.bashrc" ]; then
        if ! grep -q "claude-team" "$HOME/.bashrc"; then
            echo "" >> "$HOME/.bashrc"
            echo "# Claude Team Framework" >> "$HOME/.bashrc"
            echo "export PATH=\"$SCRIPT_DIR:\$PATH\"" >> "$HOME/.bashrc"
            log_success "Added to ~/.bashrc"
        fi
    fi
    
    # Zsh
    if [ -f "$HOME/.zshrc" ]; then
        if ! grep -q "claude-team" "$HOME/.zshrc"; then
            echo "" >> "$HOME/.zshrc"
            echo "# Claude Team Framework" >> "$HOME/.zshrc"
            echo "export PATH=\"$SCRIPT_DIR:\$PATH\"" >> "$HOME/.zshrc"
            log_success "Added to ~/.zshrc"
        fi
    fi
    
    # Fish
    if [ -d "$HOME/.config/fish" ]; then
        mkdir -p "$HOME/.config/fish/conf.d"
        echo "set -gx PATH $SCRIPT_DIR \$PATH" > "$HOME/.config/fish/conf.d/claude-team.fish"
        log_success "Added to fish config"
    fi
    
    # Current session
    export PATH="$SCRIPT_DIR:$PATH"
}

# Create onboarding guide
create_onboarding() {
    log_info "Creating onboarding documentation..."
    
    cat > "$INSTALL_DIR/TEAM_ONBOARDING.md" << 'EOF'
# Claude Team Development Framework - Quick Start

## Welcome to Team Claude Development!

You now have access to a context-aware development framework that learns from your team's collective knowledge.

## Getting Started (5-minute setup)

### 1. Initialize Your First Project
```bash
cd /path/to/your/project
claude-team init
```

### 2. Try Your First Workflow
```bash
# Context-aware code review
claude-team code-review --focus "authentication module"

# Security audit with team knowledge
claude-team security-audit --focus "API endpoints"

# Performance analysis with project history
claude-team performance-audit --focus "database queries"
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

## Next Steps

1. Initialize your current project: `claude-team init`
2. Try a workflow: `claude-team code-review --help`
3. Share this framework with your team members
4. Watch your AI assistant get smarter over time!

---

**Need help?** Run `claude-team help` for command reference.
EOF
    
    log_success "Onboarding guide created"
}

# Run installation
main() {
    print_banner
    
    log_info "Starting Claude Team Development Framework installation..."
    log_info "This will install to: $INSTALL_DIR"
    
    # Installation steps
    check_prerequisites
    create_directories
    install_dependencies
    install_core_files
    create_command_wrapper
    copy_workflow_system
    setup_path
    create_onboarding
    
    # Success message
    echo ""
    log_success "Installation completed successfully!"
    echo ""
    echo -e "${GREEN}┌─────────────────────────────────────────────────────────────┐${NC}"
    echo -e "${GREEN}│                    🎉 READY TO GO! 🎉                      │${NC}"
    echo -e "${GREEN}└─────────────────────────────────────────────────────────────┘${NC}"
    echo ""
    echo -e "${BLUE}Quick Start:${NC}"
    echo "  1. Restart your terminal (or run: source ~/.bashrc)"
    echo "  2. cd /path/to/your/project"
    echo "  3. claude-team init"
    echo "  4. claude-team code-review --help"
    echo ""
    echo -e "${BLUE}Documentation:${NC} $INSTALL_DIR/TEAM_ONBOARDING.md"
    echo ""
    echo -e "${YELLOW}Note:${NC} You may need to restart your terminal for the 'claude-team' command to be available."
}

# Run main function
main "$@"