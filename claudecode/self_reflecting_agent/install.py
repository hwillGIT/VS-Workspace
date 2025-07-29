#!/usr/bin/env python3
"""
Installation script for Self-Reflecting Claude Code Agent system.

This script sets up the agent system globally for use across all Claude Code instances.
"""

import subprocess
import sys
import os
from pathlib import Path
import shutil


def run_command(command, check=True):
    """Run a shell command and return the result."""
    print(f"Running: {command}")
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=check,
            capture_output=True,
            text=True
        )
        if result.stdout:
            print(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {command}")
        print(f"Error output: {e.stderr}")
        if check:
            raise
        return e


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required. Current version: {version.major}.{version.minor}")
        return False
    print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    return True


def install_package():
    """Install the package using pip."""
    print("\n📦 Installing Self-Reflecting Agent package...")
    
    # Get the directory containing this script
    script_dir = Path(__file__).parent
    
    # Install in development mode so it's editable
    result = run_command(f'pip install -e "{script_dir}"')
    
    if result.returncode == 0:
        print("✅ Package installed successfully!")
        return True
    else:
        print("❌ Package installation failed!")
        return False


def verify_installation():
    """Verify the installation by testing CLI commands."""
    print("\n🔍 Verifying installation...")
    
    # Test if the CLI commands are available
    commands_to_test = ['sra', 'self-reflecting-agent']
    
    for cmd in commands_to_test:
        result = run_command(f'{cmd} --help', check=False)
        if result.returncode == 0:
            print(f"✅ Command '{cmd}' is available")
        else:
            print(f"❌ Command '{cmd}' is not available")
            return False
    
    return True


def create_global_config():
    """Create initial global configuration."""
    print("\n⚙️ Setting up global configuration...")
    
    try:
        # Import here to ensure package is installed
        from self_reflecting_agent.global_manager import GlobalAgentManager
        
        # This will create the global config directory and files
        manager = GlobalAgentManager()
        print(f"✅ Global configuration created at: {manager.global_config_dir}")
        return True
        
    except ImportError as e:
        print(f"❌ Could not import GlobalAgentManager: {e}")
        return False
    except Exception as e:
        print(f"❌ Error creating global configuration: {e}")
        return False


def show_usage_info():
    """Show usage information after successful installation."""
    print("\n" + "="*60)
    print("🎉 INSTALLATION COMPLETE!")
    print("="*60)
    print("\n📋 Available Commands:")
    print("  sra task \"description\"                    - Execute a development task")
    print("  sra workflow domain workflow \"description\" - Execute domain workflow")
    print("  sra info                                  - Show system information")
    print("\n📖 Usage Examples:")
    print('  sra task "Create a REST API for user management"')
    print('  sra workflow software_development architecture_review "Review my design"')
    print('  sra workflow software_development comprehensive_project_planning "Plan an e-commerce app"')
    print("\n🗂️ The agent system will automatically:")
    print("  • Detect your project type and context")
    print("  • Parse CLAUDE.md files for project configuration")
    print("  • Use appropriate domain-specific agents")
    print("  • Maintain persistent memory across sessions")
    print("\n🌐 Global Configuration:")
    print("  • Windows: %APPDATA%\\SelfReflectingAgent\\")
    print("  • Unix/Mac: ~/.self_reflecting_agent/")
    print("\n📚 For more information, see the README.md file.")


def main():
    """Main installation process."""
    print("🚀 Self-Reflecting Claude Code Agent - Installation")
    print("="*60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install the package
    if not install_package():
        print("\n❌ Installation failed at package installation step!")
        sys.exit(1)
    
    # Verify installation
    if not verify_installation():
        print("\n❌ Installation verification failed!")
        print("The package was installed but CLI commands are not working.")
        print("You may need to restart your terminal or check your PATH.")
        sys.exit(1)
    
    # Create global configuration
    if not create_global_config():
        print("\n⚠️ Package installed but global configuration setup failed.")
        print("You can still use the system, but some features may not work optimally.")
    
    # Show usage information
    show_usage_info()


if __name__ == "__main__":
    main()