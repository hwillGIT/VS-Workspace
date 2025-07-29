#!/usr/bin/env python3
"""
Neo4j Windows Native Setup

Downloads and configures Neo4j Community Edition directly on Windows
without requiring Docker.
"""

import subprocess
import sys
import os
import urllib.request
import zipfile
from pathlib import Path
import time


def check_java():
    """Check if Java 11+ is available"""
    try:
        result = subprocess.run(["java", "-version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Java found: {result.stderr.split()[2]}")
            return True
        else:
            print("Java not found")
            return False
    except:
        print("Java not available")
        return False


def download_neo4j():
    """Download Neo4j Community Edition"""
    neo4j_url = "https://dist.neo4j.org/neo4j-community-5.15.0-windows.zip"
    neo4j_dir = Path("./neo4j")
    neo4j_zip = neo4j_dir / "neo4j-community-5.15.0-windows.zip"
    
    neo4j_dir.mkdir(exist_ok=True)
    
    if neo4j_zip.exists():
        print("Neo4j already downloaded")
        return str(neo4j_dir / "neo4j-community-5.15.0")
    
    print("Downloading Neo4j Community Edition...")
    try:
        urllib.request.urlretrieve(neo4j_url, neo4j_zip)
        print("Download complete")
        
        print("Extracting...")
        with zipfile.ZipFile(neo4j_zip, 'r') as zip_ref:
            zip_ref.extractall(neo4j_dir)
        
        return str(neo4j_dir / "neo4j-community-5.15.0")
        
    except Exception as e:
        print(f"Download failed: {e}")
        return None


def configure_neo4j(neo4j_path):
    """Configure Neo4j for architecture intelligence"""
    config_file = Path(neo4j_path) / "conf" / "neo4j.conf"
    
    config_updates = [
        "# Architecture Intelligence Configuration",
        "dbms.default_database=architecture",
        "dbms.security.auth_enabled=true",
        "dbms.connector.bolt.listen_address=:7687",
        "dbms.connector.http.listen_address=:7474",
        "dbms.directories.data=data",
        "dbms.directories.logs=logs"
    ]
    
    try:
        with open(config_file, 'a') as f:
            f.write('\n'.join(config_updates))
        print("Neo4j configured")
        return True
    except Exception as e:
        print(f"Configuration failed: {e}")
        return False


def set_neo4j_password(neo4j_path):
    """Set Neo4j password"""
    bin_path = Path(neo4j_path) / "bin"
    
    try:
        # Set initial password
        result = subprocess.run([
            str(bin_path / "neo4j-admin.bat"),
            "dbms", "set-initial-password", "architecture123"
        ], cwd=neo4j_path, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("Password set successfully")
            return True
        else:
            print(f"Password setting failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"Password setup error: {e}")
        return False


def start_neo4j(neo4j_path):
    """Start Neo4j service"""
    bin_path = Path(neo4j_path) / "bin"
    
    try:
        print("Starting Neo4j...")
        process = subprocess.Popen([
            str(bin_path / "neo4j.bat"), "console"
        ], cwd=neo4j_path)
        
        print(f"Neo4j started with PID: {process.pid}")
        print("Waiting for startup...")
        time.sleep(10)
        
        return process
        
    except Exception as e:
        print(f"Failed to start Neo4j: {e}")
        return None


def create_batch_scripts(neo4j_path):
    """Create convenient batch scripts"""
    
    start_script = f"""@echo off
echo Starting Neo4j Architecture Intelligence...
cd /d "{neo4j_path}"
bin\\neo4j.bat console
"""
    
    stop_script = f"""@echo off
echo Stopping Neo4j...
cd /d "{neo4j_path}"
bin\\neo4j.bat stop
"""
    
    # Write scripts
    with open("start_neo4j.bat", 'w') as f:
        f.write(start_script)
    
    with open("stop_neo4j.bat", 'w') as f:
        f.write(stop_script)
    
    print("Created start_neo4j.bat and stop_neo4j.bat")


def install_python_deps():
    """Install Python dependencies"""
    print("Installing Python dependencies...")
    packages = ["neo4j", "chromadb"]
    
    for package in packages:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", package], 
                         capture_output=True, text=True)
            print(f"Installed {package}")
        except:
            print(f"Warning: Could not install {package}")


def main():
    print("Neo4j Windows Native Setup")
    print("=" * 30)
    
    # Check Java
    if not check_java():
        print("\nERROR: Java 11+ is required for Neo4j")
        print("Download from: https://adoptium.net/")
        return False
    
    # Download Neo4j
    neo4j_path = download_neo4j()
    if not neo4j_path:
        return False
    
    # Configure
    if not configure_neo4j(neo4j_path):
        return False
    
    # Set password
    if not set_neo4j_password(neo4j_path):
        print("Warning: Password setup failed, you may need to set it manually")
    
    # Create scripts
    create_batch_scripts(neo4j_path)
    
    # Install Python deps
    install_python_deps()
    
    print("\n" + "=" * 50)
    print("Setup Complete!")
    print("\nTo start Neo4j:")
    print("  1. Run: start_neo4j.bat")
    print("  2. Wait ~30 seconds for startup")
    print("  3. Open: http://localhost:7474")
    print("  4. Login: neo4j / architecture123")
    
    print("\nTo stop Neo4j:")
    print("  1. Run: stop_neo4j.bat")
    print("  2. Or press Ctrl+C in the console window")
    
    # Ask if user wants to start now
    start_now = input("\nStart Neo4j now? (y/n): ").lower()
    if start_now == 'y':
        process = start_neo4j(neo4j_path)
        if process:
            print("\nNeo4j is starting...")
            print("Open http://localhost:7474 in ~30 seconds")
            print("Press Ctrl+C to stop")
            try:
                process.wait()
            except KeyboardInterrupt:
                print("\nStopping Neo4j...")
                process.terminate()
    
    return True


if __name__ == "__main__":
    main()