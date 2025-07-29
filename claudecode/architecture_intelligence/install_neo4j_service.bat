@echo off
REM Neo4j Windows Service Installation Script
REM Run as Administrator

echo Installing Neo4j as Windows Service...

REM Check if running as administrator
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo ERROR: This script must be run as Administrator
    echo Right-click and select "Run as administrator"
    pause
    exit /b 1
)

REM Set Neo4j path (adjust if needed)
set NEO4J_PATH=%~dp0neo4j\neo4j-community-5.15.0

if not exist "%NEO4J_PATH%" (
    echo ERROR: Neo4j not found at %NEO4J_PATH%
    echo Run setup_neo4j_windows.py first
    pause
    exit /b 1
)

echo Installing Neo4j service...
cd /d "%NEO4J_PATH%"
bin\neo4j.bat install-service

if %errorLevel% equ 0 (
    echo Service installed successfully
    
    echo Setting service to start automatically...
    sc config Neo4j start= auto
    
    echo Starting Neo4j service...
    sc start Neo4j
    
    echo.
    echo Neo4j service installed and started!
    echo Browser: http://localhost:7474
    echo Username: neo4j
    echo Password: architecture123
    
) else (
    echo Service installation failed
)

pause