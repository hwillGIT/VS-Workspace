@echo off
REM Windows batch file to quickly generate context

set PROJECT=%1
set WORK=%2

if "%PROJECT%"=="" (
    echo Usage: auto_context.bat PROJECT "work description"
    echo Example: auto_context.bat trading_system "implementing REST API"
    exit /b 1
)

if "%WORK%"=="" (
    echo Creating daily context for %PROJECT%...
    python smart_context_export.py %PROJECT% daily
) else (
    echo Creating context for: %WORK%
    python smart_context_export.py %PROJECT% work "%WORK%"
)

echo.
echo Context file created! Include it in your message to Claude.