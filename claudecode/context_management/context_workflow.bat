@echo off
REM Smart context workflow manager

set PROJECT=%1
set MODE=%2

if "%PROJECT%"=="" (
    echo Context Workflow Manager
    echo.
    echo Usage: context_workflow.bat PROJECT MODE [args]
    echo.
    echo Modes:
    echo   start     - Start daily work (creates daily context)
    echo   work      - Focus on specific work (requires description)
    echo   research  - Research topics (requires query terms)
    echo   save      - Archive current context
    echo   all       - Generate all context types
    echo.
    echo Examples:
    echo   context_workflow.bat trading_system start
    echo   context_workflow.bat trading_system work "API endpoints"
    echo   context_workflow.bat trading_system research security authentication
    echo   context_workflow.bat trading_system save
    exit /b 1
)

cd /d "%~dp0"

if "%MODE%"=="start" (
    echo === Starting daily work session for %PROJECT% ===
    python smart_context_export.py %PROJECT% daily -o CONTEXT_DAILY.md
    copy CONTEXT_DAILY.md CONTEXT.md >nul
    echo.
    echo ✓ Daily context ready in CONTEXT.md
    echo ✓ Backup saved as CONTEXT_DAILY.md
    
) else if "%MODE%"=="work" (
    set WORK_DESC=%3
    if "%WORK_DESC%"=="" (
        echo ERROR: Work description required
        echo Usage: context_workflow.bat %PROJECT% work "description"
        exit /b 1
    )
    echo === Generating work-specific context ===
    python smart_context_export.py %PROJECT% work %3 %4 %5 %6 %7 %8 %9
    echo.
    echo ✓ Work context ready in CONTEXT.md
    
) else if "%MODE%"=="research" (
    shift
    shift
    set QUERIES=%1 %2 %3 %4 %5 %6 %7 %8 %9
    echo === Researching: %QUERIES% ===
    python smart_context_export.py %PROJECT% query %QUERIES%
    echo.
    echo ✓ Research context ready in CONTEXT.md
    
) else if "%MODE%"=="save" (
    if not exist context_archive mkdir context_archive
    set TIMESTAMP=%date:~-4%%date:~4,2%%date:~7,2%_%time:~0,2%%time:~3,2%
    set TIMESTAMP=%TIMESTAMP: =0%
    copy CONTEXT.md "context_archive\CONTEXT_%PROJECT%_%TIMESTAMP%.md" >nul
    echo ✓ Archived current context
    
) else if "%MODE%"=="all" (
    echo === Generating all context types ===
    
    echo.
    echo 1. Daily context...
    python smart_context_export.py %PROJECT% daily -o CONTEXT_DAILY.md
    
    echo.
    echo 2. Common work contexts...
    python smart_context_export.py %PROJECT% work "current implementation" -o CONTEXT_WORK.md
    
    echo.
    echo 3. Research topics...
    python smart_context_export.py %PROJECT% query "decisions" "patterns" "issues" -o CONTEXT_RESEARCH.md
    
    echo.
    echo ✓ Generated:
    echo   - CONTEXT_DAILY.md
    echo   - CONTEXT_WORK.md  
    echo   - CONTEXT_RESEARCH.md
    
) else (
    echo ERROR: Unknown mode '%MODE%'
    context_workflow.bat
)

echo.
echo TIP: Copy CONTEXT.md content to include with your Claude message