#!/bin/bash
# Pre-commit hook to check for circular dependencies
# Place this file in .git/hooks/pre-commit and make it executable

set -e

echo "🔍 Checking for circular dependencies..."

# Check if dependency analyzer exists
ANALYZER_PATH="architecture_intelligence/dependency_analyzer.py"
if [ ! -f "$ANALYZER_PATH" ]; then
    echo "⚠️  Dependency analyzer not found at $ANALYZER_PATH"
    echo "🔄 Skipping circular dependency check"
    exit 0
fi

# Run quick analysis (don't export files)
python "$ANALYZER_PATH" . > /dev/null 2>&1

# Check exit code
if [ $? -eq 0 ]; then
    echo "✅ No circular dependencies detected"
    exit 0
else
    echo ""
    echo "❌ CIRCULAR DEPENDENCIES DETECTED!"
    echo ""
    echo "🔧 To fix:"
    echo "   1. Run: python $ANALYZER_PATH . --export analysis.json"
    echo "   2. Review the analysis results"
    echo "   3. Refactor using strategies in CIRCULAR_DEPENDENCY_PREVENTION.md"
    echo "   4. Commit again after fixing"
    echo ""
    echo "💡 Or override with: git commit --no-verify"
    echo ""
    exit 1
fi