#!/usr/bin/env python3
"""
Comprehensive Test Runner for Trading System

This script runs all tests and generates coverage reports.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_tests():
    """Run all tests with coverage reporting."""
    print("="*80)
    print(" TRADING SYSTEM - COMPREHENSIVE TEST SUITE")
    print("="*80)
    print()
    
    # Ensure we're in the correct directory
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    # Check if pytest is available
    try:
        import pytest
        import pytest_cov
        print("✓ pytest and pytest-cov are available")
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("\nTo install required packages:")
        print("pip install pytest pytest-cov pytest-asyncio")
        return False
    
    # Run tests with coverage
    print("\n" + "-"*60)
    print(" RUNNING TESTS WITH COVERAGE")
    print("-"*60)
    
    test_command = [
        sys.executable, "-m", "pytest",
        "trading_system/tests/",
        "--verbose",
        "--tb=short",
        "--cov=trading_system",
        "--cov-report=html:htmlcov",
        "--cov-report=term-missing",
        "--cov-report=xml",
        "--cov-fail-under=80",
        "-x",  # Stop on first failure for faster feedback
    ]
    
    try:
        result = subprocess.run(test_command, capture_output=False, text=True)
        
        if result.returncode == 0:
            print("\n" + "="*80)
            print(" ✓ ALL TESTS PASSED!")
            print("="*80)
            print("\nCoverage reports generated:")
            print("- HTML report: htmlcov/index.html")
            print("- XML report: coverage.xml")
            print("\nTo view HTML coverage report:")
            print("  python -m http.server 8000 --directory htmlcov")
            print("  Then open: http://localhost:8000")
            return True
        else:
            print("\n" + "="*80)
            print(" ✗ SOME TESTS FAILED")
            print("="*80)
            print(f"Exit code: {result.returncode}")
            return False
            
    except FileNotFoundError:
        print("✗ pytest not found. Please install: pip install pytest pytest-cov")
        return False
    except Exception as e:
        print(f"✗ Error running tests: {e}")
        return False

def run_specific_test_suites():
    """Run specific test suites individually."""
    test_suites = [
        ("Technical Analysis Agent", "test_technical_analysis_agent.py"),
        ("ML Ensemble Agent", "test_ml_ensemble_agent.py"),
        ("Risk Modeling Agent", "test_risk_modeling_agent.py"),
        ("Recommendation Agent", "test_recommendation_agent.py"),
        ("Data Universe Agent", "test_data_universe.py"),
    ]
    
    print("\n" + "-"*60)
    print(" INDIVIDUAL TEST SUITE RESULTS")
    print("-"*60)
    
    results = {}
    
    for suite_name, test_file in test_suites:
        test_path = f"trading_system/tests/{test_file}"
        
        if not Path(test_path).exists():
            print(f"⚠  {suite_name}: Test file not found ({test_file})")
            results[suite_name] = "MISSING"
            continue
        
        print(f"\nRunning {suite_name}...")
        
        cmd = [
            sys.executable, "-m", "pytest",
            test_path,
            "--tb=line",
            "-q"  # Quiet mode
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Count passed tests
                output_lines = result.stdout.split('\n')
                for line in output_lines:
                    if "passed" in line:
                        print(f"✓ {suite_name}: {line.strip()}")
                        results[suite_name] = "PASSED"
                        break
                else:
                    print(f"✓ {suite_name}: Tests passed")
                    results[suite_name] = "PASSED"
            else:
                print(f"✗ {suite_name}: Tests failed")
                if result.stdout:
                    print(f"  Output: {result.stdout.strip()}")
                if result.stderr:
                    print(f"  Error: {result.stderr.strip()}")
                results[suite_name] = "FAILED"
                
        except Exception as e:
            print(f"✗ {suite_name}: Error running tests - {e}")
            results[suite_name] = "ERROR"
    
    # Summary
    print("\n" + "="*60)
    print(" TEST SUITE SUMMARY")
    print("="*60)
    
    passed = sum(1 for r in results.values() if r == "PASSED")
    failed = sum(1 for r in results.values() if r == "FAILED")
    missing = sum(1 for r in results.values() if r == "MISSING")
    errors = sum(1 for r in results.values() if r == "ERROR")
    
    for suite_name, result in results.items():
        status_symbol = {
            "PASSED": "✓",
            "FAILED": "✗",
            "MISSING": "⚠",
            "ERROR": "!"
        }.get(result, "?")
        
        print(f"{status_symbol} {suite_name}: {result}")
    
    print(f"\nTotal: {len(results)} suites")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Missing: {missing}")
    print(f"Errors: {errors}")
    
    return passed, failed, missing, errors

def generate_test_coverage_report():
    """Generate a detailed test coverage report."""
    print("\n" + "-"*60)
    print(" GENERATING DETAILED COVERAGE ANALYSIS")
    print("-"*60)
    
    try:
        # Try to read coverage data
        coverage_cmd = [
            sys.executable, "-m", "coverage", "report",
            "--show-missing",
            "--skip-covered"
        ]
        
        result = subprocess.run(coverage_cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("\nDetailed Coverage Report:")
            print(result.stdout)
        else:
            print("Coverage report not available. Run tests first.")
            
    except FileNotFoundError:
        print("Coverage tool not found. Install with: pip install coverage")
    except Exception as e:
        print(f"Error generating coverage report: {e}")

def main():
    """Main test runner function."""
    print("Trading System Test Suite")
    print("Comprehensive testing with coverage analysis")
    print()
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("✗ Python 3.8+ required")
        return 1
    
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    
    # Run individual test suites first for quick feedback
    passed, failed, missing, errors = run_specific_test_suites()
    
    if failed > 0 or errors > 0:
        print(f"\n⚠  Some test suites failed. Skipping comprehensive test run.")
        print("Fix failing tests and run again.")
        return 1
    
    # Run comprehensive tests
    success = run_tests()
    
    # Generate additional coverage analysis
    generate_test_coverage_report()
    
    if success:
        print("\n🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        print("\nTest coverage has been significantly improved.")
        print("The trading system now has comprehensive test coverage.")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED")
        print("Please review the test output and fix any issues.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)