"""
Simple test runner script.

Usage:
    python run_tests.py              # Run all tests
    python run_tests.py --coverage   # Run with coverage report
"""

import sys
import subprocess

def main():
    args = ["python", "-m", "pytest", "tests/", "-v"]
    
    if "--coverage" in sys.argv:
        args.extend(["--cov=src", "--cov-report=html", "--cov-report=term"])
    
    result = subprocess.run(args)
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()
