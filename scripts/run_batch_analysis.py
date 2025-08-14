#!/usr/bin/env python3
"""
Simple runner script for batch intelligence analysis.
Uses the virtual environment and provides clear output.
"""

import subprocess
import sys
import os
from pathlib import Path


def main():
    """Run the batch intelligence analysis."""
    print("🚀 Starting Batch Intelligence Analysis")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("intelligence_analysis_queries.md").exists():
        print("❌ Error: intelligence_analysis_queries.md not found!")
        print("Please run this script from the project root directory.")
        sys.exit(1)
    
    # Check if virtual environment exists
    venv_python = Path(".venv/Scripts/python.exe")
    if not venv_python.exists():
        print("❌ Error: Virtual environment not found!")
        print("Please ensure .venv/Scripts/python.exe exists.")
        sys.exit(1)
    
    # Run the batch analysis
    try:
        print(f"📋 Using Python: {venv_python}")
        print(f"📁 Working directory: {os.getcwd()}")
        print("\n" + "=" * 50)
        
        # Run the batch analysis script
        result = subprocess.run([
            str(venv_python), "batch_intelligence_analysis.py"
        ], capture_output=False, text=True)
        
        if result.returncode == 0:
            print("\n✅ Batch analysis completed successfully!")
        else:
            print(f"\n❌ Batch analysis failed with return code: "
                  f"{result.returncode}")
            sys.exit(result.returncode)
            
    except Exception as e:
        print(f"❌ Error running batch analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
