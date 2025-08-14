#!/usr/bin/env python3
"""
Minimal API test to isolate startup issues
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test if all required imports work."""
    print("Testing imports...")
    
    try:
        from fastapi import FastAPI
        print("✅ FastAPI imported")
    except Exception as e:
        print(f"❌ FastAPI import failed: {e}")
        return False
    
    try:
        from pydantic import BaseModel
        print("✅ Pydantic imported")
    except Exception as e:
        print(f"❌ Pydantic import failed: {e}")
        return False
    
    try:
        from loguru import logger
        print("✅ Loguru imported")
    except Exception as e:
        print(f"❌ Loguru import failed: {e}")
        return False
    
    try:
        from src.config.config import config
        print("✅ Config imported")
    except Exception as e:
        print(f"❌ Config import failed: {e}")
        return False
    
    return True

def test_app_creation():
    """Test if the FastAPI app can be created."""
    print("\nTesting app creation...")
    
    try:
        from src.api.main import app
        print("✅ FastAPI app imported successfully")
        return True
    except Exception as e:
        print(f"❌ FastAPI app import failed: {e}")
        return False

if __name__ == "__main__":
    print("=== Minimal API Test ===")
    
    if test_imports():
        test_app_creation()
    else:
        print("❌ Import tests failed")
