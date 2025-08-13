#!/usr/bin/env python3
"""
Simple test script to verify Phase 7.4 module imports
"""

import sys
import os

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_imports():
    """Test importing Phase 7.4 modules"""
    print("Testing Phase 7.4 module imports...")
    
    try:
        # Test big data config import
        print("1. Testing big_data_config import...")
        from src.config.big_data_config import get_big_data_config
        print("   ✓ big_data_config imported successfully")
        
        # Test big data modules
        print("2. Testing big data modules...")
        from src.core.big_data import DistributedProcessor, DataLakeIntegration, BatchProcessor, DataGovernance
        print("   ✓ All big data modules imported successfully")
        
        # Test data pipelines modules
        print("3. Testing data pipelines modules...")
        from src.core.data_pipelines import DataQualityManager, SchemaManager, DataCatalog
        print("   ✓ All data pipelines modules imported successfully")
        
        # Test basic initialization
        print("4. Testing basic initialization...")
        config = get_big_data_config()
        print(f"   ✓ Config loaded: {type(config).__name__}")
        
        print("\n✅ All Phase 7.4 imports successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
