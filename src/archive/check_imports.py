#!/usr/bin/env python3
"""
Comprehensive import checker for the Sentiment Analysis project.
This script helps identify and resolve import errors.
"""

import os
import sys
import importlib
import subprocess
from pathlib import Path
from typing import List, Dict, Any

def add_project_root_to_path():
    """Add the project root directory to Python path."""
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root

def check_python_syntax(file_path: Path) -> Dict[str, Any]:
    """Check Python syntax for a file."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", str(file_path)],
            capture_output=True,
            text=True,
            timeout=30
        )
        return {
            "file": str(file_path),
            "syntax_ok": result.returncode == 0,
            "error": result.stderr if result.returncode != 0 else None
        }
    except Exception as e:
        return {
            "file": str(file_path),
            "syntax_ok": False,
            "error": str(e)
        }

def check_module_import(module_path: str) -> Dict[str, Any]:
    """Check if a module can be imported."""
    try:
        module = importlib.import_module(module_path)
        return {
            "module": module_path,
            "import_ok": True,
            "error": None
        }
    except ImportError as e:
        return {
            "module": module_path,
            "import_ok": False,
            "error": str(e)
        }
    except Exception as e:
        return {
            "module": module_path,
            "import_ok": False,
            "error": f"Unexpected error: {str(e)}"
        }

def find_python_files(directory: Path) -> List[Path]:
    """Find all Python files in a directory recursively."""
    python_files = []
    for root, dirs, files in os.walk(directory):
        # Skip virtual environment and cache directories
        dirs[:] = [d for d in dirs if d not in ['.venv', '__pycache__', '.git', 'node_modules']]
        
        for file in files:
            if file.endswith('.py'):
                python_files.append(Path(root) / file)
    return python_files

def check_dependencies() -> Dict[str, Any]:
    """Check if all required dependencies are installed."""
    required_packages = [
        'fastapi', 'uvicorn', 'pydantic', 'chromadb', 'transformers',
        'torch', 'numpy', 'pandas', 'requests', 'beautifulsoup4',
        'aiohttp', 'librosa', 'opencv-python', 'pillow', 'nltk',
        'spacy', 'scikit-learn', 'matplotlib', 'streamlit', 'loguru',
        'rich', 'click', 'typer', 'strands', 'ollama', 'fastmcp',
        'yt-dlp', 'PyPDF2', 'PyMuPDF'
    ]
    
    results = {}
    for package in required_packages:
        try:
            importlib.import_module(package)
            results[package] = {"installed": True, "error": None}
        except ImportError as e:
            results[package] = {"installed": False, "error": str(e)}
    
    return results

def main():
    """Main function to run comprehensive import checks."""
    print("🔍 Comprehensive Import Checker for Sentiment Analysis Project")
    print("=" * 70)
    
    # Add project root to path
    project_root = add_project_root_to_path()
    print(f"📁 Project root: {project_root}")
    
    # Check dependencies
    print("\n📦 Checking dependencies...")
    deps = check_dependencies()
    missing_deps = [pkg for pkg, info in deps.items() if not info["installed"]]
    
    if missing_deps:
        print(f"❌ Missing dependencies: {missing_deps}")
        print("💡 Run: .venv/Scripts/pip.exe install -e .")
    else:
        print("✅ All dependencies are installed")
    
    # Check syntax for all Python files
    print("\n🔍 Checking Python syntax...")
    python_files = find_python_files(project_root)
    syntax_errors = []
    
    for file_path in python_files:
        result = check_python_syntax(file_path)
        if not result["syntax_ok"]:
            syntax_errors.append(result)
    
    if syntax_errors:
        print(f"❌ Found {len(syntax_errors)} syntax errors:")
        for error in syntax_errors:
            print(f"   - {error['file']}: {error['error']}")
    else:
        print("✅ All Python files have valid syntax")
    
    # Check key module imports
    print("\n📚 Checking key module imports...")
    key_modules = [
        "src.agents.unified_text_agent",
        "src.agents.unified_audio_agent", 
        "src.agents.unified_vision_agent",
        "src.agents.knowledge_graph_agent",
        "src.core.ollama_integration",
        "src.core.vector_db",
        "src.config.settings",
        "main"
    ]
    
    import_errors = []
    for module in key_modules:
        result = check_module_import(module)
        if not result["import_ok"]:
            import_errors.append(result)
        else:
            print(f"✅ {module}")
    
    if import_errors:
        print(f"\n❌ Found {len(import_errors)} import errors:")
        for error in import_errors:
            print(f"   - {error['module']}: {error['error']}")
    else:
        print("✅ All key modules can be imported successfully")
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 SUMMARY:")
    print(f"   - Dependencies: {'✅ All installed' if not missing_deps else f'❌ {len(missing_deps)} missing'}")
    print(f"   - Syntax: {'✅ All valid' if not syntax_errors else f'❌ {len(syntax_errors)} errors'}")
    print(f"   - Imports: {'✅ All working' if not import_errors else f'❌ {len(import_errors)} errors'}")
    
    if not missing_deps and not syntax_errors and not import_errors:
        print("\n🎉 All checks passed! Your project is ready to run.")
        print("💡 To run the main application:")
        print("   .venv/Scripts/python.exe main.py")
    else:
        print("\n🔧 Please fix the issues above before running the application.")

if __name__ == "__main__":
    main()
