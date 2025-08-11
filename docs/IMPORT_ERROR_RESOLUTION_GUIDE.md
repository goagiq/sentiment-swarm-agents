# Import Error Resolution Guide

This guide provides a comprehensive approach to resolving unresolved import errors in the Sentiment Analysis project.

## Quick Start

### 1. Use the Virtual Environment Python Executable
Always use the virtual environment's Python executable to ensure you're using the correct environment:

```bash
# Windows
.venv/Scripts/python.exe your_script.py

# Linux/Mac
.venv/bin/python your_script.py
```

### 2. Install Dependencies
Ensure all dependencies are installed:

```bash
# Install the project in editable mode
.venv/Scripts/pip.exe install -e .

# Or install specific missing packages
.venv/Scripts/pip.exe install package_name
```

### 3. Add Project Root to Python Path
When running scripts, add the project root to the Python path:

```python
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))
```

## Common Import Error Types and Solutions

### 1. ModuleNotFoundError
**Error**: `ModuleNotFoundError: No module named 'module_name'`

**Solutions**:
- Install the missing package: `pip install package_name`
- Check if the module is in the correct directory
- Verify the virtual environment is activated
- Add the project root to Python path

### 2. ImportError
**Error**: `ImportError: cannot import name 'function_name' from 'module_name'`

**Solutions**:
- Check if the function/class exists in the module
- Verify the module's `__init__.py` file exports the function
- Check for circular imports
- Ensure the module is properly installed

### 3. SyntaxError in Imported Files
**Error**: `SyntaxError: invalid syntax`

**Solutions**:
- Run syntax check: `python -m py_compile file.py`
- Fix syntax errors in the imported file
- Check for missing parentheses, brackets, or colons
- Verify async/await usage is correct

### 4. AttributeError
**Error**: `AttributeError: module 'module_name' has no attribute 'function_name'`

**Solutions**:
- Check if the function is actually defined in the module
- Verify the module's `__all__` list includes the function
- Check for typos in function names
- Ensure the module is properly imported

## Step-by-Step Resolution Process

### Step 1: Identify the Error
Run the comprehensive import checker:

```bash
python scripts/check_imports.py
```

This will show:
- Missing dependencies
- Syntax errors
- Import errors

### Step 2: Fix Dependencies
Install missing packages:

```bash
.venv/Scripts/pip.exe install -e .
```

### Step 3: Fix Syntax Errors
For each syntax error:
1. Open the file mentioned in the error
2. Go to the line number specified
3. Fix the syntax issue
4. Run syntax check: `python -m py_compile file.py`

### Step 4: Fix Import Errors
For each import error:
1. Check if the module exists
2. Verify the import path is correct
3. Check if the module's `__init__.py` exports the function
4. Test the import manually

### Step 5: Verify Fixes
Run the import checker again:

```bash
python scripts/check_imports.py
```

## Common Issues and Solutions

### Issue: Async/Await Outside Async Function
**Problem**: Using `await` in a non-async function

**Solution**: Make the function async:
```python
# Before
def my_function():
    await some_async_operation()

# After
async def my_function():
    await some_async_operation()
```

### Issue: Circular Imports
**Problem**: Two modules importing each other

**Solution**: 
- Move imports inside functions
- Use lazy imports
- Restructure the code to avoid circular dependencies

### Issue: Missing __init__.py Files
**Problem**: Python can't find modules in directories

**Solution**: Create `__init__.py` files in all package directories

### Issue: Wrong Python Path
**Problem**: Python can't find the project modules

**Solution**: Add project root to Python path:
```python
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
```

## Testing Your Fixes

### 1. Syntax Check
```bash
python -m py_compile file.py
```

### 2. Import Test
```python
import sys
sys.path.append('.')
import your_module
```

### 3. Run the Application
```bash
.venv/Scripts/python.exe main.py
```

## Prevention Tips

### 1. Use Type Hints
Type hints help catch import errors early:
```python
from typing import List, Dict
from src.agents.base_agent import BaseAgent
```

### 2. Use Absolute Imports
Prefer absolute imports over relative imports:
```python
# Good
from src.agents.unified_text_agent import UnifiedTextAgent

# Avoid
from .unified_text_agent import UnifiedTextAgent
```

### 3. Keep Dependencies Updated
Regularly update your `pyproject.toml` and install dependencies:
```bash
.venv/Scripts/pip.exe install -e .
```

### 4. Use the Import Checker
Run the import checker regularly:
```bash
python scripts/check_imports.py
```

## Troubleshooting Checklist

- [ ] Virtual environment is activated
- [ ] All dependencies are installed
- [ ] Project root is in Python path
- [ ] No syntax errors in imported files
- [ ] Modules have proper `__init__.py` files
- [ ] No circular imports
- [ ] Function/class names are spelled correctly
- [ ] Import paths are correct

## Getting Help

If you're still having issues:

1. Run the comprehensive import checker
2. Check the error messages carefully
3. Look at the specific line numbers mentioned
4. Test imports manually in Python REPL
5. Check the project documentation
6. Review similar issues in the codebase

## Example: Fixing a Real Import Error

Let's say you get this error:
```
ModuleNotFoundError: No module named 'src.agents.unified_text_agent'
```

**Step 1**: Check if the file exists
```bash
ls src/agents/unified_text_agent.py
```

**Step 2**: Check if the directory is a package
```bash
ls src/agents/__init__.py
```

**Step 3**: Test the import manually
```python
import sys
sys.path.append('.')
from src.agents.unified_text_agent import UnifiedTextAgent
```

**Step 4**: If it works manually, the issue is likely in your script's Python path setup.

This systematic approach should resolve most import errors in the project.
