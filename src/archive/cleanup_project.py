#!/usr/bin/env python3
"""
Comprehensive Project Cleanup Script

This script performs a thorough cleanup of the project structure:
1. Organizes test files by moving them to appropriate subdirectories
2. Removes duplicate or outdated files
3. Ensures proper file organization according to the design framework
4. Cleans up temporary files and artifacts
"""

import os
import shutil
import glob
from pathlib import Path
from datetime import datetime

def log_message(message: str):
    """Log a message with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def cleanup_test_files():
    """Organize test files into appropriate subdirectories."""
    test_dir = Path("Test")
    
    # Create subdirectories if they don't exist
    subdirs = ["unit", "integration", "performance", "multilingual", "mcp"]
    for subdir in subdirs:
        (test_dir / subdir).mkdir(exist_ok=True)
    
    # Move MCP-related test files
    mcp_tests = [
        "test_mcp_server_simple.py",
        "test_strands_integration_final.py", 
        "test_strands_mcp_integration.py",
        "test_consolidated_mcp_simple.py",
        "test_consolidated_mcp_server.py",
        "test_mcp_business_intelligence.py",
        "test_mcp_content_analysis.py"
    ]
    
    for test_file in mcp_tests:
        src = test_dir / test_file
        dst = test_dir / "mcp" / test_file
        if src.exists():
            shutil.move(str(src), str(dst))
            log_message(f"Moved {test_file} to Test/mcp/")
    
    # Move multilingual test files
    multilingual_tests = [
        "test_multilingual_integration.py",
        "test_classical_chinese_*.py",
        "test_russian_*.py",
        "test_language_*.py",
        "debug_*_pdf.py",
        "debug_*_entities.py",
        "debug_*_language_detection.py"
    ]
    
    for pattern in multilingual_tests:
        for src in test_dir.glob(pattern):
            if src.is_file():
                dst = test_dir / "multilingual" / src.name
                shutil.move(str(src), str(dst))
                log_message(f"Moved {src.name} to Test/multilingual/")
    
    # Move performance test files
    performance_tests = [
        "test_performance_*.py",
        "test_step8_performance.py",
        "test_phase2_performance_*.py"
    ]
    
    for pattern in performance_tests:
        for src in test_dir.glob(pattern):
            if src.is_file():
                dst = test_dir / "performance" / src.name
                shutil.move(str(src), str(dst))
                log_message(f"Moved {src.name} to Test/performance/")
    
    # Move integration test files
    integration_tests = [
        "integration_test.py",
        "test_phase*_integration.py",
        "test_main_integration*.py",
        "test_final_integration.py"
    ]
    
    for pattern in integration_tests:
        for src in test_dir.glob(pattern):
            if src.is_file():
                dst = test_dir / "integration" / src.name
                shutil.move(str(src), str(dst))
                log_message(f"Moved {src.name} to Test/integration/")

def cleanup_duplicate_files():
    """Remove duplicate or outdated files."""
    # Remove empty or nearly empty files
    for root, dirs, files in os.walk("Test"):
        for file in files:
            file_path = Path(root) / file
            if file_path.stat().st_size < 100:  # Less than 100 bytes
                file_path.unlink()
                log_message(f"Removed small file: {file_path}")
    
    # Remove duplicate phase test files (keep the most recent)
    test_dir = Path("Test")
    phase_files = {}
    
    for file in test_dir.rglob("test_phase*.py"):
        phase_num = file.name.split("_")[1]  # Extract phase number
        if phase_num not in phase_files:
            phase_files[phase_num] = []
        phase_files[phase_num].append(file)
    
    for phase_num, files in phase_files.items():
        if len(files) > 1:
            # Keep the most recent file, remove others
            files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            for file in files[1:]:
                file.unlink()
                log_message(f"Removed duplicate: {file}")

def cleanup_documentation():
    """Ensure all documentation is properly organized."""
    docs_dir = Path("docs")
    
    # Create subdirectories for better organization
    subdirs = ["guides", "summaries", "plans", "checklists"]
    for subdir in subdirs:
        (docs_dir / subdir).mkdir(exist_ok=True)
    
    # Move files to appropriate subdirectories
    file_moves = {
        "guides": ["STRANDS_INTEGRATION_SUMMARY.md", "PROJECT_STRUCTURE.md"],
        "summaries": ["MCP_CONSOLIDATION_FINAL_SUMMARY.md", "PRODUCTION_DEPLOYMENT_SUMMARY.md"],
        "plans": ["MCP_TOOLS_CONSOLIDATION_PLAN.md", "CONTENT_ANALYSIS_IMPLEMENTATION_PLAN.md"],
        "checklists": ["PRODUCTION_READINESS_CHECKLIST.md", "CLEANUP_SUMMARY.md"]
    }
    
    for subdir, files in file_moves.items():
        for file in files:
            src = docs_dir / file
            dst = docs_dir / subdir / file
            if src.exists():
                shutil.move(str(src), str(dst))
                log_message(f"Moved {file} to docs/{subdir}/")

def cleanup_results():
    """Organize results directory."""
    results_dir = Path("Results")
    
    # Create subdirectories if they don't exist
    subdirs = ["test_results", "reports", "exports", "knowledge_graphs", "semantic_search", "reflection"]
    for subdir in subdirs:
        (results_dir / subdir).mkdir(exist_ok=True)
    
    # Move JSON result files to test_results
    for json_file in results_dir.glob("*.json"):
        if json_file.parent == results_dir:
            dst = results_dir / "test_results" / json_file.name
            shutil.move(str(json_file), str(dst))
            log_message(f"Moved {json_file.name} to Results/test_results/")

def cleanup_cache():
    """Clean up cache directory."""
    cache_dir = Path("cache")
    if cache_dir.exists():
        # Remove temporary cache files but keep important ones
        for item in cache_dir.iterdir():
            if item.is_file() and item.suffix in ['.tmp', '.log', '.cache']:
                item.unlink()
                log_message(f"Removed cache file: {item}")
            elif item.is_dir() and item.name in ['temp', 'tmp']:
                shutil.rmtree(item)
                log_message(f"Removed cache directory: {item}")

def main():
    """Main cleanup function."""
    log_message("Starting comprehensive project cleanup...")
    
    try:
        # Perform cleanup tasks
        cleanup_test_files()
        cleanup_duplicate_files()
        cleanup_documentation()
        cleanup_results()
        cleanup_cache()
        
        log_message("✅ Project cleanup completed successfully!")
        
        # Print summary
        print("\n" + "="*60)
        print("CLEANUP SUMMARY")
        print("="*60)
        print("✅ Test files organized into subdirectories")
        print("✅ Duplicate files removed")
        print("✅ Documentation properly organized")
        print("✅ Results files organized")
        print("✅ Cache cleaned up")
        print("✅ Project structure optimized")
        print("="*60)
        
    except Exception as e:
        log_message(f"❌ Error during cleanup: {e}")
        raise

if __name__ == "__main__":
    main()
