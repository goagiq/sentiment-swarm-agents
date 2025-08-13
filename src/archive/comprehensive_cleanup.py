#!/usr/bin/env python3
"""
Comprehensive cleanup script for the Sentiment Analysis System.
This script clears all cache directories, databases, and temporary files.
"""

import shutil
import time
import os
from pathlib import Path
from loguru import logger


def clear_cache_directories():
    """Clear all cache directories."""
    cache_dirs = [
        "cache/image_processing",
        "cache/ocr", 
        "cache/video",
        "cache/audio",
        "temp"
    ]
    
    logger.info("Clearing cache directories...")
    
    for dir_path in cache_dirs:
        dir_path = Path(dir_path)
        if dir_path.exists():
            try:
                shutil.rmtree(dir_path)
                logger.info(f"Cleared: {dir_path}")
            except PermissionError:
                logger.warning(f"Could not clear {dir_path} (in use)")
        
        # Recreate empty directory
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Recreated: {dir_path}")


def clear_vector_database():
    """Clear the ChromaDB vector database."""
    try:
        chroma_db_path = Path("cache/chroma_db")
        
        if chroma_db_path.exists():
            logger.info(f"Clearing ChromaDB at: {chroma_db_path}")
            
            # Try to remove the entire directory
            try:
                shutil.rmtree(chroma_db_path)
                logger.info("ChromaDB directory removed successfully")
            except PermissionError as e:
                logger.warning(f"Permission error: {e}")
                logger.info("Trying to remove individual files...")
                
                # Remove individual files
                for item in chroma_db_path.rglob("*"):
                    try:
                        if item.is_file():
                            item.unlink()
                            logger.info(f"Removed file: {item}")
                        elif item.is_dir():
                            shutil.rmtree(item)
                            logger.info(f"Removed directory: {item}")
                    except PermissionError:
                        logger.warning(f"Could not remove: {item} (in use)")
                
                # Try to remove main directory again
                try:
                    shutil.rmtree(chroma_db_path)
                    logger.info("ChromaDB directory removed after cleanup")
                except PermissionError:
                    logger.warning("ChromaDB still in use - will be fresh on restart")
            
            # Recreate directory
            chroma_db_path.mkdir(parents=True, exist_ok=True)
            logger.info("ChromaDB directory recreated")
        else:
            logger.info("ChromaDB directory not found")
            
    except Exception as e:
        logger.error(f"Error clearing ChromaDB: {e}")
        raise


def clear_knowledge_graph_database():
    """Clear knowledge graph database files."""
    try:
        kg_files = [
            "Results/enhanced_knowledge_graph_data.json",
            "Results/knowledge_graphs/knowledge_graph.pkl",
            "enhanced_knowledge_graph_data.json"
        ]
        
        kg_dirs = ["Results/knowledge_graphs"]
        
        logger.info("Clearing knowledge graph database...")
        
        # Remove files
        for file_path in kg_files:
            file_path = Path(file_path)
            if file_path.exists():
                try:
                    file_path.unlink()
                    logger.info(f"Removed: {file_path}")
                except PermissionError:
                    logger.warning(f"Could not remove {file_path} (in use)")
            else:
                logger.info(f"File not found: {file_path}")
        
        # Remove and recreate directories
        for dir_path in kg_dirs:
            dir_path = Path(dir_path)
            if dir_path.exists():
                try:
                    shutil.rmtree(dir_path)
                    logger.info(f"Removed directory: {dir_path}")
                except PermissionError:
                    logger.warning(f"Could not remove {dir_path} (in use)")
            
            # Recreate directory
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Recreated: {dir_path}")
            
    except Exception as e:
        logger.error(f"Error clearing knowledge graph: {e}")
        raise


def clear_logs():
    """Clear log files."""
    try:
        log_files = [
            "logs/database_flush.log",
            "logs/app.log",
            "logs/error.log"
        ]
        
        logger.info("Clearing log files...")
        
        for log_file in log_files:
            log_file = Path(log_file)
            if log_file.exists():
                try:
                    log_file.unlink()
                    logger.info(f"Removed log: {log_file}")
                except PermissionError:
                    logger.warning(f"Could not remove {log_file} (in use)")
                    
    except Exception as e:
        logger.error(f"Error clearing logs: {e}")


def comprehensive_cleanup():
    """Perform comprehensive cleanup of all databases and cache."""
    logger.info("🚀 Starting comprehensive cleanup...")
    
    try:
        # Clear all cache directories
        clear_cache_directories()
        
        # Clear vector database
        clear_vector_database()
        
        # Clear knowledge graph database
        clear_knowledge_graph_database()
        
        # Clear logs
        clear_logs()
        
        logger.info("✅ Comprehensive cleanup completed successfully!")
        logger.info("🎉 Your system is now completely fresh!")
        
    except Exception as e:
        logger.error(f"❌ Error during comprehensive cleanup: {e}")
        raise


def restart_instructions():
    """Provide restart instructions."""
    print("\n🔄 Restart Instructions:")
    print("=" * 50)
    print("To complete the cleanup and start fresh:")
    print()
    print("1. Stop any running Python processes:")
    print("   - Close any running applications")
    print("   - Stop any background services")
    print("   - Close IDEs or editors accessing the files")
    print()
    print("2. Restart your application:")
    print("   python main.py")
    print()
    print("3. Or start the MCP server:")
    print("   python -m src.mcp.server")
    print()
    print("4. Verify the cleanup:")
    print("   - Check that ChromaDB is empty")
    print("   - Verify knowledge graph is fresh")
    print("   - Confirm cache directories are clean")
    print()
    print("🎯 Your system will now start with completely fresh databases!")


if __name__ == "__main__":
    # Configure logging
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    logger.add("logs/comprehensive_cleanup.log", rotation="1 MB", retention="7 days")
    
    print("🧹 Comprehensive Cleanup Utility")
    print("=" * 50)
    print("This will completely clear:")
    print("- All cache directories (image, OCR, video, audio)")
    print("- ChromaDB vector database")
    print("- Knowledge graph database files")
    print("- Temporary files and logs")
    print()
    print("⚠️  WARNING: This will delete ALL cached data!")
    print()
    
    response = input("Are you sure you want to perform comprehensive cleanup? (yes/no): ")
    
    if response.lower() in ['yes', 'y']:
        try:
            comprehensive_cleanup()
            print("\n✅ Comprehensive cleanup completed!")
            restart_instructions()
        except Exception as e:
            print(f"\n❌ Cleanup failed: {e}")
            restart_instructions()
    else:
        print("\n❌ Cleanup cancelled.")
