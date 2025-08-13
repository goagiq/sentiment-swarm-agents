#!/usr/bin/env python3
"""
Robust script to flush both vector database (ChromaDB) and knowledge graph database files.
This will completely clear all stored data to start fresh.
"""

import shutil
import time
import os
from pathlib import Path
from loguru import logger


def flush_vector_database():
    """Flush the ChromaDB vector database."""
    try:
        # Path to ChromaDB directory
        chroma_db_path = Path("cache/chroma_db")
        
        if chroma_db_path.exists():
            logger.info(f"Flushing ChromaDB at: {chroma_db_path}")
            
            # Try to remove the entire ChromaDB directory
            try:
                shutil.rmtree(chroma_db_path)
                logger.info("ChromaDB directory removed successfully")
            except PermissionError as e:
                logger.warning(f"Permission error removing ChromaDB: {e}")
                logger.info("Trying alternative method: removing individual files...")
                
                # Alternative: Remove individual files
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
                
                # Try to remove the main directory again
                try:
                    shutil.rmtree(chroma_db_path)
                    logger.info("ChromaDB directory removed after file cleanup")
                except PermissionError:
                    logger.warning("ChromaDB directory still in use, will recreate on next startup")
            
            # Recreate the directory structure
            chroma_db_path.mkdir(parents=True, exist_ok=True)
            logger.info("ChromaDB directory recreated")
        else:
            logger.info("ChromaDB directory not found, nothing to flush")

    except Exception as e:
        logger.error(f"Error flushing ChromaDB: {e}")
        raise


def flush_knowledge_graph_database():
    """Flush the knowledge graph database files."""
    try:
        # Paths to knowledge graph files
        kg_files = [
            "Results/enhanced_knowledge_graph_data.json",
            "Results/knowledge_graphs/knowledge_graph.pkl",
            "enhanced_knowledge_graph_data.json"  # Root level file if exists
        ]
        
        kg_dirs = [
            "Results/knowledge_graphs"
        ]
        
        logger.info("Flushing knowledge graph database files...")
        
        # Remove knowledge graph files
        for file_path in kg_files:
            file_path = Path(file_path)
            if file_path.exists():
                try:
                    file_path.unlink()
                    logger.info(f"Removed: {file_path}")
                except PermissionError:
                    logger.warning(f"Could not remove {file_path} (file in use)")
            else:
                logger.info(f"File not found: {file_path}")
        
        # Remove knowledge graph directories (but keep Results directory)
        for dir_path in kg_dirs:
            dir_path = Path(dir_path)
            if dir_path.exists():
                try:
                    shutil.rmtree(dir_path)
                    logger.info(f"Removed directory: {dir_path}")
                except PermissionError:
                    logger.warning(f"Could not remove directory {dir_path} (in use)")
                
                # Recreate the directory
                dir_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Recreated directory: {dir_path}")
            else:
                logger.info(f"Directory not found: {dir_path}")

    except Exception as e:
        logger.error(f"Error flushing knowledge graph database: {e}")
        raise


def flush_all_databases():
    """Flush both vector and knowledge graph databases."""
    logger.info("Starting database flush process...")
    
    try:
        # Flush vector database
        flush_vector_database()
        
        # Flush knowledge graph database
        flush_knowledge_graph_database()
        
        logger.info("✅ All databases flushed successfully!")
        logger.info("You can now start fresh with clean databases.")
        
    except Exception as e:
        logger.error(f"❌ Error during database flush: {e}")
        raise


def manual_flush_instructions():
    """Provide manual flush instructions."""
    print("\n📋 Manual Flush Instructions:")
    print("=" * 50)
    print("If the automatic flush failed due to locked files:")
    print()
    print("1. Stop any running Python processes or applications")
    print("2. Close any IDEs or editors that might be accessing the files")
    print("3. Run these commands manually:")
    print()
    print("   # Remove ChromaDB")
    print("   rm -rf cache/chroma_db")
    print("   mkdir -p cache/chroma_db")
    print()
    print("   # Remove knowledge graph files")
    print("   rm -f Results/enhanced_knowledge_graph_data.json")
    print("   rm -f Results/knowledge_graphs/knowledge_graph.pkl")
    print("   rm -rf Results/knowledge_graphs")
    print("   mkdir -p Results/knowledge_graphs")
    print()
    print("4. Restart your application")


if __name__ == "__main__":
    # Configure logging
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    logger.add("logs/database_flush.log", rotation="1 MB", retention="7 days")
    
    print("🗑️  Robust Database Flush Utility")
    print("=" * 50)
    print("This will completely clear:")
    print("- ChromaDB vector database")
    print("- Knowledge graph JSON files")
    print("- Knowledge graph pickle files")
    print()
    
    response = input("Are you sure you want to flush all databases? (yes/no): ")
    
    if response.lower() in ['yes', 'y']:
        try:
            flush_all_databases()
            print("\n✅ Database flush completed successfully!")
        except Exception as e:
            print(f"\n❌ Database flush failed: {e}")
            manual_flush_instructions()
    else:
        print("\n❌ Database flush cancelled.")
