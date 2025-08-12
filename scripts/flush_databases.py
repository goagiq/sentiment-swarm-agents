#!/usr/bin/env python3
"""
Script to flush both vector database (ChromaDB) and knowledge graph database files.
This will completely clear all stored data to start fresh.
"""

import shutil
from pathlib import Path
from loguru import logger


def flush_vector_database():
    """Flush the ChromaDB vector database."""
    try:
        # Path to ChromaDB directory
        chroma_db_path = Path("cache/chroma_db")
        
        if chroma_db_path.exists():
            logger.info(f"Flushing ChromaDB at: {chroma_db_path}")
            
            # Remove the entire ChromaDB directory
            shutil.rmtree(chroma_db_path)
            logger.info("ChromaDB directory removed successfully")
            
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
                file_path.unlink()
                logger.info(f"Removed: {file_path}")
            else:
                logger.info(f"File not found: {file_path}")
        
        # Remove knowledge graph directories (but keep Results directory)
        for dir_path in kg_dirs:
            dir_path = Path(dir_path)
            if dir_path.exists():
                shutil.rmtree(dir_path)
                logger.info(f"Removed directory: {dir_path}")
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

if __name__ == "__main__":
    # Configure logging
    logger.add("logs/database_flush.log", rotation="1 MB", retention="7 days")
    
    print("🗑️  Database Flush Utility")
    print("=" * 50)
    print("This will completely clear:")
    print("- ChromaDB vector database")
    print("- Knowledge graph JSON files")
    print("- Knowledge graph pickle files")
    print()
    
    response = input("Are you sure you want to flush all databases? (yes/no): ")
    
    if response.lower() in ['yes', 'y']:
        flush_all_databases()
        print("\n✅ Database flush completed successfully!")
    else:
        print("\n❌ Database flush cancelled.")
