#!/usr/bin/env python3
"""
Script to flush the ChromaDB database and remove all files in Results/reports directory.
Usage: python flushdb.py
"""

import shutil
from pathlib import Path
import chromadb
from chromadb.config import Settings
from loguru import logger


def flush_database():
    """Flush the ChromaDB database."""
    try:
        # Initialize ChromaDB client
        client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get all collections
        collections = client.list_collections()
        logger.info(f"Found {len(collections)} collections to flush")
        
        # Delete all collections
        for collection in collections:
            logger.info(f"Deleting collection: {collection.name}")
            client.delete_collection(name=collection.name)
        
        logger.success("Database flushed successfully")
        
    except Exception as e:
        logger.error(f"Error flushing database: {e}")
        raise


def remove_reports():
    """Remove all files in the Results/reports directory."""
    try:
        reports_dir = Path("./Results/reports")
        
        if not reports_dir.exists():
            logger.warning("Reports directory does not exist")
            return
        
        # List all files in the directory
        files = list(reports_dir.glob("*"))
        logger.info(f"Found {len(files)} files/directories in reports directory")
        
        # Remove all files and directories
        for item in files:
            if item.is_file():
                logger.info(f"Removing file: {item.name}")
                item.unlink()
            elif item.is_dir():
                logger.info(f"Removing directory: {item.name}")
                shutil.rmtree(item)
        
        logger.success("All reports removed successfully")
        
    except Exception as e:
        logger.error(f"Error removing reports: {e}")
        raise


def main():
    """Main function to flush database and remove reports."""
    logger.info("Starting database flush and reports cleanup...")
    
    try:
        # Flush the database
        flush_database()
        
        # Remove reports
        remove_reports()
        
        logger.success("Database flush and reports cleanup completed successfully!")
        
    except Exception as e:
        logger.error(f"Failed to complete cleanup: {e}")
        raise


if __name__ == "__main__":
    main()
