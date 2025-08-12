"""
Vector Database Metadata Fix
Fixes the metadata compatibility issues with ChromaDB.
"""

import json
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


def sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize metadata to ensure ChromaDB compatibility.
    
    ChromaDB only accepts primitive types: str, int, float, bool, None
    This function converts incompatible types to compatible ones.
    """
    if not isinstance(metadata, dict):
        return {"value": str(metadata)}
    
    sanitized = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool, type(None))):
            # These types are directly compatible
            sanitized[key] = value
        elif isinstance(value, dict):
            # Convert dict to JSON string
            try:
                sanitized[key] = json.dumps(value, ensure_ascii=False)
            except (TypeError, ValueError):
                sanitized[key] = str(value)
        elif isinstance(value, (list, tuple)):
            # Convert list/tuple to JSON string
            try:
                sanitized[key] = json.dumps(value, ensure_ascii=False)
            except (TypeError, ValueError):
                sanitized[key] = str(value)
        else:
            # Convert other types to string
            sanitized[key] = str(value)
    
    return sanitized


def sanitize_metadata_list(metadatas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Sanitize a list of metadata dictionaries.
    
    Args:
        metadatas: List of metadata dictionaries
        
    Returns:
        List of sanitized metadata dictionaries
    """
    if not metadatas:
        return []
    
    return [sanitize_metadata(metadata) for metadata in metadatas]


class VectorDBMetadataFix:
    """
    Utility class for fixing vector database metadata issues.
    """
    
    @staticmethod
    def apply_metadata_fix():
        """
        Apply the metadata fix to the vector database.
        This should be called before any vector database operations.
        """
        try:
            # Import the vector database manager
            from src.core.vector_db import VectorDBManager
            
            # Monkey patch the add_texts method to include metadata sanitization
            original_add_texts = VectorDBManager.add_texts
            
            async def fixed_add_texts(
                self,
                collection_name: str,
                texts: List[str],
                metadatas: Optional[List[Dict[str, Any]]] = None,
                ids: Optional[List[str]] = None
            ) -> List[str]:
                """Add texts to a specific collection with metadata sanitization."""
                try:
                    # Sanitize metadata before adding
                    if metadatas is not None:
                        metadatas = sanitize_metadata_list(metadatas)
                    
                    # Call the original method
                    return await original_add_texts(self, collection_name, texts, metadatas, ids)
                    
                except Exception as e:
                    logger.error(f"Failed to add texts to collection {collection_name}: {e}")
                    raise
            
            # Replace the method
            VectorDBManager.add_texts = fixed_add_texts
            logger.info("Vector database metadata fix applied successfully")
            
        except Exception as e:
            logger.error(f"Failed to apply vector database metadata fix: {e}")


def apply_translation_service_fix():
    """
    Apply fix to translation service metadata handling.
    """
    try:
        from src.core.translation_service import TranslationService
        
        # Monkey patch the store_translation method
        original_store_translation = TranslationService.store_translation
        
        def fixed_store_translation(self, original: str, translated: str, metadata: Dict[str, Any] = None):
            """Store translation with proper metadata handling."""
            if metadata is None:
                metadata = {}
            
            # Ensure metadata is compatible with vector database
            sanitized_metadata = sanitize_metadata(metadata)
            
            try:
                # Use the original method with sanitized metadata
                return original_store_translation(self, original, translated, sanitized_metadata)
            except Exception as e:
                logger.warning(f"Failed to store translation in memory: {e}")
                # Continue processing even if storage fails
        
        # Replace the method
        TranslationService.store_translation = fixed_store_translation
        logger.info("Translation service metadata fix applied successfully")
        
    except Exception as e:
        logger.error(f"Failed to apply translation service fix: {e}")


def apply_all_fixes():
    """
    Apply all metadata-related fixes.
    """
    logger.info("Applying vector database metadata fixes...")
    VectorDBMetadataFix.apply_metadata_fix()
    apply_translation_service_fix()
    logger.info("All metadata fixes applied successfully")


if __name__ == "__main__":
    # Apply fixes when run directly
    apply_all_fixes()
