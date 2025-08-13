"""
Pattern Storage Service

This module provides persistent pattern storage capabilities including:
- Pattern database management
- Pattern persistence and retrieval
- Pattern metadata management
- Pattern versioning and history
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
from loguru import logger

from src.core.error_handler import with_error_handling


class PatternStorage:
    """
    Manages persistent storage of patterns and pattern metadata.
    """
    
    def __init__(self, storage_path: str = "data/pattern_storage"):
        self.storage_path = Path(storage_path)
        self.patterns_db = {}
        self.metadata_db = {}
        self.version_history = {}
        
        # Create storage directory if it doesn't exist
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize storage files
        self.patterns_file = self.storage_path / "patterns.json"
        self.metadata_file = self.storage_path / "metadata.json"
        self.history_file = self.storage_path / "version_history.json"
        
        # Load existing data
        self._load_existing_data()
        
        logger.info(f"PatternStorage initialized at {self.storage_path}")
    
    def _load_existing_data(self):
        """Load existing pattern data from storage files."""
        try:
            # Load patterns
            if self.patterns_file.exists():
                with open(self.patterns_file, 'r') as f:
                    self.patterns_db = json.load(f)
            
            # Load metadata
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    self.metadata_db = json.load(f)
            
            # Load version history
            if self.history_file.exists():
                with open(self.history_file, 'r') as f:
                    self.version_history = json.load(f)
                    
            logger.info(f"Loaded {len(self.patterns_db)} patterns from storage")
            
        except Exception as e:
            logger.error(f"Failed to load existing pattern data: {e}")
    
    @with_error_handling("pattern_storage")
    async def store_pattern(
        self, 
        pattern_id: str, 
        pattern_data: Dict[str, Any],
        pattern_type: str = "general",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Store a pattern in the database.
        
        Args:
            pattern_id: Unique identifier for the pattern
            pattern_data: The pattern data to store
            pattern_type: Type of pattern (temporal, seasonal, trend, etc.)
            metadata: Additional metadata for the pattern
            
        Returns:
            Dictionary containing storage result
        """
        try:
            # Create pattern entry
            pattern_entry = {
                "pattern_id": pattern_id,
                "pattern_type": pattern_type,
                "pattern_data": pattern_data,
                "created_at": datetime.now().isoformat(),
                "version": 1,
                "status": "active"
            }
            
            # Store pattern
            self.patterns_db[pattern_id] = pattern_entry
            
            # Store metadata
            if metadata:
                self.metadata_db[pattern_id] = {
                    "pattern_id": pattern_id,
                    "metadata": metadata,
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat()
                }
            
            # Create version history entry
            self.version_history[pattern_id] = [{
                "version": 1,
                "timestamp": datetime.now().isoformat(),
                "pattern_type": pattern_type,
                "status": "created"
            }]
            
            # Persist to disk
            await self._persist_data()
            
            logger.info(f"Stored pattern {pattern_id} successfully")
            
            return {
                "status": "success",
                "pattern_id": pattern_id,
                "stored_at": datetime.now().isoformat(),
                "pattern_type": pattern_type
            }
            
        except Exception as e:
            logger.error(f"Failed to store pattern {pattern_id}: {e}")
            return {"error": str(e)}
    
    @with_error_handling("pattern_retrieval")
    async def get_pattern(self, pattern_id: str) -> Dict[str, Any]:
        """
        Retrieve a pattern from the database.
        
        Args:
            pattern_id: Unique identifier for the pattern
            
        Returns:
            Dictionary containing the pattern data
        """
        try:
            if pattern_id not in self.patterns_db:
                return {"error": f"Pattern {pattern_id} not found"}
            
            pattern = self.patterns_db[pattern_id]
            metadata = self.metadata_db.get(pattern_id, {})
            
            return {
                "pattern": pattern,
                "metadata": metadata,
                "version_history": self.version_history.get(pattern_id, [])
            }
            
        except Exception as e:
            logger.error(f"Failed to retrieve pattern {pattern_id}: {e}")
            return {"error": str(e)}
    
    @with_error_handling("pattern_update")
    async def update_pattern(
        self, 
        pattern_id: str, 
        pattern_data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Update an existing pattern.
        
        Args:
            pattern_id: Unique identifier for the pattern
            pattern_data: Updated pattern data
            metadata: Updated metadata
            
        Returns:
            Dictionary containing update result
        """
        try:
            if pattern_id not in self.patterns_db:
                return {"error": f"Pattern {pattern_id} not found"}
            
            # Get current pattern
            current_pattern = self.patterns_db[pattern_id]
            current_version = current_pattern["version"]
            
            # Update pattern
            updated_pattern = {
                **current_pattern,
                "pattern_data": pattern_data,
                "version": current_version + 1,
                "updated_at": datetime.now().isoformat()
            }
            
            self.patterns_db[pattern_id] = updated_pattern
            
            # Update metadata
            if metadata:
                if pattern_id in self.metadata_db:
                    self.metadata_db[pattern_id]["metadata"] = metadata
                    self.metadata_db[pattern_id]["updated_at"] = datetime.now().isoformat()
                else:
                    self.metadata_db[pattern_id] = {
                        "pattern_id": pattern_id,
                        "metadata": metadata,
                        "created_at": datetime.now().isoformat(),
                        "updated_at": datetime.now().isoformat()
                    }
            
            # Update version history
            if pattern_id in self.version_history:
                self.version_history[pattern_id].append({
                    "version": current_version + 1,
                    "timestamp": datetime.now().isoformat(),
                    "pattern_type": current_pattern["pattern_type"],
                    "status": "updated"
                })
            
            # Persist to disk
            await self._persist_data()
            
            logger.info(f"Updated pattern {pattern_id} to version {current_version + 1}")
            
            return {
                "status": "success",
                "pattern_id": pattern_id,
                "new_version": current_version + 1,
                "updated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to update pattern {pattern_id}: {e}")
            return {"error": str(e)}
    
    @with_error_handling("pattern_deletion")
    async def delete_pattern(self, pattern_id: str) -> Dict[str, Any]:
        """
        Delete a pattern from the database.
        
        Args:
            pattern_id: Unique identifier for the pattern
            
        Returns:
            Dictionary containing deletion result
        """
        try:
            if pattern_id not in self.patterns_db:
                return {"error": f"Pattern {pattern_id} not found"}
            
            # Mark pattern as deleted (soft delete)
            self.patterns_db[pattern_id]["status"] = "deleted"
            self.patterns_db[pattern_id]["deleted_at"] = datetime.now().isoformat()
            
            # Update version history
            if pattern_id in self.version_history:
                current_version = self.patterns_db[pattern_id]["version"]
                self.version_history[pattern_id].append({
                    "version": current_version + 1,
                    "timestamp": datetime.now().isoformat(),
                    "pattern_type": self.patterns_db[pattern_id]["pattern_type"],
                    "status": "deleted"
                })
            
            # Persist to disk
            await self._persist_data()
            
            logger.info(f"Deleted pattern {pattern_id}")
            
            return {
                "status": "success",
                "pattern_id": pattern_id,
                "deleted_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to delete pattern {pattern_id}: {e}")
            return {"error": str(e)}
    
    @with_error_handling("pattern_search")
    async def search_patterns(
        self, 
        pattern_type: Optional[str] = None,
        status: str = "active",
        limit: int = 100
    ) -> Dict[str, Any]:
        """
        Search for patterns based on criteria.
        
        Args:
            pattern_type: Filter by pattern type
            status: Filter by pattern status
            limit: Maximum number of results
            
        Returns:
            Dictionary containing search results
        """
        try:
            results = []
            
            for pattern_id, pattern in self.patterns_db.items():
                # Apply filters
                if pattern_type and pattern["pattern_type"] != pattern_type:
                    continue
                
                if pattern["status"] != status:
                    continue
                
                results.append({
                    "pattern_id": pattern_id,
                    "pattern_type": pattern["pattern_type"],
                    "created_at": pattern["created_at"],
                    "version": pattern["version"],
                    "status": pattern["status"]
                })
                
                if len(results) >= limit:
                    break
            
            return {
                "results": results,
                "total_found": len(results),
                "search_criteria": {
                    "pattern_type": pattern_type,
                    "status": status,
                    "limit": limit
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to search patterns: {e}")
            return {"error": str(e)}
    
    @with_error_handling("pattern_metadata")
    async def get_pattern_metadata(self, pattern_id: str) -> Dict[str, Any]:
        """
        Get metadata for a specific pattern.
        
        Args:
            pattern_id: Unique identifier for the pattern
            
        Returns:
            Dictionary containing pattern metadata
        """
        try:
            if pattern_id not in self.metadata_db:
                return {"error": f"Metadata for pattern {pattern_id} not found"}
            
            return self.metadata_db[pattern_id]
            
        except Exception as e:
            logger.error(f"Failed to get metadata for pattern {pattern_id}: {e}")
            return {"error": str(e)}
    
    @with_error_handling("pattern_history")
    async def get_pattern_history(self, pattern_id: str) -> Dict[str, Any]:
        """
        Get version history for a specific pattern.
        
        Args:
            pattern_id: Unique identifier for the pattern
            
        Returns:
            Dictionary containing pattern version history
        """
        try:
            if pattern_id not in self.version_history:
                return {"error": f"History for pattern {pattern_id} not found"}
            
            return {
                "pattern_id": pattern_id,
                "version_history": self.version_history[pattern_id],
                "total_versions": len(self.version_history[pattern_id])
            }
            
        except Exception as e:
            logger.error(f"Failed to get history for pattern {pattern_id}: {e}")
            return {"error": str(e)}
    
    async def _persist_data(self):
        """Persist data to disk."""
        try:
            # Save patterns
            with open(self.patterns_file, 'w') as f:
                json.dump(self.patterns_db, f, indent=2)
            
            # Save metadata
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata_db, f, indent=2)
            
            # Save version history
            with open(self.history_file, 'w') as f:
                json.dump(self.version_history, f, indent=2)
                
            logger.debug("Pattern data persisted to disk successfully")
            
        except Exception as e:
            logger.error(f"Failed to persist pattern data: {e}")
            raise
    
    @with_error_handling("pattern_summary")
    async def get_storage_summary(self) -> Dict[str, Any]:
        """Get a summary of the pattern storage."""
        try:
            active_patterns = [p for p in self.patterns_db.values() if p["status"] == "active"]
            deleted_patterns = [p for p in self.patterns_db.values() if p["status"] == "deleted"]
            
            pattern_types = {}
            for pattern in active_patterns:
                pattern_type = pattern["pattern_type"]
                pattern_types[pattern_type] = pattern_types.get(pattern_type, 0) + 1
            
            return {
                "total_patterns": len(self.patterns_db),
                "active_patterns": len(active_patterns),
                "deleted_patterns": len(deleted_patterns),
                "pattern_types": pattern_types,
                "total_metadata": len(self.metadata_db),
                "storage_path": str(self.storage_path),
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get storage summary: {e}")
            return {"error": str(e)}
    
    @with_error_handling("pattern_cleanup")
    async def cleanup_deleted_patterns(self, older_than_days: int = 30) -> Dict[str, Any]:
        """
        Permanently remove deleted patterns older than specified days.
        
        Args:
            older_than_days: Remove patterns deleted more than this many days ago
            
        Returns:
            Dictionary containing cleanup result
        """
        try:
            cutoff_date = datetime.now().timestamp() - (older_than_days * 24 * 60 * 60)
            removed_count = 0
            
            patterns_to_remove = []
            
            for pattern_id, pattern in self.patterns_db.items():
                if pattern["status"] == "deleted":
                    deleted_at = datetime.fromisoformat(pattern["deleted_at"]).timestamp()
                    if deleted_at < cutoff_date:
                        patterns_to_remove.append(pattern_id)
            
            # Remove patterns
            for pattern_id in patterns_to_remove:
                del self.patterns_db[pattern_id]
                if pattern_id in self.metadata_db:
                    del self.metadata_db[pattern_id]
                if pattern_id in self.version_history:
                    del self.version_history[pattern_id]
                removed_count += 1
            
            # Persist changes
            if removed_count > 0:
                await self._persist_data()
            
            return {
                "status": "success",
                "removed_patterns": removed_count,
                "cutoff_date": datetime.fromtimestamp(cutoff_date).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to cleanup deleted patterns: {e}")
            return {"error": str(e)}
