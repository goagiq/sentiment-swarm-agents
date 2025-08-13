"""
Model Versioning System

This module provides model versioning and lifecycle management capabilities
for tracking, storing, and managing different versions of machine learning models.
"""

import os
import json
import logging
import shutil
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from datetime import datetime
import hashlib
import pickle

from src.core.error_handling_service import ErrorHandlingService
from src.config.advanced_ml_config import get_advanced_ml_config

logger = logging.getLogger(__name__)
error_handler = ErrorHandlingService()


class ModelVersioning:
    """Model versioning and lifecycle management system."""
    
    def __init__(self):
        self.config = get_advanced_ml_config()
        self.registry_path = self.config.model_versioning.registry["registry_path"]
        self.versioning_enabled = self.config.model_versioning.versioning["enabled"]
        self.backup_versions = self.config.model_versioning.versioning["backup_versions"]
        
        # Create registry directory if it doesn't exist
        os.makedirs(self.registry_path, exist_ok=True)
        
        # Initialize registry
        self.registry_file = os.path.join(self.registry_path, "model_registry.json")
        self.registry = self._load_registry()
        
        logger.info(f"Initialized ModelVersioning with registry at {self.registry_path}")
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load the model registry from file."""
        try:
            if os.path.exists(self.registry_file):
                with open(self.registry_file, 'r') as f:
                    return json.load(f)
            else:
                # Create new registry
                registry = {
                    "models": {},
                    "versions": {},
                    "metadata": {
                        "created_at": datetime.now().isoformat(),
                        "last_updated": datetime.now().isoformat(),
                        "total_models": 0,
                        "total_versions": 0
                    }
                }
                self._save_registry(registry)
                return registry
                
        except Exception as e:
            error_handler.handle_error(f"Error loading registry: {str(e)}", e)
            return {"models": {}, "versions": {}, "metadata": {}}
    
    def _save_registry(self, registry: Optional[Dict[str, Any]] = None) -> bool:
        """Save the model registry to file."""
        try:
            if registry is None:
                registry = self.registry
            
            registry["metadata"]["last_updated"] = datetime.now().isoformat()
            
            with open(self.registry_file, 'w') as f:
                json.dump(registry, f, indent=2)
            
            return True
            
        except Exception as e:
            error_handler.handle_error(f"Error saving registry: {str(e)}", e)
            return False
    
    def create_version(self, model_name: str, model: Any, 
                      metadata: Dict[str, Any], version_format: str = "semantic") -> str:
        """Create a new version of a model."""
        try:
            if not self.versioning_enabled:
                logger.warning("Model versioning is disabled")
                return "1.0.0"
            
            # Generate version number
            if version_format == "semantic":
                version = self._generate_semantic_version(model_name)
            else:
                version = self._generate_timestamp_version()
            
            # Create version directory
            version_dir = os.path.join(self.registry_path, model_name, version)
            os.makedirs(version_dir, exist_ok=True)
            
            # Save model
            model_path = os.path.join(version_dir, "model.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            # Save metadata
            metadata_path = os.path.join(version_dir, "metadata.json")
            version_metadata = {
                "model_name": model_name,
                "version": version,
                "created_at": datetime.now().isoformat(),
                "model_hash": self._calculate_model_hash(model),
                **metadata
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(version_metadata, f, indent=2)
            
            # Update registry
            if model_name not in self.registry["models"]:
                self.registry["models"][model_name] = {
                    "versions": [],
                    "latest_version": version,
                    "created_at": datetime.now().isoformat(),
                    "last_updated": datetime.now().isoformat()
                }
            
            self.registry["models"][model_name]["versions"].append(version)
            self.registry["models"][model_name]["latest_version"] = version
            self.registry["models"][model_name]["last_updated"] = datetime.now().isoformat()
            
            self.registry["versions"][f"{model_name}_{version}"] = {
                "model_name": model_name,
                "version": version,
                "path": version_dir,
                "created_at": datetime.now().isoformat(),
                "metadata": version_metadata
            }
            
            # Update metadata
            self.registry["metadata"]["total_models"] = len(self.registry["models"])
            self.registry["metadata"]["total_versions"] = len(self.registry["versions"])
            
            # Save registry
            self._save_registry()
            
            # Cleanup old versions if needed
            self._cleanup_old_versions(model_name)
            
            logger.info(f"Created version {version} for model {model_name}")
            return version
            
        except Exception as e:
            error_handler.handle_error(f"Error creating version: {str(e)}", e)
            return "1.0.0"
    
    def _generate_semantic_version(self, model_name: str) -> str:
        """Generate a semantic version number."""
        try:
            if model_name in self.registry["models"]:
                versions = self.registry["models"][model_name]["versions"]
                if versions:
                    # Get the latest version
                    latest_version = max(versions, key=lambda v: [int(x) for x in v.split('.')])
                    major, minor, patch = map(int, latest_version.split('.'))
                    return f"{major}.{minor}.{patch + 1}"
            
            return "1.0.0"
            
        except Exception as e:
            error_handler.handle_error(f"Error generating semantic version: {str(e)}", e)
            return "1.0.0"
    
    def _generate_timestamp_version(self) -> str:
        """Generate a timestamp-based version number."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            return f"v{timestamp}"
            
        except Exception as e:
            error_handler.handle_error(f"Error generating timestamp version: {str(e)}", e)
            return f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def _calculate_model_hash(self, model: Any) -> str:
        """Calculate a hash of the model for integrity checking."""
        try:
            # Convert model to bytes for hashing
            model_bytes = pickle.dumps(model)
            return hashlib.sha256(model_bytes).hexdigest()
            
        except Exception as e:
            error_handler.handle_error(f"Error calculating model hash: {str(e)}", e)
            return ""
    
    def _cleanup_old_versions(self, model_name: str) -> None:
        """Clean up old versions to maintain the backup limit."""
        try:
            if model_name not in self.registry["models"]:
                return
            
            versions = self.registry["models"][model_name]["versions"]
            if len(versions) <= self.backup_versions:
                return
            
            # Sort versions and keep only the latest ones
            sorted_versions = sorted(versions, key=lambda v: [int(x) for x in v.split('.')])
            versions_to_remove = sorted_versions[:-self.backup_versions]
            
            for version in versions_to_remove:
                self.delete_version(model_name, version)
                
        except Exception as e:
            error_handler.handle_error(f"Error cleaning up old versions: {str(e)}", e)
    
    def load_version(self, model_name: str, version: Optional[str] = None) -> Optional[Any]:
        """Load a specific version of a model."""
        try:
            if version is None:
                version = self.registry["models"][model_name]["latest_version"]
            
            version_key = f"{model_name}_{version}"
            if version_key not in self.registry["versions"]:
                logger.error(f"Version {version} not found for model {model_name}")
                return None
            
            version_info = self.registry["versions"][version_key]
            model_path = os.path.join(version_info["path"], "model.pkl")
            
            if not os.path.exists(model_path):
                logger.error(f"Model file not found at {model_path}")
                return None
            
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            logger.info(f"Loaded version {version} of model {model_name}")
            return model
            
        except Exception as e:
            error_handler.handle_error(f"Error loading version: {str(e)}", e)
            return None
    
    def get_version_metadata(self, model_name: str, version: Optional[str] = None) -> Dict[str, Any]:
        """Get metadata for a specific version of a model."""
        try:
            if version is None:
                version = self.registry["models"][model_name]["latest_version"]
            
            version_key = f"{model_name}_{version}"
            if version_key not in self.registry["versions"]:
                logger.error(f"Version {version} not found for model {model_name}")
                return {}
            
            version_info = self.registry["versions"][version_key]
            metadata_path = os.path.join(version_info["path"], "metadata.json")
            
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    return json.load(f)
            else:
                return version_info.get("metadata", {})
                
        except Exception as e:
            error_handler.handle_error(f"Error getting version metadata: {str(e)}", e)
            return {}
    
    def list_versions(self, model_name: str) -> List[Dict[str, Any]]:
        """List all versions of a model."""
        try:
            if model_name not in self.registry["models"]:
                logger.warning(f"Model {model_name} not found in registry")
                return []
            
            versions = []
            for version in self.registry["models"][model_name]["versions"]:
                version_metadata = self.get_version_metadata(model_name, version)
                versions.append({
                    "version": version,
                    "created_at": version_metadata.get("created_at", ""),
                    "model_hash": version_metadata.get("model_hash", ""),
                    "metadata": version_metadata
                })
            
            # Sort by version number
            versions.sort(key=lambda v: [int(x) for x in v["version"].split('.')])
            return versions
            
        except Exception as e:
            error_handler.handle_error(f"Error listing versions: {str(e)}", e)
            return []
    
    def delete_version(self, model_name: str, version: str) -> bool:
        """Delete a specific version of a model."""
        try:
            version_key = f"{model_name}_{version}"
            if version_key not in self.registry["versions"]:
                logger.warning(f"Version {version} not found for model {model_name}")
                return False
            
            version_info = self.registry["versions"][version_key]
            version_dir = version_info["path"]
            
            # Remove version directory
            if os.path.exists(version_dir):
                shutil.rmtree(version_dir)
            
            # Update registry
            if model_name in self.registry["models"]:
                if version in self.registry["models"][model_name]["versions"]:
                    self.registry["models"][model_name]["versions"].remove(version)
                
                # Update latest version if needed
                if self.registry["models"][model_name]["latest_version"] == version:
                    versions = self.registry["models"][model_name]["versions"]
                    if versions:
                        self.registry["models"][model_name]["latest_version"] = max(
                            versions, key=lambda v: [int(x) for x in v.split('.')]
                        )
                    else:
                        # No versions left, remove model from registry
                        del self.registry["models"][model_name]
            
            # Remove from versions registry
            del self.registry["versions"][version_key]
            
            # Update metadata
            self.registry["metadata"]["total_models"] = len(self.registry["models"])
            self.registry["metadata"]["total_versions"] = len(self.registry["versions"])
            
            # Save registry
            self._save_registry()
            
            logger.info(f"Deleted version {version} of model {model_name}")
            return True
            
        except Exception as e:
            error_handler.handle_error(f"Error deleting version: {str(e)}", e)
            return False
    
    def compare_versions(self, model_name: str, version1: str, version2: str) -> Dict[str, Any]:
        """Compare two versions of a model."""
        try:
            metadata1 = self.get_version_metadata(model_name, version1)
            metadata2 = self.get_version_metadata(model_name, version2)
            
            comparison = {
                "model_name": model_name,
                "version1": {
                    "version": version1,
                    "metadata": metadata1
                },
                "version2": {
                    "version": version2,
                    "metadata": metadata2
                },
                "differences": {}
            }
            
            # Compare metadata fields
            all_keys = set(metadata1.keys()) | set(metadata2.keys())
            for key in all_keys:
                if key not in metadata1:
                    comparison["differences"][key] = {
                        "type": "added_in_version2",
                        "value": metadata2[key]
                    }
                elif key not in metadata2:
                    comparison["differences"][key] = {
                        "type": "removed_in_version2",
                        "value": metadata1[key]
                    }
                elif metadata1[key] != metadata2[key]:
                    comparison["differences"][key] = {
                        "type": "changed",
                        "value1": metadata1[key],
                        "value2": metadata2[key]
                    }
            
            return comparison
            
        except Exception as e:
            error_handler.handle_error(f"Error comparing versions: {str(e)}", e)
            return {}
    
    def get_model_summary(self, model_name: str) -> Dict[str, Any]:
        """Get a summary of a model's versioning history."""
        try:
            if model_name not in self.registry["models"]:
                logger.warning(f"Model {model_name} not found in registry")
                return {}
            
            model_info = self.registry["models"][model_name]
            versions = self.list_versions(model_name)
            
            summary = {
                "model_name": model_name,
                "total_versions": len(versions),
                "latest_version": model_info["latest_version"],
                "created_at": model_info["created_at"],
                "last_updated": model_info["last_updated"],
                "versions": versions
            }
            
            return summary
            
        except Exception as e:
            error_handler.handle_error(f"Error getting model summary: {str(e)}", e)
            return {}
    
    def export_model(self, model_name: str, version: Optional[str] = None,
                    export_path: Optional[str] = None) -> bool:
        """Export a model version to a specified path."""
        try:
            if version is None:
                version = self.registry["models"][model_name]["latest_version"]
            
            version_key = f"{model_name}_{version}"
            if version_key not in self.registry["versions"]:
                logger.error(f"Version {version} not found for model {model_name}")
                return False
            
            version_info = self.registry["versions"][version_key]
            source_dir = version_info["path"]
            
            if export_path is None:
                export_path = os.path.join(os.getcwd(), f"{model_name}_v{version}")
            
            # Copy version directory to export path
            if os.path.exists(export_path):
                shutil.rmtree(export_path)
            
            shutil.copytree(source_dir, export_path)
            
            logger.info(f"Exported model {model_name} version {version} to {export_path}")
            return True
            
        except Exception as e:
            error_handler.handle_error(f"Error exporting model: {str(e)}", e)
            return False
    
    def import_model(self, import_path: str, model_name: str, 
                    version: Optional[str] = None) -> bool:
        """Import a model from a specified path."""
        try:
            if not os.path.exists(import_path):
                logger.error(f"Import path {import_path} does not exist")
                return False
            
            # Load model and metadata
            model_path = os.path.join(import_path, "model.pkl")
            metadata_path = os.path.join(import_path, "metadata.json")
            
            if not os.path.exists(model_path):
                logger.error(f"Model file not found at {model_path}")
                return False
            
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            # Load metadata
            metadata = {}
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            
            # Create version
            if version is None:
                version = metadata.get("version", "1.0.0")
            
            # Override model name if specified
            if model_name != metadata.get("model_name", model_name):
                metadata["model_name"] = model_name
                metadata["imported_at"] = datetime.now().isoformat()
            
            self.create_version(model_name, model, metadata)
            
            logger.info(f"Imported model {model_name} version {version} from {import_path}")
            return True
            
        except Exception as e:
            error_handler.handle_error(f"Error importing model: {str(e)}", e)
            return False
    
    def get_registry_summary(self) -> Dict[str, Any]:
        """Get a summary of the entire model registry."""
        try:
            summary = {
                "registry_path": self.registry_path,
                "total_models": self.registry["metadata"]["total_models"],
                "total_versions": self.registry["metadata"]["total_versions"],
                "created_at": self.registry["metadata"]["created_at"],
                "last_updated": self.registry["metadata"]["last_updated"],
                "models": {}
            }
            
            for model_name in self.registry["models"]:
                summary["models"][model_name] = self.get_model_summary(model_name)
            
            return summary
            
        except Exception as e:
            error_handler.handle_error(f"Error getting registry summary: {str(e)}", e)
            return {}
