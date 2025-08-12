"""
Configuration for duplicate detection settings.
"""

from typing import Dict, Any
from pydantic import BaseModel, Field


class DuplicateDetectionConfig(BaseModel):
    """Configuration for duplicate detection behavior."""
    
    # Enable/disable duplicate detection
    enabled: bool = Field(
        default=True,
        description="Enable duplicate detection"
    )
    
    # Duplicate detection strategies
    check_file_path: bool = Field(
        default=True,
        description="Check for duplicate file paths"
    )
    
    check_content_hash: bool = Field(
        default=True,
        description="Check for duplicate content hashes"
    )
    
    check_similarity: bool = Field(
        default=False,
        description="Check for similar content using vector similarity"
    )
    
    # Thresholds
    similarity_threshold: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Similarity threshold for near-duplicate detection"
    )
    
    content_hash_threshold: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Threshold for exact content hash matches"
    )
    
    # File size limits
    max_file_size_for_hash: int = Field(
        default=100 * 1024 * 1024,  # 100MB
        description="Maximum file size for content hash computation"
    )
    
    # Behavior settings
    default_action: str = Field(
        default="skip",
        description="Default action for duplicates: skip, update, or reprocess"
    )
    
    allow_force_reprocess: bool = Field(
        default=True,
        description="Allow forcing reprocessing of duplicates"
    )
    
    # Cache settings
    cache_duplicate_results: bool = Field(
        default=True,
        description="Cache duplicate detection results"
    )
    
    cache_ttl_hours: int = Field(
        default=24,
        description="Cache TTL in hours for duplicate detection results"
    )
    
    # Database settings
    cleanup_old_records_days: int = Field(
        default=30,
        description="Days to keep old processing records"
    )
    
    # Logging
    log_duplicate_detections: bool = Field(
        default=True,
        description="Log duplicate detection events"
    )
    
    # Data type specific settings
    data_type_settings: Dict[str, Dict[str, Any]] = Field(
        default_factory=lambda: {
            "pdf": {
                "check_file_path": True,
                "check_content_hash": True,
                "similarity_threshold": 0.98,
                "default_action": "skip"
            },
            "text": {
                "check_file_path": False,
                "check_content_hash": True,
                "similarity_threshold": 0.95,
                "default_action": "skip"
            },
            "image": {
                "check_file_path": True,
                "check_content_hash": True,
                "similarity_threshold": 0.90,
                "default_action": "skip"
            },
            "video": {
                "check_file_path": True,
                "check_content_hash": False,  # Videos are large
                "similarity_threshold": 0.85,
                "default_action": "skip"
            },
            "audio": {
                "check_file_path": True,
                "check_content_hash": False,  # Audio files are large
                "similarity_threshold": 0.90,
                "default_action": "skip"
            }
        },
        description="Data type specific duplicate detection settings"
    )
    
    def get_settings_for_data_type(self, data_type: str) -> Dict[str, Any]:
        """Get settings for a specific data type."""
        base_settings = {
            "check_file_path": self.check_file_path,
            "check_content_hash": self.check_content_hash,
            "check_similarity": self.check_similarity,
            "similarity_threshold": self.similarity_threshold,
            "default_action": self.default_action
        }
        
        # Override with data type specific settings
        if data_type in self.data_type_settings:
            base_settings.update(self.data_type_settings[data_type])
        
        return base_settings


# Default configuration
default_config = DuplicateDetectionConfig()

# Configuration presets
presets = {
    "strict": DuplicateDetectionConfig(
        similarity_threshold=0.98,
        default_action="skip",
        check_file_path=True,
        check_content_hash=True,
        check_similarity=True
    ),
    "lenient": DuplicateDetectionConfig(
        similarity_threshold=0.90,
        default_action="update",
        check_file_path=True,
        check_content_hash=False,
        check_similarity=False
    ),
    "disabled": DuplicateDetectionConfig(
        enabled=False
    )
}
