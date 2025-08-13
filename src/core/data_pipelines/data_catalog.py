"""
Data Catalog

Automated metadata management with:
- Dataset discovery
- Metadata indexing
- Search capabilities
- Data lineage tracking
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import json

from loguru import logger

from src.config.big_data_config import get_big_data_config
from src.core.error_handling_service import ErrorHandlingService


@dataclass
class DatasetMetadata:
    """Represents dataset metadata."""
    dataset_id: str
    name: str
    description: str = ""
    source: str = ""
    format: str = ""
    size_bytes: int = 0
    record_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)
    schema: Dict[str, Any] = field(default_factory=dict)
    quality_score: float = 0.0


class DataCatalog:
    """
    Automated metadata management.
    """
    
    def __init__(self):
        """Initialize data catalog."""
        self.config = get_big_data_config()
        self.error_handler = ErrorHandlingService()
        
        # Metadata storage
        self.datasets: Dict[str, DatasetMetadata] = {}
        self.metadata_index: Dict[str, List[str]] = {}
        
        logger.info("DataCatalog initialized")
    
    async def register_dataset(self, metadata: DatasetMetadata) -> bool:
        """Register a dataset in the catalog."""
        try:
            self.datasets[metadata.dataset_id] = metadata
            
            # Update metadata index
            await self._update_index(metadata)
            
            logger.info(f"Registered dataset: {metadata.dataset_id}")
            return True
            
        except Exception as e:
            await self.error_handler.handle_error(
                "dataset_registration_error",
                f"Failed to register dataset: {str(e)}",
                error_data={'dataset_id': metadata.dataset_id, 'error': str(e)}
            )
            return False
    
    async def _update_index(self, metadata: DatasetMetadata) -> None:
        """Update metadata index."""
        # Index by tags
        for tag in metadata.tags:
            if tag not in self.metadata_index:
                self.metadata_index[tag] = []
            self.metadata_index[tag].append(metadata.dataset_id)
        
        # Index by format
        if metadata.format not in self.metadata_index:
            self.metadata_index[metadata.format] = []
        self.metadata_index[metadata.format].append(metadata.dataset_id)
        
        # Index by source
        if metadata.source not in self.metadata_index:
            self.metadata_index[metadata.source] = []
        self.metadata_index[metadata.source].append(metadata.dataset_id)
    
    async def search_datasets(self, query: str) -> List[DatasetMetadata]:
        """Search datasets by query."""
        try:
            results = []
            query_lower = query.lower()
            
            for dataset_id, metadata in self.datasets.items():
                # Search in name
                if query_lower in metadata.name.lower():
                    results.append(metadata)
                    continue
                
                # Search in description
                if query_lower in metadata.description.lower():
                    results.append(metadata)
                    continue
                
                # Search in tags
                for tag in metadata.tags:
                    if query_lower in tag.lower():
                        results.append(metadata)
                        break
            
            return results
            
        except Exception as e:
            await self.error_handler.handle_error(
                "dataset_search_error",
                f"Failed to search datasets: {str(e)}",
                error_data={'query': query, 'error': str(e)}
            )
            return []
    
    async def get_datasets_by_tag(self, tag: str) -> List[DatasetMetadata]:
        """Get datasets by tag."""
        try:
            dataset_ids = self.metadata_index.get(tag, [])
            return [self.datasets[dataset_id] for dataset_id in dataset_ids 
                   if dataset_id in self.datasets]
            
        except Exception as e:
            await self.error_handler.handle_error(
                "tag_search_error",
                f"Failed to get datasets by tag: {str(e)}",
                error_data={'tag': tag, 'error': str(e)}
            )
            return []
    
    async def get_datasets_by_format(self, format_type: str) -> List[DatasetMetadata]:
        """Get datasets by format."""
        try:
            dataset_ids = self.metadata_index.get(format_type, [])
            return [self.datasets[dataset_id] for dataset_id in dataset_ids 
                   if dataset_id in self.datasets]
            
        except Exception as e:
            await self.error_handler.handle_error(
                "format_search_error",
                f"Failed to get datasets by format: {str(e)}",
                error_data={'format': format_type, 'error': str(e)}
            )
            return []
    
    async def get_dataset_metadata(self, dataset_id: str) -> Optional[DatasetMetadata]:
        """Get metadata for a specific dataset."""
        return self.datasets.get(dataset_id)
    
    async def update_dataset_metadata(self, dataset_id: str, 
                                    updates: Dict[str, Any]) -> bool:
        """Update dataset metadata."""
        try:
            if dataset_id not in self.datasets:
                logger.error(f"Dataset {dataset_id} not found")
                return False
            
            metadata = self.datasets[dataset_id]
            
            # Update fields
            for key, value in updates.items():
                if hasattr(metadata, key):
                    setattr(metadata, key, value)
            
            # Update last_updated timestamp
            metadata.last_updated = datetime.now()
            
            # Update index
            await self._update_index(metadata)
            
            logger.info(f"Updated metadata for dataset: {dataset_id}")
            return True
            
        except Exception as e:
            await self.error_handler.handle_error(
                "metadata_update_error",
                f"Failed to update metadata: {str(e)}",
                error_data={'dataset_id': dataset_id, 'error': str(e)}
            )
            return False
    
    async def get_catalog_summary(self) -> Dict[str, Any]:
        """Get catalog summary statistics."""
        try:
            total_datasets = len(self.datasets)
            total_size = sum(dataset.size_bytes for dataset in self.datasets.values())
            total_records = sum(dataset.record_count for dataset in self.datasets.values())
            
            # Get unique tags, formats, and sources
            all_tags = set()
            all_formats = set()
            all_sources = set()
            
            for metadata in self.datasets.values():
                all_tags.update(metadata.tags)
                all_formats.add(metadata.format)
                all_sources.add(metadata.source)
            
            return {
                'total_datasets': total_datasets,
                'total_size_bytes': total_size,
                'total_records': total_records,
                'unique_tags': len(all_tags),
                'unique_formats': len(all_formats),
                'unique_sources': len(all_sources),
                'recent_datasets': list(self.datasets.keys())[-10:]  # Last 10
            }
            
        except Exception as e:
            await self.error_handler.handle_error(
                "catalog_summary_error",
                f"Failed to get catalog summary: {str(e)}",
                error_data={'error': str(e)}
            )
            return {}
    
    async def export_catalog(self, export_path: str) -> bool:
        """Export catalog to file."""
        try:
            export_data = {
                'datasets': {
                    dataset_id: {
                        'dataset_id': metadata.dataset_id,
                        'name': metadata.name,
                        'description': metadata.description,
                        'source': metadata.source,
                        'format': metadata.format,
                        'size_bytes': metadata.size_bytes,
                        'record_count': metadata.record_count,
                        'created_at': metadata.created_at.isoformat(),
                        'last_updated': metadata.last_updated.isoformat(),
                        'tags': metadata.tags,
                        'schema': metadata.schema,
                        'quality_score': metadata.quality_score
                    } for dataset_id, metadata in self.datasets.items()
                },
                'metadata_index': self.metadata_index
            }
            
            # Write to file
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Exported catalog to: {export_path}")
            return True
            
        except Exception as e:
            await self.error_handler.handle_error(
                "catalog_export_error",
                f"Failed to export catalog: {str(e)}",
                error_data={'export_path': export_path, 'error': str(e)}
            )
            return False
