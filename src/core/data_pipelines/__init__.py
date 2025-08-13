"""
Data Pipelines Module

Advanced data pipeline capabilities including:
- Real-time data pipelines
- Data quality management
- Schema evolution
- Data catalog
"""

from .data_quality_manager import DataQualityManager
from .schema_manager import SchemaManager
from .data_catalog import DataCatalog

__all__ = [
    'DataQualityManager',
    'SchemaManager', 
    'DataCatalog'
]
