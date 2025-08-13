"""
Big Data Processing Module

Advanced big data processing capabilities including:
- Distributed processing with Apache Spark
- Data lake integration
- Batch processing
- Data governance
"""

from .distributed_processor import DistributedProcessor
from .data_lake_integration import DataLakeIntegration
from .batch_processor import BatchProcessor
from .data_governance import DataGovernance

__all__ = [
    'DistributedProcessor',
    'DataLakeIntegration', 
    'BatchProcessor',
    'DataGovernance'
]
