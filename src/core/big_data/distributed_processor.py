"""
Distributed Processor

Apache Spark integration for distributed data processing with:
- Spark session management
- Distributed data operations
- Performance optimization
- Resource management
"""

import os
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

from loguru import logger

from src.config.big_data_config import get_big_data_config
from src.core.error_handling_service import ErrorHandlingService


@dataclass
class SparkJob:
    """Represents a Spark job configuration."""
    name: str
    data_source: str
    transformations: List[str]
    output_destination: str
    config: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1
    timeout: int = 3600


@dataclass
class ProcessingMetrics:
    """Metrics for distributed processing performance."""
    jobs_completed: int = 0
    total_processing_time: float = 0.0
    data_processed_gb: float = 0.0
    errors: Dict[str, int] = field(default_factory=dict)
    start_time: datetime = field(default_factory=datetime.now)


class DistributedProcessor:
    """
    Distributed data processing using Apache Spark.
    """
    
    def __init__(self):
        """Initialize the distributed processor."""
        self.config = get_big_data_config()
        self.error_handler = ErrorHandlingService()
        
        # Spark session management
        self.spark_session = None
        self.spark_context = None
        
        # Job management
        self.active_jobs: Dict[str, SparkJob] = {}
        self.job_queue: List[SparkJob] = []
        self.metrics = ProcessingMetrics()
        
        # Resource management
        self.executor_pool = ThreadPoolExecutor(
            max_workers=self.config.get('max_workers', 4)
        )
        
        # Performance tracking
        self.performance_history: List[Dict[str, Any]] = []
        
        logger.info("DistributedProcessor initialized")
    
    async def initialize_spark(self) -> bool:
        """Initialize Apache Spark session."""
        try:
            # Conditional import for Spark
            try:
                from pyspark.sql import SparkSession
                from pyspark import SparkConf
            except ImportError:
                logger.warning("PySpark not available. Install with: pip install pyspark")
                return False
            
            # Configure Spark
            spark_config = self.config.get('spark', {})
            conf = SparkConf()
            
            # Set Spark configuration
            for key, value in spark_config.items():
                conf.set(key, str(value))
            
            # Create Spark session
            self.spark_session = SparkSession.builder \
                .config(conf=conf) \
                .appName("SentimentAnalytics") \
                .getOrCreate()
            
            self.spark_context = self.spark_session.sparkContext
            
            logger.info("Apache Spark session initialized successfully")
            return True
            
        except Exception as e:
            await self.error_handler.handle_error(
                "spark_initialization_error",
                f"Failed to initialize Spark: {str(e)}",
                error_data={'error': str(e)}
            )
            return False
    
    async def submit_job(self, job: SparkJob) -> str:
        """Submit a Spark job for processing."""
        try:
            job_id = f"{job.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Add to job queue
            self.job_queue.append(job)
            self.active_jobs[job_id] = job
            
            # Process job asynchronously
            asyncio.create_task(self._process_job(job_id, job))
            
            logger.info(f"Submitted Spark job: {job_id}")
            return job_id
            
        except Exception as e:
            await self.error_handler.handle_error(
                "job_submission_error",
                f"Failed to submit job {job.name}: {str(e)}",
                error_data={'job': job.name, 'error': str(e)}
            )
            return ""
    
    async def _process_job(self, job_id: str, job: SparkJob) -> None:
        """Process a Spark job."""
        start_time = datetime.now()
        
        try:
            logger.info(f"Processing Spark job: {job_id}")
            
            # Load data
            df = await self._load_data(job.data_source)
            
            # Apply transformations
            for transformation in job.transformations:
                df = await self._apply_transformation(df, transformation)
            
            # Save results
            await self._save_results(df, job.output_destination)
            
            # Update metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            self.metrics.jobs_completed += 1
            self.metrics.total_processing_time += processing_time
            
            # Remove from active jobs
            del self.active_jobs[job_id]
            
            logger.info(f"Completed Spark job: {job_id} in {processing_time:.2f}s")
            
        except Exception as e:
            await self.error_handler.handle_error(
                "job_processing_error",
                f"Failed to process job {job_id}: {str(e)}",
                error_data={'job_id': job_id, 'error': str(e)}
            )
            
            # Update error metrics
            self.metrics.errors[job_id] = self.metrics.errors.get(job_id, 0) + 1
    
    async def _load_data(self, data_source: str):
        """Load data from source."""
        try:
            if data_source.startswith('s3://'):
                return self.spark_session.read.parquet(data_source)
            elif data_source.startswith('hdfs://'):
                return self.spark_session.read.parquet(data_source)
            else:
                return self.spark_session.read.csv(data_source, header=True, inferSchema=True)
        except Exception as e:
            logger.error(f"Failed to load data from {data_source}: {str(e)}")
            raise
    
    async def _apply_transformation(self, df, transformation: str):
        """Apply a transformation to the DataFrame."""
        try:
            # Basic transformations
            if transformation == "drop_duplicates":
                return df.dropDuplicates()
            elif transformation == "fill_null":
                return df.fillna(0)
            elif transformation.startswith("filter_"):
                # Custom filter logic
                return df.filter(transformation[7:])  # Remove "filter_" prefix
            else:
                logger.warning(f"Unknown transformation: {transformation}")
                return df
        except Exception as e:
            logger.error(f"Failed to apply transformation {transformation}: {str(e)}")
            raise
    
    async def _save_results(self, df, output_destination: str) -> None:
        """Save results to destination."""
        try:
            if output_destination.startswith('s3://'):
                df.write.mode('overwrite').parquet(output_destination)
            elif output_destination.startswith('hdfs://'):
                df.write.mode('overwrite').parquet(output_destination)
            else:
                df.write.mode('overwrite').csv(output_destination, header=True)
        except Exception as e:
            logger.error(f"Failed to save results to {output_destination}: {str(e)}")
            raise
    
    async def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get status of a specific job."""
        if job_id in self.active_jobs:
            return {
                'status': 'running',
                'job': self.active_jobs[job_id].name,
                'start_time': self.metrics.start_time.isoformat()
            }
        else:
            return {
                'status': 'completed',
                'job_id': job_id
            }
    
    async def get_processing_metrics(self) -> Dict[str, Any]:
        """Get processing metrics."""
        return {
            'jobs_completed': self.metrics.jobs_completed,
            'active_jobs': len(self.active_jobs),
            'total_processing_time': self.metrics.total_processing_time,
            'errors': self.metrics.errors,
            'uptime': (datetime.now() - self.metrics.start_time).total_seconds()
        }
    
    async def stop_spark(self) -> None:
        """Stop Spark session."""
        try:
            if self.spark_session:
                self.spark_session.stop()
                self.spark_session = None
                self.spark_context = None
                logger.info("Apache Spark session stopped")
        except Exception as e:
            await self.error_handler.handle_error(
                "spark_stop_error",
                f"Failed to stop Spark: {str(e)}",
                error_data={'error': str(e)}
            )
    
    def __del__(self):
        """Cleanup on deletion."""
        if self.executor_pool:
            self.executor_pool.shutdown(wait=False)
