"""
Batch Processor

Large-scale batch data processing with:
- Batch job management
- Data partitioning
- Parallel processing
- Progress tracking
"""

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import os

from loguru import logger

from src.config.big_data_config import get_big_data_config
from src.core.error_handling_service import ErrorHandlingService


@dataclass
class BatchJob:
    """Represents a batch processing job."""
    id: str
    name: str
    data_source: str
    processor: Callable
    config: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1
    timeout: int = 3600
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class BatchMetrics:
    """Metrics for batch processing performance."""
    jobs_completed: int = 0
    total_processing_time: float = 0.0
    records_processed: int = 0
    errors: Dict[str, int] = field(default_factory=dict)
    start_time: datetime = field(default_factory=datetime.now)


class BatchProcessor:
    """
    Large-scale batch data processing.
    """
    
    def __init__(self):
        """Initialize the batch processor."""
        self.config = get_big_data_config()
        self.error_handler = ErrorHandlingService()
        
        # Job management
        self.active_jobs: Dict[str, BatchJob] = {}
        self.job_queue: List[BatchJob] = []
        self.completed_jobs: List[BatchJob] = []
        self.metrics = BatchMetrics()
        
        # Processing resources
        self.executor_pool = ThreadPoolExecutor(
            max_workers=self.config.get('batch_max_workers', 4)
        )
        
        # Data partitioning
        self.chunk_size = self.config.get('chunk_size', 1000)
        self.max_memory_mb = self.config.get('max_memory_mb', 1024)
        
        logger.info("BatchProcessor initialized")
    
    async def submit_batch_job(self, job: BatchJob) -> str:
        """Submit a batch job for processing."""
        try:
            # Add to job queue
            self.job_queue.append(job)
            self.active_jobs[job.id] = job
            
            # Process job asynchronously
            asyncio.create_task(self._process_batch_job(job))
            
            logger.info(f"Submitted batch job: {job.id}")
            return job.id
            
        except Exception as e:
            await self.error_handler.handle_error(
                "batch_job_submission_error",
                f"Failed to submit batch job {job.name}: {str(e)}",
                error_data={'job': job.name, 'error': str(e)}
            )
            return ""
    
    async def _process_batch_job(self, job: BatchJob) -> None:
        """Process a batch job."""
        start_time = datetime.now()
        
        try:
            logger.info(f"Processing batch job: {job.id}")
            
            # Load data in chunks
            data_chunks = await self._load_data_chunks(job.data_source)
            
            # Process each chunk
            total_records = 0
            for chunk_idx, chunk in enumerate(data_chunks):
                try:
                    # Process chunk
                    result = await self._process_chunk(chunk, job.processor)
                    total_records += len(chunk)
                    
                    logger.info(
                        f"Processed chunk {chunk_idx + 1}/{len(data_chunks)} "
                        f"for job {job.id}"
                    )
                    
                except Exception as e:
                    await self.error_handler.handle_error(
                        "chunk_processing_error",
                        f"Failed to process chunk {chunk_idx} for job {job.id}: {str(e)}",
                        error_data={'job_id': job.id, 'chunk_idx': chunk_idx, 'error': str(e)}
                    )
            
            # Update metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            self.metrics.jobs_completed += 1
            self.metrics.total_processing_time += processing_time
            self.metrics.records_processed += total_records
            
            # Move to completed jobs
            self.completed_jobs.append(job)
            del self.active_jobs[job.id]
            
            logger.info(
                f"Completed batch job: {job.id} in {processing_time:.2f}s, "
                f"processed {total_records} records"
            )
            
        except Exception as e:
            await self.error_handler.handle_error(
                "batch_job_processing_error",
                f"Failed to process batch job {job.id}: {str(e)}",
                error_data={'job_id': job.id, 'error': str(e)}
            )
            
            # Update error metrics
            self.metrics.errors[job.id] = self.metrics.errors.get(job.id, 0) + 1
    
    async def _load_data_chunks(self, data_source: str) -> List[List[Dict[str, Any]]]:
        """Load data from source in chunks."""
        try:
            chunks = []
            
            # Determine file type and load accordingly
            if data_source.endswith('.json'):
                chunks = await self._load_json_chunks(data_source)
            elif data_source.endswith('.csv'):
                chunks = await self._load_csv_chunks(data_source)
            elif data_source.endswith('.parquet'):
                chunks = await self._load_parquet_chunks(data_source)
            else:
                # Default to text file processing
                chunks = await self._load_text_chunks(data_source)
            
            logger.info(f"Loaded {len(chunks)} chunks from {data_source}")
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to load data chunks from {data_source}: {str(e)}")
            raise
    
    async def _load_json_chunks(self, file_path: str) -> List[List[Dict[str, Any]]]:
        """Load JSON data in chunks."""
        try:
            chunks = []
            current_chunk = []
            
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            record = json.loads(line.strip())
                            current_chunk.append(record)
                            
                            if len(current_chunk) >= self.chunk_size:
                                chunks.append(current_chunk)
                                current_chunk = []
                                
                        except json.JSONDecodeError:
                            logger.warning(f"Invalid JSON line: {line.strip()}")
                            continue
            
            # Add remaining records
            if current_chunk:
                chunks.append(current_chunk)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to load JSON chunks: {str(e)}")
            raise
    
    async def _load_csv_chunks(self, file_path: str) -> List[List[Dict[str, Any]]]:
        """Load CSV data in chunks."""
        try:
            import pandas as pd
            
            chunks = []
            chunk_iterator = pd.read_csv(
                file_path, 
                chunksize=self.chunk_size,
                encoding='utf-8'
            )
            
            for chunk_df in chunk_iterator:
                chunk_records = chunk_df.to_dict('records')
                chunks.append(chunk_records)
            
            return chunks
            
        except ImportError:
            logger.warning("pandas not available for CSV processing")
            return await self._load_text_chunks(file_path)
        except Exception as e:
            logger.error(f"Failed to load CSV chunks: {str(e)}")
            raise
    
    async def _load_parquet_chunks(self, file_path: str) -> List[List[Dict[str, Any]]]:
        """Load Parquet data in chunks."""
        try:
            import pandas as pd
            
            chunks = []
            chunk_iterator = pd.read_parquet(
                file_path, 
                chunksize=self.chunk_size
            )
            
            for chunk_df in chunk_iterator:
                chunk_records = chunk_df.to_dict('records')
                chunks.append(chunk_records)
            
            return chunks
            
        except ImportError:
            logger.warning("pandas not available for Parquet processing")
            return []
        except Exception as e:
            logger.error(f"Failed to load Parquet chunks: {str(e)}")
            raise
    
    async def _load_text_chunks(self, file_path: str) -> List[List[Dict[str, Any]]]:
        """Load text data in chunks."""
        try:
            chunks = []
            current_chunk = []
            
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        record = {'text': line.strip()}
                        current_chunk.append(record)
                        
                        if len(current_chunk) >= self.chunk_size:
                            chunks.append(current_chunk)
                            current_chunk = []
            
            # Add remaining records
            if current_chunk:
                chunks.append(current_chunk)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to load text chunks: {str(e)}")
            raise
    
    async def _process_chunk(self, chunk: List[Dict[str, Any]], 
                           processor: Callable) -> List[Dict[str, Any]]:
        """Process a data chunk."""
        try:
            # Process chunk in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor_pool, 
                processor, 
                chunk
            )
            
            return result if result else []
            
        except Exception as e:
            logger.error(f"Failed to process chunk: {str(e)}")
            raise
    
    async def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get status of a specific batch job."""
        if job_id in self.active_jobs:
            job = self.active_jobs[job_id]
            return {
                'status': 'running',
                'job_name': job.name,
                'created_at': job.created_at.isoformat(),
                'priority': job.priority
            }
        else:
            # Check completed jobs
            for job in self.completed_jobs:
                if job.id == job_id:
                    return {
                        'status': 'completed',
                        'job_name': job.name,
                        'created_at': job.created_at.isoformat()
                    }
            
            return {
                'status': 'not_found',
                'job_id': job_id
            }
    
    async def get_batch_metrics(self) -> Dict[str, Any]:
        """Get batch processing metrics."""
        return {
            'jobs_completed': self.metrics.jobs_completed,
            'active_jobs': len(self.active_jobs),
            'queued_jobs': len(self.job_queue),
            'total_processing_time': self.metrics.total_processing_time,
            'records_processed': self.metrics.records_processed,
            'errors': self.metrics.errors,
            'uptime': (datetime.now() - self.metrics.start_time).total_seconds()
        }
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a batch job."""
        try:
            if job_id in self.active_jobs:
                del self.active_jobs[job_id]
                
                # Remove from queue if present
                self.job_queue = [job for job in self.job_queue if job.id != job_id]
                
                logger.info(f"Cancelled batch job: {job_id}")
                return True
            else:
                logger.warning(f"Job {job_id} not found or already completed")
                return False
                
        except Exception as e:
            await self.error_handler.handle_error(
                "job_cancellation_error",
                f"Failed to cancel job {job_id}: {str(e)}",
                error_data={'job_id': job_id, 'error': str(e)}
            )
            return False
    
    def __del__(self):
        """Cleanup on deletion."""
        if self.executor_pool:
            self.executor_pool.shutdown(wait=False)
