"""
Parallel processing service for multilingual content processing.
Implements concurrent processing for large documents and entity extraction.
"""

import asyncio
import time
from typing import List, Dict, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ProcessingTask:
    """Represents a processing task with metadata."""
    task_id: str
    content: str
    language: str
    task_type: str
    priority: int = 1
    created_at: float = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()


@dataclass
class ProcessingResult:
    """Represents the result of a processing task."""
    task_id: str
    success: bool
    result: Any
    processing_time: float
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ParallelProcessor:
    """Parallel processing system for multilingual content."""
    
    def __init__(self, 
                 max_workers: int = None,
                 max_thread_workers: int = 10,
                 max_process_workers: int = 4,
                 chunk_size: int = 1000):
        self.max_workers = max_workers or min(32, (multiprocessing.cpu_count() or 1) + 4)
        self.max_thread_workers = max_thread_workers
        self.max_process_workers = max_process_workers
        self.chunk_size = chunk_size
        
        # Semaphores for controlling concurrency
        self.thread_semaphore = asyncio.Semaphore(max_thread_workers)
        self.process_semaphore = asyncio.Semaphore(max_process_workers)
        
        # Task queues
        self.high_priority_queue = asyncio.Queue()
        self.normal_priority_queue = asyncio.Queue()
        self.low_priority_queue = asyncio.Queue()
        
        # Processing statistics
        self.stats = {
            "tasks_processed": 0,
            "tasks_failed": 0,
            "total_processing_time": 0.0,
            "average_processing_time": 0.0,
            "concurrent_tasks": 0,
            "max_concurrent_tasks": 0
        }
        
        # Active tasks
        self.active_tasks = set()
        
        # Initialize executors
        self.thread_executor = ThreadPoolExecutor(max_workers=max_thread_workers)
        self.process_executor = ProcessPoolExecutor(max_workers=max_process_workers)
    
    async def process_pdf_pages(self, pdf_path: str, language: str = "en") -> List[ProcessingResult]:
        """Process PDF pages in parallel."""
        try:
            # Extract pages from PDF
            pages = await self._extract_pdf_pages(pdf_path)
            
            # Create processing tasks for each page
            tasks = []
            for i, page_content in enumerate(pages):
                task = ProcessingTask(
                    task_id=f"pdf_page_{i}",
                    content=page_content,
                    language=language,
                    task_type="pdf_page_processing"
                )
                tasks.append(task)
            
            # Process pages in parallel
            results = await self._process_tasks_parallel(tasks)
            return results
            
        except Exception as e:
            logger.error(f"Error processing PDF pages: {e}")
            return []
    
    async def parallel_entity_extraction(self, texts: List[str], language: str = "en") -> List[ProcessingResult]:
        """Extract entities from multiple texts in parallel."""
        try:
            # Create processing tasks
            tasks = []
            for i, text in enumerate(texts):
                task = ProcessingTask(
                    task_id=f"entity_extraction_{i}",
                    content=text,
                    language=language,
                    task_type="entity_extraction"
                )
                tasks.append(task)
            
            # Process in parallel
            results = await self._process_tasks_parallel(tasks)
            return results
            
        except Exception as e:
            logger.error(f"Error in parallel entity extraction: {e}")
            return []
    
    async def parallel_translation(self, texts: List[str], source_lang: str, target_lang: str) -> List[ProcessingResult]:
        """Translate multiple texts in parallel."""
        try:
            # Create processing tasks
            tasks = []
            for i, text in enumerate(texts):
                task = ProcessingTask(
                    task_id=f"translation_{i}",
                    content=text,
                    language=f"{source_lang}_{target_lang}",
                    task_type="translation"
                )
                tasks.append(task)
            
            # Process in parallel
            results = await self._process_tasks_parallel(tasks)
            return results
            
        except Exception as e:
            logger.error(f"Error in parallel translation: {e}")
            return []
    
    async def _process_tasks_parallel(self, tasks: List[ProcessingTask]) -> List[ProcessingResult]:
        """Process multiple tasks in parallel with load balancing."""
        try:
            # Sort tasks by priority
            high_priority = [t for t in tasks if t.priority == 3]
            normal_priority = [t for t in tasks if t.priority == 2]
            low_priority = [t for t in tasks if t.priority == 1]
            
            # Process high priority tasks first
            all_results = []
            
            # Process high priority tasks
            if high_priority:
                high_results = await self._process_task_batch(high_priority)
                all_results.extend(high_results)
            
            # Process normal priority tasks
            if normal_priority:
                normal_results = await self._process_task_batch(normal_priority)
                all_results.extend(normal_results)
            
            # Process low priority tasks
            if low_priority:
                low_results = await self._process_task_batch(low_priority)
                all_results.extend(low_results)
            
            return all_results
            
        except Exception as e:
            logger.error(f"Error in parallel task processing: {e}")
            return []
    
    async def _process_task_batch(self, tasks: List[ProcessingTask]) -> List[ProcessingResult]:
        """Process a batch of tasks with controlled concurrency."""
        try:
            # Create semaphore for this batch
            semaphore = asyncio.Semaphore(self.max_workers)
            
            async def process_single_task(task: ProcessingTask) -> ProcessingResult:
                async with semaphore:
                    return await self._process_single_task(task)
            
            # Process tasks concurrently
            tasks_futures = [process_single_task(task) for task in tasks]
            results = await asyncio.gather(*tasks_futures, return_exceptions=True)
            
            # Handle exceptions
            processed_results = []
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Task processing error: {result}")
                    processed_results.append(ProcessingResult(
                        task_id="unknown",
                        success=False,
                        result=None,
                        processing_time=0.0,
                        error=str(result)
                    ))
                else:
                    processed_results.append(result)
            
            return processed_results
            
        except Exception as e:
            logger.error(f"Error in task batch processing: {e}")
            return []
    
    async def _process_single_task(self, task: ProcessingTask) -> ProcessingResult:
        """Process a single task with appropriate executor."""
        start_time = time.time()
        task_id = task.task_id
        
        try:
            # Update active tasks
            self.active_tasks.add(task_id)
            self.stats["concurrent_tasks"] = len(self.active_tasks)
            self.stats["max_concurrent_tasks"] = max(
                self.stats["max_concurrent_tasks"], 
                self.stats["concurrent_tasks"]
            )
            
            # Choose appropriate executor based on task type
            if task.task_type in ["pdf_page_processing", "entity_extraction"]:
                # CPU-intensive tasks use process executor
                async with self.process_semaphore:
                    result = await self._run_in_process_executor(task)
            else:
                # I/O intensive tasks use thread executor
                async with self.thread_semaphore:
                    result = await self._run_in_thread_executor(task)
            
            processing_time = time.time() - start_time
            
            # Update statistics
            self.stats["tasks_processed"] += 1
            self.stats["total_processing_time"] += processing_time
            self.stats["average_processing_time"] = (
                self.stats["total_processing_time"] / self.stats["tasks_processed"]
            )
            
            return ProcessingResult(
                task_id=task_id,
                success=True,
                result=result,
                processing_time=processing_time,
                metadata={"task_type": task.task_type, "language": task.language}
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.stats["tasks_failed"] += 1
            
            logger.error(f"Error processing task {task_id}: {e}")
            
            return ProcessingResult(
                task_id=task_id,
                success=False,
                result=None,
                processing_time=processing_time,
                error=str(e),
                metadata={"task_type": task.task_type, "language": task.language}
            )
        
        finally:
            # Remove from active tasks
            self.active_tasks.discard(task_id)
            self.stats["concurrent_tasks"] = len(self.active_tasks)
    
    async def _run_in_thread_executor(self, task: ProcessingTask) -> Any:
        """Run task in thread executor."""
        loop = asyncio.get_event_loop()
        
        if task.task_type == "translation":
            return await loop.run_in_executor(
                self.thread_executor,
                self._translate_text,
                task.content,
                task.language
            )
        else:
            return await loop.run_in_executor(
                self.thread_executor,
                self._process_text,
                task.content,
                task.language,
                task.task_type
            )
    
    async def _run_in_process_executor(self, task: ProcessingTask) -> Any:
        """Run task in process executor."""
        loop = asyncio.get_event_loop()
        
        if task.task_type == "pdf_page_processing":
            return await loop.run_in_executor(
                self.process_executor,
                self._process_pdf_page,
                task.content,
                task.language
            )
        elif task.task_type == "entity_extraction":
            return await loop.run_in_executor(
                self.process_executor,
                self._extract_entities,
                task.content,
                task.language
            )
        else:
            return await loop.run_in_executor(
                self.process_executor,
                self._process_text,
                task.content,
                task.language,
                task.task_type
            )
    
    def _translate_text(self, text: str, language_pair: str) -> str:
        """Translate text (placeholder implementation)."""
        # This would integrate with the actual translation service
        try:
            # Placeholder translation logic
            source_lang, target_lang = language_pair.split("_", 1)
            # Here you would call the actual translation service
            return f"Translated: {text[:50]}..."  # Placeholder
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return text
    
    def _process_pdf_page(self, page_content: str, language: str) -> Dict[str, Any]:
        """Process a single PDF page."""
        try:
            # Placeholder PDF page processing
            return {
                "text_length": len(page_content),
                "language": language,
                "entities": [],  # Would contain actual entity extraction
                "summary": page_content[:200] + "..." if len(page_content) > 200 else page_content
            }
        except Exception as e:
            logger.error(f"PDF page processing error: {e}")
            return {"error": str(e)}
    
    def _extract_entities(self, text: str, language: str) -> Dict[str, Any]:
        """Extract entities from text."""
        try:
            # Placeholder entity extraction
            return {
                "text_length": len(text),
                "language": language,
                "entities": [],  # Would contain actual entities
                "entity_count": 0
            }
        except Exception as e:
            logger.error(f"Entity extraction error: {e}")
            return {"error": str(e)}
    
    def _process_text(self, text: str, language: str, task_type: str) -> Dict[str, Any]:
        """Generic text processing."""
        try:
            return {
                "text_length": len(text),
                "language": language,
                "task_type": task_type,
                "processed": True
            }
        except Exception as e:
            logger.error(f"Text processing error: {e}")
            return {"error": str(e)}
    
    async def _extract_pdf_pages(self, pdf_path: str) -> List[str]:
        """Extract pages from PDF (placeholder implementation)."""
        try:
            # This would integrate with the actual PDF extraction service
            # For now, return placeholder content
            return [
                f"Page 1 content from {pdf_path}",
                f"Page 2 content from {pdf_path}",
                f"Page 3 content from {pdf_path}"
            ]
        except Exception as e:
            logger.error(f"PDF extraction error: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            "tasks_processed": self.stats["tasks_processed"],
            "tasks_failed": self.stats["tasks_failed"],
            "total_processing_time": self.stats["total_processing_time"],
            "average_processing_time": self.stats["average_processing_time"],
            "concurrent_tasks": self.stats["concurrent_tasks"],
            "max_concurrent_tasks": self.stats["max_concurrent_tasks"],
            "success_rate": (
                self.stats["tasks_processed"] / 
                (self.stats["tasks_processed"] + self.stats["tasks_failed"])
                if (self.stats["tasks_processed"] + self.stats["tasks_failed"]) > 0 
                else 0.0
            ),
            "active_tasks": len(self.active_tasks),
            "max_workers": self.max_workers,
            "max_thread_workers": self.max_thread_workers,
            "max_process_workers": self.max_process_workers
        }
    
    async def shutdown(self):
        """Shutdown the parallel processor."""
        try:
            # Wait for active tasks to complete
            while self.active_tasks:
                await asyncio.sleep(0.1)
            
            # Shutdown executors
            self.thread_executor.shutdown(wait=True)
            self.process_executor.shutdown(wait=True)
            
            logger.info("Parallel processor shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


# Global parallel processor instance
_global_parallel_processor: Optional[ParallelProcessor] = None


def get_global_parallel_processor() -> ParallelProcessor:
    """Get global parallel processor instance."""
    global _global_parallel_processor
    if _global_parallel_processor is None:
        _global_parallel_processor = ParallelProcessor()
    return _global_parallel_processor


def set_global_parallel_processor(processor: ParallelProcessor):
    """Set global parallel processor instance."""
    global _global_parallel_processor
    _global_parallel_processor = processor
