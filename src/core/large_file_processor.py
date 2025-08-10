#!/usr/bin/env python3
"""
Large file processing utilities for audio and video files.
Implements chunking, progressive processing, and memory-efficient streaming.
"""

import asyncio
import os
import tempfile
import time
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Generator, Callable
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import subprocess
import json

from loguru import logger


@dataclass
class ProcessingChunk:
    """Represents a chunk of audio/video content for processing."""
    chunk_id: str
    start_time: float
    end_time: float
    file_path: str
    duration: float
    size_bytes: int
    metadata: Dict[str, Any] = None


@dataclass
class ProcessingProgress:
    """Tracks processing progress for large files."""
    stage: str
    percentage: float
    message: str
    timestamp: float
    current_chunk: int = 0
    total_chunks: int = 0
    estimated_time_remaining: float = 0.0


class LargeFileProcessor:
    """Handles large audio and video files with chunking and progressive processing."""
    
    def __init__(
        self,
        chunk_duration: int = 300,  # 5 minutes
        max_workers: int = 4,
        cache_dir: str = "./cache",
        temp_dir: str = "./temp"
    ):
        self.chunk_duration = chunk_duration
        self.max_workers = max_workers
        self.cache_dir = Path(cache_dir)
        self.temp_dir = Path(temp_dir)
        
        # Create directories
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Progress tracking
        self.progress_callback: Optional[Callable] = None
        self.start_time = 0

    def set_progress_callback(self, callback: Callable[[ProcessingProgress], None]):
        """Set callback for progress updates."""
        self.progress_callback = callback

    def _update_progress(self, stage: str, percentage: float, message: str,
                        current_chunk: int = 0, total_chunks: int = 0):
        """Update progress and call callback if set."""
        if self.progress_callback:
            progress = ProcessingProgress(
                stage=stage,
                percentage=percentage,
                message=message,
                timestamp=time.time(),
                current_chunk=current_chunk,
                total_chunks=total_chunks,
                estimated_time_remaining=self._calculate_eta(percentage)
            )
            self.progress_callback(progress)
        
        # Also log to console for immediate feedback
        eta_str = ""
        if progress.estimated_time_remaining > 0:
            eta_str = f" (ETA: {progress.estimated_time_remaining:.1f}s)"
        
        logger.info(f"[{stage.upper()}] {percentage:.1f}% - {message}{eta_str}")
    
    def _calculate_eta(self, percentage: float) -> float:
        """Calculate estimated time remaining."""
        if percentage <= 0:
            return 0.0
        
        elapsed = time.time() - self.start_time
        estimated_total = elapsed / (percentage / 100.0)
        return max(0.0, estimated_total - elapsed)
    
    def _get_file_hash(self, file_path: str) -> str:
        """Generate hash for file caching."""
        return hashlib.md5(file_path.encode()).hexdigest()
    
    def _get_cached_result(self, file_hash: str, stage: str) -> Optional[Dict]:
        """Retrieve cached processing results."""
        cache_file = self.cache_dir / f"{file_hash}_{stage}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        return None
    
    def _save_cached_result(self, file_hash: str, stage: str, result: Dict):
        """Save processing results to cache."""
        cache_file = self.cache_dir / f"{file_hash}_{stage}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump(result, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    async def chunk_video_by_time(self, video_path: str) -> List[ProcessingChunk]:
        """Split large video into manageable time-based chunks."""
        self.start_time = time.time()
        self._update_progress("metadata_extraction", 0, "Extracting video metadata...")
        
        try:
            # Get video duration using ffprobe
            duration = await self._get_video_duration(video_path)
            if not duration:
                raise ValueError("Could not determine video duration")
            
            # Calculate number of chunks
            num_chunks = max(1, int(duration / self.chunk_duration))
            chunks = []
            
            self._update_progress("chunking", 10, f"Creating {num_chunks} chunks...")
            
            # Create chunks
            for i in range(num_chunks):
                start_time = i * self.chunk_duration
                end_time = min((i + 1) * self.chunk_duration, duration)
                
                chunk_file = self.temp_dir / f"chunk_{i:04d}.mp4"
                
                # Extract chunk using ffmpeg
                await self._extract_video_chunk(video_path, chunk_file, start_time, end_time)
                
                chunk = ProcessingChunk(
                    chunk_id=f"chunk_{i:04d}",
                    start_time=start_time,
                    end_time=end_time,
                    file_path=str(chunk_file),
                    duration=end_time - start_time,
                    size_bytes=chunk_file.stat().st_size if chunk_file.exists() else 0
                )
                chunks.append(chunk)
                
                progress = ((i + 1) / num_chunks) * 20 + 10  # 10-30%
                self._update_progress("chunking", progress, f"Created chunk {i+1}/{num_chunks}")
            
            self._update_progress("chunking", 30, f"Successfully created {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Video chunking failed: {e}")
            raise
    
    async def chunk_audio_by_time(self, audio_path: str) -> List[ProcessingChunk]:
        """Split large audio into manageable time-based chunks."""
        self.start_time = time.time()
        self._update_progress("metadata_extraction", 0, "Extracting audio metadata...")
        
        try:
            # Get audio duration using ffprobe
            duration = await self._get_audio_duration(audio_path)
            if not duration:
                raise ValueError("Could not determine audio duration")
            
            # Calculate number of chunks
            num_chunks = max(1, int(duration / self.chunk_duration))
            chunks = []
            
            self._update_progress("chunking", 10, f"Creating {num_chunks} audio chunks...")
            
            # Create chunks
            for i in range(num_chunks):
                start_time = i * self.chunk_duration
                end_time = min((i + 1) * self.chunk_duration, duration)
                
                chunk_file = self.temp_dir / f"audio_chunk_{i:04d}.mp3"
                
                # Extract chunk using ffmpeg
                await self._extract_audio_chunk(audio_path, chunk_file, start_time, end_time)
                
                chunk = ProcessingChunk(
                    chunk_id=f"audio_chunk_{i:04d}",
                    start_time=start_time,
                    end_time=end_time,
                    file_path=str(chunk_file),
                    duration=end_time - start_time,
                    size_bytes=chunk_file.stat().st_size if chunk_file.exists() else 0
                )
                chunks.append(chunk)
                
                progress = ((i + 1) / num_chunks) * 20 + 10  # 10-30%
                self._update_progress("chunking", progress, f"Created audio chunk {i+1}/{num_chunks}")
            
            self._update_progress("chunking", 30, f"Successfully created {len(chunks)} audio chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Audio chunking failed: {e}")
            raise
    
    async def progressive_video_analysis(self, video_path: str, 
                                       processor_func: Callable) -> Dict[str, Any]:
        """Process video progressively with user feedback."""
        stages = [
            ("metadata_extraction", "Extracting video metadata..."),
            ("chunking", "Splitting video into chunks..."),
            ("content_analysis", "Analyzing video content..."),
            ("summarization", "Generating summary..."),
            ("cleanup", "Cleaning up temporary files...")
        ]
        
        results = {}
        
        try:
            # Stage 1: Metadata extraction
            self._update_progress("metadata_extraction", 0, "Extracting video metadata...")
            metadata = await self._extract_video_metadata(video_path)
            results["metadata"] = metadata
            self._update_progress("metadata_extraction", 100, "Metadata extraction complete")
            
            # Stage 2: Chunking
            self._update_progress("chunking", 0, "Splitting video into chunks...")
            chunks = await self.chunk_video_by_time(video_path)
            results["chunks"] = chunks
            self._update_progress("chunking", 100, f"Created {len(chunks)} chunks")
            
            # Stage 3: Content analysis
            self._update_progress("content_analysis", 0, "Analyzing video content...")
            chunk_results = []
            
            for i, chunk in enumerate(chunks):
                self._update_progress("content_analysis", 
                                    (i / len(chunks)) * 100, 
                                    f"Processing chunk {i+1}/{len(chunks)}")
                
                # Process chunk
                chunk_result = await processor_func(chunk.file_path)
                chunk_result["chunk_id"] = chunk.chunk_id
                chunk_result["start_time"] = chunk.start_time
                chunk_result["end_time"] = chunk.end_time
                chunk_results.append(chunk_result)
            
            results["chunk_results"] = chunk_results
            self._update_progress("content_analysis", 100, "Content analysis complete")
            
            # Stage 4: Summarization
            self._update_progress("summarization", 0, "Generating summary...")
            summary = await self._combine_chunk_results(chunk_results, metadata)
            results["summary"] = summary
            self._update_progress("summarization", 100, "Summary generation complete")
            
            # Stage 5: Cleanup
            self._update_progress("cleanup", 0, "Cleaning up temporary files...")
            await self._cleanup_chunks(chunks)
            self._update_progress("cleanup", 100, "Cleanup complete")
            
            return results
            
        except Exception as e:
            logger.error(f"Progressive video analysis failed: {e}")
            # Cleanup on error
            if "chunks" in results:
                await self._cleanup_chunks(results["chunks"])
            raise
    
    async def progressive_audio_analysis(self, audio_path: str, 
                                       processor_func: Callable) -> Dict[str, Any]:
        """Process audio progressively with user feedback."""
        stages = [
            ("metadata_extraction", "Extracting audio metadata..."),
            ("chunking", "Splitting audio into chunks..."),
            ("content_analysis", "Analyzing audio content..."),
            ("summarization", "Generating summary..."),
            ("cleanup", "Cleaning up temporary files...")
        ]
        
        results = {}
        
        try:
            # Stage 1: Metadata extraction
            self._update_progress("metadata_extraction", 0, "Extracting audio metadata...")
            metadata = await self._extract_audio_metadata(audio_path)
            results["metadata"] = metadata
            self._update_progress("metadata_extraction", 100, "Metadata extraction complete")
            
            # Stage 2: Chunking
            self._update_progress("chunking", 0, "Splitting audio into chunks...")
            chunks = await self.chunk_audio_by_time(audio_path)
            results["chunks"] = chunks
            self._update_progress("chunking", 100, f"Created {len(chunks)} chunks")
            
            # Stage 3: Content analysis
            self._update_progress("content_analysis", 0, "Analyzing audio content...")
            chunk_results = []
            
            for i, chunk in enumerate(chunks):
                self._update_progress("content_analysis", 
                                    (i / len(chunks)) * 100, 
                                    f"Processing audio chunk {i+1}/{len(chunks)}")
                
                # Process chunk
                chunk_result = await processor_func(chunk.file_path)
                chunk_result["chunk_id"] = chunk.chunk_id
                chunk_result["start_time"] = chunk.start_time
                chunk_result["end_time"] = chunk.end_time
                chunk_results.append(chunk_result)
            
            results["chunk_results"] = chunk_results
            self._update_progress("content_analysis", 100, "Audio analysis complete")
            
            # Stage 4: Summarization
            self._update_progress("summarization", 0, "Generating audio summary...")
            summary = await self._combine_chunk_results(chunk_results, metadata)
            results["summary"] = summary
            self._update_progress("summarization", 100, "Audio summary complete")
            
            # Stage 5: Cleanup
            self._update_progress("cleanup", 0, "Cleaning up temporary files...")
            await self._cleanup_chunks(chunks)
            self._update_progress("cleanup", 100, "Cleanup complete")
            
            return results
            
        except Exception as e:
            logger.error(f"Progressive audio analysis failed: {e}")
            # Cleanup on error
            if "chunks" in results:
                await self._cleanup_chunks(results["chunks"])
            raise
    
    async def stream_video_analysis(self, video_path: str, 
                                  chunk_size: int = 1024*1024) -> Generator[Dict, None, None]:
        """Process video in memory-efficient chunks."""
        try:
            # Get video metadata first
            metadata = await self._extract_video_metadata(video_path)
            yield {"type": "metadata", "data": metadata}
            
            # Process in chunks
            chunks = await self.chunk_video_by_time(video_path)
            
            for i, chunk in enumerate(chunks):
                # Process each chunk
                chunk_result = await self._process_video_chunk(chunk)
                yield {
                    "type": "chunk_result",
                    "chunk_id": chunk.chunk_id,
                    "progress": (i + 1) / len(chunks),
                    "data": chunk_result
                }
            
            # Final summary
            yield {"type": "complete", "total_chunks": len(chunks)}
            
        except Exception as e:
            logger.error(f"Stream video analysis failed: {e}")
            yield {"type": "error", "error": str(e)}
    
    async def stream_audio_analysis(self, audio_path: str, 
                                  chunk_size: int = 1024*1024) -> Generator[Dict, None, None]:
        """Process audio in memory-efficient chunks."""
        try:
            # Get audio metadata first
            metadata = await self._extract_audio_metadata(audio_path)
            yield {"type": "metadata", "data": metadata}
            
            # Process in chunks
            chunks = await self.chunk_audio_by_time(audio_path)
            
            for i, chunk in enumerate(chunks):
                # Process each chunk
                chunk_result = await self._process_audio_chunk(chunk)
                yield {
                    "type": "chunk_result",
                    "chunk_id": chunk.chunk_id,
                    "progress": (i + 1) / len(chunks),
                    "data": chunk_result
                }
            
            # Final summary
            yield {"type": "complete", "total_chunks": len(chunks)}
            
        except Exception as e:
            logger.error(f"Stream audio analysis failed: {e}")
            yield {"type": "error", "error": str(e)}
    
    # Helper methods for ffmpeg operations
    async def _get_video_duration(self, video_path: str) -> Optional[float]:
        """Get video duration using ffprobe."""
        try:
            cmd = [
                "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                "-of", "csv=p=0", video_path
            ]
            result = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await result.communicate()
            
            if result.returncode == 0:
                duration = float(stdout.decode().strip())
                return duration
            else:
                logger.error(f"ffprobe failed: {stderr.decode()}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to get video duration: {e}")
            return None
    
    async def _get_audio_duration(self, audio_path: str) -> Optional[float]:
        """Get audio duration using ffprobe."""
        try:
            cmd = [
                "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                "-of", "csv=p=0", audio_path
            ]
            result = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await result.communicate()
            
            if result.returncode == 0:
                duration = float(stdout.decode().strip())
                return duration
            else:
                logger.error(f"ffprobe failed: {stderr.decode()}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to get audio duration: {e}")
            return None
    
    async def _extract_video_chunk(self, input_path: str, output_path: str, 
                                 start_time: float, end_time: float):
        """Extract video chunk using ffmpeg."""
        try:
            duration = end_time - start_time
            cmd = [
                "ffmpeg", "-i", input_path, "-ss", str(start_time), "-t", str(duration),
                "-c", "copy", "-avoid_negative_ts", "make_zero", str(output_path), "-y"
            ]
            
            result = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await result.communicate()
            
            if result.returncode != 0:
                logger.error(f"ffmpeg chunk extraction failed: {stderr.decode()}")
                raise Exception(f"Failed to extract video chunk: {stderr.decode()}")
                
        except Exception as e:
            logger.error(f"Video chunk extraction failed: {e}")
            raise
    
    async def _extract_audio_chunk(self, input_path: str, output_path: str, 
                                 start_time: float, end_time: float):
        """Extract audio chunk using ffmpeg."""
        try:
            duration = end_time - start_time
            cmd = [
                "ffmpeg", "-i", input_path, "-ss", str(start_time), "-t", str(duration),
                "-vn", "-acodec", "mp3", "-ar", "44100", "-ac", "2", "-b:a", "192k",
                str(output_path), "-y"
            ]
            
            result = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await result.communicate()
            
            if result.returncode != 0:
                logger.error(f"ffmpeg audio chunk extraction failed: {stderr.decode()}")
                raise Exception(f"Failed to extract audio chunk: {stderr.decode()}")
                
        except Exception as e:
            logger.error(f"Audio chunk extraction failed: {e}")
            raise
    
    async def _extract_video_metadata(self, video_path: str) -> Dict[str, Any]:
        """Extract video metadata using ffprobe."""
        try:
            cmd = [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_format", "-show_streams", video_path
            ]
            result = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await result.communicate()
            
            if result.returncode == 0:
                return json.loads(stdout.decode())
            else:
                logger.error(f"ffprobe metadata extraction failed: {stderr.decode()}")
                return {}
                
        except Exception as e:
            logger.error(f"Failed to extract video metadata: {e}")
            return {}
    
    async def _extract_audio_metadata(self, audio_path: str) -> Dict[str, Any]:
        """Extract audio metadata using ffprobe."""
        try:
            cmd = [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_format", "-show_streams", audio_path
            ]
            result = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await result.communicate()
            
            if result.returncode == 0:
                return json.loads(stdout.decode())
            else:
                logger.error(f"ffprobe audio metadata extraction failed: {stderr.decode()}")
                return {}
                
        except Exception as e:
            logger.error(f"Failed to extract audio metadata: {e}")
            return {}
    
    async def _process_video_chunk(self, chunk: ProcessingChunk) -> Dict[str, Any]:
        """Process a single video chunk."""
        # This would be implemented by the specific processor
        return {
            "chunk_id": chunk.chunk_id,
            "duration": chunk.duration,
            "size": chunk.size_bytes,
            "processed": True
        }
    
    async def _process_audio_chunk(self, chunk: ProcessingChunk) -> Dict[str, Any]:
        """Process a single audio chunk."""
        # This would be implemented by the specific processor
        return {
            "chunk_id": chunk.chunk_id,
            "duration": chunk.duration,
            "size": chunk.size_bytes,
            "processed": True
        }
    
    async def _combine_chunk_results(self, chunk_results: List[Dict], 
                                   metadata: Dict) -> Dict[str, Any]:
        """Combine results from all chunks into a summary."""
        return {
            "total_chunks": len(chunk_results),
            "total_duration": sum(r.get("duration", 0) for r in chunk_results),
            "chunk_results": chunk_results,
            "metadata": metadata,
            "summary": "Combined analysis of all chunks"
        }
    
    async def _cleanup_chunks(self, chunks: List[ProcessingChunk]):
        """Clean up temporary chunk files."""
        for chunk in chunks:
            try:
                if os.path.exists(chunk.file_path):
                    os.remove(chunk.file_path)
            except Exception as e:
                logger.warning(f"Failed to cleanup chunk {chunk.chunk_id}: {e}")
