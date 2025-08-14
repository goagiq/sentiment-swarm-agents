"""
Enhanced File Extraction Agent with advanced parallel processing and multilingual optimizations.
"""

import logging
import time
import asyncio
import psutil
import re
from pathlib import Path
from typing import Dict, Optional, Any, List, Tuple
from datetime import datetime
import io
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
from dataclasses import dataclass

# PDF processing libraries
try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False
    logging.warning("PyPDF2 not available")

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    logging.warning("PyMuPDF not available")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logging.warning("PIL not available")

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logging.warning("Ollama not available")

from src.agents.base_agent import StrandsBaseAgent
from src.core.models import (
    AnalysisRequest, AnalysisResult, ProcessingStatus, 
    SentimentResult, DataType, PageData
)
from src.core.vector_db import vector_db
from src.config.config import config
from src.core.ollama_integration import OllamaIntegration
from src.config.file_extraction_config import (
    get_extraction_config, get_optimal_chunk_size, 
    get_optimal_workers, get_performance_config
)
from src.config.language_specific_config import detect_primary_language

logger = logging.getLogger(__name__)


@dataclass
class EnhancedPageResult:
    """Enhanced result of processing a single page."""
    page_number: int
    content: str
    content_length: int
    extraction_method: str
    confidence: float
    processing_time: float
    language_detected: str = "unknown"
    text_quality_score: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class EnhancedFileExtractionAgent(StrandsBaseAgent):
    """
    Enhanced agent for extracting text from PDF files with advanced optimizations.
    
    Features:
    - Dynamic parallel processing based on language and system resources
    - Adaptive chunking strategies
    - Language-specific text validation and filtering
    - Memory monitoring and optimization
    - Real-time progress tracking
    - Multilingual OCR optimization
    """
    
    def __init__(
        self,
        agent_id: Optional[str] = None,
        max_capacity: int = 5,
        model_name: str = "llava:latest",
        enable_chroma_storage: bool = True
    ):
        super().__init__(agent_id, max_capacity, model_name)
        
        self.model_name = model_name
        self.enable_chroma_storage = enable_chroma_storage
        self.performance_config = get_performance_config()
        
        # Initialize Ollama integration
        self.ollama_integration = OllamaIntegration()
        
        # Processing statistics
        self.stats = {
            "total_files": 0,
            "successful_extractions": 0,
            "failed_extractions": 0,
            "pages_processed": 0,
            "total_processing_time": 0.0,
            "pypdf2_success": 0,
            "vision_ocr_success": 0,
            "memory_cleanups": 0,
            "languages_processed": set()
        }
        
        # Memory monitoring
        self.memory_monitor = MemoryMonitor()
        
        logger.info(f"Initialized EnhancedFileExtractionAgent {self.agent_id}")
    
    async def can_process(self, request: AnalysisRequest) -> bool:
        """Check if this agent can process the given request."""
        if request.data_type != DataType.PDF:
            return False
        
        if isinstance(request.content, str):
            file_path = Path(request.content)
            return file_path.exists() and file_path.suffix.lower() == '.pdf'
        elif isinstance(request.content, bytes):
            return True
        
        return False
    
    async def process(self, request: AnalysisRequest) -> AnalysisResult:
        """Process PDF extraction request with enhanced optimizations."""
        start_time = time.time()
        
        try:
            # Determine file path
            if isinstance(request.content, str):
                file_path = Path(request.content)
            else:
                temp_file = Path(config.test_dir) / f"temp_pdf_{request.id}.pdf"
                with open(temp_file, 'wb') as f:
                    f.write(request.content)
                file_path = temp_file
            
            # Detect language if not specified
            language = request.language
            if language == "auto":
                # Quick language detection from first few pages
                language = await self._detect_language_quick(file_path)
            
            # Get language-specific configuration
            extraction_config = get_extraction_config(language)
            
            # Extract text with enhanced processing
            extraction_result = await self._extract_pdf_enhanced(
                file_path, request.id, language, extraction_config
            )
            
            if not extraction_result["success"]:
                return AnalysisResult(
                    request_id=request.id,
                    data_type=request.data_type,
                    sentiment=SentimentResult(
                        label="negative",
                        confidence=0.0,
                        reasoning=f"Extraction failed: {extraction_result['error']}"
                    ),
                    processing_time=time.time() - start_time,
                    status="failed",
                    metadata={"error": extraction_result["error"]}
                )
            
            # Store in vector database if enabled
            if self.enable_chroma_storage:
                await self._store_in_vector_db(
                    extraction_result["extracted_text"], 
                    request.id, 
                    language
                )
            
            # Update statistics
            self.stats["total_files"] += 1
            self.stats["successful_extractions"] += 1
            self.stats["pages_processed"] += extraction_result["pages_processed"]
            self.stats["languages_processed"].add(language)
            
            # Convert EnhancedPageResult objects to PageData objects
            page_data_list = []
            if extraction_result.get("page_results"):
                for page_result in extraction_result["page_results"]:
                    page_data = PageData(
                        page_number=page_result.page_number,
                        content=page_result.content,
                        content_length=page_result.content_length,
                        extraction_method=page_result.extraction_method,
                        confidence=page_result.confidence,
                        processing_time=page_result.processing_time,
                        error_message=page_result.error_message,
                        metadata=page_result.metadata or {}
                    )
                    page_data_list.append(page_data)
            
            return AnalysisResult(
                request_id=request.id,
                data_type=request.data_type,
                sentiment=SentimentResult(
                    label="positive",
                    confidence=extraction_result["confidence"],
                    reasoning="Text extraction successful"
                ),
                processing_time=time.time() - start_time,
                status="completed",
                extracted_text=extraction_result["extracted_text"],
                pages=page_data_list,
                metadata={
                    "extraction_method": extraction_result["method"],
                    "pages_processed": extraction_result["pages_processed"],
                    "language": language,
                    "text_length": len(extraction_result["extracted_text"]),
                    "confidence": extraction_result["confidence"]
                }
            )
            
        except Exception as e:
            logger.error(f"Enhanced PDF extraction failed: {e}")
            self.stats["failed_extractions"] += 1
            
            return AnalysisResult(
                request_id=request.id,
                data_type=request.data_type,
                sentiment=SentimentResult(
                    label="negative",
                    confidence=0.0,
                    reasoning=f"Extraction failed: {str(e)}"
                ),
                processing_time=time.time() - start_time,
                status="failed",
                metadata={"error": str(e)}
            )
    
    async def _extract_pdf_enhanced(
        self, 
        file_path: Path, 
        request_id: str, 
        language: str,
        extraction_config
    ) -> Dict[str, Any]:
        """Enhanced PDF extraction with language-specific optimizations."""
        
        # Monitor memory before processing
        if self.performance_config["enable_memory_monitoring"]:
            self.memory_monitor.check_memory_usage()
        
        try:
            # Try PyPDF2 first for text-based PDFs
            if PYPDF2_AVAILABLE:
                pypdf2_result = await self._extract_with_pypdf2_enhanced(
                    file_path, language, extraction_config
                )
                if pypdf2_result["success"]:
                    return pypdf2_result
            
            # Fall back to vision OCR
            if PYMUPDF_AVAILABLE and OLLAMA_AVAILABLE:
                return await self._extract_with_vision_ocr_enhanced(
                    file_path, request_id, language, extraction_config
                )
            
            return {"success": False, "error": "No suitable extraction method available"}
            
        except Exception as e:
            logger.error(f"Enhanced extraction failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _extract_with_pypdf2_enhanced(
        self, 
        file_path: Path, 
        language: str,
        extraction_config
    ) -> Dict[str, Any]:
        """Enhanced PyPDF2 extraction with language-specific validation."""
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                total_pages = len(reader.pages)
                
                if total_pages == 0:
                    return {"success": False, "error": "PDF has no pages"}
                
                logger.info(f"Processing {total_pages} pages with enhanced PyPDF2")
                
                # Get optimal processing parameters
                optimal_workers = get_optimal_workers(language)
                optimal_chunk_size = get_optimal_chunk_size(language, total_pages * 1000)
                
                # Process pages with enhanced parallel processing
                page_results = await self._process_pages_enhanced_parallel(
                    reader, total_pages, language, extraction_config,
                    optimal_workers, optimal_chunk_size
                )
                
                if not page_results:
                    return {"success": False, "error": "No text extracted from any page"}
                
                # Combine and validate text
                combined_text = self._combine_and_validate_text(
                    page_results, language, extraction_config
                )
                
                successful_pages = len([p for p in page_results if not p.error_message])
                
                return {
                    "success": True,
                    "method": "pypdf2_enhanced",
                    "extracted_text": combined_text,
                    "pages_processed": successful_pages,
                    "page_results": page_results,
                    "confidence": min(0.9, successful_pages / total_pages),
                    "stats": {
                        "total_pages": total_pages,
                        "successful_pages": successful_pages,
                        "failed_pages": total_pages - successful_pages,
                        "text_length": len(combined_text),
                        "language": language,
                        "workers_used": optimal_workers,
                        "chunk_size_used": optimal_chunk_size
                    }
                }
                
        except Exception as e:
            logger.error(f"Enhanced PyPDF2 extraction failed: {e}")
            return {"success": False, "error": f"PyPDF2 extraction failed: {e}"}
    
    async def _extract_with_vision_ocr_enhanced(
        self, 
        file_path: Path, 
        request_id: str, 
        language: str,
        extraction_config
    ) -> Dict[str, Any]:
        """Enhanced vision OCR with language-specific optimizations."""
        try:
            doc = fitz.open(file_path)
            total_pages = len(doc)
            
            if total_pages == 0:
                return {"success": False, "error": "PDF has no pages"}
            
            logger.info(f"Processing {total_pages} pages with enhanced vision OCR")
            
            # Get optimal processing parameters
            optimal_workers = get_optimal_workers(language)
            optimal_chunk_size = get_optimal_chunk_size(language, total_pages * 2000)
            
            # Process pages with enhanced parallel processing
            page_results = await self._process_pages_enhanced_parallel(
                doc, total_pages, language, extraction_config,
                optimal_workers, optimal_chunk_size, use_ocr=True
            )
            
            doc.close()
            
            if not page_results:
                return {"success": False, "error": "No text extracted from any page"}
            
            # Combine and validate text
            combined_text = self._combine_and_validate_text(
                page_results, language, extraction_config
            )
            
            successful_pages = len([p for p in page_results if not p.error_message])
            
            return {
                "success": True,
                "method": "vision_ocr_enhanced",
                "extracted_text": combined_text,
                "pages_processed": successful_pages,
                "page_results": page_results,
                "confidence": min(0.8, successful_pages / total_pages),
                "stats": {
                    "total_pages": total_pages,
                    "successful_pages": successful_pages,
                    "failed_pages": total_pages - successful_pages,
                    "text_length": len(combined_text),
                    "language": language,
                    "workers_used": optimal_workers,
                    "chunk_size_used": optimal_chunk_size
                }
            }
            
        except Exception as e:
            logger.error(f"Enhanced vision OCR extraction failed: {e}")
            return {"success": False, "error": f"Vision OCR extraction failed: {e}"}
    
    async def _process_pages_enhanced_parallel(
        self, 
        doc, 
        total_pages: int, 
        language: str,
        extraction_config,
        max_workers: int,
        chunk_size: int,
        use_ocr: bool = False
    ) -> List[EnhancedPageResult]:
        """Enhanced parallel page processing with language-specific optimizations."""
        page_results = []
        successful_pages = 0
        
        # Create progress tracking
        progress_callback = self._create_enhanced_progress_callback(total_pages)
        
        # Process pages in optimized chunks
        for chunk_start in range(0, total_pages, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_pages)
            chunk_pages = list(range(chunk_start, chunk_end))
            
            # Process chunk in parallel with adaptive timeouts
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        self._process_single_page_enhanced,
                        doc, page_num, language, extraction_config, use_ocr
                    ): page_num for page_num in chunk_pages
                }
                
                # Collect results with enhanced error handling
                for i, future in enumerate(as_completed(futures), chunk_start + 1):
                    page_num = futures[future]
                    try:
                        result = future.result(timeout=extraction_config.timeout_per_page)
                        page_results.append(result)
                        
                        if not result.error_message:
                            successful_pages += 1
                        
                        # Update progress
                        progress_callback(i, total_pages, successful_pages)
                        
                    except Exception as e:
                        logger.error(f"Page {page_num + 1} processing failed: {e}")
                        progress_callback(i, total_pages, successful_pages, error=True)
                        
                        # Add failed page result
                        page_results.append(EnhancedPageResult(
                            page_number=page_num + 1,
                            content="",
                            content_length=0,
                            extraction_method="enhanced_ocr" if use_ocr else "enhanced_pypdf2",
                            confidence=0.0,
                            processing_time=0.0,
                            language_detected=language,
                            text_quality_score=0.0,
                            error_message=str(e)
                        ))
            
            # Memory cleanup and monitoring
            if self.performance_config["enable_memory_monitoring"]:
                self.memory_monitor.cleanup_if_needed()
        
        return page_results
    
    def _process_single_page_enhanced(
        self, 
        doc, 
        page_num: int, 
        language: str,
        extraction_config,
        use_ocr: bool = False
    ) -> EnhancedPageResult:
        """Process a single PDF page with enhanced language-specific processing."""
        start_time = time.time()
        
        try:
            if use_ocr:
                # Vision OCR processing
                page = doc[page_num]
                img_data = self._convert_page_to_image_enhanced(page, language)
                ocr_text = self._ocr_with_vision_model_enhanced(img_data, page_num + 1, language)
                
                # Validate and clean text
                cleaned_text = self._validate_and_clean_text(
                    ocr_text, language, extraction_config
                )
                
                # Calculate quality score
                quality_score = self._calculate_text_quality_score(
                    cleaned_text, language, extraction_config
                )
                
                processing_time = time.time() - start_time
                
                return EnhancedPageResult(
                    page_number=page_num + 1,
                    content=cleaned_text,
                    content_length=len(cleaned_text),
                    extraction_method="enhanced_ocr",
                    confidence=min(extraction_config.ocr_confidence_threshold, quality_score),
                    processing_time=processing_time,
                    language_detected=language,
                    text_quality_score=quality_score,
                    metadata={
                        "ocr_confidence": quality_score,
                        "image_quality": "enhanced",
                        "language": language
                    }
                )
            else:
                # PyPDF2 processing
                page = doc.pages[page_num]
                text = page.extract_text()
                
                # Validate and clean text
                cleaned_text = self._validate_and_clean_text(
                    text, language, extraction_config
                )
                
                # Calculate quality score
                quality_score = self._calculate_text_quality_score(
                    cleaned_text, language, extraction_config
                )
                
                processing_time = time.time() - start_time
                
                return EnhancedPageResult(
                    page_number=page_num + 1,
                    content=cleaned_text,
                    content_length=len(cleaned_text),
                    extraction_method="enhanced_pypdf2",
                    confidence=min(0.9, quality_score),
                    processing_time=processing_time,
                    language_detected=language,
                    text_quality_score=quality_score,
                    metadata={
                        "extraction_confidence": quality_score,
                        "language": language
                    }
                )
                
        except Exception as e:
            processing_time = time.time() - start_time
            return EnhancedPageResult(
                page_number=page_num + 1,
                content="",
                content_length=0,
                extraction_method="enhanced_ocr" if use_ocr else "enhanced_pypdf2",
                confidence=0.0,
                processing_time=processing_time,
                language_detected=language,
                text_quality_score=0.0,
                error_message=str(e)
            )
    
    def _validate_and_clean_text(
        self, 
        text: str, 
        language: str, 
        extraction_config
    ) -> str:
        """Validate and clean text based on language-specific patterns."""
        if not text:
            return ""
        
        # Apply language-specific validation patterns
        for pattern in extraction_config.text_validation_patterns:
            if not re.search(pattern, text, re.IGNORECASE):
                return ""  # Text doesn't match language patterns
        
        # Remove noise based on language-specific patterns
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Check if line matches noise patterns
            is_noise = False
            for noise_pattern in extraction_config.noise_filter_patterns:
                if re.match(noise_pattern, line, re.IGNORECASE):
                    is_noise = True
                    break
            
            if not is_noise and len(line.strip()) >= extraction_config.min_text_length:
                cleaned_lines.append(line.strip())
        
        return '\n'.join(cleaned_lines)
    
    def _calculate_text_quality_score(
        self, 
        text: str, 
        language: str, 
        extraction_config
    ) -> float:
        """Calculate text quality score based on language-specific criteria."""
        if not text:
            return 0.0
        
        score = 0.0
        
        # Length-based scoring
        if len(text) >= extraction_config.min_text_length:
            score += 0.3
        
        # Pattern matching scoring
        pattern_matches = 0
        for pattern in extraction_config.text_validation_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                pattern_matches += 1
        
        score += (pattern_matches / len(extraction_config.text_validation_patterns)) * 0.4
        
        # Noise reduction scoring
        noise_lines = 0
        total_lines = len(text.split('\n'))
        
        for line in text.split('\n'):
            for noise_pattern in extraction_config.noise_filter_patterns:
                if re.match(noise_pattern, line, re.IGNORECASE):
                    noise_lines += 1
                    break
        
        if total_lines > 0:
            score += (1 - (noise_lines / total_lines)) * 0.3
        
        return min(1.0, score)
    
    def _combine_and_validate_text(
        self, 
        page_results: List[EnhancedPageResult], 
        language: str,
        extraction_config
    ) -> str:
        """Combine page results and apply final validation."""
        combined_parts = []
        
        for result in sorted(page_results, key=lambda x: x.page_number):
            if result.content and not result.error_message:
                # Apply final quality threshold
                if result.text_quality_score >= extraction_config.ocr_confidence_threshold:
                    combined_parts.append(f"--- Page {result.page_number} ---\n{result.content}")
        
        return "\n\n".join(combined_parts)
    
    async def _detect_language_quick(self, file_path: Path) -> str:
        """Quick language detection from first few pages."""
        try:
            if PYPDF2_AVAILABLE:
                with open(file_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    sample_text = ""
                    
                    # Extract text from first 2 pages for language detection
                    for i in range(min(2, len(reader.pages))):
                        sample_text += reader.pages[i].extract_text() + " "
                    
                    if sample_text:
                        return detect_primary_language(sample_text)
            
            return "en"  # Default fallback
            
        except Exception as e:
            logger.warning(f"Quick language detection failed: {e}")
            return "en"
    
    def _create_enhanced_progress_callback(self, total_pages: int):
        """Create enhanced progress callback with detailed tracking."""
        start_time = time.time()
        last_update = start_time
        
        def callback(current: int, total: int, successful: int = 0, error: bool = False):
            nonlocal last_update
            current_time = time.time()
            
            # Update progress at specified intervals
            if (current_time - last_update) >= self.performance_config["progress_update_interval"]:
                progress = (current / total) * 100
                elapsed = current_time - start_time
                eta = (elapsed / current) * (total - current) if current > 0 else 0
                
                status = "✅" if not error else "❌"
                logger.info(
                    f"{status} Progress: {progress:.1f}% ({current}/{total}) "
                    f"Success: {successful} ETA: {eta:.1f}s"
                )
                last_update = current_time
        
        return callback
    
    async def _store_in_vector_db(self, text: str, request_id: str, language: str):
        """Store extracted text in vector database."""
        try:
            if self.enable_chroma_storage and text:
                # Create AnalysisResult object for storage
                from src.core.models import AnalysisResult, SentimentResult, ProcessingStatus, DataType
                
                result = AnalysisResult(
                    request_id=request_id,
                    data_type=DataType.TEXT,
                    raw_content=text,
                    extracted_text=text,
                    sentiment=SentimentResult(
                        label="neutral",
                        confidence=1.0
                    ),
                    processing_time=0.0,
                    status=ProcessingStatus.COMPLETED,
                    model_used="enhanced_file_extraction",
                    language=language,
                    metadata={"source": "enhanced_extraction"}
                )
                
                await vector_db.store_result(result)
        except Exception as e:
            logger.warning(f"Failed to store in vector DB: {e}")

    # Interface method for MCP server compatibility
    async def extract_text_from_pdf(self, content: str, options: dict = None) -> dict:
        """
        Extract text from PDF content - interface method for MCP server.
        
        Args:
            content: The PDF content to extract text from
            options: Extraction options including:
                - language: Language code (e.g., "zh", "en")
                - pdf_type: Type of Chinese PDF ("classical_chinese", "modern_chinese", "mixed_chinese")
                - sample_only: If True, only extract a sample for language detection
                - enhanced_processing: If True, use enhanced processing for Chinese PDFs
            
        Returns:
            Text extraction result
        """
        try:
            options = options or {}
            
            # Handle sample extraction for language detection
            if options.get("sample_only"):
                # Extract only first few pages for language detection
                try:
                    import fitz
                    doc = fitz.open(content)
                    sample_text = ""
                    for page_num in range(min(3, len(doc))):
                        page = doc[page_num]
                        sample_text += page.get_text()
                        if len(sample_text) > 1000:
                            break
                    doc.close()
                    
                    return {
                        "status": "success",
                        "extracted_text": sample_text,
                        "processing_time": 0.0,
                        "metadata": {"sample_only": True},
                        "language": options.get("language", "auto")
                    }
                except Exception as e:
                    logger.error(f"Sample extraction failed: {e}")
                    return {
                        "status": "error",
                        "error": str(e),
                        "extracted_text": ""
                    }
            
            # Create analysis request with enhanced options
            request = AnalysisRequest(
                data_type=DataType.PDF,
                content=content,
                language=options.get("language", "en"),
                metadata={
                    **options,
                    "pdf_type": options.get("pdf_type"),
                    "enhanced_processing": options.get("enhanced_processing", True),
                    "language_specific": True if options.get("language") in ["zh", "ru", "ar", "ja", "ko", "hi"] else False
                }
            )
            
            # Process using the main process method
            result = await self.process(request)
            
            return {
                "status": "success",
                "extracted_text": result.extracted_text,
                "processing_time": result.processing_time,
                "metadata": result.metadata,
                "language": getattr(result, 'language', options.get("language", "en")),
                "pdf_type": options.get("pdf_type")
            }
        except Exception as e:
            logger.error(f"PDF text extraction failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "extracted_text": ""
            }


class MemoryMonitor:
    """Memory monitoring and cleanup utility."""
    
    def __init__(self):
        self.last_cleanup = time.time()
        self.cleanup_interval = 30  # seconds
    
    def check_memory_usage(self):
        """Check current memory usage."""
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            logger.debug(f"Current memory usage: {memory_mb:.1f} MB")
            return memory_mb
        except Exception:
            return 0
    
    def cleanup_if_needed(self):
        """Perform memory cleanup if needed."""
        current_time = time.time()
        
        if current_time - self.last_cleanup > self.cleanup_interval:
            try:
                gc.collect()
                self.last_cleanup = current_time
                logger.debug("Memory cleanup performed")
            except Exception as e:
                logger.warning(f"Memory cleanup failed: {e}")
    
    def force_cleanup(self):
        """Force immediate memory cleanup."""
        try:
            gc.collect()
            logger.debug("Forced memory cleanup performed")
        except Exception as e:
            logger.warning(f"Forced memory cleanup failed: {e}")



