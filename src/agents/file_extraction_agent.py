"""
Optimized File Extraction Agent for PDF processing with parallel OCR capabilities.

This agent provides efficient PDF text extraction using multiple strategies:
- PyPDF2 for text-based PDFs
- Vision OCR for image-based PDFs
- Structured page data output
- Enhanced error handling and performance monitoring
"""

import logging
import time
from pathlib import Path
from typing import Dict, Optional, Any, List, Tuple
from datetime import datetime
import io
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import asyncio
from dataclasses import dataclass

# PDF processing libraries
try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False
    logging.warning("PyPDF2 not available. Install with: pip install PyPDF2")

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    logging.warning("PyMuPDF not available. Install with: pip install PyMuPDF")

# Image processing
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logging.warning("PIL not available. Install with: pip install Pillow")

# Vision model integration
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logging.warning("Ollama not available. Install with: pip install ollama")

from src.agents.base_agent import StrandsBaseAgent
from src.core.models import (
    AnalysisRequest, AnalysisResult, ProcessingStatus, 
    SentimentResult, DataType, PageData
)
from src.core.vector_db import vector_db
from src.config.config import config
from src.core.ollama_integration import OllamaIntegration

logger = logging.getLogger(__name__)


@dataclass
class ExtractionConfig:
    """Configuration for PDF extraction."""
    max_workers: int = 4
    chunk_size: int = 1
    retry_attempts: int = 1
    timeout_per_page: int = 60
    max_image_size: int = 1024
    image_quality: int = 85
    memory_cleanup_threshold: int = 10


@dataclass
class PageResult:
    """Result of processing a single page."""
    page_number: int
    content: str
    content_length: int
    extraction_method: str
    confidence: float
    processing_time: float
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class FileExtractionAgent(StrandsBaseAgent):
    """
    Optimized agent for extracting text from PDF files using multiple strategies.
    
    Features:
    - Intelligent PDF type detection
    - Parallel processing for image-based PDFs
    - Structured page data output
    - Memory-efficient processing
    - Comprehensive error handling
    - Performance monitoring
    """
    
    def __init__(
        self,
        agent_id: Optional[str] = None,
        max_capacity: int = 5,
        model_name: str = "llava:latest",
        max_workers: int = 4,
        chunk_size: int = 1,
        retry_attempts: int = 1,
        enable_chroma_storage: bool = True
    ):
        super().__init__(agent_id, max_capacity, model_name)
        
        self.model_name = model_name
        self.config = ExtractionConfig(
            max_workers=max_workers,
            chunk_size=chunk_size,
            retry_attempts=retry_attempts
        )
        self.enable_chroma_storage = enable_chroma_storage
        
        # Initialize Ollama integration for vision processing
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
            "memory_cleanups": 0
        }
        
        logger.info(f"Initialized optimized FileExtractionAgent {self.agent_id}")
    
    def _get_tools(self) -> list:
        """Get list of tools for this agent."""
        return [
            {
                "name": "extract_pdf_text",
                "description": "Extract text from PDF files using multiple strategies",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the PDF file"
                        },
                        "output_to_chroma": {
                            "type": "boolean",
                            "description": "Whether to store results in ChromaDB",
                            "default": True
                        }
                    },
                    "required": ["file_path"]
                }
            }
        ]
    
    async def can_process(self, request: AnalysisRequest) -> bool:
        """Check if this agent can process the given request."""
        if request.data_type != DataType.PDF:
            return False
        
        # Check if content is a file path or bytes
        if isinstance(request.content, str):
            file_path = Path(request.content)
            return file_path.exists() and file_path.suffix.lower() == '.pdf'
        elif isinstance(request.content, bytes):
            return True
        
        return False
    
    async def process(self, request: AnalysisRequest) -> AnalysisResult:
        """Process PDF extraction request with optimized performance."""
        start_time = time.time()
        
        try:
            # Determine file path
            if isinstance(request.content, str):
                file_path = Path(request.content)
            else:
                # For bytes content, save to temp file
                temp_file = Path(config.test_dir) / f"temp_pdf_{request.id}.pdf"
                with open(temp_file, 'wb') as f:
                    f.write(request.content)
                file_path = temp_file
            
            # Extract text from PDF
            extraction_result = await self._extract_pdf_text(file_path, request.id)
            
            # Store in ChromaDB if requested and enabled
            if (self.enable_chroma_storage and 
                extraction_result.get("success", False)):
                await self._store_in_chroma(extraction_result, request)
            
            # Create structured page data
            pages = self._create_structured_pages(extraction_result)
            
            # Create analysis result
            result = AnalysisResult(
                request_id=request.id,
                data_type=DataType.PDF,
                sentiment=SentimentResult(
                    label="neutral",
                    confidence=1.0,
                    reasoning="PDF text extraction completed"
                ),
                processing_time=time.time() - start_time,
                status=ProcessingStatus.COMPLETED,
                extracted_text=extraction_result.get("extracted_text", ""),
                pages=pages,
                metadata=self._create_enhanced_metadata(
                    extraction_result, pages, file_path
                ),
                model_used=self.model_name,
                quality_score=extraction_result.get("confidence", 0.0)
            )
            
            # Update statistics
            self._update_stats(extraction_result)
            
            return result
            
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            return AnalysisResult(
                request_id=request.id,
                data_type=DataType.PDF,
                sentiment=SentimentResult(
                    label="neutral",
                    confidence=0.0,
                    reasoning=f"PDF extraction failed: {str(e)}"
                ),
                processing_time=time.time() - start_time,
                status=ProcessingStatus.FAILED,
                metadata={
                    "agent_id": self.agent_id,
                    "error": str(e),
                    "file_path": str(file_path) if 'file_path' in locals() else "unknown"
                }
            )
    
    async def _extract_pdf_text(self, file_path: Path, request_id: str) -> Dict[str, Any]:
        """Extract text from PDF using optimized strategies."""
        logger.info(f"Starting optimized PDF extraction for {file_path}")
        
        # Analyze PDF content first
        pdf_analysis = await self._analyze_pdf_content(file_path)
        
        # If PDF is text-based, use PyPDF2
        if pdf_analysis.get("type") == "text" and PYPDF2_AVAILABLE:
            logger.info("PDF is text-based, using PyPDF2 extraction")
            pypdf2_result = await self._extract_with_pypdf2(file_path)
            if pypdf2_result.get("success", False):
                self.stats["pypdf2_success"] += 1
                return pypdf2_result
            else:
                logger.warning("PyPDF2 extraction failed for text-based PDF")
                return pypdf2_result
        
        # If PDF is NOT text-based (image-based), use PyMuPDF + vision OCR
        elif pdf_analysis.get("type") == "image":
            logger.info("PDF is image-based, using PyMuPDF + vision OCR extraction")
            vision_result = await self._extract_with_vision_ocr(file_path, request_id)
            if vision_result.get("success", False):
                self.stats["vision_ocr_success"] += 1
                return vision_result
            else:
                logger.warning("Vision OCR extraction failed for image-based PDF")
                return vision_result
        
        # If analysis failed or type is unknown, try PyPDF2 first, then vision OCR
        else:
            logger.warning("PDF analysis failed, trying PyPDF2 first, then vision OCR")
            if PYPDF2_AVAILABLE:
                pypdf2_result = await self._extract_with_pypdf2(file_path)
                if pypdf2_result.get("success", False):
                    self.stats["pypdf2_success"] += 1
                    return pypdf2_result
            
            # Fallback to vision OCR
            vision_result = await self._extract_with_vision_ocr(file_path, request_id)
            if vision_result.get("success", False):
                self.stats["vision_ocr_success"] += 1
                return vision_result
            
            # If all methods fail
            return {
                "success": False,
                "error": "All extraction methods failed",
                "method": "none",
                "extracted_text": "",
                "pages_processed": 0,
                "confidence": 0.0
            }
    
    async def _analyze_pdf_content(self, file_path: Path) -> Dict[str, Any]:
        """Analyze PDF to determine if it's text-based or image-based."""
        try:
            if not PYPDF2_AVAILABLE:
                return {"type": "image", "pages": 1, "error": "PyPDF2 not available"}
            
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                if len(pdf_reader.pages) == 0:
                    return {"type": "empty", "pages": 0}
                
                # Check first page for text content
                first_page = pdf_reader.pages[0]
                text_content = first_page.extract_text()
                
                if text_content and len(text_content.strip()) > 50:
                    return {
                        "type": "text",
                        "pages": len(pdf_reader.pages),
                        "text_length": len(text_content),
                        "sample_text": text_content[:200] + "..." if len(text_content) > 200 else text_content
                    }
                else:
                    return {
                        "type": "image",
                        "pages": len(pdf_reader.pages),
                        "text_length": len(text_content) if text_content else 0
                    }
                    
        except Exception as e:
            logger.warning(f"Could not analyze PDF {file_path.name}: {e}")
            return {"type": "image", "pages": 1, "error": str(e)}
    
    async def _extract_with_pypdf2(self, file_path: Path) -> Dict[str, Any]:
        """Extract text using PyPDF2 with optimized page structure."""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text_content = ""
                page_results = []
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    page_start_time = time.time()
                    page_text = page.extract_text()
                    page_processing_time = time.time() - page_start_time
                    
                    if page_text:
                        text_content += f"\n--- Page {page_num} ---\n{page_text}"
                        
                        # Create page result
                        page_result = PageResult(
                            page_number=page_num,
                            content=page_text,
                            content_length=len(page_text),
                            extraction_method="pypdf2",
                            confidence=0.9 if len(page_text.strip()) > 50 else 0.6,
                            processing_time=page_processing_time,
                            metadata={
                                "page_width": page.mediabox.width if hasattr(page, 'mediabox') else None,
                                "page_height": page.mediabox.height if hasattr(page, 'mediabox') else None,
                                "rotation": page.get('/Rotate', 0) if hasattr(page, 'get') else 0
                            }
                        )
                    else:
                        # Handle empty pages
                        page_result = PageResult(
                            page_number=page_num,
                            content="",
                            content_length=0,
                            extraction_method="pypdf2",
                            confidence=0.0,
                            processing_time=page_processing_time,
                            error_message="No text content found"
                        )
                    
                    page_results.append(page_result)
                
                if not text_content.strip():
                    return {"success": False, "error": "No text content found"}
                
                return {
                    "success": True,
                    "method": "pypdf2",
                    "extracted_text": text_content,
                    "pages_processed": len(pdf_reader.pages),
                    "confidence": 0.9 if len(text_content.strip()) > 100 else 0.5,
                    "page_results": page_results,
                    "stats": {
                        "total_pages": len(pdf_reader.pages),
                        "text_length": len(text_content),
                        "extraction_method": "direct_text",
                        "successful_pages": len([p for p in page_results if not p.error_message]),
                        "failed_pages": len([p for p in page_results if p.error_message])
                    }
                }
                
        except Exception as e:
            logger.error(f"PyPDF2 extraction failed: {e}")
            return {"success": False, "error": f"PyPDF2 extraction failed: {e}"}
    
    async def _extract_with_vision_ocr(self, file_path: Path, request_id: str) -> Dict[str, Any]:
        """Extract text using vision-based OCR with optimized parallel processing."""
        try:
            if not PYMUPDF_AVAILABLE:
                return {"success": False, "error": "PyMuPDF not available"}
            
            # Open PDF with PyMuPDF
            doc = fitz.open(file_path)
            total_pages = len(doc)
            
            if total_pages == 0:
                return {"success": False, "error": "PDF has no pages"}
            
            logger.info(f"Processing {total_pages} pages with optimized vision OCR")
            
            # Process pages in parallel with improved memory management
            page_results = await self._process_pages_parallel(doc, total_pages, request_id)
            
            # Clean up
            doc.close()
            self._cleanup_memory()
            
            if not page_results:
                return {"success": False, "error": "No text extracted from any page"}
            
            # Combine all text
            combined_text = "\n\n".join([
                f"--- Page {p.page_number} ---\n{p.content}" 
                for p in page_results if p.content
            ])
            
            successful_pages = len([p for p in page_results if not p.error_message])
            
            return {
                "success": True,
                "method": "vision_ocr",
                "extracted_text": combined_text,
                "pages_processed": successful_pages,
                "page_results": page_results,
                "confidence": min(0.8, successful_pages / total_pages),
                "stats": {
                    "total_pages": total_pages,
                    "successful_pages": successful_pages,
                    "failed_pages": total_pages - successful_pages,
                    "text_length": len(combined_text),
                    "extraction_method": "vision_ocr"
                }
            }
            
        except Exception as e:
            logger.error(f"Vision OCR extraction failed: {e}")
            return {"success": False, "error": f"Vision OCR extraction failed: {e}"}
    
    async def _process_pages_parallel(self, doc, total_pages: int, request_id: str) -> List[PageResult]:
        """Process pages in parallel with improved memory management."""
        page_results = []
        successful_pages = 0
        
        # Create progress tracking
        progress_callback = self._create_progress_callback(total_pages, request_id)
        
        # Process pages in chunks to manage memory
        chunk_size = min(self.config.chunk_size, total_pages)
        
        for chunk_start in range(0, total_pages, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_pages)
            chunk_pages = list(range(chunk_start, chunk_end))
            
            # Process chunk in parallel
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = {
                    executor.submit(
                        self._process_single_page_optimized,
                        doc, page_num, self.config.retry_attempts
                    ): page_num for page_num in chunk_pages
                }
                
                # Collect results with progress updates
                for i, future in enumerate(as_completed(futures), chunk_start + 1):
                    page_num = futures[future]
                    try:
                        result = future.result(timeout=self.config.timeout_per_page)
                        page_results.append(result)
                        
                        if not result.error_message:
                            successful_pages += 1
                        
                        # Update progress
                        progress_callback(i, total_pages)
                        
                    except Exception as e:
                        logger.error(f"Page {page_num + 1} processing failed: {e}")
                        progress_callback(i, total_pages, error=True)
                        
                        # Add failed page result
                        page_results.append(PageResult(
                            page_number=page_num + 1,
                            content="",
                            content_length=0,
                            extraction_method="vision_ocr",
                            confidence=0.0,
                            processing_time=0.0,
                            error_message=str(e)
                        ))
            
            # Memory cleanup after each chunk
            if len(page_results) % self.config.memory_cleanup_threshold == 0:
                self._cleanup_memory()
        
        return page_results
    
    def _process_single_page_optimized(self, doc, page_num: int, retry_attempts: int) -> PageResult:
        """Process a single PDF page with optimized retry logic."""
        start_time = time.time()
        
        for attempt in range(retry_attempts + 1):
            try:
                # Get page
                page = doc[page_num]
                
                # Convert page to optimized image
                img_data = self._convert_page_to_image(page)
                
                # Use vision model for OCR
                if OLLAMA_AVAILABLE:
                    ocr_text = self._ocr_with_vision_model(img_data, page_num + 1)
                    processing_time = time.time() - start_time
                    
                    return PageResult(
                        page_number=page_num + 1,
                        content=ocr_text,
                        content_length=len(ocr_text),
                        extraction_method="vision_ocr",
                        confidence=0.7,  # Base confidence for OCR
                        processing_time=processing_time,
                        metadata={
                            "ocr_confidence": 0.7,
                            "image_quality": "optimized"
                        }
                    )
                else:
                    return PageResult(
                        page_number=page_num + 1,
                        content="",
                        content_length=0,
                        extraction_method="vision_ocr",
                        confidence=0.0,
                        processing_time=time.time() - start_time,
                        error_message="Ollama not available"
                    )
                
            except Exception as e:
                logger.warning(f"Page {page_num + 1} attempt {attempt + 1} failed: {e}")
                if attempt == retry_attempts:
                    return PageResult(
                        page_number=page_num + 1,
                        content="",
                        content_length=0,
                        extraction_method="vision_ocr",
                        confidence=0.0,
                        processing_time=time.time() - start_time,
                        error_message=f"All attempts failed: {e}"
                    )
                
                # Clean up and retry
                self._cleanup_memory()
                time.sleep(1)  # Brief pause before retry
        
        return PageResult(
            page_number=page_num + 1,
            content="",
            content_length=0,
            extraction_method="vision_ocr",
            confidence=0.0,
            processing_time=time.time() - start_time,
            error_message="Unexpected error in page processing"
        )
    
    def _convert_page_to_image(self, page) -> bytes:
        """Convert PDF page to optimized image for OCR."""
        # Convert page to image
        mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better quality
        pix = page.get_pixmap(matrix=mat)
        
        # Convert to PIL format
        img_data = pix.tobytes("png")
        
        # Optimize image for OCR
        if PIL_AVAILABLE:
            img = Image.open(io.BytesIO(img_data))
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize if too large
            if max(img.size) > self.config.max_image_size:
                ratio = self.config.max_image_size / max(img.size)
                new_size = tuple(int(dim * ratio) for dim in img.size)
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            # Save optimized image
            img_buffer = io.BytesIO()
            img.save(img_buffer, format="JPEG", quality=self.config.image_quality, optimize=True)
            img_data = img_buffer.getvalue()
        
        return img_data
    
    def _ocr_with_vision_model(self, image_data: bytes, page_num: int) -> str:
        """Extract text from image using vision model with optimized prompt."""
        try:
            # Create optimized prompt for OCR
            prompt = f"""Extract all text from this PDF page {page_num}. Return only the extracted text, preserving formatting."""

            # Call vision model
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [image_data]
                    }
                ],
                options={
                    "temperature": 0.1,
                    "num_predict": 2000
                }
            )
            
            extracted_text = response['message']['content'].strip()
            
            # Clean up the extracted text
            if extracted_text:
                # Remove common vision model artifacts
                extracted_text = extracted_text.replace("Extracted text:", "").strip()
                extracted_text = extracted_text.replace("Here's the extracted text:", "").strip()
                
                return extracted_text
            else:
                return f"[Page {page_num} - No text detected]"
                
        except Exception as e:
            logger.error(f"Vision OCR failed for page {page_num}: {e}")
            return f"[Page {page_num} - OCR failed: {str(e)}]"
    
    def _create_progress_callback(self, total_pages: int, request_id: str):
        """Create a progress callback function."""
        start_time = time.time()
        
        def progress_callback(current: int, total: int, error: bool = False):
            elapsed = time.time() - start_time
            progress = (current / total) * 100
            eta = (elapsed / current) * (total - current) if current > 0 else 0
            
            status = "ERROR" if error else "PROCESSING"
            logger.info(
                f"[{request_id}] {status}: {current}/{total} pages "
                f"({progress:.1f}%) - Elapsed: {elapsed:.1f}s, ETA: {eta:.1f}s"
            )
        
        return progress_callback
    
    def _cleanup_memory(self):
        """Perform memory cleanup."""
        gc.collect()
        self.stats["memory_cleanups"] += 1
    
    def _create_enhanced_metadata(self, extraction_result: Dict[str, Any], 
                                pages: List[PageData], file_path: Path) -> Dict[str, Any]:
        """Create enhanced metadata for the analysis result."""
        return {
            "agent_id": self.agent_id,
            "method": extraction_result.get("method", "unknown"),
            "pages_processed": extraction_result.get("pages_processed", 0),
            "total_pages": len(pages) if pages else 0,
            "extraction_stats": extraction_result.get("stats", {}),
            "file_path": str(file_path),
            "file_size": file_path.stat().st_size if file_path.exists() else 0,
            "page_extraction_details": {
                "successful_pages": len([p for p in pages if not p.error_message]) if pages else 0,
                "failed_pages": len([p for p in pages if p.error_message]) if pages else 0,
                "average_confidence": sum(p.confidence for p in pages) / len(pages) if pages else 0.0,
                "total_processing_time": sum(p.processing_time or 0 for p in pages) if pages else 0.0
            },
            "performance_metrics": {
                "memory_cleanups": self.stats["memory_cleanups"],
                "average_page_time": (
                    sum(p.processing_time or 0 for p in pages) / len(pages) 
                    if pages else 0.0
                )
            }
        }
    
    def _create_structured_pages(self, extraction_result: Dict[str, Any]) -> List[PageData]:
        """Create structured page data from extraction result."""
        pages = []
        
        # If we have structured page results from the extraction method
        if "page_results" in extraction_result:
            for page_result in extraction_result["page_results"]:
                pages.append(PageData(
                    page_number=page_result.page_number,
                    content=page_result.content,
                    content_length=page_result.content_length,
                    extraction_method=page_result.extraction_method,
                    confidence=page_result.confidence,
                    processing_time=page_result.processing_time,
                    error_message=page_result.error_message,
                    metadata=page_result.metadata or {}
                ))
            return pages
        
        # Fallback: Parse the extracted text to create structured pages
        extracted_text = extraction_result.get("extracted_text", "")
        if not extracted_text:
            return pages
        
        # Use the successful page splitting logic
        page_splits = extracted_text.split("--- Page")
        method = extraction_result.get("method", "unknown")
        
        for i, split in enumerate(page_splits):
            if split.strip():
                # Remove the page number and clean up
                if "---" in split:
                    content = split.split("---", 1)[1].strip()
                else:
                    content = split.strip()
                
                if content:
                    pages.append(PageData(
                        page_number=i + 1,
                        content=content,
                        content_length=len(content),
                        extraction_method=method,
                        confidence=extraction_result.get("confidence", 0.5),
                        processing_time=None,
                        error_message=None,
                        metadata={
                            "parsed_from_text": True,
                            "original_split_index": i
                        }
                    ))
        
        logger.info(f"Created {len(pages)} structured pages from extraction result")
        return pages
    
    async def _store_in_chroma(self, extraction_result: Dict[str, Any], request: AnalysisRequest):
        """Store extraction results in ChromaDB."""
        try:
            # Create a document for ChromaDB
            document_text = extraction_result.get("extracted_text", "")
            if not document_text:
                return
            
            metadata = {
                "request_id": request.id,
                "data_type": "pdf_extraction",
                "method": extraction_result.get("method", "unknown"),
                "pages_processed": extraction_result.get("pages_processed", 0),
                "confidence": extraction_result.get("confidence", 0.0),
                "agent_id": self.agent_id,
                "timestamp": datetime.now().isoformat(),
                "file_path": str(request.content) if isinstance(request.content, str) else "bytes_content",
                "extraction_stats": json.dumps(extraction_result.get("stats", {}))
            }
            
            # Store in ChromaDB
            await vector_db.store_result(
                AnalysisResult(
                    request_id=request.id,
                    data_type=DataType.PDF,
                    sentiment=SentimentResult(label="neutral", confidence=1.0),
                    processing_time=0.0,
                    status=ProcessingStatus.COMPLETED,
                    extracted_text=document_text,
                    metadata=metadata,
                    model_used=self.model_name
                )
            )
            
            logger.info(f"Stored PDF extraction result in ChromaDB for request {request.id}")
            
        except Exception as e:
            logger.error(f"Failed to store in ChromaDB: {e}")
    
    def _update_stats(self, extraction_result: Dict[str, Any]):
        """Update processing statistics."""
        self.stats["total_files"] += 1
        
        if extraction_result.get("success", False):
            self.stats["successful_extractions"] += 1
            self.stats["pages_processed"] += extraction_result.get("pages_processed", 0)
        else:
            self.stats["failed_extractions"] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current processing statistics."""
        return {
            **self.stats,
            "agent_id": self.agent_id,
            "model_name": self.model_name,
            "config": {
                "max_workers": self.config.max_workers,
                "chunk_size": self.config.chunk_size,
                "retry_attempts": self.config.retry_attempts,
                "timeout_per_page": self.config.timeout_per_page
            }
        }
    
    async def reset_stats(self):
        """Reset processing statistics."""
        self.stats = {
            "total_files": 0,
            "successful_extractions": 0,
            "failed_extractions": 0,
            "pages_processed": 0,
            "total_processing_time": 0.0,
            "pypdf2_success": 0,
            "vision_ocr_success": 0,
            "memory_cleanups": 0
        }
        logger.info(f"Reset statistics for agent {self.agent_id}")


# Global instance
file_extraction_agent = FileExtractionAgent()
