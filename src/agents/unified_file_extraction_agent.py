"""
Unified File Extraction Agent for PDF and Image processing.

This agent combines the capabilities of OCR and File Extraction agents:
- PDF text extraction (PyPDF2 + Vision OCR)
- Image OCR using Ollama/Llama Vision
- Document analysis and structured data extraction
- Batch processing and caching
- Performance optimization
"""

import logging
import time
import asyncio
from pathlib import Path
from typing import Dict, Optional, Any, List, Union
from datetime import datetime
import io
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc

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

# Vision model integration
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logging.warning("Ollama not available. Install with: pip install ollama")

from src.agents.base_agent import StrandsBaseAgent
from src.core.strands_mock import tool
from src.core.models import (
    AnalysisRequest, AnalysisResult, ProcessingStatus, 
    SentimentResult, DataType, PageData
)
from src.core.vector_db import vector_db
from src.config.config import config
from src.core.ollama_integration import OllamaIntegration
from src.core.image_processing_service import ImageProcessingService
from src.core.processing_service import ProcessingService
from src.core.error_handling_service import ErrorHandlingService, ErrorContext
from src.core.model_management_service import ModelManagementService

logger = logging.getLogger(__name__)


class UnifiedFileExtractionAgent(StrandsBaseAgent):
    """
    Unified agent for extracting text from PDF files and images.
    
    Features:
    - Intelligent PDF type detection and processing
    - Image OCR using vision models
    - Document analysis and structured data extraction
    - Parallel processing for large documents
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
        self.max_workers = max_workers
        self.chunk_size = chunk_size
        self.retry_attempts = retry_attempts
        self.enable_chroma_storage = enable_chroma_storage
        
        # Initialize services
        self.image_processing = ImageProcessingService()
        self.processing_service = ProcessingService()
        self.error_handler = ErrorHandlingService()
        self.model_management = ModelManagementService()
        
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
            "image_ocr_success": 0,
            "memory_cleanups": 0
        }
        
        logger.info(f"Initialized UnifiedFileExtractionAgent {self.agent_id}")
    
    def _get_tools(self) -> list:
        """Get list of tools for this agent."""
        return [
            self.extract_pdf_text,
            self.extract_image_text,
            self.analyze_document,
            self.batch_extract_text,
            self.extract_structured_data,
            self.generate_extraction_report,
            self.optimize_for_ocr,
            self.get_extraction_statistics
        ]
    
    async def can_process(self, request: AnalysisRequest) -> bool:
        """Check if this agent can process the given request."""
        if request.data_type not in [DataType.PDF, DataType.IMAGE]:
            return False
        
        # Check if content is a file path or bytes
        if isinstance(request.content, str):
            file_path = Path(request.content)
            if request.data_type == DataType.PDF:
                return file_path.exists() and file_path.suffix.lower() == '.pdf'
            elif request.data_type == DataType.IMAGE:
                return (file_path.exists() and 
                       self.image_processing.is_supported_format(file_path))
        elif isinstance(request.content, bytes):
            return True
        
        return False
    
    async def process(self, request: AnalysisRequest) -> AnalysisResult:
        """Process file extraction request."""
        start_time = time.time()
        
        try:
            logger.info(f"📄 Starting unified extraction for: {request.content}")
            
            # Determine file path
            if isinstance(request.content, str):
                file_path = Path(request.content)
            else:
                # For bytes content, save to temp file
                temp_file = Path(config.test_dir) / f"temp_{request.id}.{request.data_type.value}"
                with open(temp_file, 'wb') as f:
                    f.write(request.content)
                file_path = temp_file
            
            # Extract text based on data type
            if request.data_type == DataType.PDF:
                extraction_result = await self.extract_pdf_text(str(file_path))
            elif request.data_type == DataType.IMAGE:
                extraction_result = await self.extract_image_text(str(file_path))
            else:
                raise ValueError(f"Unsupported data type: {request.data_type}")
            
            # Store in ChromaDB if requested and enabled
            if (self.enable_chroma_storage and 
                extraction_result.get("success", False)):
                await self._store_in_chroma(extraction_result, request)
            
            # Create structured page data
            pages = self._create_structured_pages(extraction_result)
            
            # Create analysis result
            result = AnalysisResult(
                request_id=request.id,
                data_type=request.data_type,
                sentiment=SentimentResult(
                    label="neutral",
                    confidence=1.0,
                    reasoning="File extraction completed"
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
            logger.error(f"File extraction failed: {e}")
            return AnalysisResult(
                request_id=request.id,
                data_type=request.data_type,
                sentiment=SentimentResult(
                    label="neutral",
                    confidence=0.0,
                    reasoning=f"File extraction failed: {str(e)}"
                ),
                processing_time=time.time() - start_time,
                status=ProcessingStatus.FAILED,
                metadata={
                    "agent_id": self.agent_id,
                    "error": str(e),
                    "file_path": str(file_path) if 'file_path' in locals() else "unknown"
                }
            )
    
    @tool
    async def extract_pdf_text(self, file_path: str) -> dict:
        """
        Extract text from PDF files using multiple strategies.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Dictionary containing extraction results
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                return {"success": False, "error": "PDF file not found"}
            
            # Analyze PDF content first
            pdf_analysis = await self._analyze_pdf_content(file_path)
            
            # If PDF is text-based, use PyPDF2
            if pdf_analysis.get("type") == "text" and PYPDF2_AVAILABLE:
                logger.info("PDF is text-based, using PyPDF2 extraction")
                pypdf2_result = await self._extract_with_pypdf2(file_path)
                if pypdf2_result.get("success", False):
                    self.stats["pypdf2_success"] += 1
                    return pypdf2_result
            
            # If PDF is image-based or PyPDF2 failed, use vision OCR
            logger.info("Using vision OCR extraction")
            vision_result = await self._extract_with_vision_ocr(file_path)
            if vision_result.get("success", False):
                self.stats["vision_ocr_success"] += 1
                return vision_result
            
            return {"success": False, "error": "All extraction methods failed"}
            
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            return {"success": False, "error": str(e)}
    
    @tool
    async def extract_image_text(self, image_path: str) -> dict:
        """
        Extract text from image using vision OCR.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing extracted text and confidence
        """
        try:
            # Check cache first
            cached_result = self.image_processing.get_cached_result(image_path)
            if cached_result:
                return cached_result
            
            # Validate file format
            if not self.image_processing.is_supported_format(image_path):
                return {
                    "success": False,
                    "error": f"Unsupported file format. Supported: {self.image_processing.supported_formats}"
                }
            
            # Preprocess image
            processed_image = self.image_processing.preprocess_image(image_path)
            
            # Convert to PIL Image for Ollama
            pil_image = self.image_processing.convert_to_pil(processed_image)
            
            # Create OCR prompt
            prompt = """
            Please extract all text from this image. 
            Return only the extracted text in a clean format.
            If there are multiple text blocks, separate them with newlines.
            Maintain the original formatting and structure as much as possible.
            """
            
            # Call Ollama with image
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {
                        'role': 'user',
                        'content': prompt,
                        'images': [pil_image]
                    }
                ]
            )
            
            extracted_text = response['message']['content']
            
            # Calculate confidence based on text length and quality
            confidence = min(1.0, len(extracted_text.strip()) / 10.0)
            
            result = {
                "success": True,
                "text": extracted_text,
                "confidence": confidence,
                "status": "success",
                "method": "vision_ocr",
                "extracted_text": extracted_text
            }
            
            # Cache result
            self.image_processing.cache_result(image_path, result)
            self.stats["image_ocr_success"] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Image text extraction failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    @tool
    async def analyze_document(self, file_path: str) -> dict:
        """
        Analyze document structure and extract key information.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary containing document analysis
        """
        try:
            file_path = Path(file_path)
            
            # Extract text first
            if file_path.suffix.lower() == '.pdf':
                text_result = await self.extract_pdf_text(str(file_path))
            else:
                text_result = await self.extract_image_text(str(file_path))
            
            if not text_result.get("success", False):
                return {"error": "Failed to extract text for analysis"}
            
            extracted_text = text_result.get("extracted_text", "")
            
            # Analyze document structure
            analysis_prompt = f"""
            Analyze this extracted text and provide a comprehensive analysis:
            
            1. Document Type: What type of document is this? (receipt, invoice, form, letter, etc.)
            2. Key Information: Extract important details like dates, amounts, names, addresses, etc.
            3. Structure: Describe the layout and organization of the document
            4. Confidence: Rate your confidence in the extraction (0-100%)
            5. Missing Information: Note any unclear or missing text
            6. Recommendations: Suggest any preprocessing that might improve extraction
            
            Text: {extracted_text}
            
            Provide your analysis in JSON format.
            """
            
            analysis_response = ollama.chat(
                model=self.model_name,
                messages=[
                    {
                        'role': 'user',
                        'content': analysis_prompt
                    }
                ]
            )
            
            return {
                "document_analysis": analysis_response['message']['content'],
                "extracted_text": extracted_text,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Document analysis failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    @tool
    async def batch_extract_text(self, file_paths: List[str]) -> dict:
        """
        Extract text from multiple files in batch.
        
        Args:
            file_paths: List of file paths
            
        Returns:
            Dictionary containing batch processing results
        """
        try:
            results = []
            total_processed = len(file_paths)
            successful = 0
            
            for i, file_path in enumerate(file_paths):
                try:
                    file_path = Path(file_path)
                    
                    # Determine extraction method based on file type
                    if file_path.suffix.lower() == '.pdf':
                        result = await self.extract_pdf_text(str(file_path))
                    else:
                        result = await self.extract_image_text(str(file_path))
                    
                    results.append({
                        "file_path": str(file_path),
                        "result": result
                    })
                    
                    if result.get("success", False):
                        successful += 1
                    
                    # Progress logging
                    if (i + 1) % 10 == 0:
                        logger.info(f"Processed {i + 1}/{total_processed} files")
                        
                except Exception as e:
                    results.append({
                        "file_path": str(file_path),
                        "error": str(e)
                    })
            
            success_rate = (successful / total_processed) * 100 if total_processed > 0 else 0
            
            return {
                "total_files": total_processed,
                "successful": successful,
                "failed": total_processed - successful,
                "success_rate": success_rate,
                "results": results,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Batch text extraction failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    @tool
    async def extract_structured_data(self, file_path: str, data_type: str = "auto") -> dict:
        """
        Extract structured data from documents.
        
        Args:
            file_path: Path to the file
            data_type: Type of data to extract (auto, receipt, invoice, form)
            
        Returns:
            Dictionary containing structured data
        """
        try:
            file_path = Path(file_path)
            
            # Extract text first
            if file_path.suffix.lower() == '.pdf':
                text_result = await self.extract_pdf_text(str(file_path))
            else:
                text_result = await self.extract_image_text(str(file_path))
            
            if not text_result.get("success", False):
                return {"error": "Failed to extract text for structured data extraction"}
            
            extracted_text = text_result.get("extracted_text", "")
            
            # Create extraction prompt based on data type
            if data_type == "receipt":
                prompt = f"""
                Extract structured data from this receipt. Return JSON with:
                - merchant_name
                - date
                - total_amount
                - items (list of items with name, quantity, price)
                - tax_amount
                - payment_method
                
                Text: {extracted_text}
                """
            elif data_type == "invoice":
                prompt = f"""
                Extract structured data from this invoice. Return JSON with:
                - invoice_number
                - date
                - due_date
                - total_amount
                - items (list of items with description, quantity, unit_price, total)
                - tax_amount
                - customer_info
                - vendor_info
                
                Text: {extracted_text}
                """
            else:  # auto
                prompt = f"""
                Extract structured data from this document. Identify the document type
                and extract relevant information in JSON format.
                
                Text: {extracted_text}
                """
            
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ]
            )
            
            return {
                "structured_data": response['message']['content'],
                "data_type": data_type,
                "extracted_text": extracted_text,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Structured data extraction failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    @tool
    async def generate_extraction_report(self, file_path: str) -> dict:
        """
        Generate a comprehensive extraction report for a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary containing comprehensive extraction report
        """
        try:
            file_path = Path(file_path)
            
            # Get file metadata
            if file_path.suffix.lower() == '.pdf':
                metadata = await self._get_pdf_metadata(file_path)
            else:
                metadata = self.image_processing.get_image_metadata(file_path)
            
            # Extract text
            if file_path.suffix.lower() == '.pdf':
                text_result = await self.extract_pdf_text(str(file_path))
            else:
                text_result = await self.extract_image_text(str(file_path))
            
            # Analyze document
            doc_analysis = await self.analyze_document(str(file_path))
            
            # Generate report
            report = {
                "file_metadata": metadata,
                "text_extraction": text_result,
                "document_analysis": doc_analysis,
                "processing_timestamp": time.time(),
                "agent_id": self.agent_id,
                "model_used": self.model_name
            }
            
            return {
                "extraction_report": report,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Extraction report generation failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    @tool
    async def optimize_for_ocr(self, file_path: str, optimization_type: str = "auto") -> dict:
        """
        Optimize file specifically for OCR processing.
        
        Args:
            file_path: Path to the file
            optimization_type: Type of optimization (auto, text, receipt, handwritten)
            
        Returns:
            Dictionary containing optimization results
        """
        try:
            file_path = Path(file_path)
            
            if file_path.suffix.lower() == '.pdf':
                return {"error": "PDF optimization not yet implemented"}
            else:
                # Use image processing service for image optimization
                return self.image_processing.optimize_for_ocr(file_path, optimization_type)
            
        except Exception as e:
            logger.error(f"File optimization failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    @tool
    async def get_extraction_statistics(self) -> dict:
        """
        Get extraction processing statistics.
        
        Returns:
            Dictionary containing extraction statistics
        """
        try:
            image_stats = self.image_processing.get_statistics()
            
            return {
                **self.stats,
                "image_processing_stats": image_stats,
                "agent_id": self.agent_id,
                "model_name": self.model_name,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Statistics retrieval failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    # Helper methods for PDF processing
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
        """Extract text using PyPDF2."""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text_content = ""
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    page_text = page.extract_text()
                    if page_text:
                        text_content += f"\n--- Page {page_num} ---\n{page_text}"
                
                if not text_content.strip():
                    return {"success": False, "error": "No text content found"}
                
                return {
                    "success": True,
                    "method": "pypdf2",
                    "extracted_text": text_content,
                    "pages_processed": len(pdf_reader.pages),
                    "confidence": 0.9 if len(text_content.strip()) > 100 else 0.5
                }
                
        except Exception as e:
            logger.error(f"PyPDF2 extraction failed: {e}")
            return {"success": False, "error": f"PyPDF2 extraction failed: {e}"}
    
    async def _extract_with_vision_ocr(self, file_path: Path) -> Dict[str, Any]:
        """Extract text using vision-based OCR."""
        try:
            if not PYMUPDF_AVAILABLE:
                return {"success": False, "error": "PyMuPDF not available"}
            
            # Open PDF with PyMuPDF
            doc = fitz.open(file_path)
            total_pages = len(doc)
            
            if total_pages == 0:
                return {"success": False, "error": "PDF has no pages"}
            
            logger.info(f"Processing {total_pages} pages with vision OCR")
            
            # Process pages
            page_results = []
            combined_text = ""
            
            for page_num in range(total_pages):
                try:
                    page = doc[page_num]
                    
                    # Convert page to image
                    mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better quality
                    pix = page.get_pixmap(matrix=mat)
                    img_data = pix.tobytes("png")
                    
                    # Use vision model for OCR
                    if OLLAMA_AVAILABLE:
                        ocr_text = self._ocr_with_vision_model(img_data, page_num + 1)
                        if ocr_text:
                            combined_text += f"\n--- Page {page_num + 1} ---\n{ocr_text}"
                            page_results.append({
                                "page_number": page_num + 1,
                                "content": ocr_text,
                                "status": "success"
                            })
                        else:
                            page_results.append({
                                "page_number": page_num + 1,
                                "content": "",
                                "status": "no_text"
                            })
                    
                except Exception as e:
                    logger.error(f"Page {page_num + 1} processing failed: {e}")
                    page_results.append({
                        "page_number": page_num + 1,
                        "content": "",
                        "status": "error",
                        "error": str(e)
                    })
            
            # Clean up
            doc.close()
            
            if not combined_text.strip():
                return {"success": False, "error": "No text extracted from any page"}
            
            successful_pages = len([p for p in page_results if p["status"] == "success"])
            
            return {
                "success": True,
                "method": "vision_ocr",
                "extracted_text": combined_text,
                "pages_processed": successful_pages,
                "page_results": page_results,
                "confidence": min(0.8, successful_pages / total_pages)
            }
            
        except Exception as e:
            logger.error(f"Vision OCR extraction failed: {e}")
            return {"success": False, "error": f"Vision OCR extraction failed: {e}"}
    
    def _ocr_with_vision_model(self, image_data: bytes, page_num: int) -> str:
        """Extract text from image using vision model."""
        try:
            prompt = f"""Extract all text from this PDF page {page_num}. Return only the extracted text, preserving formatting."""

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
                extracted_text = extracted_text.replace("Extracted text:", "").strip()
                extracted_text = extracted_text.replace("Here's the extracted text:", "").strip()
                return extracted_text
            else:
                return f"[Page {page_num} - No text detected]"
                
        except Exception as e:
            logger.error(f"Vision OCR failed for page {page_num}: {e}")
            return f"[Page {page_num} - OCR failed: {str(e)}]"
    
    async def _get_pdf_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Get PDF metadata."""
        try:
            if not PYPDF2_AVAILABLE:
                return {"error": "PyPDF2 not available"}
            
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                return {
                    "pages": len(pdf_reader.pages),
                    "file_size": file_path.stat().st_size,
                    "file_size_mb": file_path.stat().st_size / (1024 * 1024),
                    "filename": file_path.name
                }
        except Exception as e:
            return {"error": str(e)}
    
    def _create_structured_pages(self, extraction_result: Dict[str, Any]) -> List[PageData]:
        """Create structured page data from extraction result."""
        pages = []
        
        # If we have structured page results from the extraction method
        if "page_results" in extraction_result:
            for page_result in extraction_result["page_results"]:
                pages.append(PageData(
                    page_number=page_result.get("page_number", 1),
                    content=page_result.get("content", ""),
                    content_length=len(page_result.get("content", "")),
                    extraction_method=extraction_result.get("method", "unknown"),
                    confidence=extraction_result.get("confidence", 0.5),
                    processing_time=None,
                    error_message=page_result.get("error"),
                    metadata=page_result.get("metadata", {})
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
                "data_type": request.data_type.value,
                "method": extraction_result.get("method", "unknown"),
                "pages_processed": extraction_result.get("pages_processed", 0),
                "confidence": extraction_result.get("confidence", 0.0),
                "agent_id": self.agent_id,
                "timestamp": datetime.now().isoformat(),
                "file_path": str(request.content) if isinstance(request.content, str) else "bytes_content"
            }
            
            # Store in ChromaDB
            await vector_db.store_result(
                AnalysisResult(
                    request_id=request.id,
                    data_type=request.data_type,
                    sentiment=SentimentResult(label="neutral", confidence=1.0),
                    processing_time=0.0,
                    status=ProcessingStatus.COMPLETED,
                    extracted_text=document_text,
                    metadata=metadata,
                    model_used=self.model_name
                )
            )
            
            logger.info(f"Stored extraction result in ChromaDB for request {request.id}")
            
        except Exception as e:
            logger.error(f"Failed to store in ChromaDB: {e}")
    
    def _create_enhanced_metadata(self, extraction_result: Dict[str, Any], 
                                pages: List[PageData], file_path: Path) -> Dict[str, Any]:
        """Create enhanced metadata for the analysis result."""
        return {
            "agent_id": self.agent_id,
            "method": extraction_result.get("method", "unknown"),
            "pages_processed": extraction_result.get("pages_processed", 0),
            "total_pages": len(pages) if pages else 0,
            "file_path": str(file_path),
            "file_size": file_path.stat().st_size if file_path.exists() else 0,
            "page_extraction_details": {
                "successful_pages": len([p for p in pages if not p.error_message]) if pages else 0,
                "failed_pages": len([p for p in pages if p.error_message]) if pages else 0,
                "average_confidence": sum(p.confidence for p in pages) / len(pages) if pages else 0.0
            }
        }
    
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
            "model_name": self.model_name
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
            "image_ocr_success": 0,
            "memory_cleanups": 0
        }
        logger.info(f"Reset statistics for agent {self.agent_id}")


# Global instance
unified_file_extraction_agent = UnifiedFileExtractionAgent()
