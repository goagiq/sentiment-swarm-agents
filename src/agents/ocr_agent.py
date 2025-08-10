#!/usr/bin/env python3
"""
OCR Agent for optical character recognition using Ollama and Llama Vision.
This agent provides comprehensive OCR capabilities including text extraction,
document analysis, and image preprocessing.
"""

import asyncio
import os
import cv2
import numpy as np
from PIL import Image
import requests
import json
import hashlib
import pickle
from typing import Any, Optional, List, Dict, Tuple
from pathlib import Path
import tempfile

from loguru import logger
from src.core.strands_mock import tool

from src.agents.base_agent import BaseAgent
from src.config.config import config
from src.core.models import (
    AnalysisRequest, AnalysisResult, DataType, SentimentResult
)
from src.core.ollama_integration import get_ollama_model


class OCRAgent(BaseAgent):
    """Agent for optical character recognition using Ollama and Llama Vision."""
    
    def __init__(
        self, 
        model_name: Optional[str] = None, 
        **kwargs
    ):
        # Use config system instead of hardcoded values
        default_model = config.model.default_vision_model
        super().__init__(model_name=model_name or default_model, **kwargs)
        self.model_name = model_name or default_model
        
        # Initialize Ollama client
        self.ollama_client = None
        
        # OCR processing settings
        self.supported_formats = [
            "jpg", "jpeg", "png", "bmp", "tiff", "tif", "webp", "gif"
        ]
        
        # Performance settings
        self.max_image_size = (2048, 2048)  # Maximum image size for processing
        self.batch_size = 4
        self.use_gpu = True
        self.cache_enabled = True
        self.cache_dir = Path("./cache/ocr")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Preprocessing settings
        self.enhance_contrast = True
        self.denoise = True
        self.sharpen = False
        self.binarize = False
        
        self.metadata["model"] = model_name or default_model
        self.metadata["supported_formats"] = self.supported_formats
        self.metadata["max_image_size"] = self.max_image_size
        self.metadata["model_type"] = "ollama_llava"
        self.metadata["capabilities"] = [
            "text_extraction", "document_analysis", "image_preprocessing",
            "batch_processing", "caching", "performance_optimization",
            "multi_language_support", "document_type_detection"
        ]
    
    def _get_tools(self) -> list:
        """Get list of tools for this agent."""
        return [
            self.extract_text,
            self.analyze_document,
            self.batch_extract_text,
            self.enhance_image,
            self.detect_document_type,
            self.extract_structured_data,
            self.generate_ocr_report,
            self.preprocess_image,
            self.optimize_for_ocr,
            self.cache_ocr_result,
            self.get_ocr_statistics
        ]
    
    async def can_process(self, request: AnalysisRequest) -> bool:
        """Check if this agent can process the request."""
        return request.data_type == DataType.IMAGE
    
    async def process(self, request: AnalysisRequest) -> AnalysisResult:
        """Process OCR request with enhanced features."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            logger.info(f"📄 Starting OCR analysis for: {request.content}")
            
            # Initialize Ollama client if not already done
            if self.ollama_client is None:
                await self._initialize_ollama()
            
            # Validate file format
            if not self._is_supported_format(request.content):
                raise ValueError(f"Unsupported file format. Supported formats: {self.supported_formats}")
            
            # Extract text from image
            extracted_text = await self.extract_text(request.content)
            
            # Analyze document structure
            document_analysis = await self.analyze_document(request.content)
            
            # Calculate processing time
            processing_time = asyncio.get_event_loop().time() - start_time
            
            # Create comprehensive result
            result = AnalysisResult(
                request_id=request.request_id,
                data_type=DataType.IMAGE,
                content=request.content,
                sentiment=SentimentResult(
                    label="neutral",
                    confidence=1.0,
                    method="ocr_analysis"
                ),
                metadata={
                    "extracted_text": extracted_text.get("text", ""),
                    "document_analysis": document_analysis,
                    "processing_time": processing_time,
                    "image_metadata": await self._get_image_metadata(request.content),
                    "ocr_confidence": extracted_text.get("confidence", 0.0),
                    "agent_id": self.agent_id
                },
                processing_time=processing_time
            )
            
            logger.info(f"✅ OCR analysis completed in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"❌ OCR analysis failed: {e}")
            return AnalysisResult(
                request_id=request.request_id,
                data_type=DataType.IMAGE,
                content=request.content,
                sentiment=SentimentResult(
                    label="error",
                    confidence=0.0,
                    method="ocr_analysis"
                ),
                error=str(e),
                processing_time=asyncio.get_event_loop().time() - start_time
            )
    
    async def _initialize_ollama(self):
        """Initialize Ollama client for OCR processing."""
        try:
            # Test Ollama connection
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code != 200:
                raise ConnectionError("Ollama service not available")
            
            # Import ollama client
            import ollama
            self.ollama_client = ollama.Client()
            
            # Check if llava model is available
            models = self.ollama_client.list()
            if not any("llava" in model["name"] for model in models["models"]):
                logger.warning("Llava model not found. Please install with: ollama pull llava")
            
            logger.info("✅ Ollama client initialized for OCR")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Ollama client: {e}")
            raise
    
    def _is_supported_format(self, file_path: str) -> bool:
        """Check if file format is supported for OCR."""
        file_ext = Path(file_path).suffix.lower().lstrip(".")
        return file_ext in self.supported_formats
    
    async def _get_image_metadata(self, image_path: str) -> Dict[str, Any]:
        """Get comprehensive image metadata."""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return {"error": "Could not read image"}
            
            height, width, channels = image.shape
            file_size = Path(image_path).stat().st_size
            
            return {
                "dimensions": {"width": width, "height": height},
                "channels": channels,
                "file_size_bytes": file_size,
                "file_size_mb": file_size / (1024 * 1024),
                "aspect_ratio": width / height if height > 0 else 0,
                "format": Path(image_path).suffix.lower()
            }
        except Exception as e:
            return {"error": str(e)}
    
    @tool
    async def extract_text(self, image_path: str) -> dict:
        """
        Extract text from image using Llama Vision.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing extracted text and confidence
        """
        try:
            # Check cache first
            if self.cache_enabled:
                cached_result = self._get_cached_result(image_path)
                if cached_result:
                    return cached_result
            
            # Preprocess image
            processed_image = await self.preprocess_image(image_path)
            
            # Convert to PIL Image for Ollama
            pil_image = Image.fromarray(processed_image)
            
            # Create OCR prompt
            prompt = """
            Please extract all text from this image. 
            Return only the extracted text in a clean format.
            If there are multiple text blocks, separate them with newlines.
            Maintain the original formatting and structure as much as possible.
            """
            
            # Call Ollama with image
            response = self.ollama_client.chat(
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
                "text": extracted_text,
                "confidence": confidence,
                "status": "success",
                "method": "llama_vision_ocr"
            }
            
            # Cache result
            if self.cache_enabled:
                self._cache_result(image_path, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            return {
                "text": "",
                "confidence": 0.0,
                "status": "error",
                "error": str(e)
            }
    
    @tool
    async def analyze_document(self, image_path: str) -> dict:
        """
        Analyze document structure and extract key information.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing document analysis
        """
        try:
            # First extract text
            text_result = await self.extract_text(image_path)
            if text_result.get("status") != "success":
                return {"error": "Failed to extract text for analysis"}
            
            extracted_text = text_result["text"]
            
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
            
            analysis_response = self.ollama_client.chat(
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
    async def batch_extract_text(self, image_paths: List[str]) -> dict:
        """
        Extract text from multiple images in batch.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            Dictionary containing batch processing results
        """
        try:
            results = []
            total_processed = len(image_paths)
            successful = 0
            
            for i, image_path in enumerate(image_paths):
                try:
                    result = await self.extract_text(image_path)
                    results.append({
                        "image_path": image_path,
                        "result": result
                    })
                    if result.get("status") == "success":
                        successful += 1
                    
                    # Progress logging
                    if (i + 1) % 10 == 0:
                        logger.info(f"Processed {i + 1}/{total_processed} images")
                        
                except Exception as e:
                    results.append({
                        "image_path": image_path,
                        "error": str(e)
                    })
            
            success_rate = (successful / total_processed) * 100 if total_processed > 0 else 0
            
            return {
                "total_images": total_processed,
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
    async def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Preprocess image for better OCR results.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image as numpy array
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Could not read image")
            
            # Resize if too large
            height, width = image.shape[:2]
            if height > self.max_image_size[0] or width > self.max_image_size[1]:
                scale = min(self.max_image_size[0]/height, self.max_image_size[1]/width)
                new_size = (int(width*scale), int(height*scale))
                image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply preprocessing techniques
            if self.denoise:
                gray = cv2.fastNlMeansDenoising(gray)
            
            if self.enhance_contrast:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                gray = clahe.apply(gray)
            
            if self.sharpen:
                kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                gray = cv2.filter2D(gray, -1, kernel)
            
            if self.binarize:
                _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            return gray
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            raise
    
    @tool
    async def enhance_image(self, image_path: str, enhancement_type: str = "auto") -> dict:
        """
        Apply specific image enhancements for OCR.
        
        Args:
            image_path: Path to the image file
            enhancement_type: Type of enhancement (auto, contrast, denoise, sharpen, binarize)
            
        Returns:
            Dictionary containing enhancement results
        """
        try:
            # Load and preprocess image
            processed_image = await self.preprocess_image(image_path)
            
            # Apply specific enhancements
            if enhancement_type == "contrast":
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                enhanced = clahe.apply(processed_image)
            elif enhancement_type == "denoise":
                enhanced = cv2.fastNlMeansDenoising(processed_image)
            elif enhancement_type == "sharpen":
                kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                enhanced = cv2.filter2D(processed_image, -1, kernel)
            elif enhancement_type == "binarize":
                _, enhanced = cv2.threshold(processed_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            else:  # auto
                enhanced = processed_image
            
            # Save enhanced image
            enhanced_path = str(Path(image_path).parent / f"enhanced_{Path(image_path).name}")
            cv2.imwrite(enhanced_path, enhanced)
            
            return {
                "original_path": image_path,
                "enhanced_path": enhanced_path,
                "enhancement_type": enhancement_type,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Image enhancement failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    @tool
    async def detect_document_type(self, image_path: str) -> dict:
        """
        Detect the type of document in the image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing document type detection results
        """
        try:
            # Extract text first
            text_result = await self.extract_text(image_path)
            if text_result.get("status") != "success":
                return {"error": "Failed to extract text for document type detection"}
            
            extracted_text = text_result["text"]
            
            # Analyze document type
            type_prompt = f"""
            Analyze this text and determine the document type. Choose from:
            - receipt
            - invoice
            - form
            - letter
            - contract
            - certificate
            - newspaper
            - book_page
            - handwritten_note
            - other
            
            Provide your answer in JSON format with:
            - document_type: the detected type
            - confidence: confidence level (0-100%)
            - reasoning: brief explanation
            
            Text: {extracted_text}
            """
            
            response = self.ollama_client.chat(
                model=self.model_name,
                messages=[
                    {
                        'role': 'user',
                        'content': type_prompt
                    }
                ]
            )
            
            return {
                "document_type_analysis": response['message']['content'],
                "extracted_text": extracted_text,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Document type detection failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    @tool
    async def extract_structured_data(self, image_path: str, data_type: str = "auto") -> dict:
        """
        Extract structured data from documents.
        
        Args:
            image_path: Path to the image file
            data_type: Type of data to extract (auto, receipt, invoice, form)
            
        Returns:
            Dictionary containing structured data
        """
        try:
            # Extract text first
            text_result = await self.extract_text(image_path)
            if text_result.get("status") != "success":
                return {"error": "Failed to extract text for structured data extraction"}
            
            extracted_text = text_result["text"]
            
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
            
            response = self.ollama_client.chat(
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
    async def generate_ocr_report(self, image_path: str) -> dict:
        """
        Generate a comprehensive OCR report for an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing comprehensive OCR report
        """
        try:
            # Get image metadata
            metadata = await self._get_image_metadata(image_path)
            
            # Extract text
            text_result = await self.extract_text(image_path)
            
            # Analyze document
            doc_analysis = await self.analyze_document(image_path)
            
            # Detect document type
            type_result = await self.detect_document_type(image_path)
            
            # Generate report
            report = {
                "image_metadata": metadata,
                "text_extraction": text_result,
                "document_analysis": doc_analysis,
                "document_type": type_result,
                "processing_timestamp": asyncio.get_event_loop().time(),
                "agent_id": self.agent_id,
                "model_used": self.model_name
            }
            
            return {
                "ocr_report": report,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"OCR report generation failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    @tool
    async def optimize_for_ocr(self, image_path: str, optimization_type: str = "auto") -> dict:
        """
        Optimize image specifically for OCR processing.
        
        Args:
            image_path: Path to the image file
            optimization_type: Type of optimization (auto, text, receipt, handwritten)
            
        Returns:
            Dictionary containing optimization results
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Could not read image")
            
            # Apply optimization based on type
            if optimization_type == "text":
                # Optimize for printed text
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                optimized = clahe.apply(gray)
                
            elif optimization_type == "receipt":
                # Optimize for receipts (often low contrast)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                optimized = clahe.apply(gray)
                # Apply slight sharpening
                kernel = np.array([[-0.5,-0.5,-0.5], [-0.5,5,-0.5], [-0.5,-0.5,-0.5]])
                optimized = cv2.filter2D(optimized, -1, kernel)
                
            elif optimization_type == "handwritten":
                # Optimize for handwritten text
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # Stronger contrast enhancement
                clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
                optimized = clahe.apply(gray)
                # Denoise
                optimized = cv2.fastNlMeansDenoising(optimized)
                
            else:  # auto
                # Use standard preprocessing
                optimized = await self.preprocess_image(image_path)
            
            # Save optimized image
            optimized_path = str(Path(image_path).parent / f"optimized_{Path(image_path).name}")
            cv2.imwrite(optimized_path, optimized)
            
            return {
                "original_path": image_path,
                "optimized_path": optimized_path,
                "optimization_type": optimization_type,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Image optimization failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _get_cache_key(self, image_path: str) -> str:
        """Generate cache key from image content."""
        try:
            with open(image_path, 'rb') as f:
                content = f.read()
            return hashlib.md5(content).hexdigest()
        except Exception:
            return hashlib.md5(image_path.encode()).hexdigest()
    
    def _get_cached_result(self, image_path: str) -> Optional[Dict[str, Any]]:
        """Get cached OCR result if available."""
        try:
            cache_key = self._get_cache_key(image_path)
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")
        
        return None
    
    def _cache_result(self, image_path: str, result: Dict[str, Any]):
        """Cache OCR result."""
        try:
            cache_key = self._get_cache_key(image_path)
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
        except Exception as e:
            logger.warning(f"Cache storage failed: {e}")
    
    @tool
    async def cache_ocr_result(self, image_path: str) -> dict:
        """
        Manually cache OCR result for an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing caching status
        """
        try:
            # Extract text to cache
            result = await self.extract_text(image_path)
            
            # Cache the result
            self._cache_result(image_path, result)
            
            return {
                "image_path": image_path,
                "cached": True,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Manual caching failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    @tool
    async def get_ocr_statistics(self) -> dict:
        """
        Get OCR processing statistics.
        
        Returns:
            Dictionary containing OCR statistics
        """
        try:
            # Count cached results
            cache_count = len(list(self.cache_dir.glob("*.pkl")))
            
            # Calculate cache size
            cache_size = sum(f.stat().st_size for f in self.cache_dir.glob("*.pkl"))
            
            return {
                "cached_results": cache_count,
                "cache_size_bytes": cache_size,
                "cache_size_mb": cache_size / (1024 * 1024),
                "model_name": self.model_name,
                "supported_formats": self.supported_formats,
                "max_image_size": self.max_image_size,
                "batch_size": self.batch_size,
                "use_gpu": self.use_gpu,
                "cache_enabled": self.cache_enabled,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Statistics retrieval failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def cleanup(self):
        """Clean up resources."""
        try:
            # Clear cache if needed
            if hasattr(self, 'cache_dir') and self.cache_dir.exists():
                for cache_file in self.cache_dir.glob("*.pkl"):
                    cache_file.unlink()
            
            logger.info("✅ OCR Agent cleanup completed")
            
        except Exception as e:
            logger.error(f"❌ OCR Agent cleanup failed: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status."""
        return {
            "agent_id": self.agent_id,
            "agent_type": "ocr_agent",
            "model": self.model_name,
            "status": "ready",
            "capabilities": self.metadata["capabilities"],
            "supported_formats": self.supported_formats,
            "cache_enabled": self.cache_enabled,
            "ollama_connected": self.ollama_client is not None
        }

