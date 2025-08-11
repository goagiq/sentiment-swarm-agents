"""
Shared Image Processing Service for OCR and File Extraction.

This service provides centralized image preprocessing capabilities including:
- Image enhancement and optimization
- Format validation and conversion
- Preprocessing for OCR
- Performance optimization
- Caching and metadata extraction
"""

import logging
import cv2
import numpy as np
from PIL import Image
import io
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
import hashlib
import pickle
import tempfile

from src.core.processing_service import ProcessingService
from src.core.error_handling_service import ErrorHandlingService, ErrorContext

logger = logging.getLogger(__name__)


class ImageProcessingService:
    """Centralized service for image processing operations."""
    
    def __init__(self):
        """Initialize the image processing service."""
        self.processing_service = ProcessingService()
        self.error_handler = ErrorHandlingService()
        
        # Supported image formats
        self.supported_formats = [
            "jpg", "jpeg", "png", "bmp", "tiff", "tif", "webp", "gif"
        ]
        
        # Performance settings
        self.max_image_size = (2048, 2048)  # Maximum image size for processing
        self.default_quality = 85
        self.cache_enabled = True
        self.cache_dir = Path("./cache/image_processing")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Preprocessing settings
        self.enhance_contrast = True
        self.denoise = True
        self.sharpen = False
        self.binarize = False
        
        logger.info("ImageProcessingService initialized")
    
    def is_supported_format(self, file_path: Union[str, Path]) -> bool:
        """Check if file format is supported for processing."""
        try:
            file_ext = Path(file_path).suffix.lower().lstrip(".")
            return file_ext in self.supported_formats
        except Exception as e:
            logger.warning(f"Format check failed: {e}")
            return False
    
    def get_image_metadata(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """Get comprehensive image metadata."""
        try:
            image_path = Path(image_path)
            if not image_path.exists():
                return {"error": "Image file not found"}
            
            # Read image with OpenCV
            image = cv2.imread(str(image_path))
            if image is None:
                return {"error": "Could not read image"}
            
            height, width, channels = image.shape
            file_size = image_path.stat().st_size
            
            # Get additional metadata with PIL
            pil_image = Image.open(image_path)
            
            metadata = {
                "dimensions": {"width": width, "height": height},
                "channels": channels,
                "file_size_bytes": file_size,
                "file_size_mb": file_size / (1024 * 1024),
                "aspect_ratio": width / height if height > 0 else 0,
                "format": image_path.suffix.lower(),
                "mode": pil_image.mode,
                "dpi": pil_image.info.get('dpi', None),
                "color_space": self._get_color_space(channels),
                "file_path": str(image_path),
                "filename": image_path.name
            }
            
            return metadata
            
        except Exception as e:
            error_context = ErrorContext(
                operation="get_image_metadata",
                error=str(e),
                file_path=str(image_path) if 'image_path' in locals() else "unknown"
            )
            self.error_handler.log_error(error_context)
            return {"error": str(e)}
    
    def _get_color_space(self, channels: int) -> str:
        """Determine color space from number of channels."""
        if channels == 1:
            return "grayscale"
        elif channels == 3:
            return "BGR"
        elif channels == 4:
            return "BGRA"
        else:
            return "unknown"
    
    def preprocess_image(self, image_path: Union[str, Path], 
                        enhancement_type: str = "auto") -> np.ndarray:
        """
        Preprocess image for better OCR results.
        
        Args:
            image_path: Path to the image file
            enhancement_type: Type of enhancement (auto, contrast, denoise, sharpen, binarize)
            
        Returns:
            Preprocessed image as numpy array
        """
        try:
            # Load image
            image = cv2.imread(str(image_path))
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
            error_context = ErrorContext(
                operation="preprocess_image",
                error=str(e),
                file_path=str(image_path)
            )
            self.error_handler.log_error(error_context)
            raise
    
    def enhance_image(self, image_path: Union[str, Path], 
                     enhancement_type: str = "auto") -> Dict[str, Any]:
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
            processed_image = self.preprocess_image(image_path)
            
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
            image_path = Path(image_path)
            enhanced_path = image_path.parent / f"enhanced_{image_path.name}"
            cv2.imwrite(str(enhanced_path), enhanced)
            
            return {
                "original_path": str(image_path),
                "enhanced_path": str(enhanced_path),
                "enhancement_type": enhancement_type,
                "status": "success"
            }
            
        except Exception as e:
            error_context = ErrorContext(
                operation="enhance_image",
                error=str(e),
                file_path=str(image_path)
            )
            self.error_handler.log_error(error_context)
            return {
                "status": "error",
                "error": str(e)
            }
    
    def optimize_for_ocr(self, image_path: Union[str, Path], 
                        optimization_type: str = "auto") -> Dict[str, Any]:
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
            image = cv2.imread(str(image_path))
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
                optimized = self.preprocess_image(image_path)
            
            # Save optimized image
            image_path = Path(image_path)
            optimized_path = image_path.parent / f"optimized_{image_path.name}"
            cv2.imwrite(str(optimized_path), optimized)
            
            return {
                "original_path": str(image_path),
                "optimized_path": str(optimized_path),
                "optimization_type": optimization_type,
                "status": "success"
            }
            
        except Exception as e:
            error_context = ErrorContext(
                operation="optimize_for_ocr",
                error=str(e),
                file_path=str(image_path)
            )
            self.error_handler.log_error(error_context)
            return {
                "status": "error",
                "error": str(e)
            }
    
    def convert_to_pil(self, image_data: Union[np.ndarray, bytes, str, Path]) -> Image.Image:
        """
        Convert various image formats to PIL Image.
        
        Args:
            image_data: Image data as numpy array, bytes, file path, or Path object
            
        Returns:
            PIL Image object
        """
        try:
            if isinstance(image_data, np.ndarray):
                # Convert numpy array to PIL
                if len(image_data.shape) == 3:
                    # Color image
                    if image_data.shape[2] == 3:
                        # BGR to RGB
                        rgb_image = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
                        return Image.fromarray(rgb_image)
                    else:
                        return Image.fromarray(image_data)
                else:
                    # Grayscale image
                    return Image.fromarray(image_data)
            
            elif isinstance(image_data, bytes):
                # Convert bytes to PIL
                return Image.open(io.BytesIO(image_data))
            
            elif isinstance(image_data, (str, Path)):
                # Load from file path
                return Image.open(image_data)
            
            else:
                raise ValueError(f"Unsupported image data type: {type(image_data)}")
                
        except Exception as e:
            error_context = ErrorContext(
                operation="convert_to_pil",
                error=str(e),
                data_type=str(type(image_data))
            )
            self.error_handler.log_error(error_context)
            raise
    
    def resize_image(self, image: Union[np.ndarray, Image.Image], 
                    max_size: Tuple[int, int]) -> Union[np.ndarray, Image.Image]:
        """
        Resize image while maintaining aspect ratio.
        
        Args:
            image: Input image (numpy array or PIL Image)
            max_size: Maximum dimensions (width, height)
            
        Returns:
            Resized image
        """
        try:
            if isinstance(image, np.ndarray):
                # Resize numpy array
                height, width = image.shape[:2]
                scale = min(max_size[0]/width, max_size[1]/height)
                new_size = (int(width*scale), int(height*scale))
                return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
            
            elif isinstance(image, Image.Image):
                # Resize PIL Image
                image.thumbnail(max_size, Image.Resampling.LANCZOS)
                return image
            
            else:
                raise ValueError(f"Unsupported image type: {type(image)}")
                
        except Exception as e:
            error_context = ErrorContext(
                operation="resize_image",
                error=str(e),
                data_type=str(type(image))
            )
            self.error_handler.log_error(error_context)
            raise
    
    def _get_cache_key(self, image_path: Union[str, Path]) -> str:
        """Generate cache key from image content."""
        try:
            with open(image_path, 'rb') as f:
                content = f.read()
            return hashlib.md5(content).hexdigest()
        except Exception:
            return hashlib.md5(str(image_path).encode()).hexdigest()
    
    def get_cached_result(self, image_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """Get cached processing result if available."""
        try:
            if not self.cache_enabled:
                return None
                
            cache_key = self._get_cache_key(image_path)
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")
        
        return None
    
    def cache_result(self, image_path: Union[str, Path], result: Dict[str, Any]):
        """Cache processing result."""
        try:
            if not self.cache_enabled:
                return
                
            cache_key = self._get_cache_key(image_path)
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
        except Exception as e:
            logger.warning(f"Cache storage failed: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get image processing statistics."""
        try:
            # Count cached results
            cache_count = len(list(self.cache_dir.glob("*.pkl")))
            
            # Calculate cache size
            cache_size = sum(f.stat().st_size for f in self.cache_dir.glob("*.pkl"))
            
            return {
                "cached_results": cache_count,
                "cache_size_bytes": cache_size,
                "cache_size_mb": cache_size / (1024 * 1024),
                "supported_formats": self.supported_formats,
                "max_image_size": self.max_image_size,
                "cache_enabled": self.cache_enabled,
                "status": "success"
            }
            
        except Exception as e:
            error_context = ErrorContext(
                operation="get_statistics",
                error=str(e)
            )
            self.error_handler.log_error(error_context)
            return {
                "status": "error",
                "error": str(e)
            }
    
    def cleanup_cache(self):
        """Clean up cached results."""
        try:
            if self.cache_dir.exists():
                for cache_file in self.cache_dir.glob("*.pkl"):
                    cache_file.unlink()
                logger.info("Image processing cache cleaned up")
        except Exception as e:
            logger.error(f"Cache cleanup failed: {e}")


# Global instance
image_processing_service = ImageProcessingService()
