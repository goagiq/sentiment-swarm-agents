"""
Configuration for the file extraction agent.
"""

from pathlib import Path
from typing import Dict, Any, List
from pydantic import Field, ConfigDict
from pydantic_settings import BaseSettings


class ExtractionConfig(BaseSettings):
    """Configuration for file extraction agent."""
    
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8"
    )
    
    # Ollama configuration
    ollama_host: str = Field(
        default="http://localhost:11434",
        description="Ollama server host URL"
    )
    ollama_timeout: int = Field(
        default=60,
        description="Ollama request timeout in seconds"
    )
    
    # Vision model configuration
    vision_model: str = Field(
        default="llava:latest",
        description="Vision model for OCR processing"
    )
    fallback_vision_model: str = Field(
        default="granite3.2-vision",
        description="Fallback vision model"
    )
    
    # Processing configuration
    max_image_size: int = Field(
        default=1024,
        description="Maximum image dimension for processing"
    )
    max_pdf_pages: int = Field(
        default=100,
        description="Maximum number of pages to process per PDF"
    )
    chunk_size: int = Field(
        default=4096,
        description="Text chunk size for vector database storage"
    )
    
    # Parallel processing
    max_workers: int = Field(
        default=4,
        description="Maximum number of parallel workers"
    )
    batch_size: int = Field(
        default=5,
        description="Batch size for parallel processing"
    )
    
    # Memory management
    memory_reserve_percent: float = Field(
        default=30.0,
        description="Percentage of RAM to reserve for system"
    )
    max_memory_usage_mb: int = Field(
        default=2048,
        description="Maximum memory usage in MB"
    )
    
    # Error handling
    max_retries: int = Field(
        default=2,
        description="Maximum retry attempts for failed extractions"
    )
    retry_delay: float = Field(
        default=1.0,
        description="Delay between retries in seconds"
    )
    
    # File type support
    supported_pdf_extensions: List[str] = Field(
        default=[".pdf"],
        description="Supported PDF file extensions"
    )
    supported_document_extensions: List[str] = Field(
        default=[".doc", ".docx", ".rtf", ".txt"],
        description="Supported document file extensions"
    )
    supported_image_extensions: List[str] = Field(
        default=[".png", ".jpg", ".jpeg", ".tiff", ".bmp"],
        description="Supported image file extensions"
    )
    
    # Output configuration
    output_format: str = Field(
        default="json",
        description="Output format for extracted data"
    )
    include_metadata: bool = Field(
        default=True,
        description="Include metadata in extraction results"
    )
    include_confidence_scores: bool = Field(
        default=True,
        description="Include confidence scores in results"
    )
    
    # Vector database configuration
    vector_db_collection_name: str = Field(
        default="extracted_documents",
        description="ChromaDB collection name for extracted documents"
    )
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Embedding model for vector database"
    )
    
    # Progress tracking
    progress_bar_enabled: bool = Field(
        default=True,
        description="Enable progress bar for processing"
    )
    detailed_logging: bool = Field(
        default=False,
        description="Enable detailed logging"
    )
    
    # Performance optimization
    use_gpu: bool = Field(
        default=False,
        description="Use GPU acceleration if available"
    )
    optimize_images: bool = Field(
        default=True,
        description="Optimize images before OCR processing"
    )
    compression_quality: int = Field(
        default=85,
        description="JPEG compression quality (1-100)"
    )
    
    # Temporary storage
    temp_dir: Path = Field(
        default_factory=lambda: Path("./temp/extraction"),
        description="Temporary directory for processing"
    )
    cleanup_temp_files: bool = Field(
        default=True,
        description="Clean up temporary files after processing"
    )
    
    def get_supported_extensions(self) -> List[str]:
        """Get all supported file extensions."""
        return (
            self.supported_pdf_extensions +
            self.supported_document_extensions +
            self.supported_image_extensions
        )
    
    def is_supported_file(self, file_path: Path) -> bool:
        """Check if file is supported based on extension."""
        return file_path.suffix.lower() in self.get_supported_extensions()
    
    def get_processing_config(self) -> Dict[str, Any]:
        """Get processing configuration as dictionary."""
        return {
            "max_image_size": self.max_image_size,
            "max_pdf_pages": self.max_pdf_pages,
            "chunk_size": self.chunk_size,
            "max_workers": self.max_workers,
            "batch_size": self.batch_size,
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
            "optimize_images": self.optimize_images,
            "compression_quality": self.compression_quality
        }


# Global configuration instance
extraction_config = ExtractionConfig()
