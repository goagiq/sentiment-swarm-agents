"""
Processing Service for common processing patterns used across agents.
Provides shared utilities for text processing, content extraction, and
analysis.
"""

import logging
import time
from typing import Any, Dict, List, Tuple
from pathlib import Path
import re

from src.core.models import (
    AnalysisRequest, AnalysisResult, SentimentResult, ProcessingStatus
)
from src.config.config import config

# Configure logger
logger = logging.getLogger(__name__)


class ProcessingService:
    """Service for common processing patterns used across agents."""

    def __init__(self):
        self.logger = logger
        self.config = config

    async def process_with_timing(
        self,
        request: AnalysisRequest,
        processing_func,
        *args,
        **kwargs
    ) -> AnalysisResult:
        """Process request with timing and error handling."""
        start_time = time.time()

        try:
            result = await processing_func(*args, **kwargs)

            processing_time = time.time() - start_time

            if isinstance(result, AnalysisResult):
                result.processing_time = processing_time
                result.status = ProcessingStatus.COMPLETED
            else:
                # Convert dict result to AnalysisResult
                sentiment = result.get('sentiment', SentimentResult(
                    label="neutral", confidence=0.0
                ))
                result = AnalysisResult(
                    request_id=request.id,
                    data_type=request.data_type,
                    sentiment=sentiment,
                    processing_time=processing_time,
                    status=ProcessingStatus.COMPLETED,
                    metadata=result.get('metadata', {}),
                    raw_content=result.get('raw_content', str(request.content)),
                    extracted_text=result.get('extracted_text', '')
                )

            self.logger.info(
                f"Processed request {request.id} in "
                f"{processing_time:.2f}s"
            )

            return result

        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(
                f"Failed to process request {request.id}: {str(e)}"
            )

            error_result = AnalysisResult(
                request_id=request.id,
                data_type=request.data_type,
                sentiment=SentimentResult(
                    label="neutral",
                    confidence=0.0,
                    metadata={"error": str(e)}
                ),
                processing_time=processing_time,
                status=ProcessingStatus.FAILED,
                metadata={"error": str(e)}
            )

            return error_result

    def extract_text_content(self, content: Any) -> str:
        """Extract text content from various input types."""
        if isinstance(content, str):
            return content
        elif isinstance(content, dict):
            # Try common text fields
            for field in ['text', 'content', 'body', 'description', 'summary']:
                if field in content and content[field]:
                    return str(content[field])
            # Fallback to string representation
            return str(content)
        elif isinstance(content, (list, tuple)):
            # Join list items
            return ' '.join(str(item) for item in content)
        else:
            return str(content)

    def clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        if not text:
            return ""

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())

        # Remove control characters
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)

        return text

    def split_text_into_chunks(
        self,
        text: str,
        chunk_size: int = 1000,
        overlap: int = 100
    ) -> List[str]:
        """Split text into overlapping chunks."""
        if not text or len(text) <= chunk_size:
            return [text] if text else []

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size

            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings
                for i in range(end, max(start, end - 200), -1):
                    if text[i - 1] in '.!?':
                        end = i
                        break

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end - overlap
            if start >= len(text):
                break

        return chunks

    def extract_metadata(self, content: Any, request: AnalysisRequest) -> Dict[str, Any]:
        """Extract common metadata from content and request."""
        metadata = {
            "request_id": request.id,
            "data_type": request.data_type.value,
            "language": request.language,
            "timestamp": time.time()
        }

        if isinstance(content, dict):
            # Extract common metadata fields
            for field in ['title', 'author', 'date', 'url', 'source']:
                if field in content:
                    metadata[field] = content[field]

        return metadata

    def validate_content(self, content: Any, data_type: str) -> Tuple[bool, str]:
        """Validate content based on data type."""
        if content is None:
            return False, "Content is None"

        if data_type == "text":
            if not isinstance(content, str) or not content.strip():
                return False, "Text content is empty or invalid"

        elif data_type == "image":
            if isinstance(content, str):
                # Check if it's a file path
                path = Path(content)
                if not path.exists():
                    return False, f"Image file not found: {content}"
            elif not hasattr(content, 'read'):  # Not a file-like object
                return False, "Invalid image content"

        elif data_type == "audio":
            if isinstance(content, str):
                path = Path(content)
                if not path.exists():
                    return False, f"Audio file not found: {content}"
            elif not hasattr(content, 'read'):
                return False, "Invalid audio content"

        elif data_type == "video":
            if isinstance(content, str):
                path = Path(content)
                if not path.exists() and not content.startswith(('http://', 'https://')):
                    return False, f"Video file not found: {content}"
            elif not hasattr(content, 'read') and not content.startswith(('http://', 'https://')):
                return False, "Invalid video content"

        return True, "Content is valid"

    def create_progress_callback(self, total_items: int, request_id: str):
        """Create a progress callback function."""
        def progress_callback(current: int, total: int = None, error: bool = False):
            if total is None:
                total = total_items

            if error:
                self.logger.error(f"Error processing item {current}/{total} for request {request_id}")
            else:
                progress = (current / total) * 100
                self.logger.info(f"Progress for request {request_id}: {progress:.1f}% ({current}/{total})")

        return progress_callback

    def merge_results(self, results: List[AnalysisResult]) -> AnalysisResult:
        """Merge multiple analysis results into a single result."""
        if not results:
            return None

        if len(results) == 1:
            return results[0]

        # Merge sentiments (average confidence, most common label)
        sentiment_labels = [r.sentiment.label for r in results if r.sentiment]
        sentiment_confidences = [r.sentiment.confidence for r in results if r.sentiment]

        if sentiment_labels and sentiment_confidences:
            # Find most common label
            from collections import Counter
            label_counts = Counter(sentiment_labels)
            most_common_label = label_counts.most_common(1)[0][0]

            # Calculate average confidence
            avg_confidence = sum(sentiment_confidences) / len(sentiment_confidences)

            merged_sentiment = SentimentResult(
                label=most_common_label,
                confidence=avg_confidence,
                metadata={
                    "merged_from": len(results),
                    "label_distribution": dict(label_counts)
                }
            )
        else:
            merged_sentiment = SentimentResult(label="neutral", confidence=0.0)

        # Merge metadata
        merged_metadata = {}
        for result in results:
            if result.metadata:
                merged_metadata.update(result.metadata)

        # Merge raw content and extracted text
        merged_raw_content = "\n".join(
            r.raw_content for r in results if r.raw_content
        )
        merged_extracted_text = "\n".join(
            r.extracted_text for r in results if r.extracted_text
        )

        # Calculate total processing time
        total_processing_time = sum(
            r.processing_time for r in results if r.processing_time
        )

        return AnalysisResult(
            request_id=results[0].request_id,
            data_type=results[0].data_type,
            sentiment=merged_sentiment,
            processing_time=total_processing_time,
            status=ProcessingStatus.COMPLETED,
            metadata=merged_metadata,
            raw_content=merged_raw_content,
            extracted_text=merged_extracted_text
        )


# Global instance
processing_service = ProcessingService()
