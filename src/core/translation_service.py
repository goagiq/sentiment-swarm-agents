"""
Translation Service for unified translation capabilities across all agents.
Provides text translation, document translation, batch translation, and language detection.
"""

import logging
import time
import re
from typing import Dict, List, Optional, Any
import requests

from src.core.models import DataType
from src.core.vector_db import VectorDBManager
from src.core.ollama_integration import OllamaIntegration

logger = logging.getLogger(__name__)


class TranslationResult:
    """Result of translation operation."""

    def __init__(
        self,
        original_text: str,
        translated_text: str,
        source_language: str,
        target_language: str = "en",
        confidence: float = 0.8,
        processing_time: float = 0.0,
        model_used: str = "",
        translation_memory_hit: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.original_text = original_text
        self.translated_text = translated_text
        self.source_language = source_language
        self.target_language = target_language
        self.confidence = confidence
        self.processing_time = processing_time
        self.model_used = model_used
        self.translation_memory_hit = translation_memory_hit
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "original_text": self.original_text,
            "translated_text": self.translated_text,
            "source_language": self.source_language,
            "target_language": self.target_language,
            "confidence": self.confidence,
            "processing_time": self.processing_time,
            "model_used": self.model_used,
            "translation_memory_hit": self.translation_memory_hit,
            "metadata": self.metadata
        }


class TranslationService:
    """Unified translation service for all agents."""

    def __init__(self):
        self.logger = logger

        # Initialize Ollama client for translation
        self.ollama_client = OllamaIntegration()

        # Initialize Chroma vector DB for translation memory
        self.vector_db = VectorDBManager()

        # Translation models configuration
        self.translation_models = {
            "primary": "mistral-small3.1:latest",
            "fallback": "llama3.2:latest",
            "vision": "llava:latest",
            "fast": "llama3.2:3b"
        }

        # Language detection patterns
        self.language_patterns = {
            "es": r"[áéíóúñü]",
            "fr": r"[àâäéèêëïîôöùûüÿç]",
            "de": r"[äöüß]",
            "it": r"[àèéìíîòóù]",
            "pt": r"[ãõâêîôûç]",
            "ru": r"[а-яё]",
            "zh": r"[\u4e00-\u9fff]",
            "ja": r"[\u3040-\u309f\u30a0-\u30ff]",
            "ko": r"[\uac00-\ud7af]",
            "ar": r"[\u0600-\u06ff]",
            "hi": r"[\u0900-\u097f]",
            "th": r"[\u0e00-\u0e7f]"
        }

        # Translation statistics
        self.stats = {
            "total_translations": 0,
            "memory_hits": 0,
            "languages_detected": {},
            "processing_times": []
        }

        logger.info("Translation Service initialized")

    async def translate_text(
        self,
        text: str,
        source_language: Optional[str] = None,
        target_language: str = "en"
    ) -> TranslationResult:
        """Translate text content."""
        start_time = time.time()

        try:
            # Check translation memory first
            memory_result = await self._check_translation_memory(text)
            if memory_result:
                self.stats["memory_hits"] += 1
                return memory_result

            # Detect language if not provided
            if not source_language:
                source_language = await self._detect_language(text)

            # Perform translation
            translated_text = await self._perform_translation(text, source_language)

            # Create result
            result = TranslationResult(
                original_text=text,
                translated_text=translated_text,
                source_language=source_language,
                target_language=target_language,
                processing_time=time.time() - start_time,
                model_used=self.translation_models["primary"]
            )

            # Store in translation memory
            await self._store_translation_memory(result)

            # Update statistics
            self.stats["total_translations"] += 1
            self.stats["languages_detected"][source_language] = (
                self.stats["languages_detected"].get(source_language, 0) + 1
            )
            self.stats["processing_times"].append(result.processing_time)

            return result

        except Exception as e:
            self.logger.error(f"Translation failed: {e}")
            return TranslationResult(
                original_text=text,
                translated_text=text,  # Return original on error
                source_language=source_language or "unknown",
                target_language=target_language,
                confidence=0.0,
                processing_time=time.time() - start_time,
                model_used="error"
            )

    async def translate_document(
        self,
        content: str,
        content_type: DataType,
        source_language: Optional[str] = None
    ) -> TranslationResult:
        """Translate document content (PDF, webpage, etc.)."""
        try:
            # Extract text content based on type
            if content_type == DataType.PDF:
                text_content = await self._extract_pdf_text(content)
            elif content_type == DataType.WEBPAGE:
                text_content = await self._extract_webpage_text(content)
            else:
                text_content = content

            # Translate the extracted text
            return await self.translate_text(text_content, source_language)

        except Exception as e:
            self.logger.error(f"Document translation failed: {e}")
            return TranslationResult(
                original_text=content,
                translated_text=content,
                source_language=source_language or "unknown",
                confidence=0.0,
                model_used="error"
            )

    async def batch_translate(
        self,
        texts: List[str],
        source_language: Optional[str] = None
    ) -> List[TranslationResult]:
        """Translate multiple texts in batch."""
        results = []

        for text in texts:
            result = await self.translate_text(text, source_language)
            results.append(result)

        return results

    async def detect_language(self, text: str) -> str:
        """Detect the language of the text."""
        return await self._detect_language(text)

    async def _detect_language(self, text: str) -> str:
        """Detect language using pattern matching and model inference."""
        # Check for specific language patterns
        for lang_code, pattern in self.language_patterns.items():
            if re.search(pattern, text):
                return lang_code

        # Use model for language detection
        try:
            prompt = f"""
            Detect the language of the following text and respond with only the ISO 639-1 language code (e.g., 'en', 'es', 'fr', 'de', 'zh', 'ja', 'ko', 'ar', 'hi', 'th', 'ru', 'pt', 'it'):

            Text: {text[:500]}

            Language code:"""

            response = await self.ollama_client.generate_text(
                prompt,
                model=self.translation_models["fast"],
                max_tokens=10
            )

            # Extract language code from response
            lang_code = response.strip().lower()
            if lang_code in self.language_patterns:
                return lang_code

        except Exception as e:
            self.logger.warning(f"Language detection failed: {e}")

        # Default to English if detection fails
        return "en"

    async def _perform_translation(
        self,
        text: str,
        source_language: str
    ) -> str:
        """Perform the actual translation using Ollama."""
        try:
            # Create translation prompt
            prompt = f"""
            Translate the following text from {source_language} to English.
            Maintain the original meaning and tone. Only return the translated text, no explanations.

            Text: {text}

            Translation:"""

            # Try primary model first
            try:
                response = await self.ollama_client.generate_text(
                    prompt,
                    model=self.translation_models["primary"],
                    max_tokens=len(text) * 2
                )
                return response.strip()
            except Exception as e:
                self.logger.warning(f"Primary model failed: {e}")

                # Fallback to secondary model
                response = await self.ollama_client.generate_text(
                    prompt,
                    model=self.translation_models["fallback"],
                    max_tokens=len(text) * 2
                )
                return response.strip()

        except Exception as e:
            self.logger.error(f"Translation failed: {e}")
            return text  # Return original text on error

    async def _check_translation_memory(self, text: str) -> Optional[TranslationResult]:
        """Check if translation exists in memory."""
        try:
            # Query vector database
            results = await self.vector_db.query(
                collection_name="translations",
                query_text=text,
                n_results=1
            )

            if results and results[0]['score'] > 0.9:  # High similarity threshold
                # Reconstruct TranslationResult from stored data
                metadata = results[0]['metadata']
                return TranslationResult(
                    original_text=metadata.get("original_text", text),
                    translated_text=metadata.get("translated_text", text),
                    source_language=metadata.get("source_language", "unknown"),
                    target_language=metadata.get("target_language", "en"),
                    confidence=metadata.get("confidence", 0.8),
                    processing_time=0.0,  # No processing time for memory hits
                    model_used=metadata.get("model_used", "memory"),
                    translation_memory_hit=True,
                    metadata=metadata
                )

        except Exception as e:
            self.logger.warning(f"Translation memory check failed: {e}")

        return None

    async def _store_translation_memory(self, translation_result: TranslationResult):
        """Store translation in memory for future use."""
        try:
            # Get metadata and sanitize it for ChromaDB compatibility
            metadata = translation_result.to_dict()
            sanitized_metadata = self.vector_db.sanitize_metadata(metadata)
            
            # Store in vector database
            await self.vector_db.add_texts(
                collection_name="translations",
                texts=[translation_result.original_text],
                metadatas=[sanitized_metadata]
            )

        except Exception as e:
            self.logger.warning(f"Failed to store translation in memory: {e}")

    async def _extract_pdf_text(self, pdf_path: str) -> str:
        """Extract text from PDF file."""
        try:
            # This would integrate with PDF extraction service
            # For now, return a placeholder
            return f"PDF content from {pdf_path}"
        except Exception as e:
            self.logger.error(f"PDF text extraction failed: {e}")
            return ""

    async def _extract_webpage_text(self, url: str) -> str:
        """Extract text from webpage."""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            # Simple text extraction (in practice, use BeautifulSoup)
            text = response.text
            # Remove HTML tags and clean up
            text = re.sub(r'<[^>]+>', '', text)
            text = re.sub(r'\s+', ' ', text).strip()

            return text[:5000]  # Limit text length

        except Exception as e:
            self.logger.error(f"Webpage text extraction failed: {e}")
            return ""

    def get_stats(self) -> Dict[str, Any]:
        """Get translation statistics."""
        avg_processing_time = 0
        if self.stats["processing_times"]:
            avg_processing_time = (
                sum(self.stats["processing_times"]) /
                len(self.stats["processing_times"])
            )

        return {
            "total_translations": self.stats["total_translations"],
            "memory_hits": self.stats["memory_hits"],
            "memory_hit_rate": (
                self.stats["memory_hits"] /
                max(self.stats["total_translations"], 1)
            ),
            "languages_detected": self.stats["languages_detected"],
            "average_processing_time": avg_processing_time,
            "available_models": list(self.translation_models.keys())
        }

    def reset_stats(self):
        """Reset translation statistics."""
        self.stats = {
            "total_translations": 0,
            "memory_hits": 0,
            "languages_detected": {},
            "processing_times": []
        }
