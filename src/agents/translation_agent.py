"""
Translation Agent for comprehensive foreign language translation to English.
Supports text, URL, audio, and video content with automatic language detection
and translation memory using Chroma vector DB.
"""

import asyncio
import logging
import time
import re
from typing import Dict, List, Optional, Any
import requests
from pydantic import BaseModel

from src.agents.base_agent import StrandsBaseAgent
from src.core.models import (
    AnalysisRequest,
    AnalysisResult,
    DataType,
    SentimentResult,
    SentimentLabel
)
from src.core.vector_db import VectorDBManager
from src.core.ollama_integration import OllamaIntegration

logger = logging.getLogger(__name__)


class TranslationResult(BaseModel):
    """Result of translation operation."""
    original_text: str
    translated_text: str
    source_language: str
    target_language: str = "en"
    confidence: float
    processing_time: float
    model_used: str
    translation_memory_hit: bool = False
    metadata: Dict[str, Any] = {}


class TranslationAgent(StrandsBaseAgent):
    """Comprehensive translation agent supporting multiple content types."""
    
    def __init__(self, agent_id: Optional[str] = None):
        super().__init__(
            agent_id=agent_id or "translation_agent",
            model_name="ollama:mistral-small3.1:latest"
        )
        
        # Initialize Ollama client for translation
        self.ollama_client = OllamaIntegration()
        
        # Simple async HTTP client for Ollama requests
        import aiohttp
        self.session = None
        
        # Initialize Chroma vector DB for translation memory
        self.vector_db = VectorDBManager()
        
        # Translation models configuration
        self.translation_models = {
            "primary": "mistral-small3.1:latest",
            "fallback": "llama3.2:latest",
            "vision": "llava:latest",
            "fast": "phi3:mini"
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
        
        logger.info(f"Translation Agent {self.agent_id} initialized")
    
    async def can_process(self, request: AnalysisRequest) -> bool:
        """Check if this agent can process the given request."""
        return request.data_type in [
            DataType.TEXT, 
            DataType.AUDIO, 
            DataType.VIDEO, 
            DataType.WEBPAGE,
            DataType.IMAGE,
            DataType.PDF
        ]
    
    async def process(self, request: AnalysisRequest) -> AnalysisResult:
        """Process translation request."""
        start_time = time.time()
        
        try:
            # Detect content type and route to appropriate handler
            if request.data_type == DataType.TEXT:
                translation_result = await self._translate_text(
                    request.content
                )
            elif request.data_type == DataType.WEBPAGE:
                translation_result = await self._translate_webpage(
                    request.content
                )
            elif request.data_type == DataType.AUDIO:
                translation_result = await self._translate_audio(
                    request.content
                )
            elif request.data_type == DataType.VIDEO:
                translation_result = await self._translate_video(
                    request.content
                )
            elif request.data_type == DataType.IMAGE:
                translation_result = await self._translate_image(
                    request.content
                )
            elif request.data_type == DataType.PDF:
                translation_result = await self._translate_pdf(
                    request.content
                )
            else:
                raise ValueError(
                    f"Unsupported data type: {request.data_type}"
                )
            
            # Store in translation memory
            await self._store_translation_memory(translation_result)
            
            # Create sentiment result (neutral for translation)
            sentiment_result = SentimentResult(
                label=SentimentLabel.NEUTRAL,
                confidence=translation_result.confidence,
                reasoning=f"Translation completed from {translation_result.source_language} to English"
            )
            
            processing_time = time.time() - start_time
            
            return AnalysisResult(
                request_id=request.id,
                data_type=request.data_type,
                sentiment=sentiment_result,
                processing_time=processing_time,
                status="completed",
                extracted_text=translation_result.translated_text,
                model_used=translation_result.model_used,
                quality_score=translation_result.confidence,
                reflection_enabled=True,
                metadata={
                    "translation": translation_result.dict(),
                    "original_language": translation_result.source_language,
                    "translation_memory_hit": translation_result.translation_memory_hit,
                    "model_used": translation_result.model_used
                }
            )
            
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return AnalysisResult(
                request_id=request.id,
                data_type=request.data_type,
                sentiment=SentimentResult(
                    label=SentimentLabel.UNCERTAIN,
                    confidence=0.0,
                    reasoning=f"Translation failed: {str(e)}"
                ),
                processing_time=time.time() - start_time,
                status="failed",
                model_used="translation_error",
                quality_score=0.0,
                reflection_enabled=False,
                metadata={"error": str(e)}
            )
    
    async def _translate_text(self, text: str) -> TranslationResult:
        """Translate text content with automatic language detection."""
        # Check translation memory first
        memory_result = await self._check_translation_memory(text)
        if memory_result:
            return memory_result
        
        # Detect source language
        source_language = await self._detect_language(text)
        
        # Skip translation if already English
        if source_language == "en":
            return TranslationResult(
                original_text=text,
                translated_text=text,
                source_language="en",
                confidence=1.0,
                processing_time=0.0,
                model_used="no_translation_needed"
            )
        
        # Perform translation
        translated_text = await self._perform_translation(text, source_language)
        
        return TranslationResult(
            original_text=text,
            translated_text=translated_text,
            source_language=source_language,
            confidence=0.95,  # High confidence for mistral model
            processing_time=0.0,  # Will be set by caller
            model_used=self.translation_models["primary"]
        )
    
    async def _translate_webpage(self, url: str) -> TranslationResult:
        """Extract and translate webpage content."""
        try:
            # Extract text from webpage
            webpage_text = await self._extract_webpage_text(url)
            
            # Translate the extracted text
            return await self._translate_text(webpage_text)
            
        except Exception as e:
            raise Exception(f"Failed to translate webpage {url}: {e}")
    
    async def _translate_audio(self, audio_path: str) -> TranslationResult:
        """Transcribe and translate audio content."""
        try:
            # First, transcribe the audio using existing audio agent capabilities
            from src.agents.audio_agent_enhanced import EnhancedAudioAgent
            
            audio_agent = EnhancedAudioAgent()
            transcription_request = AnalysisRequest(
                data_type=DataType.AUDIO,
                content=audio_path,
                language="auto"  # Let the audio agent detect language
            )
            
            transcription_result = await audio_agent.process(transcription_request)
            
            if not transcription_result.extracted_text:
                raise Exception("Failed to transcribe audio")
            
            # Translate the transcribed text
            return await self._translate_text(transcription_result.extracted_text)
            
        except Exception as e:
            raise Exception(f"Failed to translate audio {audio_path}: {e}")
    
    async def _translate_video(self, video_path: str) -> TranslationResult:
        """Extract audio/visual content and translate."""
        try:
            # Use existing video analysis capabilities
            from src.agents.video_summarization_agent import VideoSummarizationAgent
            
            video_agent = VideoSummarizationAgent()
            video_request = AnalysisRequest(
                data_type=DataType.VIDEO,
                content=video_path,
                language="auto"
            )
            
            video_result = await video_agent.process(video_request)
            
            # Combine audio transcript and visual analysis
            combined_text = ""
            if video_result.extracted_text:
                combined_text += f"Audio transcript: {video_result.extracted_text}\n"
            if video_result.metadata.get("visual_analysis"):
                combined_text += f"Visual content: {video_result.metadata['visual_analysis']}\n"
            
            if not combined_text.strip():
                raise Exception("Failed to extract content from video")
            
            # Translate the combined content
            return await self._translate_text(combined_text)
            
        except Exception as e:
            raise Exception(f"Failed to translate video {video_path}: {e}")
    
    async def _translate_image(self, image_path: str) -> TranslationResult:
        """Extract and translate text from images using vision model for best results."""
        try:
            # Use vision model for better text extraction from images
            import aiohttp
            
            # Initialize session if needed
            if self.session is None:
                self.session = aiohttp.ClientSession()
            
            # Use vision model to extract text from image
            vision_prompt = f"""Analyze this image and extract all visible text. Return only the extracted text without any explanations or formatting.

Image: {image_path}"""
            
            vision_payload = {
                "model": self.translation_models["vision"],
                "prompt": vision_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 500
                }
            }
            
            async with self.session.post(
                "http://localhost:11434/api/generate",
                json=vision_payload,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    extracted_text = result.get("response", "").strip()
                    
                    if not extracted_text:
                        # Fallback to OCR agent if vision model fails
                        from src.agents.ocr_agent import OCRAgent
                        ocr_agent = OCRAgent()
                        ocr_result = await ocr_agent.extract_text(image_path)
                        extracted_text = ocr_result.get("extracted_text", "")
                    
                    if not extracted_text:
                        raise Exception("Failed to extract text from image")
                    
                    # Translate the extracted text
                    return await self._translate_text(extracted_text)
                else:
                    raise Exception(f"Vision model API error: {response.status}")
            
        except Exception as e:
            raise Exception(f"Failed to translate image {image_path}: {e}")
    
    async def _translate_pdf(self, pdf_path: str) -> TranslationResult:
        """Extract text from PDF and translate it."""
        try:
            import PyPDF2
            import io
            
            # Extract text from PDF
            extracted_text = ""
            
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    if page_text:
                        extracted_text += page_text + "\n"
            
            if not extracted_text.strip():
                raise Exception("No text could be extracted from PDF")
            
            # Translate the extracted text
            return await self._translate_text(extracted_text.strip())
            
        except ImportError:
            # Fallback if PyPDF2 is not available
            try:
                import fitz  # PyMuPDF
                
                extracted_text = ""
                doc = fitz.open(pdf_path)
                
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    page_text = page.get_text()
                    if page_text:
                        extracted_text += page_text + "\n"
                
                doc.close()
                
                if not extracted_text.strip():
                    raise Exception("No text could be extracted from PDF")
                
                # Translate the extracted text
                return await self._translate_text(extracted_text.strip())
                
            except ImportError:
                raise Exception("PDF processing requires PyPDF2 or PyMuPDF. Please install one of these packages.")
            except Exception as e:
                raise Exception(f"Failed to extract text from PDF: {e}")
        except Exception as e:
            raise Exception(f"Failed to translate PDF {pdf_path}: {e}")
    
    async def _detect_language(self, text: str) -> str:
        """Detect the source language of the text."""
        # Use pattern matching for quick detection
        for lang_code, pattern in self.language_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                return lang_code
        
        # Use Ollama for more accurate detection
        try:
            import aiohttp
            
            # Initialize session if needed
            if self.session is None:
                self.session = aiohttp.ClientSession()
            
            prompt = f"""Detect the language of the following text and respond with only the ISO 639-1 language code (e.g., 'en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'zh', 'ja', 'ko', 'ar', 'hi', 'th'):

Text: "{text[:500]}"

Language code:"""
            
            payload = {
                "model": self.translation_models["primary"],
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 10
                }
            }
            
            async with self.session.post(
                "http://localhost:11434/api/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    detected_lang = result.get("response", "").strip().lower()
                    
                    # Validate the response
                    if detected_lang in self.language_patterns.keys() or detected_lang == "en":
                        return detected_lang
                    
                    return "en"  # Default to English if detection fails
                else:
                    return "en"  # Default to English if API fails
            
        except Exception as e:
            logger.warning(f"Language detection failed: {e}, defaulting to English")
            return "en"
    
    async def _perform_translation(self, text: str, source_language: str) -> str:
        """Perform the actual translation using Ollama."""
        try:
            import aiohttp
            import json
            
            # Initialize session if needed
            if self.session is None:
                self.session = aiohttp.ClientSession()
            
            prompt = f"""Translate the following text from {source_language} to English. Provide only the translated text without any explanations or additional formatting:

Original text: "{text}"

English translation:"""
            
            # Prepare request payload
            payload = {
                "model": self.translation_models["primary"],
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": max(100, len(text) * 3)  # More generous token limit
                }
            }
            
            async with self.session.post(
                "http://localhost:11434/api/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("response", "").strip()
                else:
                    raise Exception(f"Ollama API error: {response.status}")
            
        except Exception as e:
            # Fallback to secondary model
            logger.warning(f"Primary translation failed: {e}, trying fallback model")
            
            try:
                payload = {
                    "model": self.translation_models["fallback"],
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": max(100, len(text) * 3)  # More generous token limit
                    }
                }
                
                async with self.session.post(
                    "http://localhost:11434/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("response", "").strip()
                    else:
                        raise Exception(f"Ollama API error: {response.status}")
                
            except Exception as e2:
                raise Exception(f"Translation failed with both models: {e2}")
    
    async def _check_translation_memory(self, text: str) -> Optional[TranslationResult]:
        """Check if translation exists in memory."""
        try:
            # Search for similar text in translation memory
            results = await self.vector_db.search_similar_results(
                query=text,
                n_results=1
            )
            
            if results and len(results) > 0:
                # Found a similar translation
                memory_entry = results[0]
                metadata = memory_entry.get("metadata", {})
                
                # Ensure we have actual translated text, not empty string
                translated_text = metadata.get("translated_text", "")
                if not translated_text or translated_text.strip() == "":
                    return None  # Don't return empty translations
                
                return TranslationResult(
                    original_text=metadata.get("original_text", text),
                    translated_text=translated_text,
                    source_language=metadata.get("source_language", "unknown"),
                    confidence=0.8,  # Default confidence for memory hits
                    processing_time=0.0,
                    model_used="translation_memory",
                    translation_memory_hit=True
                )
            
            return None
            
        except Exception as e:
            logger.warning(f"Translation memory check failed: {e}")
            return None

    async def comprehensive_translate_and_analyze(self, text: str, include_analysis: bool = True) -> Dict[str, Any]:
        """
        Provide comprehensive translation with automatic analysis.
        
        Args:
            text: Text to translate
            include_analysis: Whether to include sentiment and summary analysis
            
        Returns:
            Dictionary containing complete translation, analysis, and metadata
        """
        start_time = time.time()
        
        # Perform translation
        translation_result = await self._translate_text(text)
        
        # Store in translation memory for future use
        await self._store_translation_memory(translation_result)
        
        result = {
            "translation": {
                "original_text": translation_result.original_text,
                "translated_text": translation_result.translated_text,
                "source_language": translation_result.source_language,
                "target_language": translation_result.target_language,
                "confidence": translation_result.confidence,
                "model_used": translation_result.model_used,
                "translation_memory_hit": translation_result.translation_memory_hit,
                "processing_time": time.time() - start_time
            }
        }
        
        if include_analysis:
            # Add sentiment analysis
            try:
                from src.agents.text_agent import TextAgent
                text_agent = TextAgent()
                
                sentiment_request = AnalysisRequest(
                    data_type=DataType.TEXT,
                    content=translation_result.translated_text,
                    language="en"
                )
                
                sentiment_result = await text_agent.process(sentiment_request)
                
                result["sentiment_analysis"] = {
                    "sentiment": sentiment_result.sentiment.label.value,
                    "confidence": sentiment_result.sentiment.confidence,
                    "reasoning": sentiment_result.sentiment.reasoning
                }
                
            except Exception as e:
                logger.warning(f"Sentiment analysis failed: {e}")
                result["sentiment_analysis"] = {
                    "error": "Sentiment analysis unavailable"
                }
            
            # Add summary analysis
            try:
                summary_prompt = f"""
                Please provide a comprehensive summary of the following translated text:
                
                {translation_result.translated_text}
                
                Include:
                1. Key themes and main points
                2. Important context and background
                3. Potential implications or significance
                4. Key entities mentioned (people, organizations, places)
                
                Format as a structured summary with clear sections.
                """
                
                summary_response = await self.ollama_client.generate_response(
                    model=self.translation_models["primary"],
                    prompt=summary_prompt,
                    max_tokens=1000
                )
                
                result["summary_analysis"] = {
                    "summary": summary_response.strip(),
                    "word_count": len(translation_result.translated_text.split()),
                    "key_themes": self._extract_key_themes(translation_result.translated_text)
                }
                
            except Exception as e:
                logger.warning(f"Summary analysis failed: {e}")
                result["summary_analysis"] = {
                    "error": "Summary analysis unavailable"
                }
        
        return result

    async def analyze_chinese_news_dynamic(self, text: str, include_timestamp: bool = True) -> Dict[str, Any]:
        """
        Dynamic Chinese news analysis with comprehensive translation, sentiment, and summary.
        This method is designed for on-the-fly news analysis without creating temporary files.
        
        Args:
            text: Chinese news text to analyze
            include_timestamp: Whether to include analysis timestamp
            
        Returns:
            Dictionary containing comprehensive analysis results
        """
        import datetime
        
        try:
            # Perform comprehensive analysis
            result = await self.comprehensive_translate_and_analyze(text, include_analysis=True)
            
            # Add news-specific metadata
            result["news_analysis"] = {
                "analysis_type": "chinese_news",
                "processing_method": "dynamic_on_the_fly",
                "content_length": len(text),
                "characters": len(text),
                "estimated_reading_time": max(1, len(text) // 200)  # Rough estimate: 200 chars per minute
            }
            
            if include_timestamp:
                result["analysis_timestamp"] = datetime.datetime.now().isoformat()
            
            return result
            
        except Exception as e:
            logger.error(f"Dynamic Chinese news analysis failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "agent": "translation_agent",
                "analysis_type": "chinese_news"
            }
    
    def _extract_key_themes(self, text: str) -> List[str]:
        """Extract key themes from translated text."""
        try:
            # Simple keyword extraction - can be enhanced with NLP
            import re
            
            # Common political and economic keywords
            political_keywords = [
                "government", "election", "party", "political", "policy", "administration",
                "democracy", "vote", "campaign", "opposition", "ruling", "legislative"
            ]
            
            economic_keywords = [
                "economy", "trade", "tariff", "business", "market", "financial",
                "economic", "industry", "commerce", "investment", "growth"
            ]
            
            international_keywords = [
                "international", "foreign", "diplomatic", "relations", "global",
                "world", "country", "nation", "border", "treaty", "agreement"
            ]
            
            text_lower = text.lower()
            themes = []
            
            # Count keyword occurrences
            for category, keywords in [
                ("Political", political_keywords),
                ("Economic", economic_keywords),
                ("International", international_keywords)
            ]:
                count = sum(1 for keyword in keywords if keyword in text_lower)
                if count > 0:
                    themes.append(f"{category} ({count} mentions)")
            
            return themes[:5]  # Return top 5 themes
            
        except Exception as e:
            logger.warning(f"Theme extraction failed: {e}")
            return ["Analysis unavailable"]
    
    async def _store_translation_memory(self, translation_result: TranslationResult):
        """Store translation in memory for future use."""
        try:
            # Create a mock AnalysisResult for storage
            from src.core.models import AnalysisResult, SentimentResult, SentimentLabel
            
            mock_result = AnalysisResult(
                request_id=f"trans_{int(time.time())}_{hash(translation_result.original_text)}",
                data_type=DataType.TEXT,
                sentiment=SentimentResult(
                    label=SentimentLabel.NEUTRAL,
                    confidence=translation_result.confidence,
                    reasoning=f"Translation from {translation_result.source_language} to English"
                ),
                processing_time=translation_result.processing_time,
                status="completed",
                extracted_text=translation_result.translated_text,
                model_used=translation_result.model_used,
                quality_score=translation_result.confidence,
                reflection_enabled=True,
                metadata={
                    "original_text": translation_result.original_text,
                    "translated_text": translation_result.translated_text,
                    "source_language": translation_result.source_language,
                    "target_language": translation_result.target_language,
                    "model_used": translation_result.model_used,
                    "confidence": translation_result.confidence,
                    "timestamp": time.time(),
                    "translation_memory": True
                }
            )
            
            # Store in vector DB
            await self.vector_db.store_result(mock_result)
            
        except Exception as e:
            logger.warning(f"Failed to store translation memory: {e}")
    
    async def _extract_webpage_text(self, url: str) -> str:
        """Extract text content from webpage."""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Simple text extraction (can be enhanced with BeautifulSoup)
            text = response.text
            
            # Remove HTML tags and clean up
            text = re.sub(r'<[^>]+>', '', text)
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
            
            # Limit text length
            if len(text) > 10000:
                text = text[:10000] + "..."
            
            return text
            
        except Exception as e:
            raise Exception(f"Failed to extract text from webpage: {e}")
    
    async def batch_translate(self, requests: List[AnalysisRequest]) -> List[AnalysisResult]:
        """Process multiple translation requests in batch."""
        results = []
        
        # Process in parallel with semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(5)  # Limit to 5 concurrent translations
        
        async def process_single(request):
            async with semaphore:
                return await self.process(request)
        
        # Create tasks for all requests
        tasks = [process_single(req) for req in requests]
        
        # Execute all tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Create error result
                error_result = AnalysisResult(
                    request_id=requests[i].id,
                    data_type=requests[i].data_type,
                    sentiment=SentimentResult(
                        label=SentimentLabel.UNCERTAIN,
                        confidence=0.0,
                        reasoning=f"Batch translation failed: {str(result)}"
                    ),
                    processing_time=0.0,
                    status="failed",
                    model_used="translation_error",
                    quality_score=0.0,
                    reflection_enabled=False,
                    metadata={"error": str(result)}
                )
                processed_results.append(error_result)
            else:
                processed_results.append(result)
        
        return processed_results
    
    def get_status(self) -> Dict[str, Any]:
        """Get translation agent status."""
        base_status = super().get_status()
        base_status.update({
            "translation_models": list(self.translation_models.values()),
            "supported_languages": list(self.language_patterns.keys()) + ["en"],
            "translation_memory_enabled": True,
            "batch_processing_enabled": True
        })
        return base_status
