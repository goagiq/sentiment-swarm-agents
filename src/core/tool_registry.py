"""
Tool Registry for centralized tool management.
Extracts tool functions from orchestrator and provides unified tool access.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Callable

from src.core.models import AnalysisRequest, DataType
from src.agents.unified_text_agent import UnifiedTextAgent
from src.agents.unified_vision_agent import UnifiedVisionAgent
from src.agents.unified_audio_agent import UnifiedAudioAgent
from src.agents.web_agent_enhanced import EnhancedWebAgent
from src.agents.unified_file_extraction_agent import UnifiedFileExtractionAgent
# YouTube analysis now handled by UnifiedVisionAgent

# Configure logger
logger = logging.getLogger(__name__)


class ToolRegistry:
    """Registry for all available tools."""

    def __init__(self):
        self.logger = logger
        self.tools: Dict[str, Callable] = {}
        self.tool_metadata: Dict[str, Dict[str, Any]] = {}
        self._register_default_tools()

    def _register_default_tools(self):
        """Register all default tools."""
        # Text processing tools
        self.register_tool(
            "text_sentiment_analysis",
            self._text_sentiment_analysis,
            "Handle text-based sentiment analysis queries",
            ["text", "sentiment"]
        )

        # Vision processing tools
        self.register_tool(
            "vision_sentiment_analysis",
            self._vision_sentiment_analysis,
            "Handle comprehensive image, video, and YouTube analysis",
            ["vision", "image", "video", "youtube"]
        )

        self.register_tool(
            "youtube_comprehensive_analysis",
            self._youtube_comprehensive_analysis,
            "Handle comprehensive YouTube video analysis with audio and visual sentiment",
            ["youtube", "video", "audio", "sentiment"]
        )

        # Audio processing tools
        self.register_tool(
            "enhanced_audio_sentiment_analysis",
            self._enhanced_audio_sentiment_analysis,
            "Handle enhanced audio sentiment analysis queries",
            ["audio", "sentiment"]
        )

        self.register_tool(
            "audio_summarization_analysis",
            self._audio_summarization_analysis,
            "Handle comprehensive audio summarization",
            ["audio", "summarization"]
        )

        # Web processing tools
        self.register_tool(
            "web_sentiment_analysis",
            self._web_sentiment_analysis,
            "Handle webpage sentiment analysis queries",
            ["web", "sentiment"]
        )

        # Swarm tools
        self.register_tool(
            "swarm_text_analysis",
            self._swarm_text_analysis,
            "Handle complex text analysis using coordinated swarm of agents",
            ["text", "swarm", "coordinated"]
        )

        # Video processing tools
        self.register_tool(
            "unified_video_analysis",
            self._unified_video_analysis,
            "Handle comprehensive video analysis for YouTube, local videos, and other platforms",
            ["video", "youtube", "local"]
        )

        self.register_tool(
            "video_summarization_analysis",
            self._video_summarization_analysis,
            "Handle comprehensive video summarization with key scenes, visual analysis, and sentiment analysis",
            ["video", "summarization", "scenes"]
        )

        # Translation tools
        self.register_tool(
            "translate_text",
            self._translate_text,
            "Translate text content to English with automatic language detection",
            ["translation", "text", "language"]
        )

        self.register_tool(
            "translate_document",
            self._translate_document,
            "Translate document content (PDF, webpage, etc.) to English",
            ["translation", "document", "pdf", "webpage"]
        )

        self.register_tool(
            "batch_translate",
            self._batch_translate,
            "Translate multiple texts in batch",
            ["translation", "batch", "multiple"]
        )

        self.register_tool(
            "detect_language",
            self._detect_language,
            "Detect the language of text content",
            ["translation", "language", "detection"]
        )

        # OCR tools
        self.register_tool(
            "ocr_analysis",
            self._ocr_analysis,
            "Handle optical character recognition using Ollama and Llama Vision",
            ["ocr", "text_extraction", "vision"]
        )

        self.register_tool(
            "ocr_text_extraction",
            self._ocr_text_extraction,
            "Extract text from images using OCR",
            ["ocr", "text_extraction"]
        )

        self.register_tool(
            "ocr_document_analysis",
            self._ocr_document_analysis,
            "Analyze document structure and extract key information",
            ["ocr", "document", "structure"]
        )

        self.register_tool(
            "ocr_batch_processing",
            self._ocr_batch_processing,
            "Process multiple images for OCR in batch",
            ["ocr", "batch", "multiple"]
        )

    def register_tool(
        self,
        name: str,
        func: Callable,
        description: str,
        tags: List[str]
    ):
        """Register a tool with metadata."""
        self.tools[name] = func
        self.tool_metadata[name] = {
            'description': description,
            'tags': tags,
            'async': asyncio.iscoroutinefunction(func)
        }
        self.logger.info(f"Registered tool: {name}")

    def get_tool(self, name: str) -> Optional[Callable]:
        """Get a tool by name."""
        return self.tools.get(name)

    def get_tool_metadata(self, name: str) -> Optional[Dict[str, Any]]:
        """Get tool metadata."""
        return self.tool_metadata.get(name)

    def list_tools(self) -> List[str]:
        """List all available tools."""
        return list(self.tools.keys())

    def get_tools_by_tag(self, tag: str) -> List[str]:
        """Get tools by tag."""
        return [
            name for name, metadata in self.tool_metadata.items()
            if tag in metadata.get('tags', [])
        ]

    async def execute_tool(self, name: str, *args, **kwargs) -> Any:
        """Execute a tool by name."""
        tool = self.get_tool(name)
        if not tool:
            raise ValueError(f"Tool not found: {name}")

        if asyncio.iscoroutinefunction(tool):
            return await tool(*args, **kwargs)
        else:
            return tool(*args, **kwargs)

    # Tool implementations
    async def _text_sentiment_analysis(self, query: str) -> dict:
        """Handle text-based sentiment analysis queries."""
        try:
            text_agent = UnifiedTextAgent(use_strands=True, use_swarm=False)

            request = AnalysisRequest(
                data_type=DataType.TEXT,
                content=query,
                language="en"
            )

            result = await text_agent.process(request)

            return {
                "status": "success",
                "content": [{
                    "json": {
                        "sentiment": result.sentiment.label,
                        "confidence": result.sentiment.confidence,
                        "method": "text_agent",
                        "agent_id": text_agent.agent_id
                    }
                }]
            }

        except Exception as e:
            self.logger.error(f"Text sentiment analysis failed: {e}")
            return {
                "status": "error",
                "content": [{"text": f"Text analysis error: {str(e)}"}]
            }

    async def _vision_sentiment_analysis(self, image_path: str) -> dict:
        """Handle comprehensive image, video, and YouTube analysis."""
        try:
            if "youtube.com" in image_path or "youtu.be" in image_path:
                vision_agent = UnifiedVisionAgent()

                request = AnalysisRequest(
                    data_type=DataType.VIDEO,
                    content=image_path,
                    language="en"
                )

                result = await vision_agent.process(request)

                return {
                    "status": "success",
                    "content": [{
                        "json": {
                            "sentiment": result.sentiment.label,
                            "confidence": result.sentiment.confidence,
                            "method": "unified_vision_agent_youtube",
                            "agent_id": vision_agent.agent_id
                        }
                    }]
                }
            else:
                vision_agent = UnifiedVisionAgent()

                request = AnalysisRequest(
                    data_type=DataType.IMAGE,
                    content=image_path,
                    language="en"
                )

                result = await vision_agent.process(request)

                return {
                    "status": "success",
                    "content": [{
                        "json": {
                            "sentiment": result.sentiment.label,
                            "confidence": result.sentiment.confidence,
                            "method": "vision_agent",
                            "agent_id": vision_agent.agent_id
                        }
                    }]
                }

        except Exception as e:
            self.logger.error(f"Vision sentiment analysis failed: {e}")
            return {
                "status": "error",
                "content": [{"text": f"Vision analysis error: {str(e)}"}]
            }

    async def _youtube_comprehensive_analysis(self, youtube_url: str) -> dict:
        """Handle comprehensive YouTube video analysis."""
        try:
            vision_agent = UnifiedVisionAgent()

            request = AnalysisRequest(
                data_type=DataType.VIDEO,
                content=youtube_url,
                language="en"
            )

            result = await vision_agent.process(request)

            return {
                "status": "success",
                "content": [{
                    "json": {
                        "sentiment": result.sentiment.label,
                        "confidence": result.sentiment.confidence,
                        "method": "unified_vision_agent_youtube",
                        "agent_id": vision_agent.agent_id
                    }
                }]
            }

        except Exception as e:
            self.logger.error(f"YouTube analysis failed: {e}")
            return {
                "status": "error",
                "content": [{"text": f"YouTube analysis error: {str(e)}"}]
            }

    async def _enhanced_audio_sentiment_analysis(self, audio_path: str) -> dict:
        """Handle enhanced audio sentiment analysis."""
        try:
            audio_agent = UnifiedAudioAgent()

            request = AnalysisRequest(
                data_type=DataType.AUDIO,
                content=audio_path,
                language="en"
            )

            result = await audio_agent.process(request)

            return {
                "status": "success",
                "content": [{
                    "json": {
                        "sentiment": result.sentiment.label,
                        "confidence": result.sentiment.confidence,
                        "method": "enhanced_audio_agent",
                        "agent_id": audio_agent.agent_id
                    }
                }]
            }

        except Exception as e:
            self.logger.error(f"Audio sentiment analysis failed: {e}")
            return {
                "status": "error",
                "content": [{"text": f"Audio analysis error: {str(e)}"}]
            }

    async def _audio_summarization_analysis(self, audio_path: str) -> dict:
        """Handle comprehensive audio summarization."""
        try:
            audio_agent = UnifiedAudioAgent()

            request = AnalysisRequest(
                data_type=DataType.AUDIO,
                content=audio_path,
                language="en"
            )

            result = await audio_agent.process(request)

            return {
                "status": "success",
                "content": [{
                    "json": {
                        "sentiment": result.sentiment.label,
                        "confidence": result.sentiment.confidence,
                        "method": "audio_summarization",
                        "agent_id": audio_agent.agent_id,
                        "summary": result.extracted_text
                    }
                }]
            }

        except Exception as e:
            self.logger.error(f"Audio summarization failed: {e}")
            return {
                "status": "error",
                "content": [{"text": f"Audio summarization error: {str(e)}"}]
            }

    async def _web_sentiment_analysis(self, url: str) -> dict:
        """Handle webpage sentiment analysis."""
        try:
            web_agent = EnhancedWebAgent()

            request = AnalysisRequest(
                data_type=DataType.WEBPAGE,
                content=url,
                language="en"
            )

            result = await web_agent.process(request)

            return {
                "status": "success",
                "content": [{
                    "json": {
                        "sentiment": result.sentiment.label,
                        "confidence": result.sentiment.confidence,
                        "method": "web_agent",
                        "agent_id": web_agent.agent_id,
                        "url": url
                    }
                }]
            }

        except Exception as e:
            self.logger.error(f"Web sentiment analysis failed: {e}")
            return {
                "status": "error",
                "content": [{"text": f"Web analysis error: {str(e)}"}]
            }

    async def _swarm_text_analysis(self, text: str) -> dict:
        """Handle complex text analysis using swarm."""
        try:
            text_agent = UnifiedTextAgent(use_strands=True, use_swarm=True)

            request = AnalysisRequest(
                data_type=DataType.TEXT,
                content=text,
                language="en"
            )

            result = await text_agent.process(request)

            return {
                "status": "success",
                "content": [{
                    "json": {
                        "sentiment": result.sentiment.label,
                        "confidence": result.sentiment.confidence,
                        "method": "swarm_text_analysis",
                        "agent_id": text_agent.agent_id
                    }
                }]
            }

        except Exception as e:
            self.logger.error(f"Swarm text analysis failed: {e}")
            return {
                "status": "error",
                "content": [{"text": f"Swarm analysis error: {str(e)}"}]
            }

    async def _unified_video_analysis(self, video_input: str) -> dict:
        """Handle unified video analysis."""
        try:
            vision_agent = UnifiedVisionAgent()

            request = AnalysisRequest(
                data_type=DataType.VIDEO,
                content=video_input,
                language="en"
            )

            result = await vision_agent.process(request)

            return {
                "status": "success",
                "content": [{
                    "json": {
                        "sentiment": result.sentiment.label,
                        "confidence": result.sentiment.confidence,
                        "method": "unified_video_analysis",
                        "agent_id": vision_agent.agent_id
                    }
                }]
            }

        except Exception as e:
            self.logger.error(f"Unified video analysis failed: {e}")
            return {
                "status": "error",
                "content": [{"text": f"Video analysis error: {str(e)}"}]
            }

    async def _video_summarization_analysis(self, video_path: str) -> dict:
        """Handle video summarization analysis."""
        try:
            vision_agent = UnifiedVisionAgent()

            request = AnalysisRequest(
                data_type=DataType.VIDEO,
                content=video_path,
                language="en"
            )

            result = await vision_agent.process(request)

            return {
                "status": "success",
                "content": [{
                    "json": {
                        "sentiment": result.sentiment.label,
                        "confidence": result.sentiment.confidence,
                        "method": "video_summarization",
                        "agent_id": vision_agent.agent_id,
                        "summary": result.extracted_text
                    }
                }]
            }

        except Exception as e:
            self.logger.error(f"Video summarization failed: {e}")
            return {
                "status": "error",
                "content": [{"text": f"Video summarization error: {str(e)}"}]
            }

    async def _ocr_analysis(self, image_path: str) -> dict:
        """Handle OCR analysis using Unified File Extraction Agent."""
        try:
            file_agent = UnifiedFileExtractionAgent()

            request = AnalysisRequest(
                data_type=DataType.IMAGE,
                content=image_path,
                language="en"
            )

            result = await file_agent.process(request)

            return {
                "status": "success",
                "content": [{
                    "json": {
                        "extracted_text": result.extracted_text,
                        "method": "ocr_analysis",
                        "agent_id": file_agent.agent_id
                    }
                }]
            }

        except Exception as e:
            self.logger.error(f"OCR analysis failed: {e}")
            return {
                "status": "error",
                "content": [{"text": f"OCR analysis error: {str(e)}"}]
            }

    async def _ocr_text_extraction(self, image_path: str) -> dict:
        """Extract text from images using OCR."""
        return await self._ocr_analysis(image_path)

    async def _ocr_document_analysis(self, image_path: str) -> dict:
        """Analyze document structure and extract key information."""
        return await self._ocr_analysis(image_path)

    async def _ocr_batch_processing(self, image_paths: List[str]) -> dict:
        """Process multiple images for OCR in batch."""
        try:
            file_agent = UnifiedFileExtractionAgent()
            results = []

            for image_path in image_paths:
                request = AnalysisRequest(
                    data_type=DataType.IMAGE,
                    content=image_path,
                    language="en"
                )

                result = await file_agent.process(request)
                results.append({
                    "image_path": image_path,
                    "extracted_text": result.extracted_text,
                    "agent_id": file_agent.agent_id
                })

            return {
                "status": "success",
                "content": [{"json": {"results": results}}]
            }

        except Exception as e:
            self.logger.error(f"OCR batch processing failed: {e}")
            return {
                "status": "error",
                "content": [{"text": f"OCR batch processing error: {str(e)}"}]
            }

    # Translation tool implementations
    async def _translate_text(self, text: str, source_language: str = None, target_language: str = "en") -> dict:
        """Translate text content using Unified Text Agent."""
        try:
            text_agent = UnifiedTextAgent()
            result = await text_agent.translate_text(text, source_language, target_language)
            return result
        except Exception as e:
            self.logger.error(f"Text translation failed: {e}")
            return {
                "status": "error",
                "content": [{"text": f"Text translation error: {str(e)}"}]
            }

    async def _translate_document(self, content: str, content_type: str, source_language: str = None) -> dict:
        """Translate document content using Unified Text Agent."""
        try:
            text_agent = UnifiedTextAgent()
            result = await text_agent.translate_document(content, content_type, source_language)
            return result
        except Exception as e:
            self.logger.error(f"Document translation failed: {e}")
            return {
                "status": "error",
                "content": [{"text": f"Document translation error: {str(e)}"}]
            }

    async def _batch_translate(self, texts: List[str], source_language: str = None) -> dict:
        """Translate multiple texts in batch using Unified Text Agent."""
        try:
            text_agent = UnifiedTextAgent()
            result = await text_agent.batch_translate(texts, source_language)
            return result
        except Exception as e:
            self.logger.error(f"Batch translation failed: {e}")
            return {
                "status": "error",
                "content": [{"text": f"Batch translation error: {str(e)}"}]
            }

    async def _detect_language(self, text: str) -> dict:
        """Detect language using Unified Text Agent."""
        try:
            text_agent = UnifiedTextAgent()
            result = await text_agent.detect_language(text)
            return result
        except Exception as e:
            self.logger.error(f"Language detection failed: {e}")
            return {
                "status": "error",
                "content": [{"text": f"Language detection error: {str(e)}"}]
            }


# Global instance
tool_registry = ToolRegistry()
