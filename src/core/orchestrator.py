"""
Orchestrator for managing the agent swarm and routing requests.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime

from loguru import logger

from src.agents.base_agent import StrandsBaseAgent as BaseAgent
from src.agents.unified_text_agent import UnifiedTextAgent
from src.agents.unified_vision_agent import UnifiedVisionAgent
from src.agents.unified_audio_agent import UnifiedAudioAgent
from src.agents.web_agent_enhanced import EnhancedWebAgent
from src.agents.knowledge_graph_agent import KnowledgeGraphAgent
from src.agents.file_extraction_agent import FileExtractionAgent
from src.core.models import (
    AnalysisRequest, AnalysisResult, DataType,
    ModelConfig
)
from src.core.model_manager import ModelManager


class SentimentOrchestrator:
    """Orchestrates the sentiment analysis agent swarm."""

    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.model_manager = ModelManager()
        self.request_cache: Dict[str, AnalysisResult] = {}
        self.cache_ttl = 3600  # 1 hour
        self.cache_timestamps: Dict[str, datetime] = {}

        # Initialize agents
        self._register_agents()

    def _register_agents(self):
        """Register all available agents."""
        # Text agent (unified with swarm mode)
        text_agent = UnifiedTextAgent(use_strands=True, use_swarm=True)
        self._register_agent(text_agent, [DataType.TEXT])

        # Vision agent (unified)
        vision_agent = UnifiedVisionAgent()
        self._register_agent(vision_agent, [DataType.IMAGE, DataType.VIDEO])

        # Audio agent (unified)
        audio_agent = UnifiedAudioAgent()
        self._register_agent(audio_agent, [DataType.AUDIO])

        # Web agent
        web_agent = EnhancedWebAgent()
        self._register_agent(web_agent, [DataType.WEBPAGE])

        # Knowledge Graph agent (GraphRAG-inspired)
        knowledge_graph_agent = KnowledgeGraphAgent()
        self._register_agent(knowledge_graph_agent, [
            DataType.TEXT, DataType.AUDIO, DataType.VIDEO,
            DataType.WEBPAGE, DataType.PDF, DataType.SOCIAL_MEDIA
        ])

        # File Extraction agent
        file_extraction_agent = FileExtractionAgent()
        self._register_agent(file_extraction_agent, [DataType.PDF])

        logger.info(
            f"Registered {len(self.agents)} unified agents including "
            f"GraphRAG-inspired Knowledge Graph Agent and File Extraction Agent"
        )

    def _register_agent(self, agent: BaseAgent, supported_types: List[DataType]):
        """Register an agent with its supported data types."""
        self.agents[agent.agent_id] = agent
        agent.metadata["supported_types"] = [dt.value for dt in supported_types]
        logger.info(f"Registered agent {agent.agent_id} for types: {supported_types}")

    async def analyze_text(self, content: str, language: str = "en", **kwargs) -> AnalysisResult:
        """Analyze text sentiment."""
        request = AnalysisRequest(
            data_type=DataType.TEXT,
            content=content,
            language=language,
            **kwargs
        )
        return await self.analyze(request)

    async def analyze_image(self, image_path: str, **kwargs) -> AnalysisResult:
        """Analyze image sentiment."""
        request = AnalysisRequest(
            data_type=DataType.IMAGE,
            content=image_path,
            **kwargs
        )
        return await self.analyze(request)

    async def analyze_video(self, video_path: str, **kwargs) -> AnalysisResult:
        """Analyze video sentiment."""
        request = AnalysisRequest(
            data_type=DataType.VIDEO,
            content=video_path,
            **kwargs
        )
        return await self.analyze(request)

    async def analyze_audio(self, audio_path: str, **kwargs) -> AnalysisResult:
        """Analyze audio sentiment."""
        request = AnalysisRequest(
            data_type=DataType.AUDIO,
            content=audio_path,
            **kwargs
        )
        return await self.analyze(request)

    async def analyze_webpage(self, url: str, **kwargs) -> AnalysisResult:
        """Analyze webpage sentiment."""
        request = AnalysisRequest(
            data_type=DataType.WEBPAGE,
            content=url,
            **kwargs
        )
        return await self.analyze(request)

    async def analyze_pdf(self, pdf_path: str, **kwargs) -> AnalysisResult:
        """Analyze PDF content and extract text."""
        request = AnalysisRequest(
            data_type=DataType.PDF,
            content=pdf_path,
            **kwargs
        )
        return await self.analyze(request)

    async def analyze(self, request: AnalysisRequest) -> AnalysisResult:
        """Analyze content using the appropriate agent."""
        # Check cache first
        cache_key = self._generate_cache_key(request)
        if cache_key in self.request_cache:
            cached_result = self.request_cache[cache_key]
            if self._is_cache_valid(cache_key):
                logger.info(f"Returning cached result for request {request.id}")
                return cached_result
            else:
                # Remove expired cache entry
                del self.request_cache[cache_key]
                del self.cache_timestamps[cache_key]

        # Find suitable agent
        agent = await self._find_suitable_agent(request)
        if not agent:
            raise ValueError(f"No suitable agent found for data type: {request.data_type}")

        # Process with reflection if enabled
        if request.reflection_enabled:
            result = await self._process_with_reflection(request, agent)
        else:
            result = await agent.process(request)

        # Cache the result
        self._cache_result(cache_key, result)

        return result

    async def _find_suitable_agent(self, request: AnalysisRequest) -> Optional[BaseAgent]:
        """Find a suitable agent for the request."""
        suitable_agents = []

        for agent in self.agents.values():
            if await agent.can_process(request):
                suitable_agents.append(agent)

        if not suitable_agents:
            return None

        # If user specified a model preference, prioritize agents that can use it
        if request.model_preference:
            preferred_agents = [
                agent for agent in suitable_agents
                if hasattr(agent, 'model_name') and
                agent.model_name == request.model_preference
            ]
            if preferred_agents:
                suitable_agents = preferred_agents

        # Return the first suitable agent (could be enhanced with load balancing)
        return suitable_agents[0]

    async def _process_with_reflection(
        self,
        request: AnalysisRequest,
        agent: BaseAgent
    ) -> AnalysisResult:
        """Process request with reflection for quality improvement."""
        logger.info(f"Processing request {request.id} with reflection")

        best_result = None
        best_confidence = 0.0
        reflection_notes = []

        for iteration in range(request.max_iterations):
            # Process the request
            result = await agent.process(request)

            # Assess confidence and quality
            confidence = result.sentiment.confidence
            reflection_note = f"Iteration {iteration + 1}: confidence {confidence:.3f}"

            # Generate alternative hypotheses if confidence is low
            if (confidence < request.confidence_threshold and
                    iteration < request.max_iterations - 1):
                alternatives = await self._generate_alternatives(request, agent, result)
                reflection_note += f", generated {len(alternatives)} alternatives"

                # Try alternatives
                for alt_request in alternatives:
                    alt_result = await agent.process(alt_request)
                    if alt_result.sentiment.confidence > confidence:
                        result = alt_result
                        confidence = alt_result.sentiment.confidence
                        reflection_note += (
                            f", found better alternative "
                            f"(confidence: {confidence:.3f})"
                        )

            reflection_notes.append(reflection_note)

            # Update best result if this iteration is better
            if confidence > best_confidence:
                best_result = result
                best_confidence = confidence

            # Check if we've reached the confidence threshold
            if confidence >= request.confidence_threshold:
                logger.info(f"Confidence threshold reached after {iteration + 1} iterations")
                break

        # Update the best result with reflection information
        if best_result:
            best_result.sentiment.reflection_notes = reflection_notes
            best_result.sentiment.iteration_count = len(reflection_notes)
            best_result.reflection_enabled = True
            best_result.quality_score = best_confidence

        return best_result or result

    async def _generate_alternatives(
        self,
        request: AnalysisRequest,
        agent: BaseAgent,
        current_result: AnalysisResult
    ) -> List[AnalysisRequest]:
        """Generate alternative analysis approaches."""
        alternatives = []

        # Try different prompts or approaches
        if request.data_type == DataType.TEXT:
            # Try different text analysis approaches
            alt_prompts = [
                "Analyze the sentiment from a different perspective",
                "Consider the context and implications",
                "Look for subtle emotional cues"
            ]

            for prompt in alt_prompts:
                alt_request = AnalysisRequest(
                    data_type=request.data_type,
                    content=f"{request.content}\n\nAnalysis instruction: {prompt}",
                    language=request.language,
                    reflection_enabled=False,  # Prevent infinite recursion
                    max_iterations=1
                )
                alternatives.append(alt_request)

        elif request.data_type in [DataType.IMAGE, DataType.VIDEO]:
            # Try different vision analysis approaches
            alt_prompts = [
                "Focus on facial expressions and body language",
                "Analyze colors, lighting, and composition",
                "Look for text, symbols, and contextual elements"
            ]

            for prompt in alt_prompts:
                alt_request = AnalysisRequest(
                    data_type=request.data_type,
                    content=request.content,
                    language=request.language,
                    reflection_enabled=False,
                    max_iterations=1
                )
                alternatives.append(alt_request)

        return alternatives

    def _generate_cache_key(self, request: AnalysisRequest) -> str:
        """Generate a cache key for the request."""
        # Simple cache key based on content hash and data type
        import hashlib
        content_hash = hashlib.md5(str(request.content).encode()).hexdigest()
        return f"{request.data_type.value}:{content_hash}"

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if a cached result is still valid."""
        if cache_key not in self.cache_timestamps:
            return False

        timestamp = self.cache_timestamps[cache_key]
        age = (datetime.now() - timestamp).total_seconds()
        return age < self.cache_ttl

    def _cache_result(self, cache_key: str, result: AnalysisResult):
        """Cache a result."""
        self.request_cache[cache_key] = result
        self.cache_timestamps[cache_key] = datetime.now()

        # Clean up old cache entries
        self._cleanup_cache()

    def _cleanup_cache(self):
        """Clean up expired cache entries."""
        current_time = datetime.now()
        expired_keys = []

        for key, timestamp in self.cache_timestamps.items():
            age = (current_time - timestamp).total_seconds()
            if age > self.cache_ttl:
                expired_keys.append(key)

        for key in expired_keys:
            del self.request_cache[key]
            del self.cache_timestamps[key]

        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")

    async def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all agents."""
        status = {}

        for agent_id, agent in self.agents.items():
            status[agent_id] = {
                "agent_type": agent.__class__.__name__,
                "supported_types": agent.metadata.get("supported_types", []),
                "model": agent.metadata.get("model", "unknown"),
                "capabilities": agent.metadata.get("capabilities", []),
                "status": "active"
            }

        return status

    async def get_available_models(self) -> List[ModelConfig]:
        """Get list of available models."""
        return self.model_manager.get_available_models()

    async def cleanup(self):
        """Cleanup resources."""
        # Cleanup agents
        for agent in self.agents.values():
            if hasattr(agent, 'cleanup'):
                await agent.cleanup()

        # Cleanup model manager
        await self.model_manager.cleanup()

        logger.info("Orchestrator cleanup completed")
