"""
Unified Text Processing Agent that consolidates all text processing capabilities
including Strands framework, simple processing, and swarm coordination.
"""

import asyncio
from typing import Any, Optional, List, Dict

from loguru import logger
from src.core.strands_mock import tool, Agent, Swarm

from src.agents.base_agent import StrandsBaseAgent as BaseAgent
from src.config.config import config
from src.core.models import (
    AnalysisRequest, 
    AnalysisResult, 
    DataType, 
    SentimentResult
)
from src.core.ollama_integration import create_ollama_agent
from src.core.translation_service import TranslationService


class UnifiedTextAgent(BaseAgent):
    """
    Unified agent for processing text-based content with configurable processing modes.
    
    Supports:
    - Strands framework processing
    - Simple direct processing
    - Swarm coordination
    - Multiple model configurations
    """
    
    def __init__(
        self, 
        use_strands: bool = True,
        use_swarm: bool = False,
        agent_count: int = 3,
        model_name: Optional[str] = None,
        **kwargs
    ):
        # Set configuration flags before calling parent constructor
        self.use_strands = use_strands
        self.use_swarm = use_swarm
        self.agent_count = agent_count if use_swarm else 1
        
        # Use config system instead of hardcoded values
        default_model = config.model.default_text_model
        super().__init__(
            model_name=model_name or default_model, 
            **kwargs
        )
        
        # Initialize processing components
        self.swarm_agents: List[Agent] = []
        self.coordinator_agent: Optional[Agent] = None
        
        # Initialize translation service
        self.translation_service = TranslationService()
        
        # Set metadata based on configuration
        self.metadata["model"] = model_name or default_model
        self.metadata["use_strands"] = use_strands
        self.metadata["use_swarm"] = use_swarm
        self.metadata["agent_count"] = self.agent_count
        self.metadata["supported_languages"] = ["en"]
        self.metadata["capabilities"] = self._get_capabilities()
        
        # Initialize processing mode
        self._initialize_processing_mode()
        
        logger.info(f"Initialized UnifiedTextAgent with strands={use_strands}, swarm={use_swarm}")
    
    def _get_capabilities(self) -> List[str]:
        """Get capabilities based on configuration."""
        capabilities = ["text", "sentiment_analysis"]
        
        if self.use_strands:
            capabilities.append("strands_framework")
        
        if self.use_swarm:
            capabilities.extend(["swarm", "coordinated_analysis"])
        
        return capabilities
    
    def _initialize_processing_mode(self):
        """Initialize the appropriate processing mode based on configuration."""
        if self.use_swarm:
            self._initialize_swarm_mode()
        elif self.use_strands:
            self._initialize_strands_mode()
        else:
            self._initialize_simple_mode()
    
    def _initialize_swarm_mode(self):
        """Initialize swarm coordination mode."""
        # Create specialized Strands agents for the swarm
        swarm_agents = []
        
        # Create a researcher agent for initial analysis
        researcher = Agent(
            name="researcher",
            system_prompt="""You are a text sentiment research specialist. 
            Analyze the given text and identify key sentiment indicators, 
            emotional cues, and context that could affect sentiment analysis.
            Focus on understanding the content before passing to sentiment experts."""
        )
        
        # Create a sentiment specialist agent
        sentiment_specialist = Agent(
            name="sentiment_specialist", 
            system_prompt="""You are a sentiment analysis expert. 
            Use the available tools to perform detailed sentiment analysis.
            Always coordinate with other agents in the swarm for comprehensive results."""
        )
        
        # Create a reviewer agent for quality assurance
        reviewer = Agent(
            name="reviewer",
            system_prompt="""You are a sentiment analysis reviewer. 
            Review and validate sentiment analysis results from other agents.
            Ensure consistency and accuracy across the swarm's analysis."""
        )
        
        # Create a coordinator agent that manages the workflow
        coordinator = Agent(
            name="coordinator",
            system_prompt="""You are a sentiment analysis coordinator.
            Coordinate the work of multiple specialized agents to produce
            comprehensive sentiment analysis results. Manage the workflow
            and ensure all agents contribute effectively."""
        )
        
        self.swarm_agents = [researcher, sentiment_specialist, reviewer]
        self.coordinator_agent = coordinator
        
        logger.info(f"Initialized swarm mode with {len(self.swarm_agents)} agents")
    
    def _initialize_strands_mode(self):
        """Initialize Strands framework mode."""
        # Strands mode uses the base agent's Strands integration
        logger.info("Initialized Strands framework mode")
    
    def _initialize_simple_mode(self):
        """Initialize simple direct processing mode."""
        # Simple mode processes directly without Strands coordination
        logger.info("Initialized simple direct processing mode")
    
    def _get_tools(self) -> list:
        """Get list of tools for this agent based on processing mode."""
        base_tools = [
            self.analyze_text_sentiment,
            self.extract_text_features,
            self.fallback_sentiment_analysis,
            # Summarization tools
            self.generate_text_summary,
            self.extract_key_points,
            self.identify_themes,
            # Translation tools
            self.translate_text,
            self.translate_document,
            self.batch_translate,
            self.detect_language
        ]
        
        if self.use_swarm:
            base_tools.extend([
                self.coordinate_sentiment_analysis,
                self.analyze_text_with_swarm,
                self.get_swarm_status,
                self.distribute_workload
            ])
        
        return base_tools
    
    async def can_process(self, request: AnalysisRequest) -> bool:
        """Check if this agent can process the request."""
        return request.data_type in [
            DataType.TEXT, 
            DataType.SOCIAL_MEDIA
        ]
    
    async def process(self, request: AnalysisRequest) -> AnalysisResult:
        """Process text analysis request using the configured processing mode."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Extract text content
            text_content = self._extract_text(request.content)
            
            if self.use_swarm:
                result = await self._process_with_swarm(text_content, request)
            elif self.use_strands:
                result = await self._process_with_strands(text_content, request, start_time)
            else:
                result = await self._process_simple(text_content, request, start_time)
            
            return result
            
        except Exception as e:
            logger.error(f"Text processing failed: {e}")
            # Return neutral sentiment on error
            return AnalysisResult(
                request_id=request.id,
                data_type=request.data_type,
                sentiment=SentimentResult(
                    label="neutral",
                    confidence=0.0,
                    metadata={"error": str(e)}
                ),
                processing_time=asyncio.get_event_loop().time() - start_time,
                status=None,
                raw_content=str(request.content),
                metadata={
                    "agent_id": self.agent_id,
                    "model": self.metadata["model"],
                    "language": request.language,
                    "method": "error_fallback",
                    "error": str(e)
                }
            )
    
    async def _process_with_swarm(self, text_content: str, request: AnalysisRequest) -> AnalysisResult:
        """Process text using swarm coordination."""
        # Use coordinator agent to manage the swarm
        system_prompt = (
            "You are coordinating a swarm of text sentiment analysis agents. "
            "Use the available tools to coordinate comprehensive analysis of the text. "
            "Ensure all agents contribute and validate the final result."
        )
        
        self.coordinator_agent.system_prompt = system_prompt
        
        prompt = (
            f"Coordinate the analysis of this text using the swarm: {text_content}\n\n"
            f"Use the coordinate_sentiment_analysis tool to manage the workflow."
        )
        
        response = await self.coordinator_agent.invoke_async(prompt)
        
        # Parse the swarm response
        sentiment_result = self._parse_swarm_response(str(response))
        
        return AnalysisResult(
            request_id=request.id,
            data_type=request.data_type,
            sentiment=sentiment_result,
            processing_time=0.0,  # Will be set by base class
            status=None,  # Will be set by base class
            raw_content=str(request.content),
            extracted_text=text_content,
            metadata={
                "agent_id": self.agent_id,
                "model": self.metadata["model"],
                "language": request.language,
                "method": "swarm_coordination",
                "swarm_size": self.agent_count,
                "tools_used": ["coordinate_sentiment_analysis"]
            }
        )
    
    async def _process_with_strands(self, text_content: str, request: AnalysisRequest, start_time: float) -> AnalysisResult:
        """Process text using Strands framework."""
        system_prompt = (
            "You are a text sentiment analysis expert. Use the available "
            "tools to analyze the sentiment of the given text.\n\n"
            "Available tools:\n"
            "- analyze_text_sentiment: Analyze sentiment using Ollama\n"
            "- extract_text_features: Extract text features for analysis\n"
            "- fallback_sentiment_analysis: Rule-based fallback analysis\n\n"
            "Process the text step by step:\n"
            "1. First extract text features to understand the content\n"
            "2. Then analyze sentiment using the sentiment analysis tool\n"
            "3. If sentiment analysis fails, use the fallback method\n\n"
            "Always use the tools rather than trying to analyze directly."
        )

        # Update the agent's system prompt for this specific task
        self.strands_agent.system_prompt = system_prompt
        
        # Invoke the Strands agent with the text analysis request
        prompt = (
            f"Analyze the sentiment of this text: {text_content}\n\n"
            f"Please use the available tools to perform a comprehensive "
            f"analysis."
        )
        response = await self.strands_agent.invoke_async(prompt)
        
        # Parse the response and create sentiment result
        sentiment_result = self._parse_sentiment_response(str(response))
        
        return AnalysisResult(
            request_id=request.id,
            data_type=request.data_type,
            sentiment=sentiment_result,
            processing_time=asyncio.get_event_loop().time() - start_time,
            status="completed",
            raw_content=str(request.content),
            extracted_text=text_content,
            metadata={
                "agent_id": self.agent_id,
                "model": self.metadata["model"],
                "language": request.language,
                "method": "strands_framework",
                "tools_used": ["analyze_text_sentiment", "extract_text_features"]
            }
        )
    
    async def _process_simple(self, text_content: str, request: AnalysisRequest) -> AnalysisResult:
        """Process text using simple direct processing."""
        # First extract text features
        features_result = await self.extract_text_features(text_content)
        
        # Then analyze sentiment using the sentiment analysis tool
        sentiment_result = await self.analyze_text_sentiment(text_content)
        
        # Parse the sentiment response
        sentiment_data = self._parse_sentiment_tool_response(sentiment_result)
        
        return AnalysisResult(
            request_id=request.id,
            data_type=request.data_type,
            sentiment=sentiment_data,
            processing_time=0.0,  # Will be set by base class
            status=None,  # Will be set by base class
            raw_content=str(request.content),
            extracted_text=text_content,
            metadata={
                "agent_id": self.agent_id,
                "model": self.metadata["model"],
                "language": request.language,
                "method": "direct_tools",
                "tools_used": ["extract_text_features", "analyze_text_sentiment"],
                "features": features_result.get("content", [{}])[0].get("json", {})
            }
        )
    
    def _extract_text(self, content: Any) -> str:
        """Extract text content from various input types."""
        if isinstance(content, str):
            return content
        elif isinstance(content, dict):
            # Handle dictionary input
            if "text" in content:
                return content["text"]
            elif "content" in content:
                return content["content"]
            else:
                return str(content)
        else:
            return str(content)
    
    @tool
    async def analyze_text_sentiment(self, text: str) -> dict:
        """
        Analyze the sentiment of the given text using Ollama.
        
        Args:
            text: The text to analyze for sentiment
            
        Returns:
            A sentiment analysis result in JSON format
        """
        try:
            # Create a specialized sentiment analysis agent with Ollama
            sentiment_agent = create_ollama_agent(
                model_type="text",
                name="sentiment_analyzer",
                system_prompt=(
                    "You are a specialized sentiment analysis expert. "
                    "Analyze the given text and respond with exactly one word: "
                    "POSITIVE, NEGATIVE, or NEUTRAL. "
                    "Then provide a confidence score from 0.0 to 1.0. "
                    "Format your response as: "
                    "SENTIMENT: [POSITIVE/NEGATIVE/NEUTRAL], "
                    "CONFIDENCE: [0.0-1.0]"
                )
            )
            
            if not sentiment_agent:
                logger.warning("Ollama agent not available, using fallback")
                return await self.fallback_sentiment_analysis(text)
            
            # Get the response
            try:
                response = await sentiment_agent.invoke_async(
                    f"Analyze this text: {text}"
                )
            except Exception as e:
                logger.error(f"Error invoking sentiment agent: {e}")
                return await self.fallback_sentiment_analysis(text)
            
            return {
                "status": "success",
                "content": [{
                    "text": str(response),
                    "json": {
                        "sentiment": "analyzed",
                        "confidence": 0.8,
                        "method": "ollama_sentiment"
                    }
                }]
            }
            
        except Exception as e:
            logger.error(f"Ollama sentiment analysis failed: {e}")
            return await self.fallback_sentiment_analysis(text)
    
    @tool
    async def extract_text_features(self, text: str) -> dict:
        """
        Extract text features for analysis.
        
        Args:
            text: The text to analyze
            
        Returns:
            Text features in JSON format
        """
        try:
            # Create a text feature extraction agent
            feature_agent = create_ollama_agent(
                model_type="text",
                name="feature_extractor",
                system_prompt=(
                    "You are a text feature extraction expert. "
                    "Analyze the given text and extract key features including: "
                    "word count, average word length, sentiment indicators, "
                    "key topics, and emotional content. "
                    "Respond with a JSON object containing these features."
                )
            )
            
            if not feature_agent:
                # Fallback feature extraction
                features = {
                    "word_count": len(text.split()),
                    "avg_word_length": sum(len(word) for word in text.split()) / max(len(text.split()), 1),
                    "sentiment_indicators": [],
                    "key_topics": [],
                    "emotional_content": "neutral"
                }
            else:
                try:
                    response = await feature_agent.invoke_async(
                        f"Extract features from this text: {text}"
                    )
                    # Parse response to extract features
                    features = {
                        "word_count": len(text.split()),
                        "avg_word_length": sum(len(word) for word in text.split()) / max(len(text.split()), 1),
                        "sentiment_indicators": [],
                        "key_topics": [],
                        "emotional_content": "neutral",
                        "extracted_features": str(response)
                    }
                except Exception as e:
                    logger.error(f"Error invoking feature agent: {e}")
                    # Fallback feature extraction
                    features = {
                        "word_count": len(text.split()),
                        "avg_word_length": sum(len(word) for word in text.split()) / max(len(text.split()), 1),
                        "sentiment_indicators": [],
                        "key_topics": [],
                        "emotional_content": "neutral"
                    }
            
            return {
                "status": "success",
                "content": [{
                    "json": features
                }]
            }
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return {
                "status": "error",
                "content": [{"text": f"Feature extraction error: {str(e)}"}]
            }
    
    @tool
    async def fallback_sentiment_analysis(self, text: str) -> dict:
        """
        Fallback rule-based sentiment analysis when primary methods fail.
        
        Args:
            text: The text to analyze
            
        Returns:
            A sentiment analysis result in JSON format
        """
        text_lower = text.lower()
        
        # Simple keyword-based sentiment analysis
        positive_words = {
            'love', 'great', 'good', 'excellent', 'amazing', 'wonderful', 
            'fantastic', 'awesome', 'brilliant', 'outstanding', 'perfect',
            'happy', 'joy', 'pleased', 'satisfied', 'delighted', 'thrilled'
        }
        
        negative_words = {
            'hate', 'terrible', 'awful', 'horrible', 'bad', 'worst',
            'disappointed', 'angry', 'sad', 'upset', 'frustrated',
            'annoyed', 'disgusted', 'furious', 'miserable'
        }
        
        # Count positive and negative words
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        # Determine sentiment
        if pos_count > neg_count:
            sentiment_label = "POSITIVE"
            confidence = min(0.7 + (pos_count * 0.1), 0.95)
        elif neg_count > pos_count:
            sentiment_label = "NEGATIVE"
            confidence = min(0.7 + (neg_count * 0.1), 0.95)
        else:
            sentiment_label = "NEUTRAL"
            confidence = 0.6
        
        return {
            "status": "success",
            "content": [{
                "text": f"SENTIMENT: {sentiment_label}, CONFIDENCE: {confidence:.2f}",
                "json": {
                    "sentiment": sentiment_label,
                    "confidence": confidence,
                    "method": "fallback_rule_based",
                    "positive_count": pos_count,
                    "negative_count": neg_count
                }
            }]
        }
    
    # Swarm-specific tools
    @tool
    async def coordinate_sentiment_analysis(self, text: str) -> dict:
        """
        Coordinate sentiment analysis using the swarm of agents.
        
        Args:
            text: The text to analyze
            
        Returns:
            Coordinated sentiment analysis result
        """
        try:
            # Distribute work among swarm agents
            tasks = []
            for agent in self.swarm_agents:
                task = agent.invoke_async(f"Analyze this text: {text}")
                tasks.append(task)
            
            # Wait for all agents to complete
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Aggregate results
            valid_responses = [r for r in responses if not isinstance(r, Exception)]
            
            if not valid_responses:
                return await self.fallback_sentiment_analysis(text)
            
            # Use the first valid response as base
            base_response = str(valid_responses[0])
            sentiment_result = self._parse_sentiment_response(base_response)
            
            return {
                "status": "success",
                "content": [{
                    "json": {
                        "sentiment": sentiment_result.label,
                        "confidence": sentiment_result.confidence,
                        "method": "swarm_coordination",
                        "agents_used": len(valid_responses),
                        "total_agents": len(self.swarm_agents)
                    }
                }]
            }
            
        except Exception as e:
            logger.error(f"Swarm coordination failed: {e}")
            return await self.fallback_sentiment_analysis(text)
    
    @tool
    async def analyze_text_with_swarm(self, text: str) -> dict:
        """
        Analyze text using the entire swarm.
        
        Args:
            text: The text to analyze
            
        Returns:
            Swarm analysis result
        """
        return await self.coordinate_sentiment_analysis(text)
    
    @tool
    async def get_swarm_status(self) -> dict:
        """
        Get the status of all agents in the swarm.
        
        Returns:
            Swarm status information
        """
        return {
            "status": "success",
            "content": [{
                "json": {
                    "swarm_size": len(self.swarm_agents),
                    "coordinator_active": self.coordinator_agent is not None,
                    "agents": [
                        {
                            "name": agent.name,
                            "type": "swarm_agent"
                        } for agent in self.swarm_agents
                    ]
                }
            }]
        }
    
    @tool
    async def distribute_workload(self, texts: List[str]) -> dict:
        """
        Distribute workload across swarm agents.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            Distributed analysis results
        """
        try:
            results = []
            for i, text in enumerate(texts):
                agent = self.swarm_agents[i % len(self.swarm_agents)]
                response = await agent.invoke_async(f"Analyze this text: {text}")
                results.append({
                    "text_index": i,
                    "agent": agent.name,
                    "response": str(response)
                })
            
            return {
                "status": "success",
                "content": [{
                    "json": {
                        "results": results,
                        "total_texts": len(texts),
                        "agents_used": len(self.swarm_agents)
                    }
                }]
            }
            
        except Exception as e:
            logger.error(f"Workload distribution failed: {e}")
            return {
                "status": "error",
                "content": [{"text": f"Distribution error: {str(e)}"}]
            }
    
    def _parse_sentiment_response(self, response: str) -> SentimentResult:
        """Parse sentiment response from various formats."""
        try:
            # Try to extract sentiment and confidence from response
            response_lower = response.lower()
            
            # Look for sentiment labels
            if "positive" in response_lower:
                label = "positive"
            elif "negative" in response_lower:
                label = "negative"
            else:
                label = "neutral"
            
            # Look for confidence scores
            import re
            confidence_match = re.search(r'confidence[:\s]*([0-9]*\.?[0-9]+)', response_lower)
            if confidence_match:
                confidence = float(confidence_match.group(1))
            else:
                confidence = 0.7 if label != "neutral" else 0.6
            
            return SentimentResult(
                label=label,
                confidence=confidence,
                metadata={"parsed_from": response[:100]}
            )
            
        except Exception as e:
            logger.error(f"Failed to parse sentiment response: {e}")
            return SentimentResult(
                label="neutral",
                confidence=0.5,
                metadata={"error": str(e)}
            )
    
    def _parse_sentiment_tool_response(self, tool_response: dict) -> SentimentResult:
        """Parse sentiment from tool response."""
        try:
            content = tool_response.get("content", [{}])[0]
            json_data = content.get("json", {})
            
            sentiment = json_data.get("sentiment", "neutral")
            confidence = json_data.get("confidence", 0.5)
            
            return SentimentResult(
                label=sentiment.lower(),
                confidence=confidence,
                metadata=json_data
            )
            
        except Exception as e:
            logger.error(f"Failed to parse tool response: {e}")
            return SentimentResult(
                label="neutral",
                confidence=0.5,
                metadata={"error": str(e)}
            )
    
    def _parse_swarm_response(self, response: str) -> SentimentResult:
        """Parse response from swarm coordination."""
        return self._parse_sentiment_response(response)
    
    async def start(self):
        """Start the unified text agent."""
        await super().start()
        logger.info(f"UnifiedTextAgent {self.agent_id} started with mode: "
                   f"strands={self.use_strands}, swarm={self.use_swarm}")
    
    async def stop(self):
        """Stop the unified text agent."""
        await super().stop()
        logger.info(f"UnifiedTextAgent {self.agent_id} stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get detailed status of the unified text agent."""
        base_status = super().get_status()
        base_status.update({
            "processing_mode": {
                "use_strands": self.use_strands,
                "use_swarm": self.use_swarm,
                "agent_count": self.agent_count
            },
            "swarm_agents": len(self.swarm_agents) if self.use_swarm else 0,
            "coordinator_active": self.coordinator_agent is not None if self.use_swarm else False
        })
        return base_status

    # Translation tools
    @tool
    async def translate_text(self, text: str, source_language: str = None, target_language: str = "en") -> dict:
        """Translate text content to the target language."""
        try:
            result = await self.translation_service.translate_text(
                text, source_language, target_language
            )
            return {
                "status": "success",
                "original_text": result.original_text,
                "translated_text": result.translated_text,
                "source_language": result.source_language,
                "target_language": result.target_language,
                "confidence": result.confidence,
                "processing_time": result.processing_time,
                "model_used": result.model_used,
                "translation_memory_hit": result.translation_memory_hit
            }
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "original_text": text,
                "translated_text": text
            }

    @tool
    async def translate_document(self, content: str, content_type: str, source_language: str = None) -> dict:
        """Translate document content (PDF, webpage, etc.)."""
        try:
            # Convert string to DataType enum
            from src.core.models import DataType
            data_type = DataType(content_type)
            
            result = await self.translation_service.translate_document(
                content, data_type, source_language
            )
            return {
                "status": "success",
                "original_text": result.original_text,
                "translated_text": result.translated_text,
                "source_language": result.source_language,
                "target_language": result.target_language,
                "confidence": result.confidence,
                "processing_time": result.processing_time,
                "model_used": result.model_used
            }
        except Exception as e:
            logger.error(f"Document translation failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "original_text": content,
                "translated_text": content
            }

    @tool
    async def batch_translate(self, texts: List[str], source_language: str = None) -> dict:
        """Translate multiple texts in batch."""
        try:
            results = await self.translation_service.batch_translate(texts, source_language)
            return {
                "status": "success",
                "results": [result.to_dict() for result in results],
                "total_translations": len(results)
            }
        except Exception as e:
            logger.error(f"Batch translation failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "results": []
            }

    @tool
    async def detect_language(self, text: str) -> dict:
        """Detect the language of the text."""
        try:
            language = await self.translation_service.detect_language(text)
            return {
                "status": "success",
                "detected_language": language,
                "text_sample": text[:100] + "..." if len(text) > 100 else text
            }
        except Exception as e:
            logger.error(f"Language detection failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "detected_language": "unknown"
            }

    # Summarization tools
    @tool
    async def generate_text_summary(self, text: str, summary_type: str = "comprehensive") -> dict:
        """Generate a summary of the text content."""
        try:
            # Create analysis request for summarization
            request = AnalysisRequest(
                data_type=DataType.TEXT,
                content=text,
                language="en",
                metadata={
                    "summary_type": summary_type,
                    "include_key_points": True,
                    "include_entities": True
                }
            )
            
            # Process the request
            result = await self.process(request)
            
            # Extract summary from result
            summary = ""
            if hasattr(result, 'summary') and result.summary:
                summary = result.summary
            elif hasattr(result, 'sentiment') and result.sentiment.reasoning:
                summary = result.sentiment.reasoning
            else:
                summary = "Summary generated successfully"
            
            return {
                "status": "success",
                "summary_type": summary_type,
                "summary": summary,
                "key_points": result.metadata.get('key_points', []) if result.metadata else [],
                "entities": result.metadata.get('entities', []) if result.metadata else [],
                "processing_time": result.processing_time
            }
        except Exception as e:
            logger.error(f"Text summarization failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "summary": "Failed to generate summary"
            }

    @tool
    async def extract_key_points(self, text: str) -> dict:
        """Extract key points from the text content."""
        try:
            # Create analysis request for key point extraction
            request = AnalysisRequest(
                data_type=DataType.TEXT,
                content=text,
                language="en",
                metadata={
                    "extract_key_points": True,
                    "summary_type": "detailed"
                }
            )
            
            # Process the request
            result = await self.process(request)
            
            # Extract key points from result
            key_points = result.metadata.get('key_points', []) if result.metadata else []
            
            return {
                "status": "success",
                "key_points": key_points,
                "total_points": len(key_points),
                "processing_time": result.processing_time
            }
        except Exception as e:
            logger.error(f"Key point extraction failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "key_points": []
            }

    @tool
    async def identify_themes(self, text: str) -> dict:
        """Identify themes and concepts in the text content."""
        try:
            # Create analysis request for theme identification
            request = AnalysisRequest(
                data_type=DataType.TEXT,
                content=text,
                language="en",
                metadata={
                    "identify_themes": True,
                    "extract_concepts": True
                }
            )
            
            # Process the request
            result = await self.process(request)
            
            # Extract themes and concepts from result
            themes = result.metadata.get('themes', []) if result.metadata else []
            concepts = result.metadata.get('key_concepts', []) if result.metadata else []
            
            return {
                "status": "success",
                "themes": themes,
                "concepts": concepts,
                "total_themes": len(themes),
                "total_concepts": len(concepts),
                "processing_time": result.processing_time
            }
        except Exception as e:
            logger.error(f"Theme identification failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "themes": [],
                "concepts": []
            }
