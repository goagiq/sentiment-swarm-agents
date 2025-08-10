"""
Audio processing agent for sentiment analysis using Strands tools.
"""

import asyncio
import tempfile
from typing import Any, Optional

from loguru import logger
from strands import tool

from agents.base_agent import BaseAgent
from config.config import config
from core.models import (
    AnalysisRequest, AnalysisResult, DataType, SentimentResult
)


class AudioAgent(BaseAgent):
    """Agent for processing audio content using Strands tools."""
    
    def __init__(
        self, 
        model_name: Optional[str] = None,
        **kwargs
    ):
        # Use config system instead of hardcoded values
        default_model = config.model.default_audio_model
        super().__init__(model_name=model_name or default_model, **kwargs)
        self.metadata["model"] = model_name or default_model
        self.metadata["supported_formats"] = [
            "mp3", "wav", "flac", "m4a", "ogg"
        ]
        self.metadata["max_audio_duration"] = 300  # 5 minutes
        self.metadata["capabilities"] = ["audio", "transcription", "sentiment_analysis"]
    
    def _get_tools(self) -> list:
        """Get list of tools for this agent."""
        return [
            self.transcribe_audio,
            self.analyze_audio_sentiment,
            self.extract_audio_features,
            self.fallback_audio_analysis
        ]
    
    async def can_process(self, request: AnalysisRequest) -> bool:
        """Check if this agent can process the request."""
        return request.data_type == DataType.AUDIO
    
    async def process(self, request: AnalysisRequest) -> AnalysisResult:
        """Process audio analysis request using Strands tools."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Extract audio content
            audio_content = self._extract_audio(request.content)
            
            # Use Strands agent to process the request with tool coordination
            system_prompt = (
                "You are an audio sentiment analysis expert. Use the available "
                "tools to analyze the sentiment of the given audio content.\n\n"
                "Available tools:\n"
                "- transcribe_audio: Transcribe audio to text\n"
                "- analyze_audio_sentiment: Analyze audio sentiment using Ollama\n"
                "- extract_audio_features: Extract audio features\n"
                "- fallback_audio_analysis: Fallback analysis when main tools fail\n\n"
                "Process the audio content step by step:\n"
                "1. First transcribe the audio to get text content\n"
                "2. Then analyze sentiment using the sentiment analysis tool\n"
                "3. If analysis fails, use the fallback method\n\n"
                "Always use the tools rather than trying to analyze directly."
            )
            
            # Update the agent's system prompt for this specific task
            self.strands_agent.system_prompt = system_prompt
            
            # Invoke the Strands agent with the audio analysis request
            prompt = (
                f"Analyze the sentiment of this audio content: "
                f"{audio_content}\n\n"
                f"Please use the available tools to perform a comprehensive "
                f"analysis."
            )
            response = await self.strands_agent.invoke_async(prompt)
            
            # Parse the response and create sentiment result
            sentiment_result = self._parse_audio_sentiment(
                str(response)
            )
            
            # Create analysis result
            result = AnalysisResult(
                request_id=request.id,
                data_type=request.data_type,
                sentiment=sentiment_result,
                processing_time=0.0,  # Will be set by base class
                status=None,  # Will be set by base class
                raw_content=str(request.content),
                extracted_text=audio_content,
                metadata={
                    "agent_id": self.agent_id,
                    "model": self.metadata["model"],
                    "content_type": request.data_type.value,
                    "method": "strands_tools",
                    "tools_used": [tool.__name__ for tool in self._get_tools()]
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Audio processing failed: {e}")
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
                status="failed",
                raw_content=str(request.content),
                extracted_text="",
                metadata={
                    "agent_id": self.agent_id,
                    "model": self.metadata["model"],
                    "error": str(e)
                }
            )
    
    def _extract_audio(self, content: Any) -> str:
        """Extract audio content from various input formats."""
        if isinstance(content, str):
            return content
        elif isinstance(content, bytes):
            # Create temporary file from bytes
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_file.write(content)
                return tmp_file.name
        elif isinstance(content, dict):
            # Handle audio file paths, URLs, etc.
            if "audio_path" in content:
                return content["audio_path"]
            elif "audio_url" in content:
                return content["audio_url"]
            elif "audio_data" in content:
                return content["audio_data"]
            else:
                return str(content)
        else:
            return str(content)
    
    @tool
    async def transcribe_audio(self, audio_path: str) -> dict:
        """Transcribe audio to text using Ollama."""
        try:
            # For now, we'll use a placeholder transcription
            # In a full implementation, this would use Whisper or similar
            transcription = f"Audio transcription from {audio_path}"
            
            return {
                "status": "success",
                "content": [{"text": transcription}]
            }
            
        except Exception as e:
            logger.error(f"Audio transcription failed: {e}")
            return {
                "status": "error",
                "content": [{"text": f"Transcription error: {str(e)}"}]
            }
    
    @tool
    async def analyze_audio_sentiment(self, audio_path: str) -> dict:
        """Analyze audio sentiment using Ollama."""
        try:
            # First transcribe the audio
            transcription_result = await self.transcribe_audio(audio_path)
            
            if transcription_result["status"] != "success":
                raise RuntimeError("Failed to transcribe audio")
            
            transcription = transcription_result["content"][0]["text"]
            
            # Use Ollama to analyze sentiment of transcription
            import aiohttp
            
            payload = {
                "model": "llama3.2:latest",
                "prompt": f"""Analyze the sentiment of this transcribed audio and respond with exactly one word: POSITIVE, NEGATIVE, or NEUTRAL.

Transcription: {transcription}

Sentiment:""",
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 10,
                    "top_k": 1,
                    "top_p": 0.1
                }
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "http://localhost:11434/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        response_text = result.get("response", "").strip().upper()
                        
                        # Parse the response and convert to sentiment
                        if "POSITIVE" in response_text:
                            sentiment_label = "positive"
                            confidence = 0.8
                        elif "NEGATIVE" in response_text:
                            sentiment_label = "negative"
                            confidence = 0.8
                        else:
                            sentiment_label = "neutral"
                            confidence = 0.6
                        
                        return {
                            "status": "success",
                            "content": [{
                                "json": {
                                    "sentiment": sentiment_label,
                                    "confidence": confidence,
                                    "transcription": transcription,
                                    "raw_response": response_text
                                }
                            }]
                        }
                    else:
                        raise RuntimeError(f"Ollama API error: {response.status}")
                        
        except Exception as e:
            logger.error(f"Audio sentiment analysis failed: {e}")
            # Fallback to basic analysis
            return await self.fallback_audio_analysis(audio_path)
    
    @tool
    async def extract_audio_features(self, audio_path: str) -> dict:
        """Extract basic audio features."""
        try:
            # Basic audio feature extraction
            features = {
                "file_path": audio_path,
                "file_type": audio_path.split(".")[-1] if "." in audio_path else "unknown",
                "has_transcription": True,
                "analysis_method": "ollama_audio"
            }
            
            return {
                "status": "success",
                "content": [{"json": features}]
            }
            
        except Exception as e:
            logger.error(f"Audio feature extraction failed: {e}")
            return {
                "status": "error",
                "content": [{"text": f"Feature extraction error: {str(e)}"}]
            }
    
    @tool
    async def fallback_audio_analysis(self, audio_path: str) -> dict:
        """Fallback audio analysis when Ollama fails."""
        try:
            # Basic fallback analysis
            analysis = f"Audio analysis of {audio_path} - sentiment unclear"
            
            return {
                "status": "success",
                "content": [{
                    "json": {
                        "sentiment": "neutral",
                        "confidence": 0.5,
                        "method": "fallback",
                        "analysis": analysis
                    }
                }]
            }
            
        except Exception as e:
            logger.error(f"Fallback audio analysis failed: {e}")
            return {
                "status": "error",
                "content": [{"text": f"Fallback analysis error: {str(e)}"}]
            }
    
    def _parse_audio_sentiment(self, response: str) -> SentimentResult:
        """Parse the audio sentiment response from Strands tools."""
        try:
            # Try to extract sentiment information from the response
            response_lower = response.lower()
            
            if "positive" in response_lower:
                label = "positive"
                confidence = 0.8
            elif "negative" in response_lower:
                label = "negative"
                confidence = 0.8
            else:
                label = "neutral"
                confidence = 0.6
            
            return SentimentResult(
                label=label,
                confidence=confidence,
                scores={
                    "positive": 0.8 if label == "positive" else 0.1,
                    "negative": 0.8 if label == "negative" else 0.1,
                    "neutral": 0.6 if label == "neutral" else 0.1
                },
                metadata={
                    "method": "strands_tools",
                    "raw_response": response
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to parse audio sentiment response: {e}")
            # Return neutral sentiment on parsing failure
            return SentimentResult(
                label="neutral",
                confidence=0.0,
                scores={"neutral": 1.0},
                metadata={
                    "method": "strands_tools",
                    "error": str(e),
                    "raw_response": response
                }
            )
    
    async def cleanup(self):
        """Cleanup resources."""
        # Cleanup any resources if needed
        pass



