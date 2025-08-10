"""
Vision processing agent for sentiment analysis of images and videos using 
Strands tools with proper Ollama integration.
"""

import asyncio
import base64
from typing import Any, Optional, List
from pathlib import Path

from PIL import Image
from transformers import pipeline
from loguru import logger
from strands import tool

from agents.base_agent import BaseAgent
from config.config import config
from core.models import (
    AnalysisRequest, AnalysisResult, DataType, SentimentResult
)
from core.ollama_integration import (
    get_ollama_model
)
from core.youtube_dl_service import YouTubeDLService, VideoInfo, AudioInfo


class VisionAgent(BaseAgent):
    """Agent for processing image and video content using Strands tools with 
    Ollama."""
    
    def __init__(
        self, 
        model_name: Optional[str] = None, 
        **kwargs
    ):
        # Use config system instead of hardcoded values
        default_model = config.model.default_vision_model
        super().__init__(model_name=model_name or default_model, **kwargs)
        
        # Initialize Ollama vision model
        self.ollama_model = None
        self.sentiment_pipeline = None
        
        # Initialize YouTube-DL service
        self.youtube_dl_service = YouTubeDLService(
            download_path=config.youtube_dl.download_path
        )
        
        self.metadata["model"] = model_name or default_model
        self.metadata["supported_formats"] = [
            "jpg", "jpeg", "png", "bmp", "mp4", "avi", "mov"
        ]
        self.metadata["max_image_size"] = 1024
        self.metadata["max_video_duration"] = 30  # seconds
        self.metadata["model_type"] = "ollama"
        self.metadata["capabilities"] = ["vision", "tool_calling", "youtube_dl"]
    
    def _get_tools(self) -> list:
        """Get list of tools for this agent."""
        return [
            self.analyze_image_sentiment,
            self.process_video_frame,
            self.extract_vision_features,
            self.fallback_vision_analysis,
            self.download_video_frames,
            self.analyze_video_sentiment,
            self.get_video_metadata
        ]
    
    async def can_process(self, request: AnalysisRequest) -> bool:
        """Check if this agent can process the request."""
        return request.data_type in [DataType.IMAGE, DataType.VIDEO]
    
    async def process(self, request: AnalysisRequest) -> AnalysisResult:
        """Process vision analysis request using Strands tools with Ollama."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Initialize models if not already done
            if self.ollama_model is None or self.sentiment_pipeline is None:
                await self._initialize_models()
            
            # Extract and process content
            if request.data_type == DataType.IMAGE:
                content = await self._process_image(request.content)
            elif request.data_type == DataType.VIDEO:
                content = await self._process_video(request.content)
            else:
                raise ValueError(f"Unsupported data type: {request.data_type}")
            
            # Use Strands agent to process the request with tool coordination
            system_prompt = (
                "You are a vision sentiment analysis expert. Use the available "
                "tools to analyze the sentiment of the given vision content.\n\n"
                "Available tools:\n"
                "- analyze_image_sentiment: Analyze image sentiment using Ollama\n"
                "- process_video_frame: Process video frames for analysis\n"
                "- extract_vision_features: Extract visual features\n"
                "- fallback_vision_analysis: Fallback analysis when main tools fail\n"
                "- download_video_frames: Download video and extract key frames\n"
                "- analyze_video_sentiment: Analyze video sentiment from URL\n"
                "- get_video_metadata: Get metadata for video URLs\n\n"
                "Process the vision content step by step:\n"
                "1. For video URLs, first get metadata to understand the content\n"
                "2. Download and extract frames for video analysis\n"
                "3. Analyze each frame for sentiment\n"
                "4. For images, extract features then analyze sentiment\n"
                "5. If analysis fails, use the fallback method\n\n"
                "Always use the tools rather than trying to analyze directly."
            )
            
            # Update the agent's system prompt for this specific task
            self.strands_agent.system_prompt = system_prompt
            
            # Invoke the Strands agent with the vision analysis request
            prompt = (
                f"Analyze the sentiment of this vision content: {content}\n\n"
                f"Please use the available tools to perform a comprehensive "
                f"analysis."
            )
            response = await self.strands_agent.invoke_async(prompt)
            
            # Parse the response and create sentiment result
            sentiment_result = self._parse_vision_sentiment(str(response))
            
            # Create analysis result
            result = AnalysisResult(
                request_id=request.id,
                data_type=request.data_type,
                sentiment=sentiment_result,
                processing_time=0.0,  # Will be set by base class
                status=None,  # Will be set by base class
                raw_content=str(request.content),
                extracted_text=content,
                metadata={
                    "agent_id": self.agent_id,
                    "model": self.metadata["model"],
                    "content_type": request.data_type.value,
                    "method": "strands_ollama",
                    "tools_used": [tool.__name__ for tool in self._get_tools()]
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Vision processing failed: {e}")
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
                    "content_type": request.data_type.value,
                    "error": str(e)
                }
            )
    
    async def _initialize_models(self):
        """Initialize the Ollama vision model and fallback sentiment pipeline."""
        try:
            # Initialize Ollama vision model using the integration module
            self.ollama_model = get_ollama_model("vision")
            
            if self.ollama_model is None:
                logger.warning("Ollama vision model not available, using fallback only")
            
            # Initialize sentiment analysis pipeline for fallback
            loop = asyncio.get_event_loop()
            self.sentiment_pipeline = await loop.run_in_executor(
                None,
                pipeline,
                "sentiment-analysis",
                "cardiffnlp/twitter-roberta-base-sentiment-latest"
            )
            
            logger.info(
                f"VisionAgent {self.agent_id} initialized with Ollama integration"
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            raise
    
    async def _process_image(self, content: Any) -> str:
        """Process image content using Ollama vision model."""
        try:
            # Handle different input formats
            if isinstance(content, str):
                image_path = content
            elif isinstance(content, bytes):
                # Create temporary file from bytes
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
                    tmp_file.write(content)
                    image_path = tmp_file.name
            elif isinstance(content, dict) and "image_path" in content:
                image_path = content["image_path"]
            else:
                raise ValueError("Unsupported image content format")
            
            # Directly use Ollama vision model for image processing
            if self.ollama_model is None:
                logger.warning("Ollama model not available, using fallback")
                fallback_result = await self.fallback_vision_analysis(image_path)
                return fallback_result.get('content', [{}])[0].get('text', '')
            
            # Read and encode image
            image_data = Path(image_path).read_bytes()
            image_b64 = base64.b64encode(image_data).decode('utf-8')
            
            # Use the Ollama model directly for vision analysis
            prompt = (
                "Describe this image in detail, focusing on emotional "
                "content, mood, and any text or symbols that might "
                "indicate sentiment. Provide a comprehensive analysis "
                "that can be used for sentiment classification."
            )
            
            try:
                # Use the Ollama model directly for vision tasks
                response = await self.ollama_model.agenerate(
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url", 
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_b64}"
                                }
                            }
                        ]
                    }]
                )
                
                description = response.choices[0].message.content
                return description
                
            except Exception as ollama_error:
                logger.warning(
                    f"Ollama vision analysis failed: {ollama_error}, using fallback"
                )
                fallback_result = await self.fallback_vision_analysis(image_path)
                return fallback_result.get('content', [{}])[0].get('text', '')
                
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            raise
    
    async def _process_video(self, content: Any) -> str:
        """Process video content using Ollama vision model."""
        try:
            # Handle different input formats
            if isinstance(content, str):
                video_path = content
            elif isinstance(content, bytes):
                # Create temporary file from bytes
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
                    tmp_file.write(content)
                    video_path = tmp_file.name
            elif isinstance(content, dict) and "video_path" in content:
                video_path = content["video_path"]
            else:
                raise ValueError("Unsupported video content format")
            
            # For now, extract a frame and process as image
            # In a full implementation, we'd extract multiple frames
            frame_path = await self._extract_video_frame(video_path)
            
            # Process the frame using the same image processing logic
            return await self._process_image(frame_path)
            
        except Exception as e:
            logger.error(f"Error processing video: {e}")
            raise
    
    async def _extract_video_frame(self, video_path: str) -> str:
        """Extract a frame from video for analysis."""
        try:
            import cv2
            import tempfile
            
            # Open video and read first frame
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                raise ValueError("Could not read video frame")
            
            # Save frame to temporary file
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
                cv2.imwrite(tmp_file.name, frame)
                return tmp_file.name
                
        except ImportError:
            logger.warning("OpenCV not available, using placeholder for video processing")
            return "placeholder_frame.jpg"
        except Exception as e:
            logger.error(f"Error extracting video frame: {e}")
            raise
    
    @tool
    async def analyze_image_sentiment(self, image_path: str) -> dict:
        """Analyze image sentiment using Ollama vision model via Strands."""
        try:
            if self.ollama_model is None:
                logger.warning("Ollama model not available, using fallback")
                return await self.fallback_vision_analysis(image_path)
            
            # Read and encode image
            image_data = Path(image_path).read_bytes()
            image_b64 = base64.b64encode(image_data).decode('utf-8')
            
            # Use the Ollama model through Strands for vision analysis
            prompt = (
                "Describe this image in detail, focusing on emotional "
                "content, mood, and any text or symbols that might "
                "indicate sentiment. Provide a comprehensive analysis "
                "that can be used for sentiment classification."
            )
            
            # Create the request with image data
            # Note: This is a simplified approach - in a full implementation,
            # we'd use the proper Strands vision model interface
            try:
                # Use the Ollama model directly for vision tasks
                response = await self.ollama_model.agenerate(
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url", 
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_b64}"
                                }
                            }
                        ]
                    }]
                )
                
                description = response.choices[0].message.content
                
                return {
                    "status": "success",
                    "content": [{"text": description}]
                }
                
            except Exception as ollama_error:
                logger.warning(
                    f"Ollama vision analysis failed: {ollama_error}, using fallback"
                )
                return await self.fallback_vision_analysis(image_path)
        
        except Exception as e:
            logger.error(f"Error analyzing image sentiment: {e}")
            return {
                "status": "error",
                "content": [{"text": f"Error analyzing image: {str(e)}"}]
            }
    
    @tool
    async def process_video_frame(self, video_path: str) -> dict:
        """Process video frame for sentiment analysis."""
        try:
            # Extract frame and analyze
            frame_path = await self._extract_video_frame(video_path)
            image_result = await self.analyze_image_sentiment(frame_path)
            
            return {
                "status": "success",
                "content": [{
                    "text": f"Video frame analysis: {image_result.get('content', [{}])[0].get('text', '')}"
                }]
            }
            
        except Exception as e:
            logger.error(f"Error processing video frame: {e}")
            return {
                "status": "error",
                "content": [{"text": f"Error processing video frame: {str(e)}"}]
            }
    
    @tool
    async def extract_vision_features(self, image_path: str) -> dict:
        """Extract visual features from image."""
        try:
            # Load image and get basic information
            image = Image.open(image_path)
            width, height = image.size
            
            # Generate basic description
            features = {
                "dimensions": f"{width}x{height}",
                "mode": image.mode,
                "format": image.format,
                "size_bytes": len(Path(image_path).read_bytes())
            }
            
            # Add basic analysis based on image properties
            if image.mode == 'RGB':
                features["type"] = "color_image"
            elif image.mode == 'L':
                features["type"] = "grayscale_image"
            
            return {
                "status": "success",
                "content": [{"json": features}]
            }
            
        except Exception as e:
            logger.error(f"Error extracting vision features: {e}")
            return {
                "status": "error",
                "content": [{"text": f"Error extracting features: {str(e)}"}]
            }
    
    @tool
    async def fallback_vision_analysis(self, image_path: str) -> dict:
        """Fallback vision analysis when Ollama fails."""
        try:
            # Basic image processing fallback
            image = Image.open(image_path)
            width, height = image.size
            
            # Generate basic description
            description = f"Image with dimensions {width}x{height} pixels"
            
            # Add basic analysis based on image properties
            if image.mode == 'RGB':
                description += ", color image"
            elif image.mode == 'L':
                description += ", grayscale image"
            
            return {
                "status": "success",
                "content": [{"text": description}]
            }
            
        except Exception as e:
            logger.error(f"Error in fallback vision analysis: {e}")
            return {
                "status": "error",
                "content": [{"text": f"Error in fallback analysis: {str(e)}"}]
            }
    
    async def _analyze_vision_sentiment(self, description: str) -> SentimentResult:
        """Analyze sentiment of the vision description."""
        try:
            # Use sentiment pipeline to analyze the description
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self.sentiment_pipeline,
                description
            )
            
            # Extract sentiment information
            if isinstance(result, list) and len(result) > 0:
                sentiment_data = result[0]
                label = sentiment_data.get('label', 'neutral')
                score = sentiment_data.get('score', 0.5)
                
                # Map label to our enum
                if label == 'POSITIVE':
                    sentiment_label = "positive"
                elif label == 'NEGATIVE':
                    sentiment_label = "negative"
                else:
                    sentiment_label = "neutral"
                
                return SentimentResult(
                    label=sentiment_label,
                    confidence=score,
                    scores={label: score},
                    reasoning=f"Analysis based on vision description: {description[:100]}...",
                    context_notes="Vision-based sentiment analysis using Ollama and Strands",
                    iteration_count=1
                )
            else:
                # Fallback to neutral sentiment
                return SentimentResult(
                    label="neutral",
                    confidence=0.5,
                    scores={"neutral": 0.5},
                    reasoning="Unable to determine sentiment from vision analysis",
                    context_notes="Vision processing completed but sentiment unclear",
                    iteration_count=1
                )
                
        except Exception as e:
            logger.error(f"Error analyzing vision sentiment: {e}")
            # Return neutral sentiment on error
            return SentimentResult(
                label="neutral",
                confidence=0.0,
                scores={"neutral": 0.0},
                reasoning=f"Error in sentiment analysis: {str(e)}",
                context_notes="Vision processing error",
                iteration_count=1
            )
    
    def _parse_vision_sentiment(self, response: str) -> SentimentResult:
        """Parse the vision sentiment response from Strands tools."""
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
                    "method": "strands_ollama",
                    "raw_response": response
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to parse vision sentiment response: {e}")
            # Return neutral sentiment on parsing failure
            return SentimentResult(
                label="neutral",
                confidence=0.0,
                scores={"neutral": 1.0},
                metadata={
                    "method": "strands_ollama",
                    "error": str(e),
                    "raw_response": response
                }
            )
    
    @tool
    async def download_video_frames(self, video_url: str, num_frames: int = 10) -> dict:
        """Download video and extract key frames for analysis."""
        try:
            # Check if URL is supported
            if not self.youtube_dl_service.is_supported_platform(video_url):
                return {
                    "status": "error",
                    "content": [{
                        "text": f"Unsupported platform: {video_url}"
                    }]
                }
            
            # Get video metadata first
            metadata = await self.youtube_dl_service.get_metadata(video_url)
            
            # Check duration limits
            if metadata.duration > config.youtube_dl.max_video_duration:
                return {
                    "status": "error",
                    "content": [{
                        "text": f"Video too long: {metadata.duration}s > "
                               f"{config.youtube_dl.max_video_duration}s"
                    }]
                }
            
            # Download video
            video_info = await self.youtube_dl_service.download_video(video_url)
            
            if not video_info.video_path:
                return {
                    "status": "error",
                    "content": [{
                        "text": "Failed to download video"
                    }]
                }
            
            # Extract frames
            frame_paths = await self.youtube_dl_service.extract_frames(
                video_info.video_path, num_frames
            )
            
            # Clean up video file
            await self.youtube_dl_service.cleanup_files([video_info.video_path])
            
            return {
                "status": "success",
                "content": [{
                    "json": {
                        "title": video_info.title,
                        "duration": video_info.duration,
                        "platform": video_info.platform,
                        "frames_extracted": len(frame_paths),
                        "frame_paths": frame_paths
                    }
                }]
            }
            
        except Exception as e:
            logger.error(f"Error downloading video frames: {e}")
            return {
                "status": "error",
                "content": [{
                    "text": f"Error downloading video frames: {str(e)}"
                }]
            }
    
    @tool
    async def analyze_video_sentiment(self, video_url: str) -> dict:
        """Analyze video sentiment by extracting frames and analyzing them."""
        try:
            # Download and extract frames
            frames_result = await self.download_video_frames(video_url)
            
            if frames_result["status"] != "success":
                return frames_result
            
            frame_data = frames_result["content"][0]["json"]
            frame_paths = frame_data["frame_paths"]
            
            # Analyze each frame
            frame_analyses = []
            for i, frame_path in enumerate(frame_paths):
                try:
                    # Analyze frame using existing image analysis
                    frame_result = await self.analyze_image_sentiment(frame_path)
                    frame_analyses.append({
                        "frame": i,
                        "analysis": frame_result.get("content", [{}])[0].get("text", "")
                    })
                except Exception as frame_error:
                    logger.warning(f"Failed to analyze frame {i}: {frame_error}")
                    frame_analyses.append({
                        "frame": i,
                        "analysis": "Analysis failed"
                    })
            
            # Clean up frame files
            await self.youtube_dl_service.cleanup_files(frame_paths)
            
            # Aggregate sentiment from all frames
            all_text = " ".join([
                analysis["analysis"] for analysis in frame_analyses
            ])
            
            # Use sentiment analysis on aggregated text
            sentiment_result = await self._analyze_vision_sentiment(all_text)
            
            return {
                "status": "success",
                "content": [{
                    "json": {
                        "title": frame_data["title"],
                        "duration": frame_data["duration"],
                        "platform": frame_data["platform"],
                        "frames_analyzed": len(frame_analyses),
                        "frame_analyses": frame_analyses,
                        "sentiment": {
                            "label": sentiment_result.label,
                            "confidence": sentiment_result.confidence,
                            "reasoning": sentiment_result.reasoning
                        }
                    }
                }]
            }
            
        except Exception as e:
            logger.error(f"Error analyzing video sentiment: {e}")
            return {
                "status": "error",
                "content": [{
                    "text": f"Error analyzing video sentiment: {str(e)}"
                }]
            }
    
    @tool
    async def get_video_metadata(self, video_url: str) -> dict:
        """Get metadata for a video URL without downloading."""
        try:
            # Check if URL is supported
            if not self.youtube_dl_service.is_supported_platform(video_url):
                return {
                    "status": "error",
                    "content": [{
                        "text": f"Unsupported platform: {video_url}"
                    }]
                }
            
            # Get metadata
            metadata = await self.youtube_dl_service.get_metadata(video_url)
            
            return {
                "status": "success",
                "content": [{
                    "json": {
                        "title": metadata.title,
                        "description": metadata.description[:200] + "..." if len(metadata.description) > 200 else metadata.description,
                        "duration": metadata.duration,
                        "platform": metadata.platform,
                        "upload_date": metadata.upload_date,
                        "view_count": metadata.view_count,
                        "like_count": metadata.like_count,
                        "available_formats": metadata.available_formats[:10]  # Limit to first 10
                    }
                }]
            }
            
        except Exception as e:
            logger.error(f"Error getting video metadata: {e}")
            return {
                "status": "error",
                "content": [{
                    "text": f"Error getting video metadata: {str(e)}"
                }]
            }
    
    async def cleanup(self):
        """Cleanup resources."""
        # Cleanup any resources if needed
        pass



