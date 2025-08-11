#!/usr/bin/env python3
"""
Comprehensive YouTube Video Analyzer
Integrates YouTubeDL, Enhanced Audio Agent, and Enhanced Vision Agent
for complete audio/visual sentiment analysis of YouTube videos.
"""

import asyncio
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

from loguru import logger

from src.core.youtube_dl_service import YouTubeDLService, VideoInfo, AudioInfo
from src.agents.unified_audio_agent import UnifiedAudioAgent
from src.agents.unified_vision_agent import UnifiedVisionAgent
from src.agents.web_agent_enhanced import EnhancedWebAgent
from src.core.models import AnalysisRequest, AnalysisResult, DataType, SentimentResult


@dataclass
class YouTubeAnalysisResult:
    """Comprehensive YouTube analysis result."""
    video_url: str
    video_metadata: Dict[str, Any]
    audio_sentiment: SentimentResult
    visual_sentiment: SentimentResult
    combined_sentiment: SentimentResult
    audio_analysis: Dict[str, Any]
    visual_analysis: Dict[str, Any]
    processing_time: float
    extracted_frames: List[str]
    audio_path: Optional[str]
    video_path: Optional[str]
    analysis_timestamp: datetime


class YouTubeComprehensiveAnalyzer:
    """Comprehensive YouTube video analyzer with full audio/visual analysis."""
    
    def __init__(self, download_path: str = "./temp/videos"):
        self.download_path = Path(download_path)
        self.download_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize services
        self.youtube_dl_service = YouTubeDLService(str(self.download_path))
        self.audio_agent = UnifiedAudioAgent()
        self.vision_agent = UnifiedVisionAgent()
        self.web_agent = EnhancedWebAgent()
        
        # Analysis settings
        self.max_frames = 10
        self.cleanup_after_analysis = True
        
    async def analyze_youtube_video(
        self, 
        video_url: str, 
        extract_audio: bool = True,
        extract_frames: bool = True,
        num_frames: int = 10
    ) -> YouTubeAnalysisResult:
        """
        Perform comprehensive analysis of a YouTube video.
        
        Args:
            video_url: YouTube video URL
            extract_audio: Whether to extract and analyze audio
            extract_frames: Whether to extract and analyze video frames
            num_frames: Number of frames to extract for analysis
            
        Returns:
            Comprehensive analysis result
        """
        start_time = asyncio.get_event_loop().time()
        analysis_timestamp = datetime.now()
        
        try:
            logger.info(f"Starting comprehensive analysis of: {video_url}")
            
            # Step 1: Get video metadata
            logger.info("Step 1: Extracting video metadata...")
            video_metadata = await self.youtube_dl_service.get_metadata(video_url)
            
            # Step 2: Download video and extract components
            logger.info("Step 2: Downloading video and extracting components...")
            video_info, audio_info, extracted_frames = await self._extract_video_components(
                video_url, extract_audio, extract_frames, num_frames
            )
            
            # Step 3: Analyze audio (if extracted)
            audio_sentiment = None
            audio_analysis = {}
            if audio_info and extract_audio:
                logger.info("Step 3: Analyzing audio content...")
                audio_sentiment, audio_analysis = await self._analyze_audio(audio_info.audio_path)
            
            # Step 4: Analyze visual content (if frames extracted)
            visual_sentiment = None
            visual_analysis = {}
            if extracted_frames and extract_frames:
                logger.info("Step 4: Analyzing visual content...")
                visual_sentiment, visual_analysis = await self._analyze_visual_content(extracted_frames)
            
            # Step 5: Combine results
            logger.info("Step 5: Combining analysis results...")
            combined_sentiment = self._combine_sentiment_results(
                audio_sentiment, visual_sentiment, video_metadata
            )
            
            # Step 6: Cleanup (if enabled)
            if self.cleanup_after_analysis:
                await self._cleanup_files(video_info, audio_info, extracted_frames)
            
            processing_time = asyncio.get_event_loop().time() - start_time
            
            return YouTubeAnalysisResult(
                video_url=video_url,
                video_metadata=video_metadata.__dict__,
                audio_sentiment=audio_sentiment or SentimentResult(label="neutral", confidence=0.0),
                visual_sentiment=visual_sentiment or SentimentResult(label="neutral", confidence=0.0),
                combined_sentiment=combined_sentiment,
                audio_analysis=audio_analysis,
                visual_analysis=visual_analysis,
                processing_time=processing_time,
                extracted_frames=extracted_frames,
                audio_path=audio_info.audio_path if audio_info else None,
                video_path=video_info.video_path if video_info else None,
                analysis_timestamp=analysis_timestamp
            )
            
        except Exception as e:
            logger.error(f"Comprehensive YouTube analysis failed: {e}")
            processing_time = asyncio.get_event_loop().time() - start_time
            
            # Return error result
            return YouTubeAnalysisResult(
                video_url=video_url,
                video_metadata={"error": str(e)},
                audio_sentiment=SentimentResult(label="neutral", confidence=0.0),
                visual_sentiment=SentimentResult(label="neutral", confidence=0.0),
                combined_sentiment=SentimentResult(label="neutral", confidence=0.0),
                audio_analysis={"error": str(e)},
                visual_analysis={"error": str(e)},
                processing_time=processing_time,
                extracted_frames=[],
                audio_path=None,
                video_path=None,
                analysis_timestamp=analysis_timestamp
            )
    
    async def _extract_video_components(
        self, 
        video_url: str, 
        extract_audio: bool, 
        extract_frames: bool, 
        num_frames: int
    ) -> tuple[VideoInfo, Optional[AudioInfo], List[str]]:
        """Extract video, audio, and frames from YouTube URL."""
        video_info = None
        audio_info = None
        extracted_frames = []
        
        try:
            # Download video
            video_info = await self.youtube_dl_service.download_video(video_url)
            logger.info(f"Video downloaded: {video_info.title}")
            
            # Extract audio if requested
            if extract_audio:
                audio_info = await self.youtube_dl_service.extract_audio(video_url)
                logger.info(f"Audio extracted: {audio_info.audio_path}")
            
            # Extract frames if requested
            if extract_frames and video_info.video_path:
                extracted_frames = await self.youtube_dl_service.extract_frames(
                    video_info.video_path, num_frames
                )
                logger.info(f"Extracted {len(extracted_frames)} frames")
            
            return video_info, audio_info, extracted_frames
            
        except Exception as e:
            logger.error(f"Failed to extract video components: {e}")
            return video_info, audio_info, extracted_frames
    
    async def _analyze_audio(self, audio_path: str) -> tuple[SentimentResult, Dict[str, Any]]:
        """Analyze audio content using Enhanced Audio Agent."""
        try:
            # Create analysis request
            request = AnalysisRequest(
                data_type=DataType.AUDIO,
                content=audio_path
            )
            
            # Process with enhanced audio agent
            result = await self.audio_agent.process(request)
            
            # Extract additional analysis data
            audio_analysis = {
                "transcription": result.extracted_text,
                "metadata": result.metadata,
                "processing_time": result.processing_time,
                "status": result.status
            }
            
            return result.sentiment, audio_analysis
            
        except Exception as e:
            logger.error(f"Audio analysis failed: {e}")
            return SentimentResult(label="neutral", confidence=0.0), {"error": str(e)}
    
    async def _analyze_visual_content(self, frame_paths: List[str]) -> tuple[SentimentResult, Dict[str, Any]]:
        """Analyze visual content using Enhanced Vision Agent."""
        try:
            # Analyze each frame
            frame_analyses = []
            frame_sentiments = []
            
            for frame_path in frame_paths:
                # Create analysis request for each frame
                request = AnalysisRequest(
                    data_type=DataType.IMAGE,
                    content=frame_path
                )
                
                # Process with enhanced vision agent
                result = await self.vision_agent.process(request)
                
                frame_analyses.append({
                    "frame_path": frame_path,
                    "sentiment": result.sentiment.label,
                    "confidence": result.sentiment.confidence,
                    "metadata": result.metadata
                })
                
                frame_sentiments.append(result.sentiment)
            
            # Combine frame sentiments
            combined_sentiment = self._combine_frame_sentiments(frame_sentiments)
            
            visual_analysis = {
                "frame_analyses": frame_analyses,
                "total_frames": len(frame_paths),
                "frame_sentiments": [s.label for s in frame_sentiments],
                "average_confidence": sum(s.confidence for s in frame_sentiments) / len(frame_sentiments) if frame_sentiments else 0.0
            }
            
            return combined_sentiment, visual_analysis
            
        except Exception as e:
            logger.error(f"Visual content analysis failed: {e}")
            return SentimentResult(label="neutral", confidence=0.0), {"error": str(e)}
    
    def _combine_frame_sentiments(self, frame_sentiments: List[SentimentResult]) -> SentimentResult:
        """Combine sentiments from multiple frames."""
        if not frame_sentiments:
            return SentimentResult("neutral", 0.0)
        
        # Count sentiment labels
        sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
        total_confidence = 0.0
        
        for sentiment in frame_sentiments:
            sentiment_counts[sentiment.label] += 1
            total_confidence += sentiment.confidence
        
        # Determine dominant sentiment
        dominant_sentiment = max(sentiment_counts.items(), key=lambda x: x[1])[0]
        average_confidence = total_confidence / len(frame_sentiments)
        
        return SentimentResult(label=dominant_sentiment, confidence=average_confidence)
    
    def _combine_sentiment_results(
        self, 
        audio_sentiment: Optional[SentimentResult], 
        visual_sentiment: Optional[SentimentResult],
        video_metadata: Dict[str, Any]
    ) -> SentimentResult:
        """Combine audio and visual sentiment results."""
        sentiments = []
        weights = []
        
        # Add audio sentiment if available
        if audio_sentiment:
            sentiments.append(audio_sentiment)
            weights.append(0.6)  # Audio gets higher weight as it often contains more sentiment
        
        # Add visual sentiment if available
        if visual_sentiment:
            sentiments.append(visual_sentiment)
            weights.append(0.4)  # Visual gets lower weight
        
        # If no sentiments available, return neutral
        if not sentiments:
            return SentimentResult(label="neutral", confidence=0.0)
        
        # Normalize weights
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        # Calculate weighted sentiment
        sentiment_scores = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
        total_confidence = 0.0
        
        for sentiment, weight in zip(sentiments, normalized_weights):
            sentiment_scores[sentiment.label] += weight * sentiment.confidence
            total_confidence += sentiment.confidence * weight
        
        # Determine combined sentiment
        combined_sentiment = max(sentiment_scores.items(), key=lambda x: x[1])[0]
        
        return SentimentResult(label=combined_sentiment, confidence=total_confidence)
    
    async def _cleanup_files(
        self, 
        video_info: Optional[VideoInfo], 
        audio_info: Optional[AudioInfo], 
        frame_paths: List[str]
    ):
        """Clean up downloaded files."""
        try:
            files_to_cleanup = []
            
            if video_info and video_info.video_path:
                files_to_cleanup.append(video_info.video_path)
            
            if audio_info and audio_info.audio_path:
                files_to_cleanup.append(audio_info.audio_path)
            
            files_to_cleanup.extend(frame_paths)
            
            await self.youtube_dl_service.cleanup_files(files_to_cleanup)
            logger.info(f"Cleaned up {len(files_to_cleanup)} files")
            
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")
    
    async def analyze_youtube_urls_batch(
        self, 
        urls: List[str], 
        extract_audio: bool = True,
        extract_frames: bool = True,
        num_frames: int = 10
    ) -> List[YouTubeAnalysisResult]:
        """Analyze multiple YouTube URLs in batch."""
        results = []
        
        for url in urls:
            try:
                result = await self.analyze_youtube_video(
                    url, extract_audio, extract_frames, num_frames
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to analyze {url}: {e}")
                # Add error result
                results.append(YouTubeAnalysisResult(
                    video_url=url,
                    video_metadata={"error": str(e)},
                    audio_sentiment=SentimentResult(label="neutral", confidence=0.0),
                    visual_sentiment=SentimentResult(label="neutral", confidence=0.0),
                    combined_sentiment=SentimentResult(label="neutral", confidence=0.0),
                    audio_analysis={"error": str(e)},
                    visual_analysis={"error": str(e)},
                    processing_time=0.0,
                    extracted_frames=[],
                    audio_path=None,
                    video_path=None,
                    analysis_timestamp=datetime.now()
                ))
        
        return results


# Example usage
async def main():
    """Example usage of YouTubeComprehensiveAnalyzer."""
    analyzer = YouTubeComprehensiveAnalyzer()
    
    # Test URL
    test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    
    try:
        print("Starting comprehensive YouTube analysis...")
        result = await analyzer.analyze_youtube_video(
            test_url,
            extract_audio=True,
            extract_frames=True,
            num_frames=5
        )
        
        print(f"\n=== YouTube Analysis Results ===")
        print(f"Video: {result.video_metadata.get('title', 'Unknown')}")
        print(f"Duration: {result.video_metadata.get('duration', 0)} seconds")
        print(f"Processing Time: {result.processing_time:.2f} seconds")
        
        print(f"\n=== Sentiment Analysis ===")
        print(f"Audio Sentiment: {result.audio_sentiment.label} (confidence: {result.audio_sentiment.confidence:.2f})")
        print(f"Visual Sentiment: {result.visual_sentiment.label} (confidence: {result.visual_sentiment.confidence:.2f})")
        print(f"Combined Sentiment: {result.combined_sentiment.label} (confidence: {result.combined_sentiment.confidence:.2f})")
        
        print(f"\n=== Analysis Details ===")
        print(f"Frames Analyzed: {len(result.extracted_frames)}")
        print(f"Audio Analysis: {result.audio_analysis.get('status', 'N/A')}")
        print(f"Visual Analysis: {result.visual_analysis.get('total_frames', 0)} frames processed")
        
    except Exception as e:
        print(f"Analysis failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
