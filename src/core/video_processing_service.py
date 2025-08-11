"""
Video Processing Service for unified video analysis.
Consolidates video processing capabilities from YouTube analyzers and provides
a unified interface for video analysis, summarization, and metadata extraction.
"""

import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import concurrent.futures

from loguru import logger

from src.core.youtube_dl_service import YouTubeDLService, VideoInfo, AudioInfo
from src.agents.unified_audio_agent import UnifiedAudioAgent
from src.agents.unified_vision_agent import UnifiedVisionAgent
from src.agents.web_agent_enhanced import EnhancedWebAgent
from src.core.models import AnalysisRequest, AnalysisResult, DataType, SentimentResult


@dataclass
class VideoAnalysisResult:
    """Comprehensive video analysis result."""
    video_url: str
    video_metadata: Dict[str, Any]
    audio_sentiment: Optional[SentimentResult]
    visual_sentiment: Optional[SentimentResult]
    combined_sentiment: SentimentResult
    audio_analysis: Dict[str, Any]
    visual_analysis: Dict[str, Any]
    processing_time: float
    extracted_frames: List[str]
    audio_path: Optional[str]
    video_path: Optional[str]
    analysis_timestamp: datetime
    parallel_processing_used: bool = False
    summary: Optional[str] = None
    key_scenes: List[Dict[str, Any]] = None
    transcript: Optional[str] = None


class VideoProcessingService:
    """Unified service for video processing and analysis."""

    def __init__(self, download_path: str = "./temp/videos", max_workers: int = 4):
        self.download_path = Path(download_path)
        self.download_path.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers

        # Initialize services
        self.youtube_dl_service = YouTubeDLService(str(self.download_path))
        self.audio_agent = UnifiedAudioAgent()
        self.vision_agent = UnifiedVisionAgent()
        self.web_agent = EnhancedWebAgent()

        # Analysis settings
        self.max_frames = 10
        self.cleanup_after_analysis = True

        # Thread pool for parallel processing
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

        logger.info(f"Initialized VideoProcessingService with max_workers={max_workers}")

    async def analyze_video(
        self,
        video_input: str,
        extract_audio: bool = True,
        extract_frames: bool = True,
        num_frames: int = 10,
        use_parallel: bool = True,
        generate_summary: bool = True,
        extract_key_scenes: bool = True,
        generate_transcript: bool = False
    ) -> VideoAnalysisResult:
        """
        Perform comprehensive analysis of a video (YouTube URL, local file, etc.).

        Args:
            video_input: Video URL or local file path
            extract_audio: Whether to extract and analyze audio
            extract_frames: Whether to extract and analyze video frames
            num_frames: Number of frames to extract for analysis
            use_parallel: Whether to use parallel processing
            generate_summary: Whether to generate video summary
            extract_key_scenes: Whether to extract key scenes
            generate_transcript: Whether to generate transcript

        Returns:
            Comprehensive video analysis result
        """
        start_time = asyncio.get_event_loop().time()
        analysis_timestamp = datetime.now()

        try:
            logger.info(f"Starting comprehensive video analysis of: {video_input}")

            # Determine if this is a YouTube URL or local file
            if self._is_youtube_url(video_input):
                return await self._analyze_youtube_video(
                    video_input, extract_audio, extract_frames, num_frames,
                    use_parallel, generate_summary, extract_key_scenes, generate_transcript,
                    start_time, analysis_timestamp
                )
            else:
                return await self._analyze_local_video(
                    video_input, extract_audio, extract_frames, num_frames,
                    use_parallel, generate_summary, extract_key_scenes, generate_transcript,
                    start_time, analysis_timestamp
                )

        except Exception as e:
            logger.error(f"Video analysis failed: {e}")
            raise

    async def _analyze_youtube_video(
        self,
        video_url: str,
        extract_audio: bool,
        extract_frames: bool,
        num_frames: int,
        use_parallel: bool,
        generate_summary: bool,
        extract_key_scenes: bool,
        generate_transcript: bool,
        start_time: float,
        analysis_timestamp: datetime
    ) -> VideoAnalysisResult:
        """Analyze YouTube video."""
        try:
            # Check if this is a search URL
            if self.youtube_dl_service.is_youtube_search_url(video_url):
                raise ValueError(
                    "Search URLs are not supported. Please provide a specific video URL."
                )

            # Get video metadata
            logger.info("Step 1: Extracting video metadata...")
            video_metadata = await self.youtube_dl_service.get_metadata(video_url)

            # Extract video components
            logger.info("Step 2: Downloading video and extracting components...")
            video_info, audio_info, extracted_frames = await self._extract_video_components(
                video_url, extract_audio, extract_frames, num_frames
            )

            # Analyze components
            if use_parallel:
                audio_sentiment, audio_analysis, visual_sentiment, visual_analysis = (
                    await self._analyze_components_parallel(
                        audio_info, extracted_frames, extract_audio, extract_frames
                    )
                )
            else:
                audio_sentiment, audio_analysis, visual_sentiment, visual_analysis = (
                    await self._analyze_components_sequential(
                        audio_info, extracted_frames, extract_audio, extract_frames
                    )
                )

            # Generate additional content
            summary = None
            key_scenes = None
            transcript = None

            if generate_summary and video_info.video_path:
                summary = await self._generate_video_summary(video_info.video_path)

            if extract_key_scenes and video_info.video_path:
                key_scenes = await self._extract_key_scenes(video_info.video_path)

            if generate_transcript and audio_info and audio_info.audio_path:
                transcript = await self._generate_transcript(audio_info.audio_path)

            # Combine sentiments
            combined_sentiment = self._combine_sentiment_results(
                audio_sentiment, visual_sentiment, video_metadata
            )

            processing_time = asyncio.get_event_loop().time() - start_time

            # Cleanup if requested
            if self.cleanup_after_analysis:
                await self._cleanup_files(video_info, audio_info, extracted_frames)

            return VideoAnalysisResult(
                video_url=video_url,
                video_metadata=(
                    video_metadata.__dict__ if hasattr(video_metadata, '__dict__')
                    else video_metadata
                ),
                audio_sentiment=audio_sentiment,
                visual_sentiment=visual_sentiment,
                combined_sentiment=combined_sentiment,
                audio_analysis=audio_analysis,
                visual_analysis=visual_analysis,
                processing_time=processing_time,
                extracted_frames=extracted_frames,
                audio_path=audio_info.audio_path if audio_info else None,
                video_path=video_info.video_path if video_info else None,
                analysis_timestamp=analysis_timestamp,
                parallel_processing_used=use_parallel,
                summary=summary,
                key_scenes=key_scenes,
                transcript=transcript
            )

        except Exception as e:
            logger.error(f"YouTube video analysis failed: {e}")
            raise

    async def _analyze_local_video(
        self,
        video_path: str,
        extract_audio: bool,
        extract_frames: bool,
        num_frames: int,
        use_parallel: bool,
        generate_summary: bool,
        extract_key_scenes: bool,
        generate_transcript: bool,
        start_time: float,
        analysis_timestamp: datetime
    ) -> VideoAnalysisResult:
        """Analyze local video file."""
        try:
            # Verify file exists
            if not Path(video_path).exists():
                raise FileNotFoundError(f"Video file not found: {video_path}")

            # Extract video metadata
            video_metadata = await self._extract_local_video_metadata(video_path)

            # Extract components
            audio_info = None
            extracted_frames = []

            if extract_audio:
                audio_info = await self._extract_audio_from_local_video(video_path)

            if extract_frames:
                extracted_frames = await self._extract_frames_from_local_video(
                    video_path, num_frames
                )

            # Analyze components
            if use_parallel:
                audio_sentiment, audio_analysis, visual_sentiment, visual_analysis = (
                    await self._analyze_components_parallel(
                        audio_info, extracted_frames, extract_audio, extract_frames
                    )
                )
            else:
                audio_sentiment, audio_analysis, visual_sentiment, visual_analysis = (
                    await self._analyze_components_sequential(
                        audio_info, extracted_frames, extract_audio, extract_frames
                    )
                )

            # Generate additional content
            summary = None
            key_scenes = None
            transcript = None

            if generate_summary:
                summary = await self._generate_video_summary(video_path)

            if extract_key_scenes:
                key_scenes = await self._extract_key_scenes(video_path)

            if generate_transcript and audio_info and audio_info.audio_path:
                transcript = await self._generate_transcript(audio_info.audio_path)

            # Combine sentiments
            combined_sentiment = self._combine_sentiment_results(
                audio_sentiment, visual_sentiment, video_metadata
            )

            processing_time = asyncio.get_event_loop().time() - start_time

            return VideoAnalysisResult(
                video_url=video_path,
                video_metadata=video_metadata,
                audio_sentiment=audio_sentiment,
                visual_sentiment=visual_sentiment,
                combined_sentiment=combined_sentiment,
                audio_analysis=audio_analysis,
                visual_analysis=visual_analysis,
                processing_time=processing_time,
                extracted_frames=extracted_frames,
                audio_path=audio_info.audio_path if audio_info else None,
                video_path=video_path,
                analysis_timestamp=analysis_timestamp,
                parallel_processing_used=use_parallel,
                summary=summary,
                key_scenes=key_scenes,
                transcript=transcript
            )

        except Exception as e:
            logger.error(f"Local video analysis failed: {e}")
            raise

    async def _extract_video_components(
        self,
        video_url: str,
        extract_audio: bool,
        extract_frames: bool,
        num_frames: int
    ) -> Tuple[VideoInfo, Optional[AudioInfo], List[str]]:
        """Extract video components from YouTube URL."""
        video_info = None
        audio_info = None
        extracted_frames = []

        try:
            # Download video
            video_info = await self.youtube_dl_service.download_video(video_url)

            # Extract audio if requested - use workaround method for better reliability
            if extract_audio and video_info.video_path:
                try:
                    # Try the workaround method first for better reliability
                    audio_info = await self.youtube_dl_service.extract_audio_workaround(video_url)
                    logger.info("Audio extracted successfully using workaround method")
                except Exception as e:
                    logger.warning(f"Audio extraction workaround failed: {e}")
                    try:
                        # Fallback to direct audio extraction
                        audio_info = await self.youtube_dl_service.extract_audio(video_url)
                        logger.info("Audio extracted successfully using direct method")
                    except Exception as e2:
                        logger.warning(f"Direct audio extraction also failed: {e2}")
                        audio_info = None

            # Extract frames if requested
            if extract_frames and video_info.video_path:
                extracted_frames = await self.youtube_dl_service.extract_frames(
                    video_info.video_path, num_frames
                )

            return video_info, audio_info, extracted_frames

        except Exception as e:
            logger.error(f"Failed to extract video components: {e}")
            raise

    async def _analyze_components_parallel(
        self,
        audio_info: Optional[AudioInfo],
        frame_paths: List[str],
        extract_audio: bool,
        extract_frames: bool
    ) -> Tuple[Optional[SentimentResult], Dict[str, Any], Optional[SentimentResult], Dict[str, Any]]:
        """Analyze audio and visual components in parallel."""
        audio_sentiment = None
        audio_analysis = {}
        visual_sentiment = None
        visual_analysis = {}

        # Create tasks for parallel execution
        tasks = []

        if extract_audio and audio_info:
            tasks.append(self._analyze_audio(audio_info.audio_path))

        if extract_frames and frame_paths:
            tasks.append(self._analyze_visual_content(frame_paths))

        # Execute tasks in parallel
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Analysis task failed: {result}")
                    continue

                if extract_audio and audio_info and i == 0:
                    audio_sentiment, audio_analysis = result
                elif extract_frames and frame_paths and (len(tasks) == 1 or i == 1):
                    visual_sentiment, visual_analysis = result

        return audio_sentiment, audio_analysis, visual_sentiment, visual_analysis

    async def _analyze_components_sequential(
        self,
        audio_info: Optional[AudioInfo],
        frame_paths: List[str],
        extract_audio: bool,
        extract_frames: bool
    ) -> Tuple[Optional[SentimentResult], Dict[str, Any], Optional[SentimentResult], Dict[str, Any]]:
        """Analyze audio and visual components sequentially."""
        audio_sentiment = None
        audio_analysis = {}
        visual_sentiment = None
        visual_analysis = {}

        # Analyze audio
        if extract_audio and audio_info:
            audio_sentiment, audio_analysis = await self._analyze_audio(audio_info.audio_path)

        # Analyze visual content
        if extract_frames and frame_paths:
            visual_sentiment, visual_analysis = await self._analyze_visual_content(frame_paths)

        return audio_sentiment, audio_analysis, visual_sentiment, visual_analysis

    async def _analyze_audio(self, audio_path: str) -> Tuple[SentimentResult, Dict[str, Any]]:
        """Analyze audio content."""
        try:
            request = AnalysisRequest(
                data_type=DataType.AUDIO,
                content=audio_path,
                language="en"
            )

            result = await self.audio_agent.process(request)

            return result.sentiment, {
                "extracted_text": result.extracted_text,
                "processing_time": result.processing_time,
                "metadata": result.metadata
            }

        except Exception as e:
            logger.error(f"Audio analysis failed: {e}")
            return SentimentResult(label="neutral", confidence=0.0), {"error": str(e)}

    async def _analyze_visual_content(self, frame_paths: List[str]) -> Tuple[SentimentResult, Dict[str, Any]]:
        """Analyze visual content from frame paths."""
        try:
            frame_sentiments = []
            frame_analyses = []

            for frame_path in frame_paths:
                request = AnalysisRequest(
                    data_type=DataType.IMAGE,
                    content=frame_path,
                    language="en"
                )

                result = await self.vision_agent.process(request)
                frame_sentiments.append(result.sentiment)
                frame_analyses.append({
                    "frame_path": frame_path,
                    "sentiment": result.sentiment,
                    "extracted_text": result.extracted_text,
                    "metadata": result.metadata
                })

            # Combine frame sentiments
            combined_sentiment = self._combine_frame_sentiments(frame_sentiments)

            return combined_sentiment, {
                "frame_analyses": frame_analyses,
                "num_frames": len(frame_paths),
                "frame_sentiments": frame_sentiments
            }

        except Exception as e:
            logger.error(f"Visual content analysis failed: {e}")
            return SentimentResult(label="neutral", confidence=0.0), {"error": str(e)}

    def _combine_frame_sentiments(self, frame_sentiments: List[SentimentResult]) -> SentimentResult:
        """Combine sentiments from multiple frames."""
        if not frame_sentiments:
            return SentimentResult(label="neutral", confidence=0.0)

        # Calculate average confidence
        avg_confidence = sum(s.confidence for s in frame_sentiments) / len(frame_sentiments)

        # Find most common label
        label_counts = {}
        for sentiment in frame_sentiments:
            label = sentiment.label
            label_counts[label] = label_counts.get(label, 0) + 1

        most_common_label = max(label_counts.items(), key=lambda x: x[1])[0]

        return SentimentResult(
            label=most_common_label,
            confidence=avg_confidence,
            metadata={
                "frame_count": len(frame_sentiments),
                "label_distribution": label_counts
            }
        )

    def _combine_sentiment_results(
        self,
        audio_sentiment: Optional[SentimentResult],
        visual_sentiment: Optional[SentimentResult],
        video_metadata: Dict[str, Any]
    ) -> SentimentResult:
        """Combine audio and visual sentiments."""
        sentiments = []
        weights = []

        if audio_sentiment:
            sentiments.append(audio_sentiment)
            weights.append(0.4)  # Audio weight

        if visual_sentiment:
            sentiments.append(visual_sentiment)
            weights.append(0.6)  # Visual weight

        if not sentiments:
            return SentimentResult(label="neutral", confidence=0.0)

        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        # Calculate weighted average confidence
        weighted_confidence = sum(s.confidence * w for s, w in zip(sentiments, weights))

        # Find most common label
        label_counts = {}
        for sentiment, weight in zip(sentiments, weights):
            label = sentiment.label
            label_counts[label] = label_counts.get(label, 0) + weight

        most_common_label = max(label_counts.items(), key=lambda x: x[1])[0]

        return SentimentResult(
            label=most_common_label,
            confidence=weighted_confidence,
            metadata={
                "audio_sentiment": audio_sentiment.label if audio_sentiment else None,
                "visual_sentiment": visual_sentiment.label if visual_sentiment else None,
                "label_distribution": label_counts,
                "video_metadata": video_metadata
            }
        )

    async def _generate_video_summary(self, video_path: str) -> Optional[str]:
        """Generate video summary."""
        try:
            # Use vision agent to generate summary
            request = AnalysisRequest(
                data_type=DataType.VIDEO,
                content=video_path,
                language="en"
            )

            result = await self.vision_agent.process(request)
            return result.extracted_text

        except Exception as e:
            logger.error(f"Video summary generation failed: {e}")
            return None

    async def _extract_key_scenes(self, video_path: str) -> Optional[List[Dict[str, Any]]]:
        """Extract key scenes from video."""
        try:
            # This would typically involve scene detection and analysis
            # For now, return a placeholder
            return [
                {
                    "timestamp": 0,
                    "description": "Scene analysis not yet implemented",
                    "sentiment": "neutral"
                }
            ]

        except Exception as e:
            logger.error(f"Key scene extraction failed: {e}")
            return None

    async def _generate_transcript(self, audio_path: str) -> Optional[str]:
        """Generate transcript from audio."""
        try:
            # Use audio agent to generate transcript
            request = AnalysisRequest(
                data_type=DataType.AUDIO,
                content=audio_path,
                language="en"
            )

            result = await self.audio_agent.process(request)
            return result.extracted_text

        except Exception as e:
            logger.error(f"Transcript generation failed: {e}")
            return None

    async def _extract_local_video_metadata(self, video_path: str) -> Dict[str, Any]:
        """Extract metadata from local video file."""
        # This would typically use ffprobe or similar
        # For now, return basic info
        return {
            "title": Path(video_path).name,
            "duration": 0,
            "format": Path(video_path).suffix,
            "size": Path(video_path).stat().st_size
        }

    async def _extract_audio_from_local_video(self, video_path: str) -> Optional[AudioInfo]:
        """Extract audio from local video file."""
        # This would typically use ffmpeg
        # For now, return None
        return None

    async def _extract_frames_from_local_video(self, video_path: str, num_frames: int) -> List[str]:
        """Extract frames from local video file."""
        # This would typically use ffmpeg
        # For now, return empty list
        return []

    def _is_youtube_url(self, url: str) -> bool:
        """Check if URL is a YouTube URL."""
        return "youtube.com" in url or "youtu.be" in url

    async def _cleanup_files(
        self,
        video_info: Optional[VideoInfo],
        audio_info: Optional[AudioInfo],
        frame_paths: List[str]
    ):
        """Clean up temporary files."""
        try:
            files_to_clean = []

            if video_info and video_info.video_path:
                files_to_clean.append(video_info.video_path)

            if audio_info and audio_info.audio_path:
                files_to_clean.append(audio_info.audio_path)

            files_to_clean.extend(frame_paths)

            for file_path in files_to_clean:
                if Path(file_path).exists():
                    Path(file_path).unlink()
                    logger.debug(f"Cleaned up: {file_path}")

        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")

    async def analyze_videos_batch(
        self,
        video_inputs: List[str],
        **kwargs
    ) -> List[VideoAnalysisResult]:
        """Analyze multiple videos in batch."""
        results = []

        for video_input in video_inputs:
            try:
                result = await self.analyze_video(video_input, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Batch analysis failed for {video_input}: {e}")
                # Add error result
                results.append(VideoAnalysisResult(
                    video_url=video_input,
                    video_metadata={},
                    audio_sentiment=None,
                    visual_sentiment=None,
                    combined_sentiment=SentimentResult(label="neutral", confidence=0.0),
                    audio_analysis={"error": str(e)},
                    visual_analysis={"error": str(e)},
                    processing_time=0.0,
                    extracted_frames=[],
                    audio_path=None,
                    video_path=None,
                    analysis_timestamp=datetime.now(),
                    parallel_processing_used=False
                ))

        return results

    def __del__(self):
        """Cleanup thread pool executor."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)


# Global instance
video_processing_service = VideoProcessingService()
