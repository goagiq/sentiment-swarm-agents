"""
YouTube-DL service for video processing and analysis.
Provides core functionality for downloading videos, extracting audio,
and processing video content for sentiment analysis.
"""

import asyncio
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yt_dlp

logger = logging.getLogger(__name__)


@dataclass
class VideoInfo:
    """Information about a downloaded video."""
    title: str
    duration: int
    platform: str
    video_path: Optional[str] = None
    audio_path: Optional[str] = None
    metadata: Dict[str, Any] = None


@dataclass
class AudioInfo:
    """Information about extracted audio."""
    audio_path: str
    format: str
    duration: int
    bitrate: int
    metadata: Dict[str, Any] = None


@dataclass
class VideoMetadata:
    """Video metadata without downloading."""
    title: str
    description: str
    duration: int
    platform: str
    upload_date: str
    view_count: int
    like_count: int
    available_formats: List[str]


class YouTubeDLError(Exception):
    """Base exception for YouTube-DL errors."""
    pass


class VideoUnavailableError(YouTubeDLError):
    """Raised when video is unavailable."""
    pass


class NetworkError(YouTubeDLError):
    """Raised when network issues occur."""
    pass


class YouTubeDLService:
    """Enhanced YouTube-DL service with better error handling and retry mechanisms."""

    def __init__(self, download_path: str = "./temp/videos"):
        self.download_path = Path(download_path)
        self.download_path.mkdir(parents=True, exist_ok=True)

        # Enhanced user agents for better compatibility
        self.user_agents = [
            ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
             'AppleWebKit/537.36 (KHTML, like Gecko) '
             'Chrome/120.0.0.0 Safari/537.36'),
            ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
             'AppleWebKit/537.36 (KHTML, like Gecko) '
             'Chrome/120.0.0.0 Safari/537.36'),
            ('Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
             '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'),
            ('Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) '
             'Gecko/20100101 Firefox/121.0'),
            ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) '
             'Gecko/20100101 Firefox/121.0')
        ]

        # Enhanced yt-dlp options with multiple extraction strategies
        self.ydl_opts = {
            'outtmpl': str(self.download_path / '%(title)s.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
            # Multiple user agents for better compatibility
            'http_headers': {
                'User-Agent': self.user_agents[0],
                'Accept': ('text/html,application/xhtml+xml,application/xml;'
                           'q=0.9,image/webp,*/*;q=0.8'),
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
                'Sec-Fetch-User': '?1',
                'Cache-Control': 'max-age=0',
            },
            # Enhanced retry configuration
            'retries': 10,
            'fragment_retries': 10,
            'skip_unavailable_fragments': True,
            'ignoreerrors': False,
            # Cookie handling
            'cookiefile': None,
            # Additional options for better compatibility
            'nocheckcertificate': True,
            'prefer_insecure': True,
            'geo_bypass': True,
            'geo_bypass_country': 'US',
            'geo_bypass_ip_block': '1.0.0.1',
            # Enhanced format selection
            'format': 'best[height<=720]/best',
            'merge_output_format': 'mp4',
        }

    def _get_audio_options(self) -> Dict[str, Any]:
        """Get options optimized for audio extraction."""
        return {
            **self.ydl_opts,
            'format': 'bestaudio[ext=m4a]/bestaudio[ext=mp3]/bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': str(self.download_path / '%(title)s.%(ext)s'),
            'ignoreerrors': False,
            'no_warnings': True,
            'extractaudio': True,
            'audioformat': 'mp3',
            'audioquality': '192K',
        }

    def _get_video_options(self) -> Dict[str, Any]:
        """Get options optimized for video download."""
        return {
            **self.ydl_opts,
            'format': 'best[height<=720]/best[height<=480]/worst',
            'outtmpl': str(self.download_path / '%(title)s.%(ext)s'),
            'ignoreerrors': False,
            'no_warnings': True,
            'merge_output_format': 'mp4',
            'prefer_ffmpeg': True,
        }

    def _get_fallback_options(self, user_agent: str) -> Dict[str, Any]:
        """Get fallback options with different user agent and extraction method."""
        return {
            **self.ydl_opts,
            'http_headers': {
                'User-Agent': user_agent,
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-us,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
            },
            'extractor_args': {
                'youtube': {
                    'player_client': ['web', 'android'],
                    'player_skip': ['webpage'],
                }
            },
            'retries': 5,
            'fragment_retries': 5,
        }

    async def _try_extraction_with_retry(self, url: str, options: Dict[str, Any], max_retries: int = 3) -> Optional[Dict[str, Any]]:
        """Try extraction with retry mechanism and different strategies."""
        last_error = None
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempt {attempt + 1}/{max_retries} for URL: {url}")
                
                # Create a fresh options dict for each attempt to avoid modification issues
                current_options = options.copy()
                
                # Try different user agent on each attempt for YouTube URLs
                if 'youtube' in url.lower() and attempt > 0:
                    user_agent_index = attempt % len(self.user_agents)
                    if user_agent_index < len(self.user_agents):
                        current_options['http_headers'] = {
                            'User-Agent': self.user_agents[user_agent_index],
                            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                            'Accept-Language': 'en-us,en;q=0.5',
                            'Accept-Encoding': 'gzip, deflate',
                            'Connection': 'keep-alive',
                        }
                
                with yt_dlp.YoutubeDL(current_options) as ydl:
                    info = ydl.extract_info(url, download=False)
                    if info:
                        logger.info(f"Successfully extracted info on attempt {attempt + 1}")
                        return info
                    
            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                
                # Wait before retry with exponential backoff
                if attempt < max_retries - 1:
                    wait_time = min(2 ** attempt, 10)  # Cap at 10 seconds
                    await asyncio.sleep(wait_time)
        
        logger.error(f"All {max_retries} attempts failed for URL: {url}")
        raise last_error or VideoUnavailableError("Failed to extract video information")

    async def get_metadata(self, url: str) -> VideoMetadata:
        """Get video metadata without downloading."""
        try:
            # Use enhanced options with better headers
            enhanced_opts = {
                'quiet': True,
                'no_warnings': True,
                'http_headers': {
                    'User-Agent': (
                        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                        '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                    ),
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Language': 'en-us,en;q=0.5',
                },
                'retries': 3,
                'ignoreerrors': True,
            }

            info = await self._try_extraction_with_retry(url, enhanced_opts)

            if not info:
                raise VideoUnavailableError("No metadata available")

            return VideoMetadata(
                title=info.get('title', 'Unknown'),
                description=info.get('description', ''),
                duration=info.get('duration', 0),
                platform=info.get('extractor', 'Unknown'),
                upload_date=info.get('upload_date', ''),
                view_count=info.get('view_count', 0),
                like_count=info.get('like_count', 0),
                available_formats=[f.get('ext', '') for f in info.get('formats', [])]
            )
        except Exception as e:
            logger.error(f"Failed to get metadata for {url}: {e}")
            if "403" in str(e) or "forbidden" in str(e).lower():
                raise VideoUnavailableError(f"Access forbidden (403): {e}")
            elif "404" in str(e) or "not found" in str(e).lower():
                raise VideoUnavailableError(f"Video not found (404): {e}")
            else:
                raise VideoUnavailableError(f"Could not get metadata: {e}")

    async def download_video(self, url: str) -> VideoInfo:
        """Download video and return metadata with enhanced error handling."""
        last_error = None
        
        # Try multiple extraction strategies with safety checks
        strategies = []
        
        # Add video options
        try:
            strategies.append(self._get_video_options())
        except Exception as e:
            logger.warning(f"Failed to create video options: {e}")
        
        # Add fallback options with user agents
        for i in range(min(3, len(self.user_agents))):
            try:
                strategies.append(self._get_fallback_options(self.user_agents[i]))
            except Exception as e:
                logger.warning(f"Failed to create fallback options {i}: {e}")
        
        # Ensure we have at least one strategy
        if not strategies:
            logger.warning("No strategies available, using minimal options")
            strategies = [{
                'outtmpl': str(self.download_path / '%(title)s.%(ext)s'),
                'format': 'worst',
                'quiet': True,
                'no_warnings': True,
                'retries': 3,
                'ignoreerrors': False,
            }]
        
        for i, options in enumerate(strategies):
            try:
                logger.info(f"Trying download strategy {i + 1}/{len(strategies)} for URL: {url}")
                
                # First try to get info without downloading
                info = await self._try_extraction_with_retry(url, options, max_retries=2)
                
                if not info:
                    continue
                
                # Now try to download
                with yt_dlp.YoutubeDL(options) as ydl:
                    download_info = ydl.extract_info(url, download=True)
                    
                    if not download_info:
                        continue

                    # Find the downloaded file
                    video_path = None
                    for file in self.download_path.iterdir():
                        if file.is_file() and file.suffix in ['.mp4', '.webm', '.mkv']:
                            video_path = str(file)
                            break

                    if video_path:
                        logger.info(f"Successfully downloaded video using strategy {i + 1}")
                        return VideoInfo(
                            title=download_info.get('title', 'Unknown'),
                            duration=download_info.get('duration', 0),
                            platform=download_info.get('extractor', 'Unknown'),
                            video_path=video_path,
                            metadata=download_info
                        )
                        
            except Exception as e:
                last_error = e
                logger.warning(f"Strategy {i + 1} failed: {e}")
                if i < len(strategies) - 1:
                    await asyncio.sleep(1)  # Brief pause between strategies
        
        # If all strategies failed, try one more time with minimal options
        try:
            logger.info("Trying minimal download strategy as last resort")
            minimal_options = {
                'outtmpl': str(self.download_path / '%(title)s.%(ext)s'),
                'format': 'worst',
                'quiet': True,
                'no_warnings': True,
                'retries': 3,
                'ignoreerrors': False,
            }
            
            with yt_dlp.YoutubeDL(minimal_options) as ydl:
                info = ydl.extract_info(url, download=True)
                
                if info:
                    video_path = None
                    for file in self.download_path.iterdir():
                        if file.is_file() and file.suffix in ['.mp4', '.webm', '.mkv']:
                            video_path = str(file)
                            break
                    
                    if video_path:
                        return VideoInfo(
                            title=info.get('title', 'Unknown'),
                            duration=info.get('duration', 0),
                            platform=info.get('extractor', 'Unknown'),
                            video_path=video_path,
                            metadata=info
                        )
                        
        except Exception as e:
            last_error = e
            logger.error(f"Minimal strategy also failed: {e}")
        
        # If we get here, all strategies failed
        logger.error(f"All download strategies failed for {url}")
        if last_error:
            if "403" in str(last_error) or "forbidden" in str(last_error).lower():
                raise VideoUnavailableError(f"Access forbidden (403): {last_error}")
            elif "404" in str(last_error) or "not found" in str(last_error).lower():
                raise VideoUnavailableError(f"Video not found (404): {last_error}")
            else:
                raise VideoUnavailableError(f"Could not download video: {last_error}")
        else:
            raise VideoUnavailableError("No video information available")

    async def extract_audio(self, url: str) -> AudioInfo:
        """Extract audio from video URL with enhanced error handling."""
        last_error = None
        
        # Try multiple extraction strategies for audio with safety checks
        audio_strategies = []
        
        # Add audio options
        try:
            audio_strategies.append(self._get_audio_options())
        except Exception as e:
            logger.warning(f"Failed to create audio options: {e}")
        
        # Add enhanced audio options
        try:
            audio_strategies.append(self._get_enhanced_audio_options())
        except Exception as e:
            logger.warning(f"Failed to create enhanced audio options: {e}")
        
        # Add fallback options with user agents
        for i in range(min(3, len(self.user_agents))):
            try:
                audio_strategies.append(self._get_fallback_options(self.user_agents[i]))
            except Exception as e:
                logger.warning(f"Failed to create audio fallback options {i}: {e}")
        
        # Add minimal audio options
        try:
            audio_strategies.append(self._get_minimal_audio_options())
        except Exception as e:
            logger.warning(f"Failed to create minimal audio options: {e}")
        
        # Ensure we have at least one strategy
        if not audio_strategies:
            logger.warning("No audio strategies available, using minimal options")
            audio_strategies = [{
                'outtmpl': str(self.download_path / '%(title)s.%(ext)s'),
                'format': 'bestaudio/best',
                'quiet': True,
                'no_warnings': True,
                'retries': 3,
                'ignoreerrors': False,
            }]
        
        for i, options in enumerate(audio_strategies):
            try:
                logger.info(f"Trying audio extraction strategy {i + 1}/{len(audio_strategies)} for URL: {url}")
                
                # First try to get info without downloading
                info = await self._try_extraction_with_retry(url, options, max_retries=3)
                
                if not info:
                    continue
                
                # Now try to download audio
                with yt_dlp.YoutubeDL(options) as ydl:
                    download_info = ydl.extract_info(url, download=True)
                    
                    if not download_info:
                        continue

                    # Find the extracted audio file
                    audio_path = None
                    for file in self.download_path.iterdir():
                        if file.is_file() and file.suffix in ['.mp3', '.m4a', '.wav']:
                            audio_path = str(file)
                            break

                    if audio_path:
                        logger.info(f"Successfully extracted audio using strategy {i + 1}")
                        return AudioInfo(
                            audio_path=audio_path,
                            format=Path(audio_path).suffix if audio_path else 'mp3',
                            duration=download_info.get('duration', 0),
                            bitrate=192,
                            metadata=download_info
                        )
                        
            except Exception as e:
                last_error = e
                logger.warning(f"Audio strategy {i + 1} failed: {e}")
                if i < len(audio_strategies) - 1:
                    await asyncio.sleep(2)  # Longer pause between strategies
        
        # If all strategies failed, try the workaround method
        try:
            logger.info("Trying workaround method as final fallback")
            return await self.extract_audio_workaround(url)
        except Exception as e:
            last_error = e
            logger.error(f"Workaround method also failed: {e}")
        
        # If we get here, all strategies failed
        logger.error(f"All audio extraction strategies failed for {url}")
        if last_error:
            if "403" in str(last_error) or "forbidden" in str(last_error).lower():
                raise VideoUnavailableError(f"Access forbidden (403) for audio extraction: {last_error}")
            elif "404" in str(last_error) or "not found" in str(last_error).lower():
                raise VideoUnavailableError(f"Video not found (404) for audio extraction: {last_error}")
            else:
                raise VideoUnavailableError(f"Could not extract audio: {last_error}")
        else:
            raise VideoUnavailableError("No audio information available")

    async def extract_metadata_only(self, url: str) -> VideoMetadata:
        """Extract only metadata when full download fails."""
        last_error = None
        
        # Try multiple strategies for metadata extraction with safety checks
        metadata_strategies = []
        
        # Add metadata options
        try:
            metadata_strategies.append(self._get_metadata_options())
        except Exception as e:
            logger.warning(f"Failed to create metadata options: {e}")
        
        # Add fallback metadata options
        try:
            metadata_strategies.append(self._get_fallback_metadata_options())
        except Exception as e:
            logger.warning(f"Failed to create fallback metadata options: {e}")
        
        # Add minimal metadata options
        try:
            metadata_strategies.append(self._get_minimal_metadata_options())
        except Exception as e:
            logger.warning(f"Failed to create minimal metadata options: {e}")
        
        # Ensure we have at least one strategy
        if not metadata_strategies:
            logger.warning("No metadata strategies available, using minimal options")
            metadata_strategies = [{
                'quiet': True,
                'no_warnings': True,
                'retries': 3,
                'ignoreerrors': True,
            }]
        
        for i, options in enumerate(metadata_strategies):
            try:
                logger.info(f"Trying metadata extraction strategy {i + 1}/{len(metadata_strategies)} for URL: {url}")
                
                info = await self._try_extraction_with_retry(url, options, max_retries=3)
                
                if info:
                    logger.info(f"Successfully extracted metadata using strategy {i + 1}")
                    return VideoMetadata(
                        title=info.get('title', 'Unknown Title'),
                        description=info.get('description', ''),
                        duration=info.get('duration', 0),
                        platform=info.get('extractor', 'youtube'),
                        upload_date=info.get('upload_date', ''),
                        view_count=info.get('view_count', 0),
                        like_count=info.get('like_count', 0),
                        available_formats=[f.get('format_id', '') for f in info.get('formats', [])]
                    )
                        
            except Exception as e:
                last_error = e
                logger.warning(f"Metadata strategy {i + 1} failed: {e}")
                if i < len(metadata_strategies) - 1:
                    await asyncio.sleep(1)
        
        # If all strategies failed
        logger.error(f"All metadata extraction strategies failed for {url}")
        if last_error:
            raise VideoUnavailableError(f"Could not extract metadata: {last_error}")
        else:
            raise VideoUnavailableError("No metadata information available")

    def _get_enhanced_audio_options(self) -> Dict[str, Any]:
        """Get enhanced audio extraction options with better compatibility."""
        return {
            'outtmpl': str(self.download_path / '%(title)s.%(ext)s'),
            'format': 'bestaudio[ext=m4a]/bestaudio[ext=mp3]/bestaudio',
            'extractaudio': True,
            'audioformat': 'mp3',
            'audioquality': '192K',
            'quiet': True,
            'no_warnings': True,
            'retries': 5,
            'fragment_retries': 5,
            'skip_unavailable_fragments': True,
            'ignoreerrors': False,
            'nocheckcertificate': True,
            'prefer_insecure': True,
            'geo_bypass': True,
            'http_headers': {
                'User-Agent': self.user_agents[1],
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
            },
            'extractor_args': {
                'youtube': {
                    'player_client': ['android', 'web', 'mweb'],
                    'player_skip': ['webpage', 'configs'],
                }
            },
        }

    def _get_minimal_audio_options(self) -> Dict[str, Any]:
        """Get minimal audio extraction options as last resort."""
        return {
            'outtmpl': str(self.download_path / '%(title)s.%(ext)s'),
            'format': 'worstaudio/worst',
            'quiet': True,
            'no_warnings': True,
            'retries': 3,
            'ignoreerrors': False,
            'extractaudio': True,
            'audioformat': 'mp3',
            'audioquality': '128K',
            'nocheckcertificate': True,
            'prefer_insecure': True,
            'http_headers': {
                'User-Agent': self.user_agents[3],
                'Accept': '*/*',
                'Accept-Language': 'en-US,en;q=0.5',
            },
        }

    async def extract_audio_from_video(self, video_path: str) -> AudioInfo:
        """Extract audio from a downloaded video file using ffmpeg."""
        try:
            import ffmpeg
            
            audio_path = str(Path(video_path).with_suffix('.mp3'))
            
            # Use ffmpeg to extract audio
            stream = ffmpeg.input(video_path)
            stream = ffmpeg.output(stream, audio_path, acodec='mp3', ab='192k')
            ffmpeg.run(stream, overwrite_output=True, quiet=True)
            
            # Get duration from the video file
            probe = ffmpeg.probe(video_path)
            duration = float(probe['format']['duration']) if 'duration' in probe['format'] else 0
            
            return AudioInfo(
                audio_path=audio_path,
                format='mp3',
                duration=int(duration),
                bitrate=192,
                metadata={'source_video': video_path}
            )
        except ImportError:
            logger.error("ffmpeg-python not available. Please install with: pip install ffmpeg-python")
            raise VideoUnavailableError("ffmpeg-python not available for audio extraction")
        except Exception as e:
            logger.error(f"Failed to extract audio from video: {e}")
            raise VideoUnavailableError(f"Could not extract audio from video: {e}")

    async def extract_audio_workaround(self, url: str) -> AudioInfo:
        """Extract audio using workaround: download video first, then extract audio."""
        try:
            logger.info(f"Using workaround method for audio extraction: {url}")
            
            # First download the video
            video_info = await self.download_video(url)
            
            if not video_info.video_path:
                raise VideoUnavailableError("No video file found after download")
            
            # Then extract audio from the video file
            audio_info = await self.extract_audio_from_video(video_info.video_path)
            
            # Clean up the video file if requested
            # await self.cleanup_files([video_info.video_path])
            
            return audio_info
            
        except Exception as e:
            logger.error(f"Workaround audio extraction failed: {e}")
            raise VideoUnavailableError(f"Could not extract audio using workaround: {e}")

    async def extract_frames(self, video_path: str, num_frames: int = 10) -> List[str]:
        """Extract key frames from video."""
        try:
            import cv2

            frames = []
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                raise ValueError("Could not open video file")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_interval = max(1, total_frames // num_frames)

            for i in range(0, total_frames, frame_interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()

                if ret:
                    frame_path = str(self.download_path / f"frame_{i:04d}.jpg")
                    cv2.imwrite(frame_path, frame)
                    frames.append(frame_path)

                if len(frames) >= num_frames:
                    break

            cap.release()
            return frames

        except ImportError:
            logger.warning("OpenCV not available, skipping frame extraction")
            return []
        except Exception as e:
            logger.error(f"Failed to extract frames from {video_path}: {e}")
            return []

    async def cleanup_files(self, file_paths: List[str]):
        """Clean up downloaded files."""
        for file_path in file_paths:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.debug(f"Cleaned up: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to cleanup {file_path}: {e}")

    def is_supported_platform(self, url: str) -> bool:
        """Check if the URL is from a supported platform."""
        supported_domains = [
            'youtube.com', 'youtu.be',
            'vimeo.com',
            'tiktok.com',
            'instagram.com',
            'facebook.com',
            'twitter.com'
        ]

        url_lower = url.lower()
        return any(domain in url_lower for domain in supported_domains)

    def is_youtube_search_url(self, url: str) -> bool:
        """Check if the URL is a YouTube search results page."""
        return 'youtube.com/results' in url or 'youtube.com/search' in url

    async def search_youtube_videos(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search for YouTube videos using a query string."""
        try:
            # Create a proper YouTube search URL
            search_url = f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}"

            # Use yt-dlp to search for videos
            search_opts = {
                **self.ydl_opts,
                'extract_flat': True,
                'quiet': True,
                'no_warnings': True,
                'playlist_items': f'1-{max_results}',
            }

            with yt_dlp.YoutubeDL(search_opts) as ydl:
                # Extract search results
                search_results = ydl.extract_info(search_url, download=False)

                if not search_results or 'entries' not in search_results:
                    logger.warning(f"No search results found for query: {query}")
                    return []

                videos = []
                for entry in search_results['entries']:
                    if entry:
                        videos.append({
                            'title': entry.get('title', 'Unknown'),
                            'url': f"https://www.youtube.com/watch?v={entry.get('id', '')}",
                            'duration': entry.get('duration', 0),
                            'uploader': entry.get('uploader', 'Unknown'),
                            'view_count': entry.get('view_count', 0),
                        })

                return videos[:max_results]

        except Exception as e:
            logger.error(f"Failed to search YouTube for '{query}': {e}")
            return []

    async def get_available_formats(self, url: str) -> List[Dict[str, Any]]:
        """Get available formats for a video."""
        try:
            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                info = ydl.extract_info(url, download=False)
                return info.get('formats', [])
        except Exception as e:
            logger.error(f"Failed to get formats for {url}: {e}")
            return []

    async def download_with_progress(self, url: str, progress_callback=None) -> VideoInfo:
        """Download video with progress tracking."""
        def progress_hook(d):
            if progress_callback and d['status'] == 'downloading':
                progress_callback(d.get('downloaded_bytes', 0), d.get('total_bytes', 0))

        options = {
            **self._get_video_options(),
            'progress_hooks': [progress_hook] if progress_callback else []
        }

        try:
            with yt_dlp.YoutubeDL(options) as ydl:
                info = ydl.extract_info(url, download=True)

                video_path = None
                for file in self.download_path.iterdir():
                    if file.is_file() and file.suffix in ['.mp4', '.webm', '.mkv']:
                        video_path = str(file)
                        break

                return VideoInfo(
                    title=info.get('title', 'Unknown'),
                    duration=info.get('duration', 0),
                    platform=info.get('extractor', 'Unknown'),
                    video_path=video_path,
                    metadata=info
                )
        except Exception as e:
            logger.error(f"Failed to download video {url}: {e}")
            raise VideoUnavailableError(f"Could not download video: {e}")

    def _get_metadata_options(self) -> Dict[str, Any]:
        """Get options for metadata extraction."""
        return {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
            'skip_download': True,
            'retries': 3,
            'fragment_retries': 3,
            'ignoreerrors': False,
            'nocheckcertificate': True,
            'prefer_insecure': True,
            'geo_bypass': True,
            'http_headers': {
                'User-Agent': self.user_agents[0],
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
            },
        }

    def _get_fallback_metadata_options(self) -> Dict[str, Any]:
        """Get fallback options for metadata extraction."""
        return {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
            'skip_download': True,
            'retries': 2,
            'ignoreerrors': False,
            'nocheckcertificate': True,
            'prefer_insecure': True,
            'http_headers': {
                'User-Agent': self.user_agents[2],
                'Accept': '*/*',
                'Accept-Language': 'en-US,en;q=0.5',
            },
        }

    def _get_minimal_metadata_options(self) -> Dict[str, Any]:
        """Get minimal options for metadata extraction as last resort."""
        return {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
            'skip_download': True,
            'retries': 1,
            'ignoreerrors': False,
            'nocheckcertificate': True,
            'http_headers': {
                'User-Agent': self.user_agents[4],
                'Accept': '*/*',
            },
        }


# Example usage
async def main():
    """Example usage of YouTubeDLService."""
    service = YouTubeDLService()

    # Test URL
    test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    try:
        # Get metadata
        print("Getting metadata...")
        metadata = await service.get_metadata(test_url)
        print(f"Title: {metadata.title}")
        print(f"Duration: {metadata.duration} seconds")
        print(f"Platform: {metadata.platform}")

        # Extract audio
        print("\nExtracting audio...")
        audio_info = await service.extract_audio(test_url)
        print(f"Audio extracted: {audio_info.audio_path}")
        print(f"Format: {audio_info.format}")
        print(f"Duration: {audio_info.duration} seconds")

        # Cleanup
        await service.cleanup_files([audio_info.audio_path])

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
