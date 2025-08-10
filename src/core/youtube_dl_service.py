"""
YouTube-DL service for video processing and analysis.
Provides core functionality for downloading videos, extracting audio,
and processing video content for sentiment analysis.
"""

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

import yt_dlp
from loguru import logger


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
    """Base exception for YouTube-DL operations."""
    pass


class VideoUnavailableError(YouTubeDLError):
    """Video is unavailable or restricted."""
    pass


class NetworkError(YouTubeDLError):
    """Network-related errors."""
    pass


class YouTubeDLService:
    """Service for YouTube-DL operations."""
    
    def __init__(self, download_path: str = "./temp/videos"):
        self.download_path = Path(download_path)
        self.download_path.mkdir(parents=True, exist_ok=True)
        
        # Default yt-dlp options
        self.ydl_opts = {
            'outtmpl': str(self.download_path / '%(title)s.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
        }
    
    def _get_audio_options(self) -> Dict[str, Any]:
        """Get options optimized for audio extraction."""
        return {
            **self.ydl_opts,
            'format': 'bestaudio[ext=m4a]/bestaudio[ext=mp3]/bestaudio',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': str(self.download_path / '%(title)s.%(ext)s'),
        }
    
    def _get_video_options(self) -> Dict[str, Any]:
        """Get options optimized for video download."""
        return {
            **self.ydl_opts,
            'format': 'best[height<=720]/best',
            'outtmpl': str(self.download_path / '%(title)s.%(ext)s'),
        }
    
    async def get_metadata(self, url: str) -> VideoMetadata:
        """Get video metadata without downloading."""
        try:
            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                info = ydl.extract_info(url, download=False)
                
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
            raise VideoUnavailableError(f"Could not get metadata: {e}")
    
    async def download_video(self, url: str) -> VideoInfo:
        """Download video and return metadata."""
        try:
            with yt_dlp.YoutubeDL(self._get_video_options()) as ydl:
                info = ydl.extract_info(url, download=True)
                
                # Find the downloaded file
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
    
    async def extract_audio(self, url: str) -> AudioInfo:
        """Extract audio from video URL."""
        try:
            with yt_dlp.YoutubeDL(self._get_audio_options()) as ydl:
                info = ydl.extract_info(url, download=True)
                
                # Find the extracted audio file
                audio_path = None
                for file in self.download_path.iterdir():
                    if file.is_file() and file.suffix in ['.mp3', '.m4a', '.wav']:
                        audio_path = str(file)
                        break
                
                return AudioInfo(
                    audio_path=audio_path,
                    format=Path(audio_path).suffix if audio_path else 'mp3',
                    duration=info.get('duration', 0),
                    bitrate=192,
                    metadata=info
                )
        except Exception as e:
            logger.error(f"Failed to extract audio from {url}: {e}")
            raise VideoUnavailableError(f"Could not extract audio: {e}")
    
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
