import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').absolute() / 'src'))

from core.youtube_dl_service import YouTubeDLService
from core.video_processing_service import VideoProcessingService
from agents.unified_vision_agent import UnifiedVisionAgent
from config.config import SentimentConfig

async def analyze_video():
    print('Analyzing video: https://www.youtube.com/watch?v=YyUM1-y_yQM')
    
    # Initialize services
    config = SentimentConfig()
    youtube_service = YouTubeDLService(download_path='./temp/video_analysis')
    video_service = VideoProcessingService(download_path='./temp/video_analysis')
    vision_agent = UnifiedVisionAgent(config)
    
    video_url = 'https://www.youtube.com/watch?v=YyUM1-y_yQM'
    
    try:
        # Step 1: Extract metadata
        print('Extracting video metadata...')
        metadata = await youtube_service.get_metadata(video_url)
        if metadata:
            print('Metadata extracted successfully')
            print(f'   Title: {metadata.title}')
            print(f'   Duration: {metadata.duration} seconds')
            print(f'   Upload Date: {metadata.upload_date}')
            print(f'   View Count: {metadata.view_count}')
            print(f'   Description: {metadata.description[:200]}...')
        else:
            print('Failed to extract metadata')
            return
        
        # Step 2: Download video for analysis
        print('Downloading video for analysis...')
        video_info = await youtube_service.download_video(video_url)
        if video_info and video_info.video_path:
            print(f'Video downloaded successfully: {video_info.video_path}')
            
            # Step 3: Extract audio for analysis
            print('Extracting audio for analysis...')
            try:
                audio_info = await youtube_service.extract_audio_workaround(video_url)
                if audio_info and audio_info.audio_path:
                    print(f'Audio extracted successfully: {audio_info.audio_path}')
                else:
                    print('Audio extraction failed, proceeding with video analysis only')
            except Exception as e:
                print(f'Audio extraction failed: {e}')
            
            # Step 4: Analyze video content
            print('Analyzing video content...')
            analysis_result = await video_service.analyze_video(
                video_input=video_url,
                extract_audio=True,
                extract_frames=True,
                num_frames=10
            )
            
            if analysis_result:
                print('Video analysis completed successfully')
                print(f'   Video Path: {analysis_result.video_path}')
                print(f'   Audio Path: {analysis_result.audio_path}')
                print(f'   Frames Extracted: {len(analysis_result.frames) if analysis_result.frames else 0}')
                
                # Step 5: Generate summary using vision agent
                print('Generating comprehensive summary...')
                summary = await vision_agent.analyze_video(video_url)
                if summary:
                    print('Summary generated successfully')
                    print('\n' + '='*80)
                    print('VIDEO SUMMARY')
                    print('='*80)
                    print(summary)
                else:
                    print('Failed to generate summary')
            else:
                print('Video analysis failed')
        else:
            print('Video download failed')
            
    except Exception as e:
        print(f'Error during video analysis: {e}')
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    asyncio.run(analyze_video())
