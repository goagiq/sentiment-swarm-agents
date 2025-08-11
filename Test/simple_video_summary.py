#!/usr/bin/env python3
"""
Simple script to generate a summary of the Innovation Workshop video.
"""

import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').absolute() / 'src'))

from core.youtube_dl_service import YouTubeDLService

async def simple_video_analysis():
    print('Analyzing video: https://www.youtube.com/watch?v=YyUM1-y_yQM')
    
    # Initialize YouTube service
    youtube_service = YouTubeDLService(download_path='./temp/video_analysis')
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
            print(f'   Description: {metadata.description[:300]}...')
            
            # Generate a basic summary from metadata
            print('\n' + '='*80)
            print('VIDEO SUMMARY')
            print('='*80)
            print(f"Title: {metadata.title}")
            print(f"Duration: {metadata.duration} seconds ({metadata.duration//60} minutes {metadata.duration%60} seconds)")
            print(f"Upload Date: {metadata.upload_date}")
            print(f"View Count: {metadata.view_count:,}")
            print(f"Platform: {metadata.platform}")
            print(f"Available Formats: {len(metadata.available_formats)}")
            print(f"\nDescription: {metadata.description}")
            
            # Step 2: Try to download video for further analysis
            print('\nAttempting to download video for content analysis...')
            try:
                video_info = await youtube_service.download_video(video_url)
                if video_info and video_info.video_path:
                    print(f'Video downloaded successfully: {video_info.video_path}')
                    print('Video is available for further analysis if needed.')
                else:
                    print('Video download failed, but metadata analysis is complete.')
            except Exception as e:
                print(f'Video download failed: {e}')
                print('Metadata analysis is still complete.')
                
        else:
            print('Failed to extract metadata')
            return
            
    except Exception as e:
        print(f'Error during video analysis: {e}')
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    asyncio.run(simple_video_analysis())
