#!/usr/bin/env python3
"""
Test script to demonstrate yt-dlp capabilities for YouTube video metadata extraction.
"""

import yt_dlp
import json


def test_ytdlp_youtube_extraction():
    """Test yt-dlp extraction of YouTube video metadata."""
    
    print("🎬 Testing yt-dlp YouTube Video Metadata Extraction")
    print("=" * 60)
    
    # Test URL
    test_url = "https://www.youtube.com/watch?v=UrT72MNbSI8"
    
    # Configure yt-dlp options
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': False,  # Get full metadata
        'writeinfojson': False,  # Don't save to file
        'skip_download': True,   # Don't download video
    }
    
    try:
        print(f"🔗 Testing URL: {test_url}")
        print("-" * 40)
        
        # Create yt-dlp instance
        ydl = yt_dlp.YoutubeDL(ydl_opts)
        
        # Extract info
        print("📥 Extracting metadata...")
        info = ydl.extract_info(test_url, download=False)
        
        # Display key metadata
        print(f"📺 Title: {info.get('title', 'N/A')}")
        print(f"👤 Channel: {info.get('uploader', 'N/A')}")
        print(f"📅 Upload Date: {info.get('upload_date', 'N/A')}")
        print(f"⏱️  Duration: {info.get('duration', 'N/A')} seconds")
        print(f"👀 View Count: {info.get('view_count', 'N/A')}")
        print(f"👍 Like Count: {info.get('like_count', 'N/A')}")
        print(f"💬 Comment Count: {info.get('comment_count', 'N/A')}")
        
        # Description (truncated)
        description = info.get('description', '')
        if description:
            print(f"📝 Description Preview: {description[:200]}...")
        else:
            print("📝 Description: N/A")
        
        # Tags
        tags = info.get('tags', [])
        if tags:
            print(f"🏷️  Tags: {', '.join(tags[:5])}{'...' if len(tags) > 5 else ''}")
        else:
            print("🏷️  Tags: N/A")
        
        # Categories
        categories = info.get('categories', [])
        if categories:
            print(f"📂 Categories: {', '.join(categories)}")
        else:
            print("📂 Categories: N/A")
        
        # Thumbnail
        thumbnails = info.get('thumbnails', [])
        if thumbnails:
            print(f"🖼️  Thumbnails: {len(thumbnails)} available")
        else:
            print("🖼️  Thumbnails: N/A")
        
        # Check for transcript availability
        print(f"📄 Has Transcript: {info.get('subtitles', {}) != {}}")
        
        # Available formats
        formats = info.get('formats', [])
        if formats:
            print(f"🎥 Available Formats: {len(formats)}")
        else:
            print("🎥 Available Formats: N/A")
        
        print("\n✅ yt-dlp extraction successful!")
        
        # Show what we could use for sentiment analysis
        print("\n🎯 Sentiment Analysis Opportunities:")
        print("1. 📝 Video title analysis")
        print("2. 📄 Description text analysis") 
        print("3. 🏷️  Tags analysis")
        print("4. 💬 Comments analysis (if available)")
        print("5. 📊 Engagement metrics (views, likes)")
        print("6. 📄 Transcript analysis (if available)")
        
        return info
        
    except Exception as e:
        print(f"❌ Error extracting metadata: {e}")
        return None


def test_ytdlp_transcript_extraction():
    """Test transcript extraction if available."""
    
    print("\n📄 Testing Transcript Extraction")
    print("=" * 40)
    
    test_url = "https://www.youtube.com/watch?v=UrT72MNbSI8"
    
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': ['en'],
        'skip_download': True,
    }
    
    try:
        ydl = yt_dlp.YoutubeDL(ydl_opts)
        info = ydl.extract_info(test_url, download=False)
        
        subtitles = info.get('subtitles', {})
        auto_subtitles = info.get('automatic_captions', {})
        
        if subtitles:
            print("✅ Manual subtitles available")
            for lang, formats in subtitles.items():
                print(f"   Language: {lang}, Formats: {len(formats)}")
        
        if auto_subtitles:
            print("✅ Automatic captions available")
            for lang, formats in auto_subtitles.items():
                print(f"   Language: {lang}, Formats: {len(formats)}")
        
        if not subtitles and not auto_subtitles:
            print("❌ No subtitles/captions available")
            
    except Exception as e:
        print(f"❌ Error extracting transcripts: {e}")


def main():
    """Main test function."""
    print("🚀 yt-dlp YouTube Integration Test")
    print("=" * 50)
    
    # Test basic metadata extraction
    info = test_ytdlp_youtube_extraction()
    
    # Test transcript extraction
    test_ytdlp_transcript_extraction()
    
    print("\n📋 Summary:")
    print("✅ yt-dlp can extract rich YouTube metadata")
    print("✅ Multiple data sources available for sentiment analysis")
    print("✅ Much better than just scraping page HTML")
    print("✅ Can potentially extract transcripts for text analysis")


if __name__ == "__main__":
    main()
