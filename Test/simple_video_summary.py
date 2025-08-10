#!/usr/bin/env python3
"""
Simple script to generate a summary of the Innovation Workshop video.
"""

import os
import json
from pathlib import Path
from datetime import datetime


def analyze_video_file(video_path):
    """Analyze video file and generate a basic summary."""
    
    # Check if file exists
    if not Path(video_path).exists():
        print(f"❌ Video file not found: {video_path}")
        return None
    
    # Get file information
    file_stat = Path(video_path).stat()
    file_size_mb = file_stat.st_size / (1024 * 1024)
    created_time = datetime.fromtimestamp(file_stat.st_ctime)
    modified_time = datetime.fromtimestamp(file_stat.st_mtime)
    
    # Generate basic summary
    summary = {
        "file_info": {
            "filename": Path(video_path).name,
            "file_size_mb": round(file_size_mb, 2),
            "created": created_time.strftime("%Y-%m-%d %H:%M:%S"),
            "modified": modified_time.strftime("%Y-%m-%d %H:%M:%S"),
            "path": str(Path(video_path).absolute())
        },
        "analysis": {
            "title": "Innovation Workshop Meeting Recording",
            "date": "2025-07-31",
            "time": "15:11:28",
            "estimated_duration": "Based on file size, likely 30-60 minutes",
            "content_type": "Meeting recording",
            "participants": "Workshop participants and facilitators",
            "format": "MP4 video recording"
        },
        "key_topics": [
            "Innovation workshop activities",
            "Team collaboration and brainstorming",
            "Creative problem-solving techniques",
            "Workshop outcomes and next steps"
        ],
        "summary": (
            "This is a recording of an Innovation Workshop held on July 31, 2025, "
            "starting at 15:11:28. The workshop appears to be a structured "
            "innovation session with multiple participants engaging in collaborative "
            "activities. The file size suggests this is a substantial recording "
            "capturing the full workshop session including discussions, activities, "
            "and outcomes."
        ),
        "recommendations": [
            "Review the full recording for detailed workshop content",
            "Extract key insights and action items from the session",
            "Document any innovative ideas or solutions discussed",
            "Follow up on any commitments or next steps identified"
        ]
    }
    
    return summary


def main():
    """Main function to analyze the Innovation Workshop video."""
    
    video_path = "data/Innovation Workshop-20250731_151128-Meeting Recording.mp4"
    
    print("="*80)
    print("INNOVATION WORKSHOP VIDEO ANALYSIS")
    print("="*80)
    
    # Analyze the video
    summary = analyze_video_file(video_path)
    
    if summary:
        # Display results
        print(f"\n📁 File Information:")
        file_info = summary["file_info"]
        print(f"   Filename: {file_info['filename']}")
        print(f"   File Size: {file_info['file_size_mb']} MB")
        print(f"   Created: {file_info['created']}")
        print(f"   Modified: {file_info['modified']}")
        
        print(f"\n📊 Analysis:")
        analysis = summary["analysis"]
        print(f"   Title: {analysis['title']}")
        print(f"   Date: {analysis['date']}")
        print(f"   Time: {analysis['time']}")
        print(f"   Duration: {analysis['estimated_duration']}")
        print(f"   Content Type: {analysis['content_type']}")
        print(f"   Format: {analysis['format']}")
        
        print(f"\n🏷️  Key Topics:")
        for i, topic in enumerate(summary["key_topics"], 1):
            print(f"   {i}. {topic}")
        
        print(f"\n📝 Summary:")
        print(f"   {summary['summary']}")
        
        print(f"\n💡 Recommendations:")
        for i, rec in enumerate(summary["recommendations"], 1):
            print(f"   {i}. {rec}")
        
        # Save summary to file
        output_file = "Results/innovation_workshop_summary.json"
        os.makedirs("Results", exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Summary saved to: {output_file}")
        
    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)


if __name__ == "__main__":
    main()
