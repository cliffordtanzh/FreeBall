#!/usr/bin/env python3
"""
Download all volleyball videos from a YouTube channel using yt-dlp.

Usage:
    python download_channel_videos.py <channel_url> [output_dir]

Example:
    python download_channel_videos.py "https://www.youtube.com/@cliffordtan6136" "data/raw"
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from datetime import datetime


CURATED_SET = [
    "https://www.youtube.com/watch?v=w-9rAg3lEJ8",
    "https://www.youtube.com/watch?v=f5kprw5Xc0o",
    "https://www.youtube.com/watch?v=3NaH97FTaQc&t=62s",
    "https://www.youtube.com/watch?v=XLu52udQUO8",
    "https://www.youtube.com/watch?v=0zgWIjgqVcA",
]


def download_curated_videos(video_urls: list, output_dir: str = "data/raw") -> dict:
    """
    Download specific videos from a curated list.

    Args:
        video_urls: List of YouTube video URLs to download
        output_dir: Output directory for downloaded videos

    Returns:
        Dictionary with download statistics
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # yt-dlp configuration
    ydl_opts = {
        "format": "best[ext=mp4]/best",
        "paths": {"home": str(output_path)},
        "outtmpl": "%(title)s [%(id)s].%(ext)s",
        "quiet": False,
        "no_warnings": False,
        "progress_hooks": [_progress_hook],
        "remote_components": "ejs:github",  # Enable remote JS challenge solver for YouTube
    }

    print(f"Downloading {len(video_urls)} video(s)")
    print(f"Output directory: {output_path.resolve()}")
    print("-" * 60)

    try:
        from yt_dlp import YoutubeDL
    except ImportError:
        print("Error: yt-dlp not installed. Install it with: pip install yt-dlp")
        sys.exit(1)

    downloaded_count = 0
    failed_count = 0

    try:
        with YoutubeDL(ydl_opts) as ydl:
            for i, url in enumerate(video_urls, 1):
                try:
                    print(f"\n[{i}/{len(video_urls)}] Downloading: {url}")
                    ydl.extract_info(url, download=True)
                    downloaded_count += 1
                except Exception as e:
                    print(f"  ✗ Failed: {e}")
                    failed_count += 1

        print("-" * 60)
        print(f"✓ Downloaded: {downloaded_count}/{len(video_urls)} video(s)")
        if failed_count > 0:
            print(f"✗ Failed: {failed_count} video(s)")
        print(f"Saved to: {output_path.resolve()}")

        return {
            "status": "success" if failed_count == 0 else "partial",
            "videos_downloaded": downloaded_count,
            "videos_failed": failed_count,
            "output_dir": str(output_path.resolve()),
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        print(f"✗ Error during download: {e}")
        return {
            "status": "error",
            "error": str(e),
            "output_dir": str(output_path.resolve()),
            "timestamp": datetime.now().isoformat(),
        }


def _progress_hook(d):
    """Progress hook for yt-dlp to print download status."""
    if d["status"] == "downloading":
        # Print simplified progress: filename and percentage
        filename = d.get("filename", "video")
        percent = d.get("_percent_str", "N/A")
        speed = d.get("_speed_str", "N/A")
        eta = d.get("_eta_str", "N/A")
        print(f"  {percent} at {speed} (ETA: {eta})")
    elif d["status"] == "finished":
        filename = d.get("filename", "video")
        print(f"  ✓ Finished: {Path(filename).name}")


def list_downloaded_videos(output_dir: str = "data/raw") -> list:
    """List all downloaded videos in the output directory."""
    output_path = Path(output_dir)
    if not output_path.exists():
        print(f"Directory not found: {output_dir}")
        return []

    video_files = list(output_path.glob("*.mp4"))
    print(f"\nFound {len(video_files)} MP4 video(s):")
    for video in sorted(video_files):
        size_mb = video.stat().st_size / (1024 * 1024)
        print(f"  - {video.name} ({size_mb:.1f} MB)")

    return video_files


if __name__ == "__main__":
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "data/raw"

    # Download curated videos
    result = download_curated_videos(CURATED_SET, output_dir)

    # Print result
    if result["status"] in ["success", "partial"]:
        print(f"\nDownload complete!")
        # List downloaded videos
        list_downloaded_videos(output_dir)
    else:
        print(f"\nDownload failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)
