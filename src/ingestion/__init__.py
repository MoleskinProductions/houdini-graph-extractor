"""Ingestion modules for video and transcript download."""

from .youtube import YouTubeIngester, VideoInfo
from .frame_extractor import FrameExtractor

__all__ = ["YouTubeIngester", "VideoInfo", "FrameExtractor"]
