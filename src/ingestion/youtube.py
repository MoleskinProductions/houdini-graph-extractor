"""YouTube video and transcript download using yt-dlp."""

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound


@dataclass
class TranscriptSegment:
    """A single transcript segment with timing."""

    start: float
    end: float
    text: str


@dataclass
class VideoInfo:
    """Information about a downloaded video."""

    video_id: str
    title: str
    duration_seconds: float
    video_path: Path
    transcript: list[TranscriptSegment]


class YouTubeIngester:
    """Download YouTube videos and fetch transcripts."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_video_id(self, url: str) -> str:
        """Extract video ID from various YouTube URL formats."""
        import re

        patterns = [
            r"(?:v=|/v/|youtu\.be/)([a-zA-Z0-9_-]{11})",
            r"(?:embed/)([a-zA-Z0-9_-]{11})",
            r"(?:shorts/)([a-zA-Z0-9_-]{11})",
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        raise ValueError(f"Could not extract video ID from URL: {url}")

    def download_video(self, url: str, resolution: str = "1080") -> tuple[Path, dict]:
        """
        Download video using yt-dlp at specified resolution.

        Returns:
            Tuple of (video_path, metadata_dict)
        """
        video_id = self.extract_video_id(url)
        output_template = str(self.output_dir / f"{video_id}.%(ext)s")

        cmd = [
            "yt-dlp",
            "-f",
            f"bestvideo[height<={resolution}]+bestaudio/best[height<={resolution}]",
            "--merge-output-format",
            "mp4",
            "-o",
            output_template,
            "--write-info-json",
            "--no-playlist",
            url,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"yt-dlp failed: {result.stderr}")

        # Find the downloaded files
        video_path = self.output_dir / f"{video_id}.mp4"
        info_path = self.output_dir / f"{video_id}.info.json"

        if not video_path.exists():
            # Try webm fallback
            video_path = self.output_dir / f"{video_id}.webm"
            if not video_path.exists():
                raise FileNotFoundError(f"Downloaded video not found for {video_id}")

        metadata = {}
        if info_path.exists():
            with open(info_path) as f:
                metadata = json.load(f)

        return video_path, metadata

    def fetch_transcript(self, video_id: str) -> list[TranscriptSegment]:
        """
        Fetch transcript for a video using youtube-transcript-api.

        Falls back to auto-generated captions if manual ones unavailable.
        """
        try:
            api = YouTubeTranscriptApi()
            transcript_list = api.list(video_id)

            # Prefer manual transcripts over auto-generated
            try:
                transcript = transcript_list.find_manually_created_transcript(["en"])
            except NoTranscriptFound:
                transcript = transcript_list.find_generated_transcript(["en"])

            raw_transcript = transcript.fetch()

            segments = []
            for entry in raw_transcript:
                segments.append(
                    TranscriptSegment(
                        start=entry.start,
                        end=entry.start + entry.duration,
                        text=entry.text,
                    )
                )
            return segments

        except (TranscriptsDisabled, NoTranscriptFound) as e:
            print(f"Warning: Could not fetch transcript: {e}")
            return []

    def fetch_transcript_via_ytdlp(self, url: str) -> list[TranscriptSegment]:
        """
        Alternative transcript fetch using yt-dlp subtitles.

        Useful when youtube-transcript-api fails.
        """
        video_id = self.extract_video_id(url)
        output_template = str(self.output_dir / f"{video_id}")

        cmd = [
            "yt-dlp",
            "--skip-download",
            "--write-auto-sub",
            "--write-sub",
            "--sub-lang",
            "en",
            "--sub-format",
            "json3",
            "-o",
            output_template,
            url,
        ]

        subprocess.run(cmd, capture_output=True, text=True)

        # Look for subtitle files
        for suffix in [".en.json3", ".en-orig.json3"]:
            sub_path = self.output_dir / f"{video_id}{suffix}"
            if sub_path.exists():
                return self._parse_json3_subtitles(sub_path)

        return []

    def _parse_json3_subtitles(self, path: Path) -> list[TranscriptSegment]:
        """Parse YouTube JSON3 subtitle format."""
        with open(path) as f:
            data = json.load(f)

        segments = []
        for event in data.get("events", []):
            if "segs" not in event:
                continue

            start_ms = event.get("tStartMs", 0)
            duration_ms = event.get("dDurationMs", 0)

            text_parts = []
            for seg in event["segs"]:
                if "utf8" in seg:
                    text_parts.append(seg["utf8"])

            text = "".join(text_parts).strip()
            if text:
                segments.append(
                    TranscriptSegment(
                        start=start_ms / 1000.0,
                        end=(start_ms + duration_ms) / 1000.0,
                        text=text,
                    )
                )

        return segments

    def ingest(self, url: str, resolution: str = "1080") -> VideoInfo:
        """
        Full ingestion pipeline: download video and fetch transcript.

        Args:
            url: YouTube video URL
            resolution: Maximum video resolution (default 1080p)

        Returns:
            VideoInfo with video path and transcript
        """
        video_id = self.extract_video_id(url)
        print(f"Ingesting video: {video_id}")

        # Download video
        print("Downloading video...")
        video_path, metadata = self.download_video(url, resolution)
        print(f"Video downloaded: {video_path}")

        # Fetch transcript
        print("Fetching transcript...")
        transcript = self.fetch_transcript(video_id)

        if not transcript:
            print("Trying yt-dlp subtitle fallback...")
            transcript = self.fetch_transcript_via_ytdlp(url)

        print(f"Transcript segments: {len(transcript)}")

        return VideoInfo(
            video_id=video_id,
            title=metadata.get("title", "Unknown"),
            duration_seconds=metadata.get("duration", 0),
            video_path=video_path,
            transcript=transcript,
        )
