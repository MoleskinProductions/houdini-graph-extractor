"""Frame extraction from video using ffmpeg."""

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExtractedFrame:
    """Information about an extracted frame."""

    path: Path
    timestamp: float
    priority: str
    source_event_type: str | None = None


class FrameExtractor:
    """Extract frames from video at specific timestamps using ffmpeg."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.frames_dir = output_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)

    def extract_single_frame(
        self,
        video_path: Path,
        timestamp: float,
        output_name: str | None = None,
    ) -> Path:
        """
        Extract a single frame at the given timestamp.

        Args:
            video_path: Path to video file
            timestamp: Time in seconds
            output_name: Optional output filename (without extension)

        Returns:
            Path to extracted frame
        """
        if output_name is None:
            output_name = f"frame_{timestamp:.2f}".replace(".", "_")

        output_path = self.frames_dir / f"{output_name}.png"

        cmd = [
            "ffmpeg",
            "-y",  # Overwrite
            "-ss",
            str(timestamp),
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-q:v",
            "2",  # High quality
            str(output_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr}")

        return output_path

    def extract_frames_at_timestamps(
        self,
        video_path: Path,
        timestamps: list[dict],
    ) -> list[ExtractedFrame]:
        """
        Extract frames at multiple timestamps.

        Args:
            video_path: Path to video file
            timestamps: List of timestamp records from TranscriptParser.get_sampling_timestamps()

        Returns:
            List of ExtractedFrame objects
        """
        frames = []

        for i, ts_record in enumerate(timestamps):
            timestamp = ts_record["time"]
            priority = ts_record.get("priority", "normal")
            source_event = ts_record.get("source_event")

            output_name = f"frame_{i:05d}_{timestamp:.2f}".replace(".", "_")

            try:
                path = self.extract_single_frame(video_path, timestamp, output_name)

                event_type = None
                if source_event is not None:
                    # Handle both Enum (ActionEvent) and string (EnhancedActionEvent)
                    et = source_event.event_type
                    event_type = et.value if hasattr(et, 'value') else et

                frames.append(
                    ExtractedFrame(
                        path=path,
                        timestamp=timestamp,
                        priority=priority,
                        source_event_type=event_type,
                    )
                )
            except RuntimeError as e:
                print(f"Warning: Failed to extract frame at {timestamp}s: {e}")

        return frames

    def extract_at_interval(
        self,
        video_path: Path,
        start_time: float = 0,
        end_time: float | None = None,
        interval: float = 1.0,
    ) -> list[ExtractedFrame]:
        """
        Extract frames at regular intervals.

        Args:
            video_path: Path to video file
            start_time: Start time in seconds
            end_time: End time in seconds (None = end of video)
            interval: Interval between frames in seconds

        Returns:
            List of ExtractedFrame objects
        """
        # Get video duration if end_time not specified
        if end_time is None:
            end_time = self._get_video_duration(video_path)

        timestamps = []
        t = start_time
        while t <= end_time:
            timestamps.append({"time": t, "priority": "normal"})
            t += interval

        return self.extract_frames_at_timestamps(video_path, timestamps)

    def _get_video_duration(self, video_path: Path) -> float:
        """Get video duration using ffprobe."""
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffprobe failed: {result.stderr}")

        return float(result.stdout.strip())

    def cleanup(self):
        """Remove all extracted frames."""
        for frame in self.frames_dir.glob("*.png"):
            frame.unlink()
