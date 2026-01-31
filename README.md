# Houdini Graph Extractor

Extract Houdini node graphs from YouTube tutorial videos using vision-language models.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Usage

```bash
houdini-extract https://youtube.com/watch?v=VIDEO_ID -o output.json
```

## Requirements

- Python 3.10+
- yt-dlp
- ffmpeg
- vLLM or Ollama with a Qwen VL model
