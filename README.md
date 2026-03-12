#  Video Analyzer Lite

Real-time gender & age prediction from webcam or video file.

## Setup

```bash
pip install -r requirements.txt
python analyzer.py
```

That's it. Models download automatically on first run (~900 MB).

## Commands

```bash
# Webcam
python analyzer.py

# Video file
python analyzer.py --source video.mp4

# Save output
python analyzer.py --source video.mp4 --output result.mp4

# Download models manually
python analyzer.py --download-models
```

Press **Q** to quit.
