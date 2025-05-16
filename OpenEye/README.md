# OpenEye

A comprehensive framework for real-time and asynchronous video processing from drone camera systems to web, VR clients, and computer vision pipelines.

## Features

- **Efficient Video Encoding/Decoding**
    - GPU acceleration (CUDA, VA-API)
    - H.264/H.265 format support
    - MPEG-TS containerization
    - MISB KLV metadata embedding for geo-referencing and telemetry

- **Camera Interface Support**
    - V4L2 and MIPI camera APIs
    - Comprehensive camera control (exposure, gain, ROI, white balance)
    - Raw image processing (demosaicing, noise reduction, HDR)

- **Streaming Capabilities**
    - WebRTC for low-latency video delivery to web clients
    - RTSP servers for compatibility with existing systems
    - Intelligent bandwidth adaptation based on network conditions

- **Computer Vision Integration**
    - Real-time object detection, tracking, and classification
    - PyTorch and TensorFlow model support
    - Model quantization for edge deployment

- **Cross-Platform Visualization**
    - Tauri v2 for desktop applications
    - PyO3 bindings to Rust components for performance-critical operations
    - Qt 6.5.2 with QML for interactive visualization

- **AI-Powered Analysis**
    - LangChain integration for natural language queries of video archives
    - OpenAI Assistants API for intelligent video analysis and summarization
    - Metadata indexing and retrieval

- **Edge Deployment**
    - Optimized for NVIDIA Jetson platforms
    - Supports various camera types and processing pipelines
    - Python-based configuration and orchestration

## Architecture

OpenEye is built with a modular architecture supporting:

```
openeye/
├── core/          # Core video processing components
├── streaming/     # Streaming protocols and servers
├── vision/        # Computer vision algorithms
├── encoding/      # Video codecs and encoding utilities
├── camera/        # Camera interfaces and controls
├── ai/            # AI integration and analysis
├── utils/         # Utility functions and helpers
└── ui/            # User interface components
      ├── qml/       # QML UI components
      └── web/       # Web interface components
```

## Requirements

- Python 3.11+
- Rust (for performance-critical components)
- CUDA 12.2.0+ (for GPU acceleration)
- Qt 6.5.2+ (for visualization)
- OpenCV 4.8.0+

## Installation

### From PyPI

```bash
pip install openeye
```

### From Source

```bash
git clone https://github.com/openeye/openeye.git
cd openeye
pip install -e ".[dev]"
```

For CUDA support:

```bash
pip install -e ".[dev,cuda]"
```

## Quick Start

### Basic Video Processing

```python
from openeye.core import VideoProcessor
from openeye.vision import YOLODetector

# Initialize detector and processor
detector = YOLODetector()
processor = VideoProcessor(detector=detector)

# Open a video file
processor.open("drone_footage.mp4")

# Process and display
processor.start()
```

### Streaming to RTSP

```python
from openeye.core import VideoProcessor
from openeye.streaming import RTSPServer

# Create processor and RTSP server
processor = VideoProcessor()
rtsp_server = RTSPServer(port=8554)

# Connect processor to server
processor.add_output(rtsp_server)
rtsp_server.start()

# Start processing
processor.open("drone_footage.mp4")
processor.start()

# RTSP stream available at: rtsp://localhost:8554/stream
```

### AI-Powered Video Analysis

```python
from openeye.ai import VideoAssistant

# Initialize the assistant
assistant = VideoAssistant()

# Analyze a video file
result = assistant.summarize_video("drone_reconnaissance.mp4")
print(result["summary"])
```

### Natural Language Queries

```python
from openeye.ai import VideoQueryEngine

# Initialize the query engine
query_engine = VideoQueryEngine()

# Index video metadata
query_engine.index_video_metadata({
      "video_id": "drone_1",
      "timestamp": 1667347200,
      "location": "34.052235,-118.243683",
      "duration": 120,
      "description": "Aerial footage of urban area",
})

# Query the video database
result = query_engine.query("Show me videos with urban areas from November 2022")
print(result["answer"])
```

## Desktop Application

OpenEye includes a desktop application built with Tauri and QML:

```bash
cd OpenEye
cargo tauri dev
```

## Documentation

Comprehensive documentation is available at [https://docs.openeye.ai](https://docs.openeye.ai)

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

OpenEye is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```

Now let's create a basic configuration file:

```yaml:OpenEye/config/default.yaml
# OpenEye Default Configuration

# Core settings
core:
    log_level: info
    num_threads: 4
    cache_dir: ~/.openeye/cache
    temp_dir: ~/.openeye/temp

# Camera settings
camera:
    default_device: 0  # Default camera device
    v4l2:
      enabled: true
      controls:
        exposure: auto
        gain: auto
        white_balance: auto
    mipi:
      enabled: false
      width: 1920
      height: 1080
      fps: 30

# Vision settings
vision:
    detector:
      type: yolo
      model: yolov8n.pt
      confidence_threshold: 0.25
      nms_threshold: 0.45
      device: cuda
    tracker:
      type: sort
      max_age: 30
      min_hits: 3
      iou_threshold: 0.3

# Streaming settings
streaming:
    rtsp:
      enabled: true
      port: 8554
      stream_name: stream
      codec: h264
      bitrate: 2000000
    webrtc:
      enabled: true
      port: 8080
      stun_server: stun:stun.l.google.com:19302
      signaling_path: /ws

# AI settings
ai:
    langchain:
      enabled: true
      db_path: ~/.openeye/vectordb
      llm_model: gpt-4o
    assistants:
      enabled: true
      cache_dir: ~/.openeye/assistants
      model: gpt-4-vision-preview

# Encoding settings
encoding:
    default_codec: h264
    default_container: mp4
    hardware_acceleration: true
    klv:
      enabled: true
      misb_standard: ST0601.16

# UI settings
ui:
    theme: dark
    layout: default
    enable_metrics: true
    fps_limit: 30
```

Let me also create a Docker configuration for deployment:

```dockerfile:OpenEye/Dockerfile
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04 as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
      python3-dev \
      python3-pip \
      build-essential \
      cmake \
      git \
      curl \
      libavcodec-dev \
      libavformat-dev \
      libswscale-dev \
      libv4l-dev \
      libxvidcore-dev \
      libx264-dev \
      libgtk-3-dev \
      libboost-all-dev \
      libgstreamer1.0-dev \
      libgstreamer-plugins-base1.0-dev \
      libdc1394-dev \
      libatlas-base-dev \
      gfortran \
      wget \
      unzip \
      && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Create app directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the source code
COPY . .

# Build Rust components
RUN cargo build --release

# Build the Python package
RUN pip3 install -e .

# Runtime image
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
      python3 \
      python3-pip \
      libavcodec59 \
      libavformat59 \
      libswscale6 \
      libv4l-0 \
      libgtk-3-0 \
      libgstreamer1.0-0 \
      libgstreamer-plugins-base1.0-0 \
      libdc1394-25 \
      libatlas-base-dev \
      && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy from builder
COPY --from=builder /app/target/release /app/target/release
COPY --from=builder /app/src /app/src
COPY --from=builder /app/config /app/config
COPY --from=builder /root/.local /root/.local

# Set environment variables
ENV PYTHONPATH=/app
ENV LD_LIBRARY_PATH=/app/target/release:${LD_LIBRARY_PATH}

# Expose ports
EXPOSE 8080 8554

# Run the application
CMD ["python3", "-m", "openeye.server"]
```