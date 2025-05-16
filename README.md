<img src="OpenEye.svg" alt="OpenEye Logo" width="150" style="display: block; margin-left: auto; margin-right: auto;"/>

# OpenEye: Intelligent Video Processing Framework

**OpenEye** is a comprehensive, high-performance framework designed for real-time and asynchronous video processing. It empowers developers to build sophisticated applications for drone camera systems, surveillance, web/VR clients, and advanced computer vision pipelines. Leveraging GPU acceleration and a modular architecture, OpenEye provides the backbone for demanding video-centric tasks.

## Overview

In an era where video data is ubiquitous, OpenEye offers a robust solution to capture, analyze, stream, and manage video feeds with unparalleled efficiency and intelligence. Whether you're tracking objects from an aerial drone, enabling low-latency streaming to a global audience, or building AI-powered video analytics, OpenEye provides the tools and infrastructure to accelerate your development and deployment.

Our mission is to deliver a professional-grade, extensible platform that integrates cutting-edge technologies in video encoding, computer vision, AI, and streaming, making them accessible and easy to use.

## Key Features

-   👁️ **Advanced Video Acquisition & Control**:
    -   Supports industry-standard camera interfaces (V4L2, MIPI).
    -   Granular control over camera parameters: exposure, gain, ROI, white balance.
    -   On-the-fly raw image processing: demosaicing, noise reduction, HDR.

-   🚀 **High-Performance Encoding/Decoding**:
    -   Hardware-accelerated video processing leveraging NVIDIA CUDA and VA-API.
    -   Support for modern codecs: H.264, H.265 for optimal compression and quality.
    -   Standard MPEG-TS containerization for broad compatibility.
    -   MISB KLV metadata embedding for critical geo-referencing and telemetry data.

-   🌐 **Versatile Streaming Capabilities**:
    -   Ultra-low latency streaming to web clients using WebRTC.
    -   Robust RTSP server implementation for integration with existing VMS and security systems.
    -   Intelligent adaptive bitrate streaming based on network conditions.

-   🧠 **Integrated Computer Vision & AI**:
    -   Real-time object detection, classification, and tracking.
    -   Seamless integration with popular deep learning frameworks: PyTorch and TensorFlow.
    -   Support for model quantization (e.g., TensorRT) for optimized performance on edge devices.
    -   LangChain integration for natural language querying of video archives.
    -   OpenAI API utilization for sophisticated video summarization and intelligent analysis.

-   💻 **Cross-Platform Desktop & Web Visualization**:
    -   Modern desktop application powered by Tauri v2, offering native performance.
    -   Rich, interactive QML (Qt 6.5+) components for custom user interfaces.
    -   High-performance Rust components bound with PyO3 for critical operations.
    -   Web interface components for browser-based interaction.

-   🛰️ **Edge Deployment Ready**:
    -   Optimized for deployment on NVIDIA Jetson platforms and other edge computing devices.
    -   Flexible Python-based configuration and orchestration for various deployment scenarios.

## Architecture

OpenEye is architected with modularity and extensibility at its core. Components are organized logically to promote separation of concerns and ease of development:

```
.
├── src/
│   ├── openeye/      # Core Python library for OpenEye
│   │   ├── core/          # Core video processing pipeline, frame management
│   │   ├── streaming/     # RTSP, WebRTC servers and clients
│   │   ├── vision/        # Computer vision algorithms (detection, tracking)
│   │   ├── encoding/      # Video/audio codecs, KLV metadata
│   │   ├── camera/        # Camera interfaces and hardware controls
│   │   ├── ai/            # AI integrations (LangChain, OpenAI)
│   │   ├── telemetry/     # Telemetry data handling
│   │   ├── utils/         # Common utilities, logging
│   │   └── ui/            # UI related backends (if any, distinct from qml/web)
│   ├── config/        # Default configuration files (e.g., default.yaml)
│   └── ui/
│       └── qml/       # QML UI components for Tauri desktop app
├── src-tauri/         # Rust backend for the Tauri desktop application
│   └── src/
│       └── main.rs    # Rust entry point for Tauri
├── web/
│   └── src/           # Web interface components (e.g., React/Vue/Svelte)
│       ├── components/
│       └── services/
├── tests/             # Unit, integration, and benchmark tests
│   ├── unit/
│   ├── integration/
│   └── benchmarks/
├── docs/              # Project documentation
├── examples/          # Example scripts and use-cases
├── .github/           # GitHub Actions workflows, issue templates, etc.
├── Dockerfile         # Docker configuration for deployment
├── pyproject.toml     # Python project metadata and dependencies (PEP 621)
├── setup.py           # Setuptools configuration (can be minimal with pyproject.toml)
├── README.md          # This file
├── DOCUMENTATION.md   # More in-depth documentation
├── LICENSE            # Project license (MIT)
└── OpenEye.svg        # Project Logo
```
*Note: The `openeye` Python package is located under `src/`. The project structure reflects a common layout for Python projects with Rust (Tauri) components.*

## System Requirements

-   **Operating System**: Linux (recommended for full CUDA and V4L2 support), macOS, Windows (Tauri works, some hardware features may vary).
-   **Python**: 3.10+
-   **Rust**: Latest stable version (for building Tauri components and performance-critical modules).
-   **NVIDIA CUDA**: 12.2.0+ (for GPU-accelerated features).
-   **Qt**: 6.5.2+ (for QML-based desktop UI).
-   **OpenCV**: 4.8.0+
-   **Build Essentials**: `cmake`, C++ compiler (gcc/clang/msvc).

## Installation

### Prerequisites

Ensure you have Python, Rust, CMake, and a C++ compiler installed. For GPU features, ensure your NVIDIA drivers and CUDA toolkit are correctly set up.

### From Source (Recommended for Developers)

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/llamasearchai/OpenEye.git
    cd OpenEye
    ```

2.  **Install Python Dependencies**:
    It's highly recommended to use a virtual environment:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```
    Install the package in editable mode with development dependencies:
    ```bash
    pip install -e ".[dev]"
    ```
    For NVIDIA CUDA accelerated features, include the `cuda` extra:
    ```bash
    pip install -e ".[dev,cuda]"
    ```

3.  **Build Rust Components**:
    If there are standalone Rust binaries or specific build steps for Rust modules not handled by `pip install` (e.g. if not using `setuptools-rust`), you might need to run:
    ```bash
    cargo build --release
    ```
    (Often, for PyO3 projects, `pip install` handles Rust compilation.)

### Using Docker

A `Dockerfile` is provided for building and running OpenEye in a containerized environment. This is an excellent option for deployment and ensuring a consistent runtime.
```bash
# Build the Docker image
docker build -t openeye-app .

# Run the Docker container (example)
docker run -p 8080:8080 -p 8554:8554 --gpus all openeye-app
```
Refer to the `Dockerfile` and potentially a `docker-compose.yml` (if added) for more detailed build and run instructions.

## Quick Start & Usage Examples

Below are conceptual examples. Actual API and module names might differ based on the final implementation within `src/openeye/`.

### 1. Basic Video File Processing with Object Detection

```python
# Ensure your Python environment is active and OpenEye is installed.
# main_app.py (example script in the project root or examples/)

from openeye.core import VideoProcessor
from openeye.vision import YOLODetector # Assuming a YOLO detector class
from openeye.utils import logger # Assuming a configured logger

def main():
    logger.info("Initializing OpenEye video processing...")

    # Initialize detector (ensure model path is correct, e.g., downloaded or in a models/ dir)
    try:
        detector = YOLODetector(model_path="yolov8n.pt", device="cuda") # or "cpu"
    except Exception as e:
        logger.error(f"Failed to initialize detector: {e}")
        return

    # Initialize video processor with the detector
    processor = VideoProcessor(detector=detector)

    video_file = "path/to/your/drone_footage.mp4" # Replace with an actual video file

    try:
        logger.info(f"Opening video file: {video_file}")
        processor.open_source(video_file)
        
        # Example: Add an output, e.g., saving to a new file or displaying
        # processor.add_output_sink(FileOutputSink("processed_output.mp4"))
        # processor.add_output_sink(DisplaySink())


        logger.info("Starting video processing...")
        processor.start() # This might be a blocking call or start a thread

        # If start() is non-blocking, you might need to wait or handle completion
        # processor.wait_for_completion() 

    except Exception as e:
        logger.error(f"Error during video processing: {e}")
    finally:
        logger.info("Shutting down video processor.")
        processor.close()

if __name__ == "__main__":
    main()
```

### 2. Streaming from a Camera to RTSP and WebRTC

```python
# stream_example.py

from openeye.core import VideoProcessor
from openeye.camera import V4L2Camera # Example camera source
from openeye.streaming import RTSPServer, WebRTCServer
from openeye.utils import logger

def main():
    logger.info("Initializing OpenEye streaming server...")

    try:
        # Initialize camera source (e.g., /dev/video0)
        camera = V4L2Camera(device_id=0) 
        # camera = MIPICamera(...) # Alternative for MIPI
    except Exception as e:
        logger.error(f"Failed to initialize camera: {e}")
        return

    processor = VideoProcessor(source=camera)

    # Configure and add RTSP server
    rtsp_server = RTSPServer(port=8554, stream_name="live/drone")
    processor.add_output_sink(rtsp_server)

    # Configure and add WebRTC server
    webrtc_server = WebRTCServer(port=8080, stun_server="stun:stun.l.google.com:19302")
    processor.add_output_sink(webrtc_server)
    
    try:
        logger.info("Starting processing and streaming...")
        rtsp_server.start() # Start individual servers
        webrtc_server.start()
        processor.start() # Start the main processing loop

        logger.info(f"RTSP stream available at: rtsp://localhost:{rtsp_server.port}/{rtsp_server.stream_name}")
        logger.info(f"WebRTC signaling available at: http://localhost:{webrtc_server.port}/ws")
        
        # Keep the main thread alive or implement proper shutdown logic
        while True:
            pass # Or use a condition variable, asyncio loop, etc.

    except KeyboardInterrupt:
        logger.info("Shutdown signal received.")
    except Exception as e:
        logger.error(f"Error during streaming: {e}")
    finally:
        logger.info("Stopping servers and processor...")
        processor.close()
        rtsp_server.stop()
        webrtc_server.stop()
        logger.info("OpenEye streaming server shut down.")

if __name__ == "__main__":
    main()
```

### 3. AI-Powered Video Analysis (Conceptual)

```python
# ai_analysis_example.py

from openeye.ai import VideoAssistant # Conceptual module
from openeye.utils import logger

def main():
    logger.info("Initializing OpenEye AI Video Assistant...")
    try:
        # Ensure API keys or necessary AI model configs are set via environment variables or a config file
        assistant = VideoAssistant(llm_model="gpt-4o", vision_model="gpt-4-vision-preview")
    except Exception as e:
        logger.error(f"Failed to initialize VideoAssistant: {e}")
        return

    video_archive_path = "path/to/your/drone_reconnaissance.mp4"
    query = "Summarize the key events in this video. Are there any vehicles present?"
    
    try:
        logger.info(f"Analyzing video: {video_archive_path} with query: '{query}'")
        result = assistant.analyze_video(video_path=video_archive_path, query=query)
        
        print("\n--- Analysis Result ---")
        print(f"Summary: {result.get('summary', 'N/A')}")
        print(f"Vehicles Present: {result.get('vehicles_detected', 'N/A')}")
        # print(f"Full AI Response: {result.get('raw_response', {})}")

    except Exception as e:
        logger.error(f"Error during AI video analysis: {e}")

if __name__ == "__main__":
    main()
```

## Running the Desktop Application (Tauri + QML)

OpenEye includes a desktop application for richer interaction and visualization.
Ensure you have Rust and Node.js (often needed for Tauri development environments) installed.

```bash
# Navigate to the project root directory if you aren't already there
# cd /path/to/OpenEye

# Run the Tauri application in development mode
cargo tauri dev
```
This command typically builds the Rust backend and the QML frontend, then launches the application. For production builds:
```bash
cargo tauri build
```

## Configuration

OpenEye uses a YAML configuration file (e.g., `src/config/default.yaml`). Key settings can be overridden via environment variables or a custom configuration file passed at runtime.

Example structure of `src/config/default.yaml`:
```yaml
# src/config/default.yaml
core:
    log_level: info
    # ... other core settings

camera:
    default_device: 0
    # ... other camera settings

vision:
    detector:
        type: yolo
        model: yolov8n.pt # Ensure this model is accessible
    # ... other vision settings

# ... other sections like streaming, ai, encoding, ui
```
To run with a specific config:
```bash
python -m openeye.main --config path/to/your/custom_config.yaml
```

## Project Documentation

For more in-depth technical details, API references, and advanced usage guides, please refer to the `DOCUMENTATION.md` file and the `docs/` directory within this repository.
We aim to host comprehensive documentation at a dedicated site in the future (e.g., `https://docs.openeye.ai` - placeholder).

## Contributing

We warmly welcome contributions from the community! Whether it's reporting a bug, submitting a feature request, improving documentation, or writing code, your help is valued.

Please see our `CONTRIBUTING.md` guide for detailed information on:
-   Setting up your development environment.
-   Coding standards and practices.
-   The pull request process.
-   How to report issues effectively.

### Development Workflow Quick Tips:
-   Create a feature branch: `git checkout -b feature/your-cool-feature`
-   Make your changes and commit them with clear messages.
-   Run linters and tests: `black .`, `ruff .`, `pytest` (configure these if not already set up).
-   Push to your fork and open a Pull Request against the `main` branch of the original repository.

## License

OpenEye is proudly open-source and licensed under the [MIT License](LICENSE). This means you are free to use, modify, and distribute the software, provided you include the original copyright and license notice.

---

**Thank you for your interest in OpenEye! We are excited to see what you build.**
For questions, support, or to share your creations, please open an issue or start a discussion on our GitHub page.