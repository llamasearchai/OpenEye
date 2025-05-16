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
      ffmpeg \
      && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Create app directory
WORKDIR /app

# Copy the source code
COPY . .

# Build Rust components (e.g., for Tauri, if any parts are bundled)
RUN cargo build --release

# Install the Python package and its dependencies, including 'cuda' extras
RUN pip3 install .[cuda]

# Runtime image
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Install runtime system dependencies
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
      ffmpeg \
      && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy application code, installed Python packages, and Rust artifacts from builder stage
COPY --from=builder /app /app
COPY --from=builder /root/.local /root/.local
COPY --from=builder /app/target/release /app/target/release

# Add sample video
RUN wget -O /app/sample.mp4 https://example.com/sample.mp4

# Configuration directory
RUN mkdir -p /etc/openeye
COPY config/default.yaml /etc/openeye/config.yaml

# Health check
HEALTHCHECK --interval=30s --timeout=3s \
    CMD curl -f http://localhost:8080/health || exit 1

# Set environment variables
ENV PYTHONPATH=/app
ENV LD_LIBRARY_PATH=/app/target/release:${LD_LIBRARY_PATH}

# Expose ports for API/WebRTC and RTSP
EXPOSE 8080 8554

# Run with config
CMD ["python3", "-m", "openeye.main", "--config", "/etc/openeye/config.yaml"] 