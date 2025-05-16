# OpenEye - Professional Video Processing Framework

## Why OpenEye?

OpenEye is the Swiss Army knife for modern video processing needs. Whether you're:
- Building drone surveillance systems
- Developing real-time video analytics
- Creating streaming platforms
- Implementing computer vision pipelines

OpenEye provides the robust foundation you need, with battle-tested components and an extensible architecture.

## Key Features Deep Dive

### 🚀 Performance Optimized
- GPU-accelerated processing (CUDA 12.2+)
- Rust-powered performance-critical components
- Intelligent frame batching for maximum throughput

### 🔌 Plug-and-Play Integration
```python
from openeye import Pipeline

pipeline = Pipeline() \
    .with_source("rtsp://camera-feed") \
    .with_detector("yolov8n.pt") \
    .with_rtsp_output() \
    .with_webrtc_output()

pipeline.run()  # That's it!
```

### 📊 Monitoring & Observability
Built-in Prometheus metrics endpoint at `/metrics`:
- Frame processing latency
- Memory usage
- GPU utilization
- Stream health metrics

## Real-World Deployment Guide

### Cloud Deployment (AWS Example)
```bash
# ECS Task Definition
aws ecs register-task-definition \
    --family openeye \
    --cpu 4096 \
    --memory 8192 \
    --requires-compatibilities EC2 \
    --container-definitions '[
        {
            "name": "openeye",
            "image": "openeye/openeye:1.0.0",
            "portMappings": [
                {"containerPort": 8080, "hostPort": 8080},
                {"containerPort": 8554, "hostPort": 8554}
            ],
            "environment": [
                {"name": "OPENEYE_CONFIG_PATH", "value": "/etc/openeye/config.yaml"}
            ]
        }
    ]'
```

### Edge Device Setup (NVIDIA Jetson)
```bash
# Flash JetPack OS
sudo ./flash.sh jetson-xavier-nx-devkit mmcblk0p1

# Install OpenEye
docker run --runtime nvidia -p 8080:8080 -p 8554:8554 \
    -v /opt/openeye/config:/etc/openeye \
    openeye/openeye:1.0.0-jetson
```

## Troubleshooting Guide

🔍 **Common Issues & Solutions**

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| High GPU memory usage | Frame batch size too large | Reduce `processing.batch_size` in config |
| RTSP stream lagging | Network bandwidth saturation | Enable adaptive bitrate in config |
| Detection accuracy low | Model input resolution mismatch | Check `detector.input_size` matches model specs |

## Contributor's Corner

We ❤️ open source! Here's how to contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b cool-feature`)
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

Pro Tip: Run `make precommit` before pushing to automatically:
- Format your code
- Run static analysis
- Execute unit tests 