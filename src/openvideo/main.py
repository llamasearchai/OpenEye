import argparse
import asyncio
import json
import os
import signal
import sys
import threading
import time
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import cv2
import numpy as np
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError
from pydantic import BaseSettings
from openvideo.core.processor import VideoProcessor
from openvideo.streaming.rtsp import RTSPServer
from openvideo.streaming.webrtc import FastAPIWebRTCServer
from openvideo.vision.detector import YOLODetector
from openvideo.telemetry.klv import KLVParser
from openvideo.ai.assistant import VideoAssistant
from openvideo.ai.query_engine import VideoQueryEngine
from .utils.logging import get_logger, setup_logging
from fastapi.exceptions import RequestValidationError
from fastapi.encoders import jsonable_encoder
from prometheus_client import make_asgi_app, Counter, Gauge, Histogram

DEFAULT_CONFIG_PATH = Path("/etc/openvideo/config.yaml")
HEALTH_CHECK_INTERVAL = 5  # seconds

class VideoConfig(BaseModel):
    """Configuration for video input sources."""
    source: str = Field(
        "0",
        description="Video source (file path, camera index, or RTSP URL)"
    )
    detector: Optional[str] = Field(
        None,
        description="Path to YOLO detector model"
    )
    gui: bool = Field(
        False,
        description="Enable OpenCV GUI display"
    )
    max_retries: int = Field(
        3,
        description="Number of retries for failed frame reads"
    )
    reconnect_delay: float = Field(
        2.0,
        description="Delay between reconnection attempts (seconds)"
    )

class ServerConfig(BaseModel):
    """Configuration for server endpoints."""
    rtsp: bool = Field(
        False,
        description="Enable RTSP server output"
    )
    rtsp_port: int = Field(
        8554,
        description="RTSP server port"
    )
    webrtc: bool = Field(
        False,
        description="Enable WebRTC server"
    )
    api_host: str = Field(
        "0.0.0.0",
        description="API server host"
    )
    api_port: int = Field(
        8080,
        description="API server port"
    )
    health_check: bool = Field(
        True,
        description="Enable health check endpoint"
    )
    metrics: bool = Field(
        True,
        description="Enable Prometheus metrics endpoint"
    )
    enable_docs: bool = Field(
        True,
        description="Enable API documentation"
    )

class AppConfig(BaseSettings):
    """Main application configuration model with enhanced validation."""
    video: VideoConfig = VideoConfig()
    server: ServerConfig = ServerConfig()
    
    class Config:
        env_prefix = "OPENVIDEO_"
        case_sensitive = False
        env_nested_delimiter = "__"

    @classmethod
    def from_yaml(cls, path: Union[str, Path]):
        """Load configuration from YAML file with detailed validation."""
        try:
            with open(path) as f:
                config_data = yaml.safe_load(f) or {}
                
            # Pre-validation of required fields
            if not config_data.get('video', {}).get('source'):
                raise ValueError("Video source is required in configuration")
                
            return cls(**config_data)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML format: {e}") from e
        except Exception as e:
            raise ValueError(f"Configuration error: {str(e)}") from e

class GracefulExit(Exception):
    """Exception for graceful shutdown."""

def signal_handler(signum, frame):
    """Handle shutdown signals."""
    raise GracefulExit()

class HealthStatus:
    def __init__(self):
        self._components = {}
        self.last_check = time.time()
        
    def register_component(self, name: str, check_fn: callable):
        self._components[name] = check_fn
        
    async def check_all(self):
        results = {}
        self.last_check = time.time()
        for name, check_fn in self._components.items():
            try:
                results[name] = await check_fn() if asyncio.iscoroutinefunction(check_fn) else check_fn()
            except Exception as e:
                results[name] = {"status": "error", "details": str(e)}
        return results

async def create_app(config: ServerConfig) -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title="OpenVideo API",
        description="Real-time video processing API",
        version="1.0.0",
        docs_url="/docs" if config.enable_docs else None,
        redoc_url=None
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Health check endpoint
    if config.health_check:
        health_status = HealthStatus()
        
        @app.get("/health", tags=["monitoring"])
        async def health_check():
            status = await health_status.check_all()
            overall_status = all(
                component.get("status") == "ok" 
                for component in status.values()
            )
            return {
                "status": "ok" if overall_status else "degraded",
                "timestamp": time.time(),
                "components": status
            }
    
    # Metrics endpoint
    if config.metrics:
        # Add Prometheus metrics route
        metrics_app = make_asgi_app()
        app.mount("/metrics", metrics_app)
        
        # Define metrics
        REQUEST_COUNT = Counter(
            'openvideo_requests_total',
            'Total number of requests',
            ['method', 'endpoint', 'status_code']
        )
        
        PROCESSING_TIME = Histogram(
            'openvideo_processing_seconds',
            'Video processing time',
            ['pipeline']
        )
        
        FRAME_RATE = Gauge(
            'openvideo_frame_rate',
            'Current frames processed per second'
        )
        
        # Add middleware to track requests
        @app.middleware("http")
        async def track_requests(request: Request, call_next):
            start_time = time.time()
            response = await call_next(request)
            process_time = time.time() - start_time
            
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=request.url.path,
                status_code=response.status_code
            ).inc()
            
            return response
    
    # Add middleware for catching unhandled exceptions
    @app.middleware("http")
    async def catch_exceptions_middleware(request: Request, call_next):
        try:
            return await call_next(request)
        except Exception as exc:
            logger = get_logger("fastapi")
            logger.error(
                f"Unhandled exception: {exc}",
                exc_info=True,
                extra={
                    "path": request.url.path,
                    "method": request.method
                }
            )
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal Server Error"},
            )

    # Enhanced validation error handler
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        logger = get_logger("fastapi")
        logger.warning(
            "Validation error",
            extra={
                "errors": exc.errors(),
                "body": exc.body
            }
        )
        return JSONResponse(
            status_code=422,
            content=jsonable_encoder({"detail": exc.errors(), "body": exc.body}),
        )
    
    return app

def load_configuration(config_path: Optional[str]) -> AppConfig:
    """Load configuration from file or environment variables.
    
    Args:
        config_path: Path to configuration file or None to use env vars
        
    Returns:
        Validated configuration object
        
    Raises:
        ValueError: If configuration is invalid
    """
    logger = get_logger("config")
    
    if config_path:
        logger.info(f"Loading configuration from: {config_path}")
        return AppConfig.from_yaml(config_path)
    elif DEFAULT_CONFIG_PATH.exists():
        logger.info(f"Loading configuration from default path: {DEFAULT_CONFIG_PATH}")
        return AppConfig.from_yaml(DEFAULT_CONFIG_PATH)
    else:
        logger.info("Using environment variables for configuration")
        return AppConfig()

async def initialize_processor(config: AppConfig) -> VideoProcessor:
    """Initialize the video processor with detector.
    
    Args:
        config: Application configuration
        
    Returns:
        Configured VideoProcessor instance
    """
    logger = get_logger("processor")
    
    # Initialize detector if needed
    detector = None
    if config.video.detector:
        logger.info(f"Initializing detector with model: {config.video.detector}")
        detector = YOLODetector(model_path=config.video.detector)
    
    # Create processor
    processor = VideoProcessor(detector=detector)
    logger.info("Video processor initialized")
    
    return processor

async def initialize_rtsp(config: AppConfig, processor: VideoProcessor) -> RTSPServer:
    """Initialize RTSP server if enabled.
    
    Args:
        config: Application configuration
        processor: Video processor instance
        
    Returns:
        Configured RTSPServer instance
    """
    logger = get_logger("rtsp")
    
    # Create RTSP server
    rtsp_server = RTSPServer(port=config.server.rtsp_port)
    
    # Start server
    rtsp_server.start()
    
    # Add as output to processor
    processor.add_output(rtsp_server.frame_callback)
    
    logger.info(f"RTSP server started on port {config.server.rtsp_port}")
    return rtsp_server

async def initialize_webrtc(config: AppConfig, processor: VideoProcessor) -> Tuple[FastAPI, FastAPIWebRTCServer]:
    """Initialize WebRTC server and FastAPI app if enabled.
    
    Args:
        config: Application configuration
        processor: Video processor instance
        
    Returns:
        Tuple of (FastAPI app, WebRTC server)
    """
    logger = get_logger("webrtc")
    
    # Create FastAPI app
    app = await create_app(config.server)
    
    # Create WebRTC server
    webrtc_server = FastAPIWebRTCServer(app=app, path="/webrtc")
    
    # Add as output to processor
    processor.add_output(webrtc_server.frame_callback)
    
    logger.info("WebRTC server initialized")
    return app, webrtc_server

async def start_processing(config: AppConfig, processor: VideoProcessor) -> bool:
    """Start video processing by opening source and starting processor.
    
    Args:
        config: Application configuration
        processor: Video processor instance
        
    Returns:
        Success status
    """
    logger = get_logger("processing")
    
    # Parse source (camera index or file/URL)
    source = int(config.video.source) if config.video.source.isdigit() else config.video.source
    logger.info(f"Opening video source: {source}")
    
    # Open source with retry
    for attempt in range(config.video.max_retries):
        if processor.open(source):
            break
        
        if attempt < config.video.max_retries - 1:
            logger.warning(f"Failed to open source, retrying ({attempt+1}/{config.video.max_retries})...")
            await asyncio.sleep(config.video.reconnect_delay)
        else:
            logger.error(f"Failed to open video source after {config.video.max_retries} attempts")
            return False
    
    # Start processor
    processor.start()
    logger.info("Video processing started")
    
    return True

async def start_api_server(config: AppConfig, app: FastAPI) -> asyncio.Task:
    """Start the FastAPI server.
    
    Args:
        config: Application configuration
        app: FastAPI application
        
    Returns:
        Server task
    """
    logger = get_logger("api")
    
    # Configure Uvicorn server
    server_config = uvicorn.Config(
        app,
        host=config.server.api_host,
        port=config.server.api_port,
        log_level="info",
        timeout_keep_alive=30
    )
    server = uvicorn.Server(server_config)
    
    # Create and start server task
    server_task = asyncio.create_task(server.serve())
    
    logger.info(f"API server started at http://{config.server.api_host}:{config.server.api_port}")
    return server_task

async def run_processing_loop(config: AppConfig, processor: VideoProcessor) -> None:
    """Run the main processing loop.
    
    Args:
        config: Application configuration
        processor: Video processor instance
    """
    logger = get_logger("loop")
    
    try:
        if config.video.gui:
            logger.info("Running in GUI mode")
            while True:
                frame = processor.get_latest_frame()
                if frame is not None:
                    cv2.imshow("OpenVideo", frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'):  # ESC or 'q' to quit
                    break
                
                await asyncio.sleep(0.01)  # Small sleep to avoid blocking
        else:
            logger.info("Running in headless mode")
            while True:
                await asyncio.sleep(HEALTH_CHECK_INTERVAL)
    except asyncio.CancelledError:
        logger.info("Processing loop cancelled")

def cleanup_resources() -> None:
    """Clean up any resources not handled by context managers."""
    logger = get_logger("cleanup")
    
    # Close any OpenCV windows if they were opened
    try:
        cv2.destroyAllWindows()
        logger.info("Closed OpenCV windows")
    except Exception as e:
        logger.warning(f"Error closing OpenCV windows: {e}")
    
    # Additional cleanup can be added here
    logger.info("Resource cleanup completed")

async def run_server(config_path: Optional[str] = None) -> int:
    """Enhanced main entry point with comprehensive resource management."""
    setup_logging()
    logger = get_logger("main")
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Load and validate configuration
        config = load_configuration(config_path)
        logger.info("Configuration loaded and validated")
        
        # Initialize components with context management
        async with AsyncExitStack() as stack:
            processor = await initialize_processor(config)
            rtsp_server = await initialize_rtsp(config, processor) if config.server.rtsp else None
            app, webrtc_server = await initialize_webrtc(config, processor) if config.server.webrtc else (None, None)
            
            # Start video processing
            if not await start_processing(config, processor):
                return 1
            
            # Start API server if enabled
            if app:
                await start_api_server(config, app)
            
            # Main processing loop
            await run_processing_loop(config, processor)
    
    except GracefulExit:
        logger.info("Received shutdown signal")
    except Exception as e:
        logger.exception("Fatal error occurred during operation")
        return 1
    finally:
        logger.info("Application shutdown complete")
        cleanup_resources()
    
    return 0

def main():
    """Command line entry point."""
    parser = argparse.ArgumentParser(description="OpenVideo Processing Server")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--version",
        action="version",
        version="OpenVideo 1.0.0"
    )
    
    args = parser.parse_args()
    return asyncio.run(run_server(args.config))

if __name__ == "__main__":
    sys.exit(main())