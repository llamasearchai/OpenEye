"""
Pipeline implementation for video processing.
"""

import logging
import threading
import time
import queue
from typing import List, Optional, Dict, Any, Callable

from .types import FrameSource, FrameSink, VideoFrame
from ..config import Configuration

logger = logging.getLogger(__name__)


class Pipeline:
    """
    Video processing pipeline that connects a source to multiple processing stages and sinks.
    """
    
    def __init__(self, source: FrameSource, name: str = "pipeline"):
        """
        Initialize a new pipeline with a frame source.
        
        Args:
            source: The source of frames for the pipeline
            name: Name for this pipeline (used in logging)
        """
        self.source = source
        self.stages: List[FrameSink] = []
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.name = name
        self.frame_count = 0
        self.start_time = 0
        self.stop_event = threading.Event()
        self.frame_queue = queue.Queue(maxsize=30)  # Buffer up to 30 frames
        self.processing_threads: List[threading.Thread] = []
        self.num_worker_threads = 4
        
    def add_stage(self, stage: FrameSink) -> 'Pipeline':
        """
        Add a processing stage to the pipeline.
        
        Args:
            stage: Frame processing stage to add
            
        Returns:
            self: For method chaining
        """
        self.stages.append(stage)
        return self
    
    def start(self, blocking: bool = False) -> None:
        """
        Start the pipeline processing.
        
        Args:
            blocking: If True, run in the current thread; otherwise start a new thread
        """
        if self.running:
            logger.warning(f"Pipeline '{self.name}' already running")
            return
            
        self.running = True
        self.stop_event.clear()
        self.start_time = time.time()
        self.frame_count = 0
        
        if blocking:
            self._run()
        else:
            self.thread = threading.Thread(target=self._run, name=f"pipeline-{self.name}")
            self.thread.daemon = True
            self.thread.start()
            
            # Start worker threads to process frames
            self.processing_threads = []
            for i in range(self.num_worker_threads):
                thread = threading.Thread(
                    target=self._process_frames_worker,
                    name=f"pipeline-worker-{self.name}-{i}"
                )
                thread.daemon = True
                thread.start()
                self.processing_threads.append(thread)
    
    def _process_frames_worker(self) -> None:
        """Worker thread function to process frames from the queue."""
        while not self.stop_event.is_set():
            try:
                # Get frame with timeout to allow checking stop_event periodically
                frame = self.frame_queue.get(timeout=0.1)
                self._process_frame(frame)
                self.frame_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing frame in pipeline '{self.name}': {e}", exc_info=True)
    
    def _run(self) -> None:
        """Main pipeline execution loop."""
        logger.info(f"Starting pipeline '{self.name}'")
        
        try:
            while self.running and not self.stop_event.is_set():
                frame = self.source.get_frame()
                if frame is None:
                    # Source has no more frames
                    logger.info(f"Source for pipeline '{self.name}' has no more frames")
                    break
                
                self.frame_count += 1
                
                # In blocking mode or if queue is full, process directly
                if len(self.processing_threads) == 0 or self.frame_queue.full():
                    self._process_frame(frame)
                else:
                    # Otherwise queue for processing by worker threads
                    try:
                        self.frame_queue.put(frame, block=False)
                    except queue.Full:
                        # If queue is full, process in the main thread
                        self._process_frame(frame)
                        
        except Exception as e:
            logger.error(f"Error in pipeline '{self.name}': {e}", exc_info=True)
        finally:
            self.stop()
    
    def _process_frame(self, frame: VideoFrame) -> None:
        """Process a single frame through all pipeline stages."""
        current_frame = frame
        try:
            for stage in self.stages:
                if current_frame is None:
                    break
                current_frame = stage.process_frame(current_frame)
        except Exception as e:
            logger.error(f"Error in pipeline stage: {e}", exc_info=True)
    
    def stop(self) -> None:
        """Stop the pipeline and release resources."""
        if not self.running:
            return
            
        logger.info(f"Stopping pipeline '{self.name}'")
        self.running = False
        self.stop_event.set()
        
        # Wait for processing threads to finish
        for thread in self.processing_threads:
            thread.join(timeout=2.0)
        
        # Wait for main thread
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=3.0)
        
        # Calculate stats
        duration = time.time() - self.start_time
        fps = self.frame_count / duration if duration > 0 else 0
        logger.info(f"Pipeline '{self.name}' processed {self.frame_count} frames in {duration:.2f}s ({fps:.2f} fps)")
        
        # Release resources
        try:
            self.source.release()
            for stage in self.stages:
                stage.release()
        except Exception as e:
            logger.error(f"Error releasing pipeline resources: {e}", exc_info=True)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get pipeline performance statistics.
        
        Returns:
            dict: Statistics about the pipeline execution
        """
        duration = time.time() - self.start_time if self.start_time > 0 else 0
        fps = self.frame_count / duration if duration > 0 else 0
        
        return {
            "name": self.name,
            "running": self.running,
            "frames_processed": self.frame_count,
            "duration_seconds": duration,
            "fps": fps,
            "num_stages": len(self.stages),
            "queue_size": self.frame_queue.qsize(),
            "source_type": type(self.source).__name__
        }        }        """