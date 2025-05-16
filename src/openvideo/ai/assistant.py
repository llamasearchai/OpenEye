import base64
import os
import time
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from openai import OpenAI

from ..utils.logging import get_logger


class VideoAssistant:
    """
    Assistant for video analysis using OpenAI's Assistants API
    with vision capabilities.
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model: str = "gpt-4-vision-preview",
                 temperature: float = 0.0,
                 config: Optional[Dict] = None):
        """
        Initialize the video assistant.
        
        Args:
            api_key: OpenAI API key (default: from OPENAI_API_KEY env var)
            model: Model to use for analysis
            temperature: Sampling temperature
            config: Additional configuration
        """
        self.logger = get_logger("VideoAssistant")
        self.config = config or {}
        
        # Set API key from argument or environment
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            self.logger.warning("No OpenAI API key provided. Set OPENAI_API_KEY environment variable.")
            self.client = None
        else:
            self.client = OpenAI(api_key=self.api_key)
        
        self.model = model
        self.temperature = temperature
    
    def analyze_frame(self, frame: np.ndarray, prompt: str) -> Dict:
        """
        Analyze a single video frame with a custom prompt.
        
        Args:
            frame: Video frame as numpy array (BGR format)
            prompt: Custom instruction for the analysis
            
        Returns:
            Dict with analysis result
        """
        if self.client is None:
            return {"error": "OpenAI client not initialized. Check API key."}
        
        try:
            # Convert frame to base64
            _, buffer = cv2.imencode(".jpg", frame)
            base64_image = base64.b64encode(buffer).decode("utf-8")
            
            # Create messages
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
            
            # Make API call
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=1000
            )
            
            # Extract response
            result = response.choices[0].message.content
            
            return {
                "result": result,
                "model": self.model,
                "processing_time": response.usage.total_tokens
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing frame: {e}", exc_info=True)
            return {"error": str(e)}
    
    def summarize_video(self, video_path: str, frame_interval: int = 100) -> Dict:
        """
        Generate a summary of a video by analyzing frames at intervals.
        
        Args:
            video_path: Path to video file
            frame_interval: Interval between frames to analyze
            
        Returns:
            Dict with video summary
        """
        if self.client is None:
            return {"error": "OpenAI client not initialized. Check API key."}
        
        try:
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {"error": f"Could not open video: {video_path}"}
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            self.logger.info(f"Summarizing video: {video_path} ({duration:.1f}s)")
            
            # Calculate number of frames to sample
            max_frames = 8  # Maximum number of frames to send
            actual_interval = max(frame_interval, frame_count // max_frames)
            
            # Extract frames
            frames = []
            frame_positions = []
            
            for frame_idx in range(0, frame_count, actual_interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
                    frame_positions.append(frame_idx / fps)  # Time in seconds
                
                if len(frames) >= max_frames:
                    break
            
            cap.release()
            
            if not frames:
                return {"error": "Could not extract frames from video"}
            
            # Convert frames to base64
            base64_images = []
            for frame in frames:
                _, buffer = cv2.imencode(".jpg", frame)
                base64_image = base64.b64encode(buffer).decode("utf-8")
                base64_images.append(base64_image)
            
            # Create content with all frames
            content = [{"type": "text", "text": 
                      "Analyze these frames from a video and provide a comprehensive summary. "
                      "Describe what's happening, the environment, any objects or people visible, "
                      "and the general context of the video. Focus on details that are consistent "
                      "across multiple frames."}]
            
            # Add each frame with timestamp
            for i, (base64_image, timestamp) in enumerate(zip(base64_images, frame_positions)):
                content.append({"type": "text", "text": f"Frame {i+1} at {timestamp:.1f}s:"})
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                })
            
            # Make API call
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": content}],
                temperature=self.temperature,
                max_tokens=2000
            )
            
            # Extract response
            summary = response.choices[0].message.content
            
            return {
                "summary": summary,
                "frame_count": len(frames),
                "duration": duration,
                "model": self.model
            }
            
        except Exception as e:
            self.logger.error(f"Error summarizing video: {e}", exc_info=True)
            return {"error": str(e)}
    
    def analyze_video_stream(self, rtsp_url: str, prompt: str, 
                           interval: int = 10) -> Dict:
        """
        Analyze a live video stream at regular intervals.
        
        Args:
            rtsp_url: RTSP URL of the stream
            prompt: Analysis prompt
            interval: Interval between frame analyses in seconds
            
        Returns:
            Dict with analysis result
        """
        if self.client is None:
            return {"error": "OpenAI client not initialized. Check API key."}
        
        try:
            # Open RTSP stream
            cap = cv2.VideoCapture(rtsp_url)
            if not cap.isOpened():
                return {"error": f"Could not open RTSP stream: {rtsp_url}"}
            
            # Capture first frame
            ret, frame = cap.read()
            if not ret:
                cap.release()
                return {"error": "Could not read frame from stream"}
            
            # Analyze first frame
            result = self.analyze_frame(frame, prompt)
            
            # Schedule periodic analysis in real scenarios...
            # For this implementation, we'll just analyze one frame
            
            cap.release()
            
            # Add stream information
            result["stream_url"] = rtsp_url
            result["interval"] = interval
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing stream: {e}", exc_info=True)
            return {"error": str(e)}