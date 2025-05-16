"""
OpenAI Assistants API integration for intelligent video analysis.
"""

import logging
import os
import time
import json
import base64
import io
from typing import Dict, Optional, Any, List, Tuple, Union, Callable
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

# Try to import OpenAI
try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    logger.warning("OpenAI package not available, Assistant features will be limited")
    OPENAI_AVAILABLE = False

# Try to import file utilities
try:
    from ..utils.file_utils import ensure_directory_exists
except ImportError:
    def ensure_directory_exists(path):
        os.makedirs(path, exist_ok=True)


class VideoAssistant:
    """
    OpenAI Assistant for intelligent video analysis.
    """
    
    def __init__(self, api_key: str = None,
                 assistant_id: str = None,
                 model: str = "gpt-4-vision-preview",
                 name: str = "OpenVideo Assistant",
                 instructions: str = None,
                 cache_dir: str = "./assistant_cache"):
        """
        Initialize OpenAI Assistant for video analysis.
        
        Args:
            api_key: OpenAI API key
            assistant_id: Existing assistant ID to use
            model: Model to use for the assistant
            name: Name for the assistant
            instructions: Instructions for the assistant
            cache_dir: Directory to cache assistant data
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package is required for VideoAssistant")
            
        # API key setup
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
            
        self.assistant_id = assistant_id
        self.model = model
        self.name = name
        self.cache_dir = cache_dir
        
        # Default instructions if not provided
        if instructions is None:
            self.instructions = (
                "You are an expert video analysis assistant for OpenVideo. "
                "You analyze videos, detect objects, identify anomalies, and provide "
                "detailed descriptions of video content. You can interpret frames, "
                "identify patterns across time, and provide insights about what's "
                "happening in the video. Be precise, informative, and helpful."
            )
        else:
            self.instructions = instructions
            
        # Ensure cache directory exists
        ensure_directory_exists(self.cache_dir)
        
        # Initialize client and assistant
        self.client = OpenAI(api_key=self.api_key)
        self._initialize_assistant()
        
        # Active threads
        self.active_threads = {}
        
    def _initialize_assistant(self) -> None:
        """Initialize or retrieve the OpenAI Assistant."""
        try:
            if self.assistant_id:
                # Try to retrieve existing assistant
                try:
                    self.assistant = self.client.beta.assistants.retrieve(self.assistant_id)
                    logger.info(f"Retrieved existing assistant: {self.assistant_id}")
                    return
                except:
                    logger.warning(f"Could not retrieve assistant {self.assistant_id}, creating new one")
            
            # Create a new assistant
            self.assistant = self.client.beta.assistants.create(
                name=self.name,
                instructions=self.instructions,
                model=self.model,
                tools=[{"type": "retrieval"}]
            )
            
            self.assistant_id = self.assistant.id
            logger.info(f"Created new assistant with ID: {self.assistant_id}")
            
            # Save assistant details to cache
            with open(os.path.join(self.cache_dir, "assistant_info.json"), "w") as f:
                json.dump({
                    "assistant_id": self.assistant_id,
                    "name": self.name,
                    "model": self.model,
                    "instructions": self.instructions,
                    "created_at": time.time()
                }, f)
                
        except Exception as e:
            logger.error(f"Error initializing assistant: {e}", exc_info=True)
            raise
            
    def create_thread(self, video_id: str = None) -> str:
        """
        Create a new thread for conversation.
        
        Args:
            video_id: Optional video ID to associate with thread
            
        Returns:
            str: Thread ID
        """
        try:
            thread = self.client.beta.threads.create()
            thread_id = thread.id
            
            # Store in active threads
            self.active_threads[thread_id] = {
                "created_at": time.time(),
                "video_id": video_id,
                "messages": []
            }
            
            logger.info(f"Created new thread {thread_id} for video {video_id}")
            return thread_id
            
        except Exception as e:
            logger.error(f"Error creating thread: {e}", exc_info=True)
            raise
            
    def analyze_frame(self, thread_id: str, frame: np.ndarray, 
                     prompt: str = None, wait_for_response: bool = True) -> Dict[str, Any]:
        """
        Analyze a video frame using the assistant.
        
        Args:
            thread_id: Thread ID to use
            frame: Video frame as numpy array
            prompt: Text prompt to send with the frame
            wait_for_response: Whether to wait for assistant response
            
        Returns:
            Dict: Analysis result
        """
        try:
            # Ensure thread exists
            if thread_id not in self.active_threads:
                thread_id = self.create_thread()
                
            # Convert frame to base64
            img = Image.fromarray(frame)
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG")
            img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
            
            # Default prompt if not provided
            if prompt is None:
                prompt = "Please analyze this video frame and describe what you see."
                
            # Create message with frame
            message = self.client.beta.threads.messages.create(
                thread_id=thread_id,
                role="user",
                content=[
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_str}"
                        }
                    }
                ]
            )
            
            # Store message in active threads
            self.active_threads[thread_id]["messages"].append({
                "id": message.id,
                "role": "user",
                "timestamp": time.time()
            })
            
            if wait_for_response:
                return self.get_response(thread_id)
            else:
                return {"message_id": message.id, "status": "message_sent"}
                
        except Exception as e:
            logger.error(f"Error analyzing frame: {e}", exc_info=True)
            return {"error": str(e)}
            
    def send_message(self, thread_id: str, message: str, 
                    wait_for_response: bool = True) -> Dict[str, Any]:
        """
        Send a text message to the assistant.
        
        Args:
            thread_id: Thread ID to use
            message: Text message to send
            wait_for_response: Whether to wait for assistant response
            
        Returns:
            Dict: Response result
        """
        try:
            # Ensure thread exists
            if thread_id not in self.active_threads:
                thread_id = self.create_thread()
                
            # Create message
            message_obj = self.client.beta.threads.messages.create(
                thread_id=thread_id,
                role="user",
                content=message
            )
            
            # Store message in active threads
            self.active_threads[thread_id]["messages"].append({
                "id": message_obj.id,
                "role": "user",
                "timestamp": time.time()
            })
            
            if wait_for_response:
                return self.get_response(thread_id)
            else:
                return {"message_id": message_obj.id, "status": "message_sent"}
                
        except Exception as e:
            logger.error(f"Error sending message: {e}", exc_info=True)
            return {"error": str(e)}
            
    def get_response(self, thread_id: str, timeout: int = 60) -> Dict[str, Any]:
        """
        Get response from the assistant.
        
        Args:
            thread_id: Thread ID to use
            timeout: Maximum time to wait for response in seconds
            
        Returns:
            Dict: Response data
        """
        try:
            # Create a run
            run = self.client.beta.threads.runs.create(
                thread_id=thread_id,
                assistant_id=self.assistant_id
            )
            
            run_id = run.id
            start_time = time.time()
            
            # Poll for completion
            while time.time() - start_time < timeout:
                run_status = self.client.beta.threads.runs.retrieve(
                    thread_id=thread_id,
                    run_id=run_id
                )
                
                if run_status.status == "completed":
                    # Get messages
                    messages = self.client.beta.threads.messages.list(
                        thread_id=thread_id
                    )
                    
                    # Get the latest assistant message
                    assistant_messages = [
                        msg for msg in messages.data 
                        if msg.role == "assistant"
                    ]
                    
                    if assistant_messages:
                        latest_msg = assistant_messages[0]
                        
                        # Extract text content
                        text_content = ""
                        for content_item in latest_msg.content:
                            if content_item.type == "text":
                                text_content += content_item.text.value
                        
                        # Store message in active threads
                        self.active_threads[thread_id]["messages"].append({
                            "id": latest_msg.id,
                            "role": "assistant",
                            "timestamp": time.time()
                        })
                        
                        return {
                            "message_id": latest_msg.id,
                            "content": text_content,
                            "role": "assistant",
                            "status": "completed"
                        }
                
                elif run_status.status == "failed":
                    return {
                        "status": "failed",
                        "error": run_status.last_error.message if hasattr(run_status, "last_error") else "Unknown error"
                    }
                    
                # Wait before polling again
                time.sleep(1)
                
            # Timeout reached
            return {"status": "timeout", "error": "Response timeout"}
            
        except Exception as e:
            logger.error(f"Error getting response: {e}", exc_info=True)
            return {"error": str(e)}
            
    def summarize_video(self, video_path: str, 
                       frame_interval: int = 30,
                       prompt: str = None) -> Dict[str, Any]:
        """
        Analyze and summarize a video file by sampling frames.
        
        Args:
            video_path: Path to video file
            frame_interval: Interval between frames to sample
            prompt: Custom prompt for analysis
            
        Returns:
            Dict: Summary result
        """
        try:
            import cv2
            
            # Open the video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {"error": f"Could not open video file: {video_path}"}
                
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps
            
            # Create a new thread
            thread_id = self.create_thread(video_id=os.path.basename(video_path))
            
            # Initialize with video info
            init_message = (
                f"I'm going to analyze a video with duration {duration:.2f} seconds, "
                f"at {fps:.2f} fps. I'll show you sample frames to analyze the content."
            )
            self.send_message(thread_id, init_message, wait_for_response=True)
            
            # Sample frames
            frames_analyzed = 0
            for frame_idx in range(0, frame_count, frame_interval):
                # Set frame position
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if not ret:
                    break
                    
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Timestamp
                timestamp = frame_idx / fps
                frame_prompt = prompt or f"This is frame {frame_idx} at timestamp {timestamp:.2f}s. Analyze this frame."
                
                # Analyze frame
                response = self.analyze_frame(thread_id, frame_rgb, frame_prompt)
                frames_analyzed += 1
                
                # Limit to max 10 frames to prevent excessive API usage
                if frames_analyzed >= 10:
                    break
            
            # Ask for overall summary
            summary_prompt = (
                f"Based on all the frames you've analyzed, please provide a comprehensive "
                f"summary of this {duration:.2f} second video. Include key objects, activities, "
                f"and any notable events or patterns."
            )
            summary = self.send_message(thread_id, summary_prompt)
            
            # Release video
            cap.release()
            
            return {
                "thread_id": thread_id,
                "summary": summary.get("content", ""),
                "frames_analyzed": frames_analyzed,
                "duration": duration,
                "status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Error summarizing video: {e}", exc_info=True)
            return {"error": str(e)}