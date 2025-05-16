"""
WebRTC streaming implementation.
"""

import asyncio
import json
import logging
import threading
import time
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder, MediaRelay

from ..utils.logging import get_logger
from ..vision.detector import Detection


class VideoStreamTrack(MediaStreamTrack):
    """
    A video stream track that captures frames from a VideoProcessor.
    """
    
    kind = "video"
    
    def __init__(self):
        super().__init__()
        self.logger = get_logger("VideoStreamTrack")
        self.frame_queue = asyncio.Queue(maxsize=10)
        self.latest_frame = None
        self.latest_frame_time = 0
        self.running = True
    
    def add_frame(self, frame: np.ndarray) -> None:
        """Add a frame to the queue."""
        if not self.running:
            return
            
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Store as latest frame
        self.latest_frame = frame_rgb
        self.latest_frame_time = time.time()
        
        # Add to queue if possible
        try:
            self.frame_queue.put_nowait(frame_rgb)
        except asyncio.QueueFull:
            # Drop oldest frame if queue is full
            try:
                self.frame_queue.get_nowait()
                self.frame_queue.put_nowait(frame_rgb)
            except (asyncio.QueueEmpty, asyncio.QueueFull):
                pass
    
    async def recv(self):
        """
        Receive the next frame.
        """
        if self.latest_frame is None:
            # Use black frame if no frame is available
            black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(black_frame, format="rgb24")
            frame.pts = int(time.time() * 1000)
            frame.time_base = fractions.Fraction(1, 1000)
            return frame
        
        try:
            # Wait for next frame with timeout
            frame_rgb = await asyncio.wait_for(self.frame_queue.get(), timeout=1.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            # Use latest frame if queue is empty
            frame_rgb = self.latest_frame
        
        # Convert to VideoFrame
        frame = av.VideoFrame.from_ndarray(frame_rgb, format="rgb24")
        frame.pts = int(time.time() * 1000)
        frame.time_base = fractions.Fraction(1, 1000)
        
        return frame
    
    def stop(self):
        """Stop the track."""
        self.running = False
        super().stop()


class WebRTCServer:
    """WebRTC server for video streaming."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize WebRTC server.
        
        Args:
            config: Configuration dictionary
        """
        self.logger = get_logger("WebRTCServer")
        self.config = config or {}
        
        # Import here to avoid dependency if not using WebRTC
        try:
            import av
            import fractions
            self.av = av
            self.fractions = fractions
        except ImportError:
            self.logger.error("Failed to import PyAV. Please install with: pip install av")
            self.av = None
        
        self.peer_connections = set()
        self.video_track = VideoStreamTrack()
        self.relay = MediaRelay()
    
    def frame_callback(self, frame: np.ndarray, detections: List[Detection] = None) -> None:
        """
        Callback to receive frames from VideoProcessor.
        
        Args:
            frame: Video frame
            detections: Optional list of detections to draw
        """
        if self.av is None:
            return
        
        # Draw detections if available
        if detections:
            frame = frame.copy()  # Create a copy to avoid modifying the original
            for detection in detections:
                x1, y1, x2, y2 = map(int, detection.bbox)
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label
                label = f"{detection.class_name}: {detection.confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Add frame to video track
        self.video_track.add_frame(frame)
    
    async def create_offer(self) -> Dict:
        """
        Create an SDP offer for a new peer connection.
        
        Returns:
            Dict with SDP offer and peer connection ID
        """
        if self.av is None:
            return {"error": "PyAV not installed"}
        
        try:
            # Create peer connection
            pc = RTCPeerConnection()
            pc_id = str(id(pc))
            self.peer_connections.add(pc)
            
            # Add video track
            pc.addTrack(self.relay.subscribe(self.video_track))
            
            # Create offer
            offer = await pc.createOffer()
            await pc.setLocalDescription(offer)
            
            # Handle ICE connection state
            @pc.on("iceconnectionstatechange")
            async def on_iceconnectionstatechange():
                if pc.iceConnectionState == "failed" or pc.iceConnectionState == "closed":
                    await self._close_peer_connection(pc)
            
            # Return offer
            return {
                "sdp": pc.localDescription.sdp,
                "type": pc.localDescription.type,
                "pc_id": pc_id
            }
            
        except Exception as e:
            self.logger.error(f"Error creating offer: {e}", exc_info=True)
            return {"error": str(e)}
    
    async def handle_answer(self, pc_id: str, sdp: str, type_: str) -> Dict:
        """
        Handle an SDP answer from a client.
        
        Args:
            pc_id: Peer connection ID
            sdp: SDP answer
            type_: SDP type
            
        Returns:
            Dict with result
        """
        # Find peer connection
        pc = None
        for connection in self.peer_connections:
            if str(id(connection)) == pc_id:
                pc = connection
                break
        
        if pc is None:
            return {"error": "Peer connection not found"}
        
        try:
            # Set remote description
            answer = RTCSessionDescription(sdp=sdp, type=type_)
            await pc.setRemoteDescription(answer)
            
            return {"success": True}
            
        except Exception as e:
            self.logger.error(f"Error handling answer: {e}", exc_info=True)
            return {"error": str(e)}
    
    async def add_ice_candidate(self, pc_id: str, candidate: Dict) -> Dict:
        """
        Add ICE candidate from client.
        
        Args:
            pc_id: Peer connection ID
            candidate: ICE candidate
            
        Returns:
            Dict with result
        """
        # Find peer connection
        pc = None
        for connection in self.peer_connections:
            if str(id(connection)) == pc_id:
                pc = connection
                break
        
        if pc is None:
            return {"error": "Peer connection not found"}
        
        try:
            # Add ICE candidate
            await pc.addIceCandidate(candidate)
            return {"success": True}
            
        except Exception as e:
            self.logger.error(f"Error adding ICE candidate: {e}", exc_info=True)
            return {"error": str(e)}
    
    async def _close_peer_connection(self, pc: RTCPeerConnection) -> None:
        """
        Close a peer connection.
        
        Args:
            pc: Peer connection to close
        """
        self.logger.info("Closing peer connection")
        
        if pc in self.peer_connections:
            self.peer_connections.remove(pc)
        
        # Close peer connection
        await pc.close()
    
    async def close_all(self) -> None:
        """Close all peer connections."""
        coros = [self._close_peer_connection(pc) for pc in list(self.peer_connections)]
        await asyncio.gather(*coros)
        self.peer_connections.clear()
        
        # Stop video track
        self.video_track.stop()


class FastAPIWebRTCServer:
    """FastAPI integration for WebRTC server."""
    
    def __init__(self, app=None, path: str = "/webrtc", config: Optional[Dict] = None):
        """
        Initialize FastAPI WebRTC server.
        
        Args:
            app: FastAPI app instance
            path: API endpoint path
            config: Configuration dictionary
        """
        self.logger = get_logger("FastAPIWebRTCServer")
        self.config = config or {}
        
        self.webrtc_server = WebRTCServer(config)
        
        # Register FastAPI endpoints if app is provided
        if app is not None:
            self.register_endpoints(app, path)
    
    def register_endpoints(self, app, path: str = "/webrtc") -> None:
        """
        Register FastAPI endpoints.
        
        Args:
            app: FastAPI app instance
            path: API endpoint path
        """
        try:
            from fastapi import APIRouter, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
            from pydantic import BaseModel
            
            # Create models
            class SDPOfferResponse(BaseModel):
                sdp: str
                type: str
                pc_id: str
            
            class SDPAnswerRequest(BaseModel):
                sdp: str
                type: str
                pc_id: str
            
            class ICECandidateRequest(BaseModel):
                candidate: Dict
                pc_id: str
            
            # Create router
            router = APIRouter()
            
            # Create offer endpoint
            @router.get("/offer", response_model=SDPOfferResponse)
            async def get_offer():
                result = await self.webrtc_server.create_offer()
                if "error" in result:
                    raise HTTPException(status_code=500, detail=result["error"])
                return result
            
            # Handle answer endpoint
            @router.post("/answer")
            async def post_answer(request: SDPAnswerRequest):
                result = await self.webrtc_server.handle_answer(
                    request.pc_id, request.sdp, request.type)
                if "error" in result:
                    raise HTTPException(status_code=500, detail=result["error"])
                return {"success": True}
            
            # Handle ICE candidate endpoint
            @router.post("/ice_candidate")
            async def post_ice_candidate(request: ICECandidateRequest):
                result = await self.webrtc_server.add_ice_candidate(
                    request.pc_id, request.candidate)
                if "error" in result:
                    raise HTTPException(status_code=500, detail=result["error"])
                return {"success": True}
            
            # Add router to app
            app.include_router(router, prefix=path)
            
            # Register on_shutdown handler
            @app.on_event("shutdown")
            async def shutdown_event():
                await self.webrtc_server.close_all()
            
            self.logger.info(f"Registered WebRTC endpoints at {path}")
            
        except ImportError as e:
            self.logger.error(f"Failed to import FastAPI dependencies: {e}")
    
    def frame_callback(self, frame: np.ndarray, detections: List[Detection] = None) -> None:
        """Proxy frame callback to WebRTC server."""
        self.webrtc_server.frame_callback(frame, detections)