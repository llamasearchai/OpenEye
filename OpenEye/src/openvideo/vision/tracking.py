"""
Object tracking implementation.
"""

import logging
import time
import numpy as np
import cv2
from typing import Dict, Optional, Any, List, Tuple, Union, Callable

from ..core.types import VideoFrame, DetectionResult, TrackingResult

logger = logging.getLogger(__name__)


class Tracker:
    """
    Base class for object trackers.
    """
    
    def __init__(self):
        """Initialize tracker."""
        self.tracks = {}
        self.next_id = 0
        
    def update(self, detections: List[DetectionResult], 
               frame: VideoFrame) -> List[TrackingResult]:
        """
        Update tracks with new detections.
        
        Args:
            detections: List of new detections
            frame: Current video frame
            
        Returns:
            List[TrackingResult]: Updated tracking results
        """
        # To be implemented by subclasses
        return []
        
    def reset(self):
        """Reset tracking state."""
        self.tracks = {}
        self.next_id = 0


class ByteTracker(Tracker):
    """
    ByteTrack implementation.
    
    Based on the paper: ByteTrack: Multi-Object Tracking by Associating 
    Every Detection Box
    """
    
    def __init__(self, max_age: int = 30, min_hits: int = 3, 
                 iou_threshold: float = 0.3):
        """
        Initialize ByteTrack tracker.
        
        Args:
            max_age: Maximum frames to keep a track alive without detection
            min_hits: Minimum detection hits to confirm a track
            iou_threshold: IoU threshold for association
        """
        super().__init__()
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.frame_count = 0
        
    def update(self, detections: List[DetectionResult], 
               frame: VideoFrame) -> List[TrackingResult]:
        """
        Update tracks with new detections.
        
        Args:
            detections: List of new detections
            frame: Current video frame
            
        Returns:
            List[TrackingResult]: Updated tracking results
        """
        self.frame_count += 1
        
        # Split detections into high and low confidence
        high_dets = [d for d in detections if d.confidence >= 0.5]
        low_dets = [d for d in detections if 0.1 <= d.confidence < 0.5]
        
        # Convert detections to format expected by association function
        high_boxes = np.array([d.bbox for d in high_dets]) if high_dets else np.empty((0, 4))
        low_boxes = np.array([d.bbox for d in low_dets]) if low_dets else np.empty((0, 4))
        
        # Step 1: First association with high confidence detections
        active_tracks = {track_id: track for track_id, track in self.tracks.items() 
                          if track['age'] < self.max_age}
        
        matched, unmatched_tracks, unmatched_dets = self._associate_detections_to_tracks(
            high_boxes, active_tracks, self.iou_threshold
        )
        
        # Update matched tracks
        for track_idx, det_idx in matched:
            track_id = list(active_tracks.keys())[track_idx]
            self._update_track(track_id, high_dets[det_idx])
            
        # Step 2: Second association with low confidence detections
        # Get remaining tracks
        remaining_tracks = {track_id: track for i, (track_id, track) in 
                            enumerate(active_tracks.items()) 
                            if i in unmatched_tracks}
        
        # Associate with low confidence detections
        if len(remaining_tracks) > 0 and len(low_dets) > 0:
            matched2, unmatched_tracks2, unmatched_dets2 = self._associate_detections_to_tracks(
                low_boxes, remaining_tracks, self.iou_threshold
            )
            
            # Update matched tracks from second association
            for track_idx, det_idx in matched2:
                track_id = list(remaining_tracks.keys())[track_idx]
                self._update_track(track_id, low_dets[det_idx])
                
            # Update unmatched tracks from second association
            for i in unmatched_tracks2:
                track_id = list(remaining_tracks.keys())[i]
                self._mark_missing(track_id)
        else:
            # Update all unmatched tracks from first association
            for i in unmatched_tracks:
                track_id = list(active_tracks.keys())[i]
                self._mark_missing(track_id)
        
        # Create new tracks from unmatched high confidence detections
        for i in unmatched_dets:
            self._create_track(high_dets[i])
            
        # Remove old tracks
        self.tracks = {k: v for k, v in self.tracks.items() if v['age'] < self.max_age}
        
        # Return current tracks as TrackingResult objects
        results = []
        for track_id, track in self.tracks.items():
            if track['hit_streak'] >= self.min_hits:
                result = TrackingResult(
                    track_id=track_id,
                    bbox=track['bbox'],
                    confidence=track['confidence'],
                    class_id=track['class_id'],
                    class_name=track['class_name'],
                    timestamp=frame.timestamp,
                    age=track['age'],
                    velocity=track['velocity'] if 'velocity' in track else [0, 0]
                )
                results.append(result)
                
        return results
        
    def _create_track(self, detection: DetectionResult) -> int:
        """
        Create a new track from detection.
        
        Args:
            detection: Detection to create track from
            
        Returns:
            int: New track ID
        """
        track_id = self.next_id
        self.next_id += 1
        
        self.tracks[track_id] = {
            'bbox': detection.bbox,
            'confidence': detection.confidence,
            'class_id': detection.class_id,
            'class_name': detection.class_name,
            'last_detection': detection,
            'time_since_update': 0,
            'hit_streak': 1,
            'age': 0,
            'history': [detection.bbox]
        }
        
        return track_id
        
    def _update_track(self, track_id: int, detection: DetectionResult) -> None:
        """
        Update existing track with new detection.
        
        Args:
            track_id: Track ID to update
            detection: New detection for the track
        """
        track = self.tracks[track_id]
        
        # Calculate velocity (simple implementation)
        prev_center = [(track['bbox'][0] + track['bbox'][2]) / 2,
                       (track['bbox'][1] + track['bbox'][3]) / 2]
        curr_center = [(detection.bbox[0] + detection.bbox[2]) / 2,
                       (detection.bbox[1] + detection.bbox[3]) / 2]
        
        velocity = [curr_center[0] - prev_center[0], 
                    curr_center[1] - prev_center[1]]
        
        # Update track data
        track['bbox'] = detection.bbox
        track['confidence'] = detection.confidence
        track['last_detection'] = detection
        track['time_since_update'] = 0
        track['hit_streak'] += 1
        track['velocity'] = velocity
        track['history'].append(detection.bbox)
        
        # Keep history limited to a certain length
        if len(track['history']) > 30:
            track['history'] = track['history'][-30:]
            
    def _mark_missing(self, track_id: int) -> None:
        """
        Mark a track as missing (no detection in current frame).
        
        Args:
            track_id: Track ID to mark as missing
        """
        track = self.tracks[track_id]
        
        # Increment time since last update
        track['time_since_update'] += 1
        track['hit_streak'] = 0
        
        # Apply Kalman filter prediction or simple motion model
        if 'velocity' in track and track['time_since_update'] < 3:
            # Simple motion model - move the box by velocity
            prev_box = track['bbox']
            vx, vy = track['velocity']
            
            # Update box position using velocity
            new_box = [
                prev_box[0] + vx,  # x_min
                prev_box[1] + vy,  # y_min
                prev_box[2] + vx,  # x_max
                prev_box[3] + vy   # y_max
            ]
            
            track['bbox'] = new_box
            
        # Increment age
        track['age'] += 1
        
    def _associate_detections_to_tracks(self, detections: np.ndarray, 
                                      tracks: Dict[int, Dict], 
                                      threshold: float) -> Tuple:
        """
        Associate detections to existing tracks using IoU.
        
        Args:
            detections: Array of detection bounding boxes
            tracks: Dictionary of existing tracks
            threshold: IoU threshold for matching
            
        Returns:
            Tuple: (matched pairs, unmatched track indices, unmatched detection indices)
        """
        if len(tracks) == 0 or len(detections) == 0:
            return [], list(range(len(tracks))), list(range(len(detections)))
            
        # Calculate IoU between detections and tracks
        track_boxes = np.array([track['bbox'] for track in tracks.values()])
        iou_matrix = np.zeros((len(track_boxes), len(detections)))
        
        for t in range(len(track_boxes)):
            for d in range(len(detections)):
                iou_matrix[t, d] = self._iou(track_boxes[t], detections[d])
                
        # Use Hungarian algorithm to find optimal matching
        try:
            from scipy.optimize import linear_sum_assignment
            track_indices, det_indices = linear_sum_assignment(-iou_matrix)
        except ImportError:
            # Fallback to greedy matching if scipy not available
            track_indices, det_indices = self._greedy_match(iou_matrix)
            
        # Filter out low IoU matches
        matches = []
        unmatched_tracks = []
        unmatched_detections = list(range(len(detections)))
        
        for t, d in zip(track_indices, det_indices):
            if iou_matrix[t, d] >= threshold:
                matches.append((t, d))
                unmatched_detections.remove(d)
            else:
                unmatched_tracks.append(t)
                
        # Add unmatched tracks
        for t in range(len(tracks)):
            if t not in track_indices:
                unmatched_tracks.append(t)
                
        return matches, unmatched_tracks, unmatched_detections
        
    def _greedy_match(self, iou_matrix: np.ndarray) -> Tuple[List[int], List[int]]:
        """
        Perform greedy matching for IoU matrix.
        
        Args:
            iou_matrix: IoU matrix of shape (num_tracks, num_detections)
            
        Returns:
            Tuple: Lists of matched track and detection indices
        """
        track_indices = []
        det_indices = []
        
        # Make a copy of the IoU matrix
        iou = iou_matrix.copy()
        
        # While there are still valid matches
        while np.max(iou) > 0:
            # Find the highest IoU
            matched_idx = np.unravel_index(np.argmax(iou), iou.shape)
            track_idx, det_idx = matched_idx
            
            # Add the match
            track_indices.append(track_idx)
            det_indices.append(det_idx)
            
            # Set the matched row and column to zero
            iou[track_idx, :] = 0
            iou[:, det_idx] = 0
            
        return track_indices, det_indices
        
    def _iou(self, box1: List[float], box2: List[float]) -> float:
        """
        Calculate IoU between two boxes.
        
        Args:
            box1: First box [x_min, y_min, x_max, y_max]
            box2: Second box [x_min, y_min, x_max, y_max]
            
        Returns:
            float: IoU value
        """
        # Calculate intersection area
        x_min = max(box1[0], box2[0])
        y_min = max(box1[1], box2[1])
        x_max = min(box1[2], box2[2])
        y_max = min(box1[3], box2[3])
        
        if x_max <= x_min or y_max <= y_min:
            return 0.0
            
        intersection = (x_max - x_min) * (y_max - y_min)
        
        # Calculate union area
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0