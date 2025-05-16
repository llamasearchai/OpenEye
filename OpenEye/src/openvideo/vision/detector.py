import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch

from ..utils.logging import get_logger


class Detection:
    """Class representing a single detection result."""
    
    def __init__(self, 
                 bbox: List[float], 
                 class_id: int, 
                 class_name: str, 
                 confidence: float):
        """
        Initialize a detection.
        
        Args:
            bbox: Bounding box coordinates [x1, y1, x2, y2]
            class_id: Class ID
            class_name: Class name
            confidence: Detection confidence
        """
        self.bbox = bbox
        self.class_id = class_id
        self.class_name = class_name
        self.confidence = confidence
    
    def __repr__(self):
        return f"Detection(class={self.class_name}, conf={self.confidence:.2f}, bbox={self.bbox})"
    
    def to_dict(self):
        """Convert to dictionary representation."""
        return {
            "bbox": self.bbox,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence
        }


class BaseDetector(ABC):
    """Abstract base class for object detectors."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the detector.
        
        Args:
            config: Configuration dictionary
        """
        self.logger = get_logger(self.__class__.__name__)
        self.config = config or {}
    
    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect objects in a frame.
        
        Args:
            frame: Image as numpy array (BGR format)
            
        Returns:
            List of Detection objects
        """
        pass
    
    @abstractmethod
    def load_model(self, model_path: str) -> bool:
        """
        Load a model from disk.
        
        Args:
            model_path: Path to model file
            
        Returns:
            bool: Success status
        """
        pass


class YOLODetector(BaseDetector):
    """YOLO object detector implementation."""
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 confidence_threshold: float = 0.25,
                 nms_threshold: float = 0.45,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 config: Optional[Dict] = None):
        """
        Initialize the YOLO detector.
        
        Args:
            model_path: Path to YOLO model file (e.g., yolov8n.pt)
            confidence_threshold: Detection confidence threshold
            nms_threshold: Non-maximum suppression IoU threshold
            device: Computation device ('cuda' or 'cpu')
            config: Additional configuration
        """
        super().__init__(config)
        
        self.model = None
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.device = device
        self.class_names = []
        
        # Import here to avoid dependency if not using YOLO
        try:
            import ultralytics
            self.ultralytics = ultralytics
        except ImportError:
            self.logger.error("Failed to import ultralytics. Please install with: pip install ultralytics")
            self.ultralytics = None
        
        # Load model if path is provided
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str) -> bool:
        """
        Load a YOLO model from disk.
        
        Args:
            model_path: Path to model file
            
        Returns:
            bool: Success status
        """
        if self.ultralytics is None:
            return False
        
        try:
            self.logger.info(f"Loading YOLO model: {model_path}")
            start_time = time.time()
            
            # Load model with ultralytics
            self.model = self.ultralytics.YOLO(model_path)
            self.model_path = model_path
            
            # Get class names from model
            self.class_names = self.model.names
            
            load_time = time.time() - start_time
            self.logger.info(f"Model loaded in {load_time:.2f}s with {len(self.class_names)} classes")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load YOLO model: {e}", exc_info=True)
            self.model = None
            return False
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect objects in a frame using YOLO.
        
        Args:
            frame: Image as numpy array (BGR format)
            
        Returns:
            List of Detection objects
        """
        if self.model is None:
            self.logger.error("Model not loaded")
            return []
        
        try:
            # Run inference
            results = self.model.predict(
                frame,
                conf=self.confidence_threshold,
                iou=self.nms_threshold,
                device=self.device
            )
            
            # Process results
            detections = []
            
            if results and len(results) > 0:
                result = results[0]  # Get first image result
                
                # Get boxes, confidence scores, and class IDs
                boxes = result.boxes
                
                for i in range(len(boxes)):
                    # Get bounding box
                    box = boxes[i].xyxy[0].tolist()  # Convert to [x1, y1, x2, y2]
                    
                    # Get class details
                    class_id = int(boxes[i].cls[0].item())
                    class_name = self.class_names[class_id]
                    
                    # Get confidence
                    confidence = boxes[i].conf[0].item()
                    
                    # Add detection
                    detection = Detection(box, class_id, class_name, confidence)
                    detections.append(detection)
            
            return detections
            
        except Exception as e:
            self.logger.error(f"Error during detection: {e}", exc_info=True)
            return []