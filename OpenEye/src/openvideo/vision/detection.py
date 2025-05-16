"""
Object detection using various models (YOLO, SSD, etc.)
"""

import logging
import time
import os
import threading
from typing import Dict, Optional, Any, List, Tuple, Union, Callable
import numpy as np
import cv2

from ..core.types import VideoFrame, DetectionResult

logger = logging.getLogger(__name__)

# Try to import PyTorch
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch not available, some detection models will be limited")
    TORCH_AVAILABLE = False

# Try to import TensorFlow
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    logger.warning("TensorFlow not available, some detection models will be limited")
    TF_AVAILABLE = False

# Try to import ONNX Runtime
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    logger.warning("ONNX Runtime not available, some detection models will be limited")
    ONNX_AVAILABLE = False


class ObjectDetector:
    """
    Base class for object detection models.
    """
    
    def __init__(self, model_path: str = None, 
                 confidence_threshold: float = 0.5,
                 nms_threshold: float = 0.4,
                 device: str = "auto"):
        """
        Initialize object detector.
        
        Args:
            model_path: Path to model file
            confidence_threshold: Minimum confidence threshold for detections
            nms_threshold: Non-maximum suppression threshold
            device: Device to run inference on ('cpu', 'cuda', or 'auto')
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        
        # Determine device
        if device == "auto":
            if TORCH_AVAILABLE and torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device
            
        self.model = None
        self.class_names = []
        self.input_size = (640, 640)  # Default input size
        self.running = False
        
    def load_model(self) -> bool:
        """
        Load the detection model.
        
        Returns:
            bool: True if model loaded successfully
        """
        # To be implemented by subclasses
        return False
        
    def preprocess(self, frame: VideoFrame) -> np.ndarray:
        """
        Preprocess the input frame.
        
        Args:
            frame: Input video frame
            
        Returns:
            np.ndarray: Preprocessed frame
        """
        # Default preprocessing - resize to input size
        if frame.data.shape[:2] != self.input_size:
            resized = cv2.resize(frame.data, self.input_size)
        else:
            resized = frame.data.copy()
            
        # Convert BGR to RGB if necessary
        if len(resized.shape) == 3 and resized.shape[2] == 3:
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        else:
            rgb = resized
            
        # Normalize to [0, 1]
        normalized = rgb.astype(np.float32) / 255.0
        
        return normalized
        
    def detect(self, frame: VideoFrame) -> List[DetectionResult]:
        """
        Detect objects in the input frame.
        
        Args:
            frame: Input video frame
            
        Returns:
            List[DetectionResult]: List of detection results
        """
        # To be implemented by subclasses
        return []
        
    def postprocess(self, predictions: Any, original_frame: VideoFrame) -> List[DetectionResult]:
        """
        Postprocess model predictions.
        
        Args:
            predictions: Raw model predictions
            original_frame: Original input frame
            
        Returns:
            List[DetectionResult]: List of detection results
        """
        # To be implemented by subclasses
        return []


class YOLODetector(ObjectDetector):
    """
    YOLO object detector implementation.
    """
    
    def __init__(self, model_path: str = None, 
                 confidence_threshold: float = 0.5,
                 nms_threshold: float = 0.4,
                 device: str = "auto",
                 model_version: str = "v8"):
        """
        Initialize YOLO object detector.
        
        Args:
            model_path: Path to model file
            confidence_threshold: Minimum confidence threshold for detections
            nms_threshold: Non-maximum suppression threshold
            device: Device to run inference on ('cpu', 'cuda', or 'auto')
            model_version: YOLO model version ('v5', 'v7', 'v8', etc.)
        """
        super().__init__(model_path, confidence_threshold, nms_threshold, device)
        self.model_version = model_version
        
        # Default paths for common YOLO models
        if not model_path:
            if model_version == "v8":
                self.model_path = os.path.join(os.path.dirname(__file__), 
                                             "models", "yolov8n.pt")
            elif model_version == "v5":
                self.model_path = os.path.join(os.path.dirname(__file__), 
                                             "models", "yolov5s.pt")
            else:
                raise ValueError(f"Unsupported YOLO version: {model_version}")
                
        # Make sure models directory exists
        os.makedirs(os.path.join(os.path.dirname(__file__), "models"), exist_ok=True)
        
    def load_model(self) -> bool:
        """
        Load the YOLO model.
        
        Returns:
            bool: True if model loaded successfully
        """
        if not TORCH_AVAILABLE:
            logger.error("PyTorch is required for YOLO models")
            return False
            
        try:
            if self.model_version == "v8":
                try:
                    from ultralytics import YOLO
                    self.model = YOLO(self.model_path)
                    
                    # Set device
                    if self.device == "cuda":
                        self.model.to("cuda")
                    
                    # Get class names
                    self.class_names = self.model.names
                    
                    logger.info(f"Loaded YOLOv8 model: {self.model_path}")
                    return True
                    
                except ImportError:
                    logger.error("ultralytics package is required for YOLOv8")
                    return False
                    
            elif self.model_version == "v5":
                try:
                    import torch.nn as nn
                    
                    # Load YOLOv5 model
                    self.model = torch.hub.load('ultralytics/yolov5', 'custom', 
                                              path=self.model_path)
                    
                    # Set device
                    if self.device == "cuda":
                        self.model.to("cuda")
                    
                    # Get class names
                    self.class_names = self.model.names
                    
                    logger.info(f"Loaded YOLOv5 model: {self.model_path}")
                    return True
                    
                except Exception as e:
                    logger.error(f"Failed to load YOLOv5 model: {e}", exc_info=True)
                    return False
                    
            else:
                logger.error(f"Unsupported YOLO version: {self.model_version}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}", exc_info=True)
            return False
            
    def detect(self, frame: VideoFrame) -> List[DetectionResult]:
        """
        Detect objects in the input frame.
        
        Args:
            frame: Input video frame
            
        Returns:
            List[DetectionResult]: List of detection results
        """
        if self.model is None:
            if not self.load_model():
                return []
                
        try:
            # Process the frame with YOLO
            if self.model_version == "v8":
                # YOLOv8 accepts BGR frames directly
                results = self.model(frame.data, conf=self.confidence_threshold, 
                                   iou=self.nms_threshold)
                
                # Convert YOLOv8 results to DetectionResult
                detections = []
                for result in results:
                    boxes = result.boxes
                    for i in range(len(boxes)):
                        box = boxes[i].xyxy[0].cpu().numpy()  # Get box coordinates
                        conf = float(boxes[i].conf[0])  # Get confidence
                        cls_id = int(boxes[i].cls[0])  # Get class ID
                        
                        # Create DetectionResult
                        detection = DetectionResult(
                            bbox=[float(x) for x in box],
                            confidence=conf,
                            class_id=cls_id,
                            class_name=self.class_names.get(cls_id, f"class_{cls_id}"),
                            timestamp=frame.timestamp
                        )
                        detections.append(detection)
                
                return detections
                
            elif self.model_version == "v5":
                # Run YOLOv5 inference
                results = self.model(frame.data)
                
                # Get the detection results
                detections = []
                
                # Convert YOLOv5 results to DetectionResult
                for *box, conf, cls_id in results.xyxy[0].cpu().numpy():
                    if conf >= self.confidence_threshold:
                        # Create DetectionResult
                        detection = DetectionResult(
                            bbox=[float(x) for x in box],
                            confidence=float(conf),
                            class_id=int(cls_id),
                            class_name=self.class_names.get(int(cls_id), f"class_{int(cls_id)}"),
                            timestamp=frame.timestamp
                        )
                        detections.append(detection)
                
                return detections
                
        except Exception as e:
            logger.error(f"Error in YOLO detection: {e}", exc_info=True)
            
        return []


class TensorflowDetector(ObjectDetector):
    """
    TensorFlow object detector implementation.
    """
    
    def __init__(self, model_path: str = None, 
                 confidence_threshold: float = 0.5,
                 device: str = "auto",
                 labels_path: str = None):
        """
        Initialize TensorFlow object detector.
        
        Args:
            model_path: Path to model file
            confidence_threshold: Minimum confidence threshold for detections
            device: Device to run inference on ('cpu', 'cuda', or 'auto')
            labels_path: Path to labels file
        """
        super().__init__(model_path, confidence_threshold, 0.5, device)
        self.labels_path = labels_path
        
        # Set default input size
        self.input_size = (300, 300)
        
    def load_model(self) -> bool:
        """
        Load the TensorFlow model.
        
        Returns:
            bool: True if model loaded successfully
        """
        if not TF_AVAILABLE:
            logger.error("TensorFlow is required for TensorFlow detector")
            return False
            
        try:
            # Configure GPU memory growth if using CUDA
            if self.device == "cuda":
                physical_devices = tf.config.list_physical_devices('GPU')
                if physical_devices:
                    for device in physical_devices:
                        tf.config.experimental.set_memory_growth(device, True)
            
            # Load saved model
            self.model = tf.saved_model.load(self.model_path)
            self.detect_fn = self.model.signatures['serving_default']
            
            # Load labels
            if self.labels_path:
                self.class_names = {}
                with open(self.labels_path, 'r') as f:
                    for i, label in enumerate(f.read().splitlines()):
                        self.class_names[i] = label
            
            logger.info(f"Loaded TensorFlow model: {self.model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load TensorFlow model: {e}", exc_info=True)
            return False
            
    def preprocess(self, frame: VideoFrame) -> np.ndarray:
        """
        Preprocess the input frame for TF model.
        
        Args:
            frame: Input video frame
            
        Returns:
            np.ndarray: Preprocessed frame
        """
        # Resize to expected size
        resized = cv2.resize(frame.data, self.input_size)
        
        # Convert to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Expand dimensions and convert to float32
        input_tensor = np.expand_dims(rgb, axis=0).astype(np.float32)
        
        return input_tensor
        
    def detect(self, frame: VideoFrame) -> List[DetectionResult]:
        """
        Detect objects in the input frame.
        
        Args:
            frame: Input video frame
            
        Returns:
            List[DetectionResult]: List of detection results
        """
        if self.model is None:
            if not self.load_model():
                return []
                
        try:
            # Preprocess image
            input_tensor = self.preprocess(frame)
            
            # Run inference
            output_dict = self.detect_fn(tf.convert_to_tensor(input_tensor))
            
            # Process outputs
            num_detections = int(output_dict['num_detections'])
            detection_classes = output_dict['detection_classes'][0].numpy().astype(np.int32)
            detection_boxes = output_dict['detection_boxes'][0].numpy()
            detection_scores = output_dict['detection_scores'][0].numpy()
            
            # Convert to list of DetectionResult
            detections = []
            h, w = frame.data.shape[:2]
            
            for i in range(num_detections):
                if detection_scores[i] >= self.confidence_threshold:
                    # TF object detection API returns normalized coordinates [y_min, x_min, y_max, x_max]
                    # Convert to [x_min, y_min, x_max, y_max] in pixel coordinates
                    y_min, x_min, y_max, x_max = detection_boxes[i]
                    x_min = int(x_min * w)
                    x_max = int(x_max * w)
                    y_min = int(y_min * h)
                    y_max = int(y_max * h)
                    
                    class_id = detection_classes[i]
                    class_name = self.class_names.get(class_id, f"class_{class_id}")
                    
                    # Create DetectionResult
                    detection = DetectionResult(
                        bbox=[x_min, y_min, x_max, y_max],
                        confidence=float(detection_scores[i]),
                        class_id=int(class_id),
                        class_name=class_name,
                        timestamp=frame.timestamp
                    )
                    detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.error(f"Error in TensorFlow detection: {e}", exc_info=True)
            
        return []