#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

use tauri::{AppHandle, Manager, State};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::sync::{Arc, Mutex};
use serde::{Deserialize, Serialize};

// Python interpreter state
struct PythonState {
    interpreter: Arc<Mutex<Python<'static>>>,
}

// Response structure for commands
#[derive(Serialize)]
struct CommandResponse {
    success: bool,
    error: Option<String>,
    data: Option<serde_json::Value>,
}

// Request structure for opening a video
#[derive(Deserialize)]
struct OpenVideoRequest {
    path: String,
}

// Request structure for analyzing a frame
#[derive(Deserialize)]
struct AnalyzeFrameRequest {
    frame_index: i32,
}

// Request structure for querying video
#[derive(Deserialize)]
struct QueryVideoRequest {
    query: String,
}

// Initialize Python
fn init_python() -> PyResult<Python<'static>> {
    pyo3::prepare_freethreaded_python();
    let py = unsafe { Python::assume_gil_acquired() };
    
    // Import required modules
    py.run(
        r#"
import sys
import os

# Add project directory to path
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

# Import OpenVideo modules
try:
    from openvideo.core.processor import VideoProcessor
    from openvideo.vision.detector import YOLODetector
    from openvideo.streaming.rtsp import RTSPServer
    from openvideo.ai.assistant import VideoAssistant
    from openvideo.ai.query_engine import VideoQueryEngine
    print("OpenVideo modules imported successfully")
except ImportError as e:
    print(f"Error importing OpenVideo modules: {e}")
        "#,
        None,
        None,
    )?;
    
    Ok(py)
}

// Initialize global modules
fn init_modules(py: Python) -> PyResult<()> {
    // Initialize modules and store them in Python's builtins
    py.run(
        r#"
import builtins
import os

# Get OpenAI API key from environment
api_key = os.environ.get('OPENAI_API_KEY')

# Initialize video processor
builtins.video_processor = None
builtins.detector = None
builtins.rtsp_server = None
builtins.assistant = VideoAssistant(api_key=api_key)
builtins.query_engine = VideoQueryEngine(api_key=api_key)

print("OpenVideo modules initialized")
        "#,
        None,
        None,
    )?;
    
    Ok(())
}

// Command to open a video source
#[tauri::command]
fn open_video(
    python_state: State<'_, PythonState>,
    request: OpenVideoRequest,
) -> CommandResponse {
    let py_mutex = python_state.interpreter.clone();
    let result = py_mutex.lock().unwrap().with_gil(|py| {
        let locals = PyDict::new(py);
        locals.set_item("video_path", request.path.clone())?;
        
        // Call Python code to open video
        py.run(
            r#"
import builtins
from openvideo.core.processor import VideoProcessor
from openvideo.vision.detector import YOLODetector

# Initialize detector if not already
if builtins.detector is None:
    try:
        builtins.detector = YOLODetector()
        print("Initialized YOLO detector")
    except Exception as e:
        print(f"Failed to initialize detector: {e}")
        builtins.detector = None

# Create video processor
if builtins.video_processor is None:
    builtins.video_processor = VideoProcessor(detector=builtins.detector)
    print("Created video processor")

# Open video
success = builtins.video_processor.open(video_path)
metadata = builtins.video_processor.get_metadata() if success else {}

# Start processing
if success:
    builtins.video_processor.start()
    
result = {
    "success": success,
    "metadata": metadata
}
            "#,
            None,
            Some(locals),
        )?;
        
        // Extract result
        let result: serde_json::Value = locals.get_item("result")?.extract()?;
        Ok(result)
    });
    
    match result {
        Ok(data) => CommandResponse {
            success: true,
            error: None,
            data: Some(data),
        },
        Err(e) => CommandResponse {
            success: false,
            error: Some(format!("Error: {}", e)),
            data: None,
        },
    }
}

// Command to analyze current frame
#[tauri::command]
fn analyze_frame(
    python_state: State<'_, PythonState>,
    request: AnalyzeFrameRequest,
) -> CommandResponse {
    let py_mutex = python_state.interpreter.clone();
    let result = py_mutex.lock().unwrap().with_gil(|py| {
        let locals = PyDict::new(py);
        locals.set_item("frame_index", request.frame_index)?;
        
        py.run(
            r#"
import builtins
import base64
import cv2
import json
import numpy as np

result = {"success": False, "analysis": None}

# Check if video processor exists
if builtins.video_processor is None:
    result["error"] = "Video processor not initialized"
else:
    # Get frame
    frame = builtins.video_processor.get_frame(frame_index=frame_index)
    
    if frame is None:
        result["error"] = f"Failed to get frame {frame_index}"
    else:
        # Analyze with assistant
        if builtins.assistant is not None:
            prompt = "Describe this frame in detail. What objects are visible? What's happening in the scene?"
            analysis = builtins.assistant.analyze_frame(frame, prompt)
            
            # Encode frame to base64 for UI
            _, buffer = cv2.imencode(".jpg", frame)
            base64_image = base64.b64encode(buffer).decode("utf-8")
            
            result["success"] = True
            result["analysis"] = analysis
            result["frame_base64"] = base64_image
        else:
            result["error"] = "Assistant not initialized"
            "#,
            None,
            Some(locals),
        )?;
        
        // Extract result
        let result: serde_json::Value = locals.get_item("result")?.extract()?;
        Ok(result)
    });
    
    match result {
        Ok(data) => CommandResponse {
            success: true,
            error: None,
            data: Some(data),
        },
        Err(e) => CommandResponse {
            success: false,
            error: Some(format!("Error: {}", e)),
            data: None,
        },
    }
}

// Command to query video
#[tauri::command]
fn query_video(
    python_state: State<'_, PythonState>,
    request: QueryVideoRequest,
) -> CommandResponse {
    let py_mutex = python_state.interpreter.clone();
    let result = py_mutex.lock().unwrap().with_gil(|py| {
        let locals = PyDict::new(py);
        locals.set_item("query_text", request.query)?;
        
        py.run(
            r#"
import builtins

result = {"success": False, "answer": None}

# Check if query engine exists
if builtins.query_engine is None:
    result["error"] = "Query engine not initialized"
else:
    # Execute query
    query_result = builtins.query_engine.query(query_text)
    
    result["success"] = True
    result["answer"] = query_result
            "#,
            None,
            Some(locals),
        )?;
        
        // Extract result
        let result: serde_json::Value = locals.get_item("result")?.extract()?;
        Ok(result)
    });
    
    match result {
        Ok(data) => CommandResponse {
            success: true,
            error: None,
            data: Some(data),
        },
        Err(e) => CommandResponse {
            success: false,
            error: Some(format!("Error: {}", e)),
            data: None,
        },
    }
}

// Command to start RTSP server
#[tauri::command]
fn start_rtsp_server(python_state: State<'_, PythonState>) -> CommandResponse {
    let py_mutex = python_state.interpreter.clone();
    let result = py_mutex.lock().unwrap().with_gil(|py| {
        let locals = PyDict::new(py);
        
        py.run(
            r#"
import builtins
from openvideo.streaming.rtsp import RTSPServer

result = {"success": False, "url": None}

# Check if video processor exists
if builtins.video_processor is None:
    result["error"] = "Video processor not initialized"
else:
    # Create RTSP server if it doesn't exist
    if builtins.rtsp_server is None:
        metadata = builtins.video_processor.get_metadata()
        builtins.rtsp_server = RTSPServer(
            port=8554,
            stream_name="drone",
            codec="h264",
            fps=metadata.get("fps", 30),
            width=metadata.get("width"),
            height=metadata.get("height")
        )
    
    # Start server and add to processor
    success = builtins.rtsp_server.start()
    if success:
        builtins.video_processor.add_output(builtins.rtsp_server)
        
    result["success"] = success
    result["url"] = builtins.rtsp_server.url if success else None
            "#,
            None,
            Some(locals),
        )?;
        
        // Extract result
        let result: serde_json::Value = locals.get_item("result")?.extract()?;
        Ok(result)
    });
    
    match result {
        Ok(data) => CommandResponse {
            success: true,
            error: None,
            data: Some(data),
        },
        Err(e) => CommandResponse {
            success: false,
            error: Some(format!("Error: {}", e)),
            data: None,
        },
    }
}

fn main() {
    tauri::Builder::default()
        .setup(|app| {
            // Initialize Python interpreter
            let python = match init_python() {
                Ok(py) => py,
                Err(e) => {
                    eprintln!("Failed to initialize Python: {}", e);
                    std::process::exit(1);
                }
            };
            
            // Initialize modules
            if let Err(e) = init_modules(python) {
                eprintln!("Failed to initialize modules: {}", e);
                std::process::exit(1);
            }
            
            // Store Python interpreter in state
            app.manage(PythonState {
                interpreter: Arc::new(Mutex::new(python)),
            });
            
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            open_video,
            analyze_frame,
            query_video,
            start_rtsp_server,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}