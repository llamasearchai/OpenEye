"""
LangChain integration for natural language query of video archives.
"""

import logging
import os
import time
import json
from typing import Dict, Optional, Any, List, Tuple, Union, Callable
import datetime

logger = logging.getLogger(__name__)

# Try to import LangChain
try:
    from langchain.chains import LLMChain, RetrievalQA
    from langchain.prompts import PromptTemplate
    from langchain.vectorstores import FAISS
    from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
    from langchain.llms import OpenAI
    from langchain.chat_models import ChatOpenAI
    from langchain.schema import Document
    
    LANGCHAIN_AVAILABLE = True
except ImportError:
    logger.warning("LangChain not available, natural language query functions will be limited")
    LANGCHAIN_AVAILABLE = False


class VideoQueryEngine:
    """
    Natural language query engine for video archives.
    """
    
    def __init__(self, api_key: str = None, 
                 embedding_model: str = "text-embedding-ada-002",
                 llm_model: str = "gpt-3.5-turbo",
                 use_local_embeddings: bool = False,
                 embedding_dimension: int = 1536,
                 db_path: str = "./video_vector_db"):
        """
        Initialize video query engine.
        
        Args:
            api_key: OpenAI API key
            embedding_model: OpenAI embedding model or HuggingFace model name
            llm_model: LLM model to use
            use_local_embeddings: Whether to use local HuggingFace embeddings
            embedding_dimension: Dimension of embeddings
            db_path: Path to vector database
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain is required for VideoQueryEngine")
            
        # API key setup
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key and not use_local_embeddings:
            logger.warning("No OpenAI API key provided, some features may be limited")
            
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.use_local_embeddings = use_local_embeddings
        self.embedding_dimension = embedding_dimension
        self.db_path = db_path
        
        # Initialize models and database
        self._initialize()
        
    def _initialize(self) -> None:
        """Initialize the models and vector database."""
        try:
            # Initialize embeddings
            if self.use_local_embeddings:
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=self.embedding_model
                )
            else:
                self.embeddings = OpenAIEmbeddings(
                    model=self.embedding_model,
                    openai_api_key=self.api_key
                )
                
            # Initialize vector store if it exists
            if os.path.exists(self.db_path):
                self.vector_store = FAISS.load_local(
                    self.db_path,
                    self.embeddings
                )
                logger.info(f"Loaded vector database from {self.db_path}")
            else:
                # Create empty vector store
                self.vector_store = FAISS.from_documents(
                    [Document(page_content="OpenVideo initialization document", 
                             metadata={"source": "init"})],
                    self.embeddings
                )
                logger.info("Created new vector database")
                
            # Initialize LLM
            if "gpt" in self.llm_model.lower():
                self.llm = ChatOpenAI(
                    model_name=self.llm_model,
                    openai_api_key=self.api_key,
                    temperature=0
                )
            else:
                self.llm = OpenAI(
                    model_name=self.llm_model,
                    openai_api_key=self.api_key,
                    temperature=0
                )
                
            # Create QA chain
            self.retriever = self.vector_store.as_retriever(
                search_kwargs={"k": 5}
            )
            
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.retriever,
                return_source_documents=True
            )
            
            logger.info("VideoQueryEngine successfully initialized")
            
        except Exception as e:
            logger.error(f"Error initializing VideoQueryEngine: {e}", exc_info=True)
            raise
            
    def index_video_metadata(self, metadata: Dict[str, Any]) -> bool:
        """
        Index video metadata into the vector database.
        
        Args:
            metadata: Video metadata dictionary
            
        Returns:
            bool: True if indexing successful
        """
        try:
            # Create a rich text description of the video
            timestamp = metadata.get("timestamp", time.time())
            dt = datetime.datetime.fromtimestamp(timestamp)
            formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S")
            
            # Handle detections
            detections_text = ""
            if "detections" in metadata and metadata["detections"]:
                detected_objects = {}
                for detection in metadata["detections"]:
                    obj_class = detection.get("class_name", "unknown")
                    if obj_class in detected_objects:
                        detected_objects[obj_class] += 1
                    else:
                        detected_objects[obj_class] = 1
                        
                # Format detection text
                detections_list = [f"{count} {obj_class}" for obj_class, count in detected_objects.items()]
                detections_text = "Objects detected: " + ", ".join(detections_list) + ". "
                
            # Create document text
            doc_text = (
                f"Video ID: {metadata.get('video_id', 'unknown')}. "
                f"Recorded at: {formatted_time}. "
                f"Location: {metadata.get('location', 'unknown')}. "
                f"Duration: {metadata.get('duration', 0)} seconds. "
                f"{detections_text}"
                f"Description: {metadata.get('description', 'No description provided')}. "
            )
            
            # Add any custom metadata fields
            for key, value in metadata.items():
                if key not in ["video_id", "timestamp", "location", "duration", "detections", "description"]:
                    if isinstance(value, (str, int, float, bool)):
                        doc_text += f"{key}: {value}. "
            
            # Create and add document
            document = Document(
                page_content=doc_text,
                metadata={
                    "video_id": metadata.get("video_id", "unknown"),
                    "timestamp": timestamp,
                    "source": "video_metadata"
                }
            )
            
            self.vector_store.add_documents([document])
            
            # Save the updated vector store
            self.vector_store.save_local(self.db_path)
            
            logger.info(f"Indexed metadata for video {metadata.get('video_id', 'unknown')}")
            return True
            
        except Exception as e:
            logger.error(f"Error indexing video metadata: {e}", exc_info=True)
            return False
            
    def index_detection_event(self, event: Dict[str, Any]) -> bool:
        """
        Index significant detection event into the vector database.
        
        Args:
            event: Detection event dictionary
            
        Returns:
            bool: True if indexing successful
        """
        try:
            # Create a rich text description of the detection event
            timestamp = event.get("timestamp", time.time())
            dt = datetime.datetime.fromtimestamp(timestamp)
            formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S")
            
            # Create document text
            doc_text = (
                f"Detection event in video {event.get('video_id', 'unknown')} at {formatted_time}. "
                f"Detected {event.get('class_name', 'object')} with confidence {event.get('confidence', 0):.2f}. "
                f"Location: {event.get('location', 'unknown')}. "
                f"Event type: {event.get('event_type', 'detection')}. "
                f"Description: {event.get('description', 'No description provided')}. "
            )
            
            # Add any custom event fields
            for key, value in event.items():
                if key not in ["video_id", "timestamp", "class_name", "confidence", "location", "event_type", "description"]:
                    if isinstance(value, (str, int, float, bool)):
                        doc_text += f"{key}: {value}. "
            
            # Create and add document
            document = Document(
                page_content=doc_text,
                metadata={
                    "video_id": event.get("video_id", "unknown"),
                    "timestamp": timestamp,
                    "class_name": event.get("class_name", "unknown"),
                    "confidence": event.get("confidence", 0),
                    "source": "detection_event"
                }
            )
            
            self.vector_store.add_documents([document])
            
            # Save the updated vector store
            self.vector_store.save_local(self.db_path)
            
            logger.info(f"Indexed detection event for {event.get('class_name', 'object')} in video {event.get('video_id', 'unknown')}")
            return True
            
        except Exception as e:
            logger.error(f"Error indexing detection event: {e}", exc_info=True)
            return False
            
    def query(self, question: str) -> Dict[str, Any]:
        """
        Query the video database with natural language.
        
        Args:
            question: Natural language query
            
        Returns:
            Dict: Query results
        """
        try:
            # Enhance the query with video context
            enhanced_question = f"This is a query about video footage. {question}"
            
            # Run the query
            result = self.qa_chain({"query": enhanced_question})
            
            # Extract sources
            sources = []
            if "source_documents" in result:
                for doc in result["source_documents"]:
                    if hasattr(doc, "metadata"):
                        sources.append({
                            "video_id": doc.metadata.get("video_id", "unknown"),
                            "timestamp": doc.metadata.get("timestamp", 0),
                            "type": doc.metadata.get("source", "unknown")
                        })
            
            # Format response
            response = {
                "answer": result.get("result", "No answer found"),
                "sources": sources
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            return {"error": str(e), "answer": "Error processing your query."}