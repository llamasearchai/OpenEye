import os
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_openai import ChatOpenAI

from ..utils.logging import get_logger


class VideoQueryEngine:
    """Natural language query engine for video archives using LangChain."""
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model_name: str = "gpt-4",
                 db_directory: str = "~/.openvideo/vectordb",
                 config: Optional[Dict] = None):
        """
        Initialize the video query engine.
        
        Args:
            api_key: OpenAI API key (default: from OPENAI_API_KEY env var)
            model_name: LLM model name to use
            db_directory: Directory to store the vector database
            config: Additional configuration
        """
        self.logger = get_logger("VideoQueryEngine")
        self.config = config or {}
        
        # Expand db directory path
        self.db_directory = os.path.expanduser(db_directory)
        os.makedirs(self.db_directory, exist_ok=True)
        
        # Set API key from argument or environment
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            self.logger.warning("No OpenAI API key provided. Set OPENAI_API_KEY environment variable.")
        
        # Initialize components
        self.embeddings = None
        self.vector_store = None
        self.retriever = None
        self.qa_chain = None
        
        # Initialize if API key is available
        if self.api_key:
            self._initialize()
    
    def _initialize(self) -> None:
        """Initialize LangChain components."""
        try:
            # Set up embeddings
            self.embeddings = OpenAIEmbeddings(api_key=self.api_key)
            
            # Create or load vector store
            if os.path.exists(os.path.join(self.db_directory, "chroma.sqlite3")):
                self.logger.info(f"Loading existing vector store from {self.db_directory}")
                self.vector_store = Chroma(
                    persist_directory=self.db_directory,
                    embedding_function=self.embeddings
                )
            else:
                self.logger.info(f"Initializing new vector store at {self.db_directory}")
                self.vector_store = Chroma(
                    persist_directory=self.db_directory,
                    embedding_function=self.embeddings
                )
            
            # Create retriever
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
            
            # Create LLM and QA chain
            llm = ChatOpenAI(
                api_key=self.api_key,
                model_name=self.model_name,
                temperature=0
            )
            
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=self.retriever,
                return_source_documents=True
            )
            
            self.logger.info("VideoQueryEngine initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing VideoQueryEngine: {e}", exc_info=True)
    
    def index_video_metadata(self, metadata: Dict) -> bool:
        """
        Index video metadata for retrieval.
        
        Args:
            metadata: Dictionary containing video metadata
            
        Returns:
            bool: Success status
        """
        if self.vector_store is None:
            self.logger.error("Vector store not initialized")
            return False
        
        try:
            # Convert metadata to text
            content = self._format_metadata_as_text(metadata)
            
            # Create document
            video_id = metadata.get("video_id", f"video_{int(time.time())}")
            doc = Document(
                page_content=content,
                metadata={
                    "source": f"video/{video_id}",
                    "video_id": video_id,
                    "timestamp": metadata.get("timestamp"),
                    "duration": metadata.get("duration"),
                }
            )
            
            # Split document if needed
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100
            )
            docs = text_splitter.split_documents([doc])
            
            # Add to vector store
            self.vector_store.add_documents(docs)
            self.vector_store.persist()  # Save to disk
            
            self.logger.info(f"Indexed metadata for video {video_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error indexing video metadata: {e}", exc_info=True)
            return False
    
    def index_video_transcript(self, video_id: str, transcript: str) -> bool:
        """
        Index video transcript for retrieval.
        
        Args:
            video_id: Unique identifier for the video
            transcript: Video transcript text
            
        Returns:
            bool: Success status
        """
        if self.vector_store is None:
            self.logger.error("Vector store not initialized")
            return False
        
        try:
            # Create document
            doc = Document(
                page_content=transcript,
                metadata={
                    "source": f"transcript/{video_id}",
                    "video_id": video_id,
                }
            )
            
            # Split document
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100
            )
            docs = text_splitter.split_documents([doc])
            
            # Add to vector store
            self.vector_store.add_documents(docs)
            self.vector_store.persist()  # Save to disk
            
            self.logger.info(f"Indexed transcript for video {video_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error indexing video transcript: {e}", exc_info=True)
            return False
    
    def index_video_detections(self, video_id: str, 
                             detections: List[Dict], 
                             timestamp: Optional[float] = None) -> bool:
        """
        Index object detections from a video for retrieval.
        
        Args:
            video_id: Unique identifier for the video
            detections: List of detection dictionaries
            timestamp: Optional timestamp for the detections
            
        Returns:
            bool: Success status
        """
        if self.vector_store is None or not detections:
            return False
        
        try:
            # Convert detections to text
            content = f"Objects detected in video {video_id}:"
            
            # Group by class name
            class_counts = {}
            for detection in detections:
                class_name = detection.get("class_name", "unknown")
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            for class_name, count in class_counts.items():
                content += f"\n- {count} {class_name}"
            
            # Add timestamp if available
            if timestamp:
                time_str = time.strftime('%H:%M:%S', time.gmtime(timestamp))
                content += f"\nTimestamp: {time_str}"
            
            # Create document
            doc = Document(
                page_content=content,
                metadata={
                    "source": f"detections/{video_id}",
                    "video_id": video_id,
                    "timestamp": timestamp,
                }
            )
            
            # Add to vector store
            self.vector_store.add_documents([doc])
            self.vector_store.persist()  # Save to disk
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error indexing video detections: {e}", exc_info=True)
            return False
    
    def query(self, query_text: str) -> Dict:
        """
        Query the video knowledge base with natural language.
        
        Args:
            query_text: Natural language query
            
        Returns:
            Dict with answer and source documents
        """
        if self.qa_chain is None:
            return {
                "answer": "Query engine not initialized. Please check API key.",
                "sources": []
            }
        
        try:
            # Run query
            result = self.qa_chain({"query": query_text})
            
            # Extract answer and sources
            answer = result.get("result", "No answer found")
            
            # Format sources
            sources = []
            source_docs = result.get("source_documents", [])
            for doc in source_docs:
                source = {
                    "content": doc.page_content[:100] + "...",  # Truncated content
                    "video_id": doc.metadata.get("video_id", "unknown"),
                    "timestamp": doc.metadata.get("timestamp"),
                    "source_type": doc.metadata.get("source", "").split("/")[0]
                }
                sources.append(source)
            
            return {
                "answer": answer,
                "sources": sources
            }
            
        except Exception as e:
            self.logger.error(f"Error querying: {e}", exc_info=True)
            return {
                "answer": f"Error processing query: {str(e)}",
                "sources": []
            }
    
    def _format_metadata_as_text(self, metadata: Dict) -> str:
        """
        Format video metadata as text for indexing.
        
        Args:
            metadata: Video metadata dictionary
            
        Returns:
            Formatted text
        """
        lines = [f"Video metadata for {metadata.get('video_id', 'unknown video')}:"]
        
        # Add basic information
        if "description" in metadata:
            lines.append(f"Description: {metadata['description']}")
        
        if "timestamp" in metadata:
            date_str = time.strftime('%Y-%m-%d %H:%M:%S', 
                                     time.gmtime(metadata['timestamp']))
            lines.append(f"Date: {date_str}")
        
        if "duration" in metadata:
            minutes, seconds = divmod(metadata.get("duration", 0), 60)
            lines.append(f"Duration: {int(minutes)}m {int(seconds)}s")
        
        if "location" in metadata:
            lines.append(f"Location: {metadata['location']}")
        
        # Add any tags
        if "tags" in metadata and metadata["tags"]:
            tags = ", ".join(metadata["tags"])
            lines.append(f"Tags: {tags}")
        
        # Add any other metadata
        for key, value in metadata.items():
            if key not in ["video_id", "description", "timestamp", 
                          "duration", "location", "tags"]:
                lines.append(f"{key}: {value}")
        
        return "\n".join(lines)