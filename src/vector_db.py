"""
Vector database module.

This module handles interactions with the Chroma vector database.
"""

import os
import time
import json
import concurrent.futures
from typing import List, Dict, Any, Optional, Tuple

from langchain.schema.document import Document
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

from src import config # Import the whole config module
from src.config import (
    DB_BASE_PATH, # Use base path for DBs
    # CHROMA_PATH, # Removed
    # DATA_PATH, # Removed (will use config.DATA_PATH + topic)
    OLLAMA_MODEL,
    OLLAMA_URL,
    BATCH_SIZE,
    MAX_WORKERS
)

def get_embeddings(ollama_url: str = OLLAMA_URL, ollama_model: str = OLLAMA_MODEL):
    """
    Initialize the Ollama embeddings model.
    
    Args:
        ollama_url: URL of the Ollama server
        ollama_model: Name of the Ollama model to use
        
    Returns:
        OllamaEmbeddings object
    """
    print(f"Initializing OllamaEmbeddings with model {ollama_model}")
    return OllamaEmbeddings(
        base_url=ollama_url,
        model=ollama_model
    )

def get_topic_paths(topic: str) -> Tuple[str, str]:
    """Helper function to get data and chroma paths for a given topic."""
    if not topic:
        raise ValueError("Topic cannot be empty")
    data_path = os.path.join(config.DATA_PATH, topic)
    chroma_path = os.path.join(DB_BASE_PATH, topic)
    return data_path, chroma_path

def load_vector_db(topic: str, embeddings = None):
    """
    Load the Chroma vector database for a specific topic.

    Args:
        topic: The topic identifier (e.g., 'jfk_files', 'literature').
        embeddings: Embeddings model to use

    Returns:
        Chroma database object
    """
    _data_path, chroma_path = get_topic_paths(topic)

    if embeddings is None:
        embeddings = get_embeddings()

    os.makedirs(chroma_path, exist_ok=True)

    print(f"Loading Chroma vector database for topic '{topic}' from {chroma_path}")
    return Chroma(
        embedding_function=embeddings,
        persist_directory=chroma_path
    )

def load_document_tracker(topic: str) -> Dict[str, Dict[str, Any]]:
    """
    Load the document tracker JSON file for a specific topic.

    Args:
        topic: The topic identifier.

    Returns:
        Document tracker dictionary
    """
    _data_path, chroma_path = get_topic_paths(topic)
    tracker_file = os.path.join(chroma_path, "document_tracker.json")
    document_tracker = {}
    
    if os.path.exists(tracker_file):
        try:
            with open(tracker_file, 'r') as f:
                document_tracker = json.load(f)
            print(f"Loaded document tracker for topic '{topic}' with {len(document_tracker)} entries")
        except Exception as e:
            print(f"Error loading document tracker for topic '{topic}': {e}")
    
    return document_tracker

def save_document_tracker(document_tracker: Dict[str, Any], topic: str):
    """
    Save the document tracker to a JSON file for a specific topic.

    Args:
        document_tracker: Dictionary of document tracking information
        topic: The topic identifier.
    """
    _data_path, chroma_path = get_topic_paths(topic)
    tracker_file = os.path.join(chroma_path, "document_tracker.json")
    os.makedirs(chroma_path, exist_ok=True)
    with open(tracker_file, 'w') as f:
        json.dump(document_tracker, f, indent=2)
    print(f"Saved document tracker for topic '{topic}' with {len(document_tracker)} entries to {tracker_file}")

def create_vector_db(
    topic: str,
    all_document_chunks: List[Document],
    ollama_url: str = OLLAMA_URL,
    ollama_model: str = OLLAMA_MODEL,
    batch_size: int = BATCH_SIZE,
    max_workers: int = MAX_WORKERS
) -> Tuple[Chroma, Dict]:
    """
    Create or update the vector database for a specific topic.

    Args:
        topic: The topic identifier.
        all_document_chunks: List of document chunks to add to the database
        ollama_url: URL of the Ollama server
        ollama_model: Name of the Ollama model to use
        batch_size: Number of chunks to process in each batch
        max_workers: Number of worker threads to use

    Returns:
        Tuple of (Chroma database, document tracker)
    """
    data_path, chroma_path = get_topic_paths(topic)

    # Create directory if it doesn't exist
    os.makedirs(chroma_path, exist_ok=True)
    os.makedirs(data_path, exist_ok=True)

    # Load document tracker for the specific topic
    document_tracker = load_document_tracker(topic)

    # Initialize embedding function
    embeddings = get_embeddings(ollama_url, ollama_model)

    # Load existing Chroma database for the specific topic
    chroma_db = load_vector_db(topic, embeddings)

    # Organize document chunks by source file
    print(f"Organizing document chunks for topic '{topic}' by source file...")
    source_to_chunks = {}
    for chunk in all_document_chunks:
        source = chunk.metadata.get("source", "")
        if source not in source_to_chunks:
            source_to_chunks[source] = []
        source_to_chunks[source].append(chunk)
    
    # Collect all available document files in the specific topic's data directory
    data_files = set()
    if os.path.exists(data_path):
        for file in os.listdir(data_path):
            file_path = os.path.join(data_path, file)
            if os.path.isfile(file_path):
                data_files.add(file_path)
    else:
        print(f"Warning: Data path '{data_path}' for topic '{topic}' does not exist.")
    
    # Check which files have been modified
    print(f"Checking for new or modified files in {data_path}...")
    modified_sources = []
    chunks_to_process = []
    files_skipped = 0
    
    for source, chunks in source_to_chunks.items():
        # Skip if source is empty
        if not source:
            print(f"Processing chunks with no source information for topic '{topic}'")
            chunks_to_process.extend(chunks)
            continue
        
        # Check if file exists and get metadata
        if os.path.exists(source):
            file_mtime = os.path.getmtime(source)
            file_size = os.path.getsize(source)
            
            # Create file signature that combines path, size and mtime
            file_signature = f"{source}:{file_size}:{file_mtime}"
            
            # Check if file has been modified since last processing
            if source in document_tracker and document_tracker[source]["signature"] == file_signature:
                print(f"Skipping unchanged file for topic '{topic}': {source}")
                files_skipped += 1
                continue
            
            # File is new or modified
            print(f"File new or modified for topic '{topic}', will process: {source}")
            modified_sources.append(source)
            chunks_to_process.extend(chunks)
            
            # Update tracker for this file
            document_tracker[source] = {
                "signature": file_signature,
                "last_processed": time.time()
            }
            
            # Remove this file from the data_files set
            if source in data_files:
                data_files.remove(source)
    
    print(f"Skipped {files_skipped} unchanged files for topic '{topic}'")
    print(f"Found {len(modified_sources)} new or modified files for topic '{topic}' with {len(chunks_to_process)} chunks")
    
    # Process missing files that weren't in source_to_chunks
    if data_files:
        # These files were not included in the original document processing
        # This would need additional code to process them
        print(f"Found {len(data_files)} files in data directory for topic '{topic}' not in source_to_chunks")
        # The required implementation would load these files and process them
    
    # If nothing to process, save tracker and return
    if not chunks_to_process:
        print(f"No new or modified files to process for topic '{topic}'")
        save_document_tracker(document_tracker, topic)
        return chroma_db, document_tracker
    
    # Process chunks in batches
    print(f"Processing {len(chunks_to_process)} chunks for topic '{topic}' in batches of {batch_size}...")
    
    total_chunks = len(chunks_to_process)
    batches = [chunks_to_process[i:i + batch_size] for i in range(0, total_chunks, batch_size)]
    print(f"Created {len(batches)} batches for topic '{topic}'")
    
    def process_batch(batch):
        try:
            texts = [doc.page_content for doc in batch]
            metadatas = [doc.metadata for doc in batch]
            chroma_db.add_texts(texts=texts, metadatas=metadatas)
            return len(batch)
        except Exception as e:
            print(f"Error processing batch for topic '{topic}': {e}")
            return 0
    
    # Process batches in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_batch, batches))
    
    processed_count = sum(results)
    print(f"Successfully processed {processed_count} chunks out of {total_chunks} for topic '{topic}'")
    
    # Save document tracker for the specific topic
    save_document_tracker(document_tracker, topic)
    
    return chroma_db, document_tracker

def validate_vector_db(topic: str) -> List[str]:
    """
    Validate that all files in the specific topic's data directory are in the vector database.

    Args:
        topic: The topic identifier.

    Returns:
        List of files that are not in the database
    """
    data_path, _chroma_path = get_topic_paths(topic)

    # Load the specific DB for the topic
    try:
        chroma_db = load_vector_db(topic)
    except Exception as e:
        print(f"Error loading vector DB for topic '{topic}': {e}. Cannot validate.")
        return []

    missing_files = []

    if not os.path.exists(data_path):
        print(f"Data path {data_path} for topic '{topic}' does not exist")
        return missing_files

    # Get all files in the specific topic's data directory
    data_files = [
        os.path.join(data_path, file)
        for file in os.listdir(data_path)
        if os.path.isfile(os.path.join(data_path, file))
    ]

    # Get all sources in the database
    db_sources = set()
    try:
        # Handle potential errors if the DB is empty or metadata is missing
        retrieved_metadata = chroma_db.get().get("metadatas", [])
        if retrieved_metadata:
             for metadata in retrieved_metadata:
                if metadata and "source" in metadata:
                    # Ensure source path normalization if necessary, though likely okay
                    # if paths were stored consistently.
                    db_sources.add(metadata["source"])
        else:
             print(f"No metadata found in vector DB for topic '{topic}'.")
    except Exception as e:
        print(f"Error retrieving sources from vector DB for topic '{topic}': {e}")
        # Decide how to handle this - potentially return an error indicator
        return []

    # Check each file
    for file_path in data_files:
        # Normalize paths for comparison if needed, e.g., os.path.abspath
        if file_path not in db_sources:
            print(f"File not in database for topic '{topic}': {file_path}")
            missing_files.append(file_path)

    if missing_files:
        print(f"Found {len(missing_files)} files for topic '{topic}' not in the database")
    else:
        print(f"All files in {data_path} are in the vector database for topic '{topic}'")

    return missing_files 