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

from src.config import (
    CHROMA_PATH,
    DATA_PATH,
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

def load_vector_db(chroma_path: str = CHROMA_PATH, embeddings = None):
    """
    Load the Chroma vector database.
    
    Args:
        chroma_path: Path to the Chroma database
        embeddings: Embeddings model to use
        
    Returns:
        Chroma database object
    """
    if embeddings is None:
        embeddings = get_embeddings()
        
    os.makedirs(chroma_path, exist_ok=True)
    
    print(f"Loading Chroma vector database from {chroma_path}")
    return Chroma(
        embedding_function=embeddings,
        persist_directory=chroma_path
    )

def load_document_tracker(chroma_path: str = CHROMA_PATH) -> Dict[str, Dict[str, Any]]:
    """
    Load the document tracker JSON file.
    
    Args:
        chroma_path: Path to the Chroma database
        
    Returns:
        Document tracker dictionary
    """
    tracker_file = os.path.join(chroma_path, "document_tracker.json")
    document_tracker = {}
    
    if os.path.exists(tracker_file):
        try:
            with open(tracker_file, 'r') as f:
                document_tracker = json.load(f)
            print(f"Loaded document tracker with {len(document_tracker)} entries")
        except Exception as e:
            print(f"Error loading document tracker: {e}")
    
    return document_tracker

def save_document_tracker(document_tracker: Dict[str, Any], chroma_path: str = CHROMA_PATH):
    """
    Save the document tracker to a JSON file.
    
    Args:
        document_tracker: Dictionary of document tracking information
        chroma_path: Path to the Chroma database
    """
    tracker_file = os.path.join(chroma_path, "document_tracker.json")
    with open(tracker_file, 'w') as f:
        json.dump(document_tracker, f, indent=2)
    print(f"Saved document tracker with {len(document_tracker)} entries")

def create_vector_db(
    all_document_chunks: List[Document],
    chroma_path: str = CHROMA_PATH,
    data_path: str = DATA_PATH,
    ollama_url: str = OLLAMA_URL,
    ollama_model: str = OLLAMA_MODEL,
    batch_size: int = BATCH_SIZE,
    max_workers: int = MAX_WORKERS
) -> Tuple[Chroma, Dict]:
    """
    Create or update the vector database with document chunks.
    
    Args:
        all_document_chunks: List of document chunks to add to the database
        chroma_path: Path to the Chroma database
        data_path: Path to the data directory
        ollama_url: URL of the Ollama server
        ollama_model: Name of the Ollama model to use
        batch_size: Number of chunks to process in each batch
        max_workers: Number of worker threads to use
        
    Returns:
        Tuple of (Chroma database, document tracker)
    """
    # Create directory if it doesn't exist
    os.makedirs(chroma_path, exist_ok=True)
    
    # Load document tracker
    document_tracker = load_document_tracker(chroma_path)
    
    # Initialize embedding function
    embeddings = get_embeddings(ollama_url, ollama_model)
    
    # Load existing Chroma database
    chroma_db = load_vector_db(chroma_path, embeddings)
    
    # Organize document chunks by source file
    print("Organizing document chunks by source file...")
    source_to_chunks = {}
    for chunk in all_document_chunks:
        source = chunk.metadata.get("source", "")
        if source not in source_to_chunks:
            source_to_chunks[source] = []
        source_to_chunks[source].append(chunk)
    
    # Collect all available document files in the data directory
    data_files = set()
    if data_path and os.path.exists(data_path):
        for file in os.listdir(data_path):
            file_path = os.path.join(data_path, file)
            if os.path.isfile(file_path):
                data_files.add(file_path)
    
    # Check which files have been modified
    print("Checking for new or modified files...")
    modified_sources = []
    chunks_to_process = []
    files_skipped = 0
    
    for source, chunks in source_to_chunks.items():
        # Skip if source is empty
        if not source:
            print(f"Processing chunks with no source information")
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
                print(f"Skipping unchanged file: {source}")
                files_skipped += 1
                continue
            
            # File is new or modified
            print(f"File new or modified, will process: {source}")
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
    
    print(f"Skipped {files_skipped} unchanged files")
    print(f"Found {len(modified_sources)} new or modified files with {len(chunks_to_process)} chunks")
    
    # Process missing files that weren't in source_to_chunks
    if data_files:
        # These files were not included in the original document processing
        # This would need additional code to process them
        print(f"Found {len(data_files)} files in data directory not in source_to_chunks")
        # The required implementation would load these files and process them
    
    # If nothing to process, save tracker and return
    if not chunks_to_process:
        print("No new or modified files to process")
        return chroma_db, document_tracker
    
    # Process chunks in batches
    print(f"Processing {len(chunks_to_process)} chunks in batches of {batch_size}...")
    
    total_chunks = len(chunks_to_process)
    batches = [chunks_to_process[i:i + batch_size] for i in range(0, total_chunks, batch_size)]
    print(f"Created {len(batches)} batches")
    
    def process_batch(batch):
        try:
            texts = [doc.page_content for doc in batch]
            metadatas = [doc.metadata for doc in batch]
            chroma_db.add_texts(texts=texts, metadatas=metadatas)
            return len(batch)
        except Exception as e:
            print(f"Error processing batch: {e}")
            return 0
    
    # Process batches in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_batch, batches))
    
    processed_count = sum(results)
    print(f"Successfully processed {processed_count} chunks out of {total_chunks}")
    
    # Save document tracker
    save_document_tracker(document_tracker, chroma_path)
    
    return chroma_db, document_tracker

def validate_vector_db(chroma_db: Chroma, data_path: str = DATA_PATH) -> List[str]:
    """
    Validate that all files in the data directory are in the vector database.
    
    Args:
        chroma_db: Chroma database object
        data_path: Path to the data directory
        
    Returns:
        List of files that are not in the database
    """
    missing_files = []
    
    if not os.path.exists(data_path):
        print(f"Data path {data_path} does not exist")
        return missing_files
    
    # Get all files in the data directory
    data_files = [
        os.path.join(data_path, file)
        for file in os.listdir(data_path)
        if os.path.isfile(os.path.join(data_path, file))
    ]
    
    # Get all sources in the database
    db_sources = set()
    for metadata in chroma_db.get()["metadatas"]:
        if metadata and "source" in metadata:
            db_sources.add(metadata["source"])
    
    # Check each file
    for file_path in data_files:
        if file_path not in db_sources:
            print(f"File not in database: {file_path}")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"Found {len(missing_files)} files not in the database")
    else:
        print("All files in the data directory are in the database")
    
    return missing_files 