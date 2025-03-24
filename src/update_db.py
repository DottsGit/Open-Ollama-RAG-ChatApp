"""
Database update script.

This script handles updating the vector database when files are added or modified.
"""

import argparse
import sys
from typing import List, Dict, Any

from src.config import (
    DATA_PATH,
    CHROMA_PATH,
    OLLAMA_MODEL,
    OLLAMA_URL
)
from src.document_processor import load_documents, process_documents
from src.vector_db import create_vector_db, validate_vector_db, load_vector_db, get_embeddings

def update_database(
    data_path: str = DATA_PATH,
    chroma_path: str = CHROMA_PATH,
    ollama_url: str = OLLAMA_URL,
    ollama_model: str = OLLAMA_MODEL,
    validate_only: bool = False
):
    """
    Update the vector database with new or modified files.
    
    Args:
        data_path: Path to the data directory
        chroma_path: Path to the Chroma database
        ollama_url: URL of the Ollama server
        ollama_model: Name of the Ollama model to use
        validate_only: Only validate the database, don't update it
    """
    print(f"Starting database update process...")
    print(f"Using model: {ollama_model}")
    print(f"Data path: {data_path}")
    print(f"Chroma path: {chroma_path}")
    
    # If validate_only is True, just validate the database
    if validate_only:
        print("Validation only mode, checking database...")
        embeddings = get_embeddings(ollama_url, ollama_model)
        chroma_db = load_vector_db(chroma_path, embeddings)
        missing_files = validate_vector_db(chroma_db, data_path)
        
        if missing_files:
            print(f"Found {len(missing_files)} files not in the database:")
            for file in missing_files:
                print(f"  - {file}")
        else:
            print("All files in the data directory are in the database")
        
        return missing_files
    
    # Load documents
    print("Loading documents...")
    documents = load_documents(data_path)
    
    if not documents:
        print("No documents found, exiting")
        return []
    
    # Process documents
    print("Processing documents...")
    document_chunks = process_documents(documents)
    
    # Create or update the vector database
    print("Creating or updating vector database...")
    chroma_db, document_tracker = create_vector_db(
        all_document_chunks=document_chunks,
        chroma_path=chroma_path,
        data_path=data_path,
        ollama_url=ollama_url,
        ollama_model=ollama_model
    )
    
    # Validate the database
    print("Validating database...")
    missing_files = validate_vector_db(chroma_db, data_path)
    
    return missing_files

def main():
    """
    Main entry point for the database update script.
    """
    parser = argparse.ArgumentParser(description="Update the RAG vector database")
    parser.add_argument("--data-path", type=str, default=DATA_PATH, help="Path to the data directory")
    parser.add_argument("--chroma-path", type=str, default=CHROMA_PATH, help="Path to the Chroma database")
    parser.add_argument("--ollama-url", type=str, default=OLLAMA_URL, help="URL of the Ollama server")
    parser.add_argument("--ollama-model", type=str, default=OLLAMA_MODEL, help="Name of the Ollama model to use")
    parser.add_argument("--validate-only", action="store_true", help="Only validate the database, don't update it")
    
    args = parser.parse_args()
    
    missing_files = update_database(
        data_path=args.data_path,
        chroma_path=args.chroma_path,
        ollama_url=args.ollama_url,
        ollama_model=args.ollama_model,
        validate_only=args.validate_only
    )
    
    # Return non-zero exit code if there are missing files
    if missing_files and args.validate_only:
        sys.exit(1)
    
    sys.exit(0)

if __name__ == "__main__":
    main() 