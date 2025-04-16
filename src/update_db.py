"""
Database update script.

This script handles updating the vector database when files are added or modified.
"""

import argparse
import sys
import os
from typing import List, Dict, Any

from src import config
from src.config import (
    TOPICS,
    OLLAMA_MODEL,
    OLLAMA_URL,
    DB_BASE_PATH
)
from src.document_processor import load_documents, process_documents
from src.vector_db import create_vector_db, validate_vector_db, load_vector_db, get_embeddings, get_topic_paths

def update_database(
    ollama_url: str = OLLAMA_URL,
    ollama_model: str = OLLAMA_MODEL,
    validate_only: bool = False,
    specific_topic: str | None = None
) -> Dict[str, List[str]]:
    """
    Update or validate vector databases for specified topics.

    Args:
        ollama_url: URL of the Ollama server
        ollama_model: Name of the Ollama model to use
        validate_only: Only validate the databases, don't update them
        specific_topic: If set, only process this topic.

    Returns:
        Dictionary mapping topic name to a list of missing files for that topic.
    """
    print(f"Starting database update process...")
    print(f"Using model: {ollama_model}")

    topics_to_process = [specific_topic] if specific_topic else TOPICS
    all_missing_files = {}

    for topic in topics_to_process:
        print(f"\n--- Processing topic: {topic} ---")
        try:
            data_path, chroma_path = get_topic_paths(topic)
        except ValueError as e:
            print(f"Error getting paths for topic '{topic}': {e}. Skipping.")
            all_missing_files[topic] = ["Error getting paths"]
            continue

        print(f"Data path: {data_path}")
        print(f"Chroma path: {chroma_path}")

        os.makedirs(data_path, exist_ok=True)
        os.makedirs(chroma_path, exist_ok=True)

        if validate_only:
            print(f"Validation only mode, checking database for topic '{topic}'...")
            missing_files = validate_vector_db(topic)
            all_missing_files[topic] = missing_files
            if missing_files:
                print(f"Found {len(missing_files)} files not in the '{topic}' database:")
                for file in missing_files:
                    print(f"  - {file}")
            else:
                print(f"All files in {data_path} are in the '{topic}' database")
            continue

        try:
            print(f"Loading documents for topic '{topic}' from {data_path}...")
            documents = load_documents(data_path)

            if not documents:
                print(f"No documents found in {data_path} for topic '{topic}', skipping update.")
                print(f"Validating existing database for topic '{topic}'...")
                all_missing_files[topic] = validate_vector_db(topic)
                continue

            print(f"Processing {len(documents)} documents for topic '{topic}'...")
            document_chunks = process_documents(documents)

            print(f"Creating or updating vector database for topic '{topic}'...")
            chroma_db, document_tracker = create_vector_db(
                topic=topic,
                all_document_chunks=document_chunks,
                ollama_url=ollama_url,
                ollama_model=ollama_model
            )

            print(f"Validating database for topic '{topic}'...")
            missing_files = validate_vector_db(topic)
            all_missing_files[topic] = missing_files

        except Exception as e:
            print(f"An error occurred processing topic '{topic}': {e}")
            all_missing_files[topic] = [f"Error processing topic: {e}"]

    print("\n--- Database update process finished ---")
    return all_missing_files

def main():
    """
    Main entry point for the database update script.
    """
    parser = argparse.ArgumentParser(description="Update the RAG vector database for specified topics")
    parser.add_argument("--topic", type=str, default=None, choices=TOPICS + [None], help="Update only a specific topic (default: all)")
    parser.add_argument("--ollama-url", type=str, default=OLLAMA_URL, help="URL of the Ollama server")
    parser.add_argument("--ollama-model", type=str, default=OLLAMA_MODEL, help="Name of the Ollama model to use")
    parser.add_argument("--validate-only", action="store_true", help="Only validate the database(s), don't update")

    args = parser.parse_args()

    missing_files_map = update_database(
        ollama_url=args.ollama_url,
        ollama_model=args.ollama_model,
        validate_only=args.validate_only,
        specific_topic=args.topic
    )

    has_missing_or_errors = any(bool(files) for files in missing_files_map.values())

    if has_missing_or_errors and args.validate_only:
        print("\nValidation issues found.")
        sys.exit(1)
    elif has_missing_or_errors:
        print("\nUpdate process completed, but some issues were found during validation.")

    print("\nUpdate completed successfully.")
    sys.exit(0)

if __name__ == "__main__":
    main() 