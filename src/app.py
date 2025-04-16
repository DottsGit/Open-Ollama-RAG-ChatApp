"""
Gradio app script.

This script runs the Gradio chat interface with Ollama.
"""

import argparse
import sys

from src.config import (
    TOPICS,
    OLLAMA_MODEL,
    OLLAMA_URL
)
from src.chat import create_chat_interface
from src.vector_db import get_embeddings, load_vector_db
from src.update_db import update_database

def main():
    """
    Main entry point for the Gradio app script.
    """
    parser = argparse.ArgumentParser(description="Run the Ollama RAG Chat App with Topic Selection")
    parser.add_argument("--topic", type=str, default=None, choices=TOPICS + [None], help="Update/Check only a specific topic (default: all)")
    parser.add_argument("--ollama-url", type=str, default=OLLAMA_URL, help="URL of the Ollama server")
    parser.add_argument("--ollama-model", type=str, default=OLLAMA_MODEL, help="Name of the Ollama model to use")
    parser.add_argument("--update-db", action="store_true", help="Update the database(s) before starting the app")
    parser.add_argument("--check-db", action="store_true", help="Check if the database(s) are up-to-date before starting")
    parser.add_argument("--port", type=int, default=7860, help="Port for the Gradio app")
    parser.add_argument("--share", action="store_true", help="Create a public link for the app")
    
    args = parser.parse_args()
    
    # Check if we need to update the database(s)
    if args.update_db:
        print("Updating database(s) before starting app...")
        update_database(
            ollama_url=args.ollama_url,
            ollama_model=args.ollama_model,
            specific_topic=args.topic
        )
    
    # Or if we just need to check them
    elif args.check_db:
        print("Checking if database(s) are up-to-date...")
        missing_files_map = update_database(
            ollama_url=args.ollama_url,
            ollama_model=args.ollama_model,
            validate_only=True,
            specific_topic=args.topic
        )
        
        has_missing_or_errors = any(bool(files) for files in missing_files_map.values())
        if has_missing_or_errors:
            print("Database check failed for one or more topics. Run with --update-db to update.")
            sys.exit(1)
        else:
            print("Database check passed for all specified topics.")
    
    # Initialize the embeddings model
    print(f"Initializing embeddings model {args.ollama_model}...")
    embeddings = get_embeddings(args.ollama_url, args.ollama_model)
    
    # Create the chat interface
    print("Creating chat interface...")
    chat_interface = create_chat_interface(embeddings=embeddings, topics=TOPICS)
    
    # Launch the Gradio app
    print(f"Launching Gradio app on port {args.port}...")
    chat_interface.launch(
        server_name="localhost",
        server_port=args.port,
        share=args.share
    )

if __name__ == "__main__":
    main() 