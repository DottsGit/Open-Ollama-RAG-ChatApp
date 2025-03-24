"""
Gradio app script.

This script runs the Gradio chat interface with Ollama.
"""

import argparse
import sys

from src.config import (
    DATA_PATH,
    CHROMA_PATH,
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
    parser = argparse.ArgumentParser(description="Run the Ollama RAG Chat App")
    parser.add_argument("--data-path", type=str, default=DATA_PATH, help="Path to the data directory")
    parser.add_argument("--chroma-path", type=str, default=CHROMA_PATH, help="Path to the Chroma database")
    parser.add_argument("--ollama-url", type=str, default=OLLAMA_URL, help="URL of the Ollama server")
    parser.add_argument("--ollama-model", type=str, default=OLLAMA_MODEL, help="Name of the Ollama model to use")
    parser.add_argument("--update-db", action="store_true", help="Update the database before starting the app")
    parser.add_argument("--check-db", action="store_true", help="Check if the database is up-to-date before starting")
    parser.add_argument("--port", type=int, default=7860, help="Port for the Gradio app")
    parser.add_argument("--share", action="store_true", help="Create a public link for the app")
    
    args = parser.parse_args()
    
    # Check if we need to update the database
    if args.update_db:
        print("Updating database before starting app...")
        update_database(
            data_path=args.data_path,
            chroma_path=args.chroma_path,
            ollama_url=args.ollama_url,
            ollama_model=args.ollama_model
        )
    
    # Or if we just need to check it
    elif args.check_db:
        print("Checking if database is up-to-date...")
        missing_files = update_database(
            data_path=args.data_path,
            chroma_path=args.chroma_path,
            ollama_url=args.ollama_url,
            ollama_model=args.ollama_model,
            validate_only=True
        )
        
        if missing_files:
            print("Database is not up-to-date. Run with --update-db to update it.")
            sys.exit(1)
    
    # Initialize the embeddings model
    print(f"Initializing embeddings model {args.ollama_model}...")
    embeddings = get_embeddings(args.ollama_url, args.ollama_model)
    
    # Load the vector database
    print(f"Loading vector database from {args.chroma_path}...")
    chroma_db = load_vector_db(args.chroma_path, embeddings)
    
    # Create the chat interface
    print("Creating chat interface...")
    chat_interface = create_chat_interface(chroma_db)
    
    # Launch the Gradio app
    print(f"Launching Gradio app on port {args.port}...")
    chat_interface.launch(
        server_name="localhost",
        server_port=args.port,
        share=args.share
    )

if __name__ == "__main__":
    main() 