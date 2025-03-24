#!/usr/bin/env python
"""
Main entry point for the Ollama RAG Chat App.

This script can be used to:
1. Update the vector database with new or modified documents
2. Launch the Gradio chat interface

Examples:
    # Update the database and launch the app
    python -m src.main --update-db
    
    # Just launch the app
    python -m src.main
    
    # Update the database only
    python -m src.main --update-db --no-app
"""

import argparse
import sys

from src.update_db import update_database
from src.app import main as app_main

def main():
    """
    Main entry point for the application.
    """
    parser = argparse.ArgumentParser(description="Ollama RAG Chat App")
    parser.add_argument("--update-db", action="store_true", help="Update the database")
    parser.add_argument("--no-app", action="store_true", help="Don't launch the app")
    
    args = parser.parse_args()
    
    if args.update_db:
        print("Updating database...")
        update_database()
    
    if not args.no_app:
        app_main()
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 