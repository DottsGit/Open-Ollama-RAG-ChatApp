"""
Chat interface module.

This module handles interactions with Ollama LLM and Gradio chat interface.
"""

import os
import re
from typing import List, Dict, Any, Tuple, Optional

import gradio as gr
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma

from src.config import (
    OLLAMA_MODEL,
    OLLAMA_URL,
)
from src.vector_db import get_embeddings, load_vector_db

def get_ollama_client(ollama_url: str = OLLAMA_URL, ollama_model: str = OLLAMA_MODEL):
    """
    Initialize the Ollama client.
    
    Args:
        ollama_url: URL of the Ollama server
        ollama_model: Name of the Ollama model to use
        
    Returns:
        Ollama LLM object
    """
    print(f"Initializing Ollama client with model {ollama_model}")
    return Ollama(base_url=ollama_url, model=ollama_model)

def search_similar_chunks(query: str, chroma_db: Chroma, top_k: int = 10) -> List[str]:
    """
    Search for similar chunks in the vector database.
    
    Args:
        query: Query string
        chroma_db: Chroma database object
        top_k: Number of top results to return
        
    Returns:
        List of document chunks as formatted strings
    """
    # Check if the query contains a document ID pattern (e.g., 104-10326-10014)
    doc_id_pattern = r'\b\d{3}-\d{5}-\d{5}\b'
    doc_ids = re.findall(doc_id_pattern, query)
    
    if doc_ids:
        print(f"Document ID(s) detected in query: {doc_ids}")
        direct_results = []
        
        # Get all documents
        all_docs = chroma_db.get()
        
        # Track if we found any direct matches
        found_direct_match = False
        
        # Search for documents matching the IDs
        for doc_id in doc_ids:
            for i, metadata in enumerate(all_docs["metadatas"]):
                if metadata and "source" in metadata and doc_id in metadata["source"]:
                    source = metadata["source"]
                    content = all_docs["documents"][i]
                    
                    # Add it to the direct results
                    direct_results.append((source, content))
                    found_direct_match = True
                    
                    # Limit to top_k results per document ID
                    if len(direct_results) >= top_k:
                        break
        
        # If we found direct matches, format and return them
        if found_direct_match:
            formatted_results = []
            for i, (source, content) in enumerate(direct_results):
                # Format the document content
                formatted_content = f"[{i+1}] {content}"
                
                # Add source information
                if source:
                    formatted_content += f"\n-- Source: {os.path.basename(source)}"
                
                formatted_results.append(formatted_content)
            
            print(f"Found {len(formatted_results)} direct document matches")
            return formatted_results
    
    # If no document IDs or no direct matches found, fall back to similarity search
    print("Performing similarity search")
    results = chroma_db.similarity_search(query, k=top_k)
    
    # Format the results
    formatted_results = []
    for i, doc in enumerate(results):
        # Get source information
        source = doc.metadata.get("source", "Unknown")
        
        # Format the document content
        content = f"[{i+1}] {doc.page_content}"
        
        # Add source information
        if source != "Unknown":
            content += f"\n-- Source: {os.path.basename(source)}"
        
        formatted_results.append(content)
    
    return formatted_results

def create_prompt(query: str, similar_chunks: List[str]) -> str:
    """
    Create a prompt for the LLM using the query and similar chunks.
    
    Args:
        query: User's query
        similar_chunks: List of similar document chunks
        
    Returns:
        Formatted prompt
    """
    # Combine the chunks into a single context string
    context = "\n\n".join(similar_chunks)
    
    # Create the prompt with context
    prompt = f"""Answer the question based on the following context:
    
Context:
{context}

Question: {query}

Answer:"""
    
    return prompt

def chat_ollama(message: str, history: List[Tuple[str, str]], chroma_db: Optional[Chroma] = None):
    """
    Process a chat message using Ollama and the vector database.
    
    Args:
        message: User's message
        history: Chat history
        chroma_db: Chroma database object
        
    Returns:
        LLM's response
    """
    # Initialize the vector database if not provided
    if chroma_db is None:
        embeddings = get_embeddings()
        chroma_db = load_vector_db(embeddings=embeddings)
    
    # Initialize the Ollama client
    ollama = get_ollama_client()
    
    # Search for similar chunks
    similar_chunks = search_similar_chunks(message, chroma_db)
    
    # Create the prompt
    if similar_chunks:
        prompt = create_prompt(message, similar_chunks)
    else:
        # If no similar chunks found, just use the message directly
        prompt = message
    
    # Get the response from Ollama
    response = ollama(prompt)
    
    return response

def create_chat_interface(chroma_db: Chroma):
    """
    Create the Gradio chat interface.
    
    Args:
        chroma_db: Chroma database object
        
    Returns:
        Gradio ChatInterface object
    """
    # Create a wrapped chat function that includes the database
    def chat_fn(message, history):
        return chat_ollama(message, history, chroma_db)
    
    # Create the chat interface
    chat_interface = gr.ChatInterface(
        chat_fn,
        examples=["What is Langchain?", "Tell me about vector databases", "How does RAG work?"],
        title="Ollama RAG ChatApp",
        description="Chat with your documents using Ollama and RAG",
    )
    
    return chat_interface 