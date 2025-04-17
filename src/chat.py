"""
Chat interface module.

This module handles interactions with Ollama LLM and Gradio chat interface.
"""

import os
import re
from typing import List, Dict, Any, Tuple, Optional

import gradio as gr
# from langchain_community.llms import Ollama # Old import
from langchain_ollama import OllamaLLM, OllamaEmbeddings # Added OllamaEmbeddings here too for clarity
from langchain_chroma import Chroma # Corrected Chroma import (already done in vector_db.py, but good practice here too if used directly)
from langchain_community.embeddings import OllamaEmbeddings

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
        OllamaLLM object # Updated return type hint
    """
    print(f"Initializing Ollama client with model {ollama_model}")
    # Use the new class name OllamaLLM
    return OllamaLLM(base_url=ollama_url, model=ollama_model)

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

def create_chat_interface(embeddings: OllamaEmbeddings, topics: List[str]):
    """
    Create the Gradio chat interface with topic selection.

    Args:
        embeddings: Initialized OllamaEmbeddings object.
        topics: List of available topic names.

    Returns:
        Gradio Blocks interface object.
    """
    print(f"Creating chat interface with topics: {topics}")

    with gr.Blocks() as chat_interface:
        gr.Markdown("## Ollama RAG ChatApp with Topic Selection")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Select Topic")
                topic_selector = gr.Radio(
                    choices=topics,
                    value=topics[0] if topics else None, # Default to first topic
                    label="Chat Topic",
                    info="Choose the knowledge base to chat with."
                )
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(label="Chat History", height=500, type='messages')
                msg_input = gr.Textbox(label="Your Message", placeholder="Type your question here...")
                clear_button = gr.Button("Clear Chat")

        # Define the core chat logic function
        def respond(message: str, chat_history: List[Dict[str, str]], selected_topic: str):
            print(f"\nReceived message: '{message}' for topic: '{selected_topic}'")
            if not selected_topic:
                 gr.Warning("Please select a topic first!")
                 # Return None for message, keep history as is
                 return None, chat_history

            # Append user message in the new format before processing
            chat_history.append({"role": "user", "content": message})

            try:
                # 1. Load the correct vector database for the selected topic
                print(f"Loading vector DB for topic: {selected_topic}")
                chroma_db = load_vector_db(topic=selected_topic, embeddings=embeddings)
                print(f"Vector DB loaded.") # Added confirmation

                # 2. Initialize the Ollama client
                print("Attempting to initialize Ollama client...") # Debug
                ollama = get_ollama_client()
                print("Ollama client initialized.") # Debug

                # 3. Search for similar chunks
                print(f"Attempting to search for similar chunks for query: '{message}'") # Debug
                similar_chunks = search_similar_chunks(message, chroma_db)
                print(f"Similarity search completed. Found {len(similar_chunks)} chunks.") # Debug

                # 4. Create the prompt & handle context display
                context_prefix = ""
                if similar_chunks:
                    print(f"Found {len(similar_chunks)} similar chunks. Creating prompt.")
                    prompt = create_prompt(message, similar_chunks)
                    context_display = "\n".join([f"- {chunk[:100]}..." for chunk in similar_chunks])
                    context_prefix = f"**Retrieved Context:**\n{context_display}\n\n---\n\n"
                    # OPTIONAL: Add context as a separate message (alternative to prefixing)
                    # chat_history.append({"role": "system", "content": f"Retrieved Context:\n{context_display}"}) 
                else:
                    print("No similar chunks found. Using direct query as prompt.")
                    prompt = message

                # 5. Get the response from Ollama
                print("Sending prompt to Ollama...")
                response_text = ollama.invoke(prompt)
                # --- Debug Print ---
                print(f">>> LLM Raw Response: {response_text}")
                # --- End Debug Print ---
                print(f"Received response from Ollama.")

                # --- Split response into Context, Thinking, and Answer ---

                # 1. Append Context (if any)
                if context_prefix:
                    chat_history.append({"role": "assistant", "content": context_prefix})

                # 2. Extract and Append Thinking (if any)
                think_match = re.search(r"<think>(.*?)</think>", response_text, re.DOTALL)
                final_answer = response_text # Default to full response

                if think_match:
                    thinking_content = think_match.group(1).strip()
                    if thinking_content: # Only append if there's actual content inside
                        chat_history.append({"role": "assistant", "content": f"**Thinking:**\n{thinking_content}"})
                    # Remove the thinking block from the original response
                    final_answer = response_text.replace(think_match.group(0), "").strip()

                # 3. Append Final Answer
                # Only append if there's a non-empty final answer left
                if final_answer:
                    chat_history.append({"role": "assistant", "content": f"**Answer:**\n{final_answer}"})

                # Return None to clear input box, return updated history
                return None, chat_history

            except Exception as e:
                print(f"Error during chat processing for topic '{selected_topic}': {e}")
                gr.Error(f"An error occurred: {e}")
                # Append error message as assistant response
                chat_history.append({"role": "assistant", "content": f"Sorry, an error occurred: {e}"}) 
                # Return None to clear input box, return updated history with error
                return None, chat_history

        # Connect components
        # The output components are now [msg_input, chatbot]
        # We expect None for msg_input and the updated history for chatbot
        msg_input.submit(respond, [msg_input, chatbot, topic_selector], [msg_input, chatbot])
        # Clear function needs to return None for msg_input and empty list for chatbot
        clear_button.click(lambda: (None, []), None, [msg_input, chatbot], queue=False)
        # Optional: Allow changing topic to clear chat? Or keep history?
        # topic_selector.change(lambda: (None, []), None, [msg_input, chatbot], queue=False) # Clears chat on topic change

    return chat_interface 