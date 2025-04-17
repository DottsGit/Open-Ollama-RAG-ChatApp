"""
Document processing module.

This module handles loading and processing documents for the RAG application.
"""

import os
import json
from typing import List, Dict, Any

from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import BSHTMLLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain.schema.document import Document

from src.config import (
    HEADERS_TO_SPLIT_ON, 
    CHUNK_SIZE, 
    CHUNK_OVERLAP
)

def load_documents(data_path: str) -> List[Document]:
    """
    Load all documents from the specified data directory.
    Handles .txt, .md, .jsonl, .htm, and .html files.

    Args:
        data_path: Path to the directory containing documents

    Returns:
        List of loaded documents
    """
    documents = []

    if not os.path.exists(data_path):
        print(f"Data path {data_path} does not exist.")
        return documents

    print(f"Scanning directory: {data_path}")
    for file in os.listdir(data_path):
        file_path = os.path.join(data_path, file)
        if not os.path.isfile(file_path):
            continue # Skip directories

        try:
            if file.lower().endswith('.jsonl'):
                print(f"  Loading JSONL file: {file}")
                with open(file_path, 'r', encoding='utf-8') as f:
                    line_num = 0
                    for line in f:
                        line_num += 1
                        try:
                            data = json.loads(line.strip())
                            # --- ASSUMPTION: Keys are 'question' and 'answer' --- 
                            # --- MODIFY HERE if your keys are different --- 
                            question = data.get('question')
                            answer = data.get('answer')

                            if question and answer:
                                page_content = f"Question: {question}\nAnswer: {answer}"
                                metadata = {"source": file_path, "row": line_num}
                                documents.append(Document(page_content=page_content, metadata=metadata))
                            else:
                                 print(f"    Warning: Skipping line {line_num} in {file} - missing 'question' or 'answer' key.")
                        except json.JSONDecodeError:
                            print(f"    Warning: Skipping line {line_num} in {file} - invalid JSON.")
                        except Exception as line_e:
                             print(f"    Warning: Skipping line {line_num} in {file} - error processing line: {line_e}")
                print(f"  Finished loading JSONL file: {file}")

            # Simple text loader for other common types (e.g., .txt, .md)
            elif file.lower().endswith(('.txt', '.md')):
                 print(f"  Loading Text file: {file}")
                 loader = TextLoader(file_path, encoding="utf-8")
                 # TextLoader returns a list, usually with one doc
                 loaded_docs = loader.load()
                 if loaded_docs:
                     # Add source manually if loader doesn't
                     if 'source' not in loaded_docs[0].metadata:
                          loaded_docs[0].metadata['source'] = file_path
                     documents.extend(loaded_docs)
                 print(f"  Finished loading Text file: {file}")

            # Modified: HTML Loader using built-in parser
            elif file.lower().endswith(('.htm', '.html')):
                 print(f"  Loading HTML file: {file} using html.parser")
                 # Explicitly use html.parser to avoid external dependencies like lxml
                 loader = BSHTMLLoader(
                     file_path,
                     open_encoding='utf-8',
                     bs_kwargs={"features": "html.parser"} # Specify the parser
                 )
                 loaded_docs = loader.load()
                 if loaded_docs:
                     if 'source' not in loaded_docs[0].metadata:
                          loaded_docs[0].metadata['source'] = file_path
                     documents.extend(loaded_docs)
                 print(f"  Finished loading HTML file: {file}")

            else:
                 print(f"  Skipping unsupported file type: {file}")

        except Exception as e:
            print(f"Error loading document {file_path}: {e}")

    print(f"Loaded {len(documents)} documents/records from {data_path}")
    return documents

def process_documents(documents: List[Document]) -> List[Document]:
    """
    Process documents by splitting them into chunks.
    
    Args:
        documents: List of documents to process
        
    Returns:
        List of document chunks
    """
    print("Starting document splitting by markdown headers...")
    text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS_TO_SPLIT_ON)
    chunks_array = []
    
    for doc in documents:
        chunks = text_splitter.split_text(doc.page_content)
        # Append source metadata to each chunk
        for chunk in chunks:
            chunk.metadata.update(doc.metadata)
        chunks_array.append(chunks)
    
    print(f"Created {len(chunks_array)} document chunk arrays after header splitting")
    
    # Character-level splitting
    print("Performing character-level splitting to create final chunks...")
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, 
        chunk_overlap=CHUNK_OVERLAP, 
        length_function=len, 
        add_start_index=True
    )
    
    chunks_array_txt_base = []
    counter = 0
    for document in chunks_array:
        for chunk in document:
            splits = char_splitter.split_documents([chunk])
            chunks_array_txt_base.append(splits)
            counter += 1
    
    print(f"Generated {counter} chunks after character-level splitting")
    
    # Flatten all chunks into a single list
    all_document_chunks = [chunk for document in chunks_array_txt_base for chunk in document]
    print(f"Total individual document chunks: {len(all_document_chunks)}")
    
    return all_document_chunks

def normalize_document_name(file_path: str) -> str:
    """
    Normalize document name by removing identifier suffixes.
    
    Args:
        file_path: The path to the document
        
    Returns:
        Normalized file path
    """
    base_name = os.path.basename(file_path)
    dir_name = os.path.dirname(file_path)
    
    # Check if filename has the specific pattern with double underscore followed by identifier
    name_parts = base_name.split('__')
    
    if len(name_parts) > 1:
        # If we have a suffix, remove it
        normalized_name = name_parts[0]
        # Add any original extension
        if '.' in base_name and '.' not in normalized_name:
            ext = base_name.split('.')[-1]
            normalized_name = f"{normalized_name}.{ext}"
        
        return os.path.join(dir_name, normalized_name)
    
    return file_path 