## initial database?
# -> Set to True if you run the notebook for the first time or if you changed the md files
initial_db = True

print("Starting Ollama RAG ChatApp...")

DATA_PATH = "data/"
OLLAMA_MODEL = "DeepSeek-R1:14b"
OLLAMA_URL = "http://localhost:11434"
CHROMA_PATH = "chroma/"

print(f"Configuration loaded: Using {OLLAMA_MODEL} model at {OLLAMA_URL}")
print(f"Data path: {DATA_PATH}, Chroma path: {CHROMA_PATH}")

## langchain split config
# md headers
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
    ("####", "Header 4"),
]

# chunk sizes
chunk_size = 500
chunk_overlap = 100
print(f"Text splitting configuration: chunk size={chunk_size}, overlap={chunk_overlap}")

from langchain.document_loaders import TextLoader
import os

print("Loading documents from data directory...")
documents = []

for file in os.listdir(DATA_PATH):
    loader = TextLoader(DATA_PATH + file, encoding="utf-8")
    documents.append(loader.load()[0])

print(f"Loaded {len(documents)} documents")

documents[0].metadata
len(documents[0].page_content)

# for doc in documents:
#     print(doc.metadata)

from langchain.text_splitter import MarkdownHeaderTextSplitter

print("Starting document splitting by markdown headers...")
text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
chunks_array= []

for doc in documents:
    chunks = text_splitter.split_text(doc.page_content)
    # append source metadata to each chunk
    for chunk in chunks:
        chunk.metadata.update(doc.metadata)  # Use update() to merge metadata
    chunks_array.append(chunks)

print(f"Created {len(chunks_array)} document chunk arrays after header splitting")

# Char-level splits
from langchain.text_splitter import RecursiveCharacterTextSplitter

print("Performing character-level splitting to create final chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len, add_start_index=True
)

chunks_array_txt_base = []
counter = 0
for document in chunks_array:
    for chunk in document:
        splits = text_splitter.split_documents([chunk])
        chunks_array_txt_base.append(splits)
        counter += 1

print(f"Generated {counter} chunks after character-level splitting")
print(f"Total document chunk arrays: {len(chunks_array_txt_base)}")

all_document_chunks = [chunk for document in chunks_array_txt_base for chunk in document]
print(f"Total individual document chunks: {len(all_document_chunks)}")

# TEST OLLAMA CONNECTION ##
from langchain_community.llms import Ollama

print(f"Testing connection to Ollama at {OLLAMA_URL}...")
ollama = Ollama(base_url=OLLAMA_URL, model=OLLAMA_MODEL)

response = ollama("Who is Alice?")
print(f"Ollama connection test response: {response}")

all_document_chunks[0]

from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from tqdm import tqdm
import time
import concurrent.futures
import os
import json
import hashlib

def create_vector_db(all_document_chunks, ollama_url, ollama_model, chroma_path, batch_size=30, max_workers=3):
    """
    Fixed version of create_vector_db that properly handles missing documents
    and prevents duplicate entries.
    """
    # Create directory if it doesn't exist
    os.makedirs(chroma_path, exist_ok=True)
    
    # Define path for document tracking file
    tracker_file = os.path.join(chroma_path, "document_tracker.json")
    
    # Initialize or load document tracker
    document_tracker = {}
    if os.path.exists(tracker_file):
        try:
            with open(tracker_file, 'r') as f:
                document_tracker = json.load(f)
            print(f"Loaded document tracker with {len(document_tracker)} entries")
        except Exception as e:
            print(f"Error loading document tracker: {e}")
            document_tracker = {}
    
    # Initialize embedding function
    print(f"Initializing OllamaEmbeddings with model {ollama_model}")
    embeddings = OllamaEmbeddings(
        base_url=ollama_url, 
        model=ollama_model
    )
    
    # Load existing Chroma database
    print(f"Loading Chroma vector database from {chroma_path}")
    chroma_db = Chroma(
        embedding_function=embeddings, 
        persist_directory=chroma_path
    )
    
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
    if DATA_PATH and os.path.exists(DATA_PATH):
        for file in os.listdir(DATA_PATH):
            file_path = os.path.join(DATA_PATH, file)
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
    
    # FIX: Improved handling of missing documents
    # Check if any documents in data directory are missing from the vector database
    missing_documents = []
    if data_files:
        print(f"Found {len(data_files)} documents in data directory not in source_to_chunks")
        # For each file we need to create new chunks and add them
        for file_path in data_files:
            print(f"Processing missing file: {file_path}")
            try:
                # Load the document
                loader = TextLoader(file_path, encoding="utf-8")
                document = loader.load()[0]
                missing_documents.append(document)
                
                # Record this as a modified source for cleanup later
                modified_sources.append(file_path)
                
                # Update the tracker for this file
                file_mtime = os.path.getmtime(file_path)
                file_size = os.path.getsize(file_path)
                file_signature = f"{file_path}:{file_size}:{file_mtime}"
                document_tracker[file_path] = {
                    "signature": file_signature,
                    "last_processed": time.time()
                }
            except Exception as e:
                print(f"Error processing missing file {file_path}: {e}")
    
    # Process any missing documents if they were found
    if missing_documents:
        print(f"Processing {len(missing_documents)} missing documents...")
        for document in missing_documents:
            # Split by markdown headers
            md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
            md_chunks = md_splitter.split_text(document.page_content)
            
            # Update metadata for each chunk
            for chunk in md_chunks:
                chunk.metadata.update(document.metadata)
            
            # Further split by character level
            char_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, 
                chunk_overlap=chunk_overlap, 
                length_function=len, 
                add_start_index=True
            )
            
            for md_chunk in md_chunks:
                splits = char_splitter.split_documents([md_chunk])
                chunks_to_process.extend(splits)
    
    # If nothing to process, save tracker and return
    if not chunks_to_process:
        print("No new or modified files to process")
        with open(tracker_file, 'w') as f:
            json.dump(document_tracker, f, indent=2)
        return chroma_db
    
    # FIX: Improved removal of old vectors for modified files
    # First check if the database has any documents before trying to delete
    try:
        if hasattr(chroma_db._collection, "count"):
            collection_count = chroma_db._collection.count()
            
            if collection_count > 0 and modified_sources and hasattr(chroma_db._collection, "get") and hasattr(chroma_db._collection, "delete"):
                print("Checking for existing vectors to update...")
                print(f"Vector database has {collection_count} existing vectors")
                
                # Get all metadatas and ids
                collection_data = chroma_db._collection.get(include=['metadatas', 'ids'])
                all_metadatas = collection_data.get('metadatas', [])
                all_ids = collection_data.get('ids', [])
                
                # Find vectors that match modified sources
                ids_to_delete = []
                for i, metadata in enumerate(all_metadatas):
                    if metadata and "source" in metadata:
                        source = metadata["source"]
                        if source in modified_sources:
                            ids_to_delete.append(all_ids[i])
                
                # Delete old vectors for modified files
                if ids_to_delete:
                    print(f"Removing {len(ids_to_delete)} existing vectors for modified files")
                    chroma_db._collection.delete(ids=ids_to_delete)
                    print("Successfully removed outdated vectors")
                else:
                    print("No existing vectors found for modified files")
    except Exception as e:
        print(f"Error removing old vectors: {e}")
        print("Will continue with adding new vectors")
    
    # Function to process a batch of documents
    def process_batch(batch):
        try:
            chroma_db.add_documents(documents=batch)
            return len(batch)
        except Exception as e:
            print(f"Error processing batch: {e}")
            # On failure, process documents one by one
            successful = 0
            for doc in batch:
                try:
                    chroma_db.add_documents(documents=[doc])
                    successful += 1
                except Exception as e:
                    print(f"Problem with document: {doc.page_content[:50]}... Error: {e}")
            return successful
    
    # Process in batches with parallel workers
    print(f"Starting batch processing with batch_size={batch_size}, max_workers={max_workers}")
    start_time = time.time()
    processed_count = 0
    
    # Create batches
    batches = [chunks_to_process[i:i+batch_size] for i in range(0, len(chunks_to_process), batch_size)]
    print(f"Created {len(batches)} batches for processing")
    
    # Process batches with progress tracking
    with tqdm(total=len(chunks_to_process), desc="Adding New Vectors") as pbar:
        # Use ThreadPoolExecutor for parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all batches to the executor
            future_to_batch = {executor.submit(process_batch, batch): batch for batch in batches}
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_batch):
                batch_count = future.result()
                processed_count += batch_count
                pbar.update(batch_count)
                
                # Persist periodically
                if processed_count % max(int(len(chunks_to_process) * 0.1), batch_size) < batch_size:
                    print(f"Persisting database after processing {processed_count} chunks...")
                    chroma_db.persist()
    
    # Final persistence
    print("Performing final database persistence...")
    chroma_db.persist()
    
    # Save tracker
    with open(tracker_file, 'w') as f:
        json.dump(document_tracker, f, indent=2)
    
    # Calculate and display metrics
    elapsed_time = time.time() - start_time
    docs_per_second = processed_count / elapsed_time if elapsed_time > 0 else 0
    
    print(f"Vector database updated with {processed_count} new document chunks!")
    print(f"Total time: {elapsed_time:.2f} seconds ({docs_per_second:.2f} docs/second)")
    
    # Validate that all documents have vectors
    print("\nValidating vector database to ensure all documents are represented...")
    validate_vector_db(chroma_db, DATA_PATH)
    
    return chroma_db

# Improved function to validate and fix the vector database
def validate_vector_db(chroma_db, data_path):
    """
    Improved validation function that provides more detailed feedback
    on missing documents.
    """
    if not os.path.exists(data_path):
        print(f"Data path {data_path} not found, skipping validation")
        return
    
    # Get all document files in the data directory
    all_files = set()
    for file in os.listdir(data_path):
        file_path = os.path.join(data_path, file)
        if os.path.isfile(file_path):
            all_files.add(file_path)
    
    print(f"Found {len(all_files)} documents in {data_path}")
    
    # Get all sources in the vector database
    try:
        print("Checking which documents have vectors in the database...")
        collection_data = chroma_db._collection.get(include=['metadatas'])
        all_metadatas = collection_data.get('metadatas', [])
        
        # Extract unique sources
        vectorized_files = set()
        for metadata in all_metadatas:
            if metadata and "source" in metadata:
                vectorized_files.add(metadata["source"])
        
        print(f"Found {len(vectorized_files)} unique document sources in the vector database")
        
        # Find files that exist but don't have vectors
        missing_vectors = all_files - vectorized_files
        if missing_vectors:
            print(f"Warning: {len(missing_vectors)} documents in {data_path} don't have vectors in the database:")
            for file in missing_vectors:
                print(f"  Missing: {file}")
            print("\nTo fix missing documents, run again with initial_db = True, or manually process these files")
        else:
            print("✓ All documents in the data directory have vectors in the database")
            
    except Exception as e:
        print(f"Error validating vector database: {e}")

# Usage
if initial_db:
    print("\nInitializing new vector database...")
    chroma_db = create_vector_db(
        all_document_chunks=all_document_chunks,
        ollama_url=OLLAMA_URL,
        ollama_model=OLLAMA_MODEL,
        chroma_path=CHROMA_PATH,
        batch_size=8,  # Adjust based on your system and model
        max_workers=100  # Adjust based on CPU cores and memory
    )
else:
    print("\nSkipping vector database creation - using existing database")

## load chroma db from disk
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma

print("Loading Chroma vector database from disk...")
chroma_db = Chroma(persist_directory=CHROMA_PATH, embedding_function=OllamaEmbeddings(base_url=OLLAMA_URL, model=OLLAMA_MODEL))
print(f"Successfully loaded Chroma database from {CHROMA_PATH}")

# test similarity search
query = "Who is Alice?"
print(f"\nRunning test similarity search with query: '{query}'")

result_docs = chroma_db.similarity_search(query)
print(f"Retrieved {len(result_docs)} relevant documents")

for doc in result_docs:
    print(doc)

from langchain_community.llms import Ollama
import re
import os

def chat_ollama(message, history):
    print(f"\nProcessing chat query: '{message}'")
    
    # initiate ollama
    ollama = Ollama(base_url=OLLAMA_URL, model=OLLAMA_MODEL)

    # Check if the query is about a specific document
    doc_pattern = r'(\d{3}-\d{5}-\d{5}(?:_redacted)?\.md)'
    doc_matches = re.findall(doc_pattern, message, re.IGNORECASE)
    
    result_chunks = []
    
    # If specific documents are mentioned, prioritize them
    if doc_matches:
        print(f"Detected specific document request for: {doc_matches}")
        for doc_name in doc_matches:
            doc_path = os.path.join(DATA_PATH, doc_name)
            
            # Check if the document exists in our data
            if os.path.exists(doc_path):
                print(f"Found requested document: {doc_path}")
                
                # Search specifically for chunks from this document
                doc_specific_chunks = []
                for doc in chroma_db.similarity_search(
                    f"Document {doc_name} content", k=25
                ):
                    if doc.metadata.get("source") == doc_path:
                        doc_specific_chunks.append(doc)
                
                if doc_specific_chunks:
                    print(f"Retrieved {len(doc_specific_chunks)} chunks from {doc_path}")
                    result_chunks.extend(doc_specific_chunks[:5])  # Limit to 5 chunks per requested doc
                else:
                    print(f"Warning: Document {doc_path} exists but no chunks found in the vector database")
            else:
                print(f"Warning: Requested document {doc_path} not found")
    
    # Then add general search results if needed
    if len(result_chunks) < 10:
        print("Searching vector database for additional relevant documents...")
        # Retrieve more chunks initially to find more unique documents
        initial_chunks = chroma_db.similarity_search(message, k=25)
        
        # Track unique sources and their chunks
        source_chunks = {}
        
        # Organize chunks by source
        for chunk in initial_chunks:
            source = chunk.metadata["source"]
            if source not in source_chunks:
                source_chunks[source] = []
            source_chunks[source].append(chunk)
        
        print(f"Found {len(source_chunks)} unique document sources from initial search")
        
        # Select the most relevant chunks while preserving source diversity
        # Get top 5 unique sources (or fewer if we already have some results)
        remaining_sources = 5 - len(set(chunk.metadata["source"] for chunk in result_chunks))
        top_sources = list(source_chunks.keys())[:remaining_sources]
        
        # Get up to 5 chunks from each top source
        for source in top_sources:
            # Add up to 5 chunks from this source
            source_docs = source_chunks[source][:5]
            result_chunks.extend(source_docs)
    
    print(f"Selected {len(result_chunks)} chunks from {len(set(chunk.metadata['source'] for chunk in result_chunks))} unique sources")
    
    # Print source information for each chunk
    print("Document chunks come from these sources:")
    for i, chunk in enumerate(result_chunks):
        source = chunk.metadata["source"]
        print(f"  Chunk {i+1}: {source}")
    
    # First pass: build a mapping of unique sources
    source_map = {}  # Maps source paths to reference numbers
    ref_counter = 1
    
    for chunk in result_chunks:
        source = chunk.metadata["source"]
        if source not in source_map:
            source_map[source] = ref_counter
            ref_counter += 1
    
    print(f"Found {len(source_map)} unique document sources in final selection")
    
    # Build knowledge context using consistent source IDs from the mapping
    chroma_knowledge = ""
    for chunk in result_chunks:
        source = chunk.metadata["source"]
        source_id = source_map[source]
        chroma_knowledge += f"[{source_id}]\n{chunk.page_content}\n"
    
    # Build the references section using the same mapping
    sources = ""
    for source, ref_id in sorted(source_map.items(), key=lambda x: x[1]):
        sources += f"[{ref_id}]\n{source}\n"

    # Create a more specific prompt if requesting specific documents
    if doc_matches:
        specific_docs = ", ".join(doc_matches)
        instruction = f"Answer the following question about document(s) {specific_docs} using ONLY the provided knowledge. If the specific document(s) mentioned are not in the references, state clearly that the requested document was not found in the database."
    else:
        instruction = "Answer the following question using the provided knowledge and the chat history:"

    print("Building prompt with retrieved knowledge and sending to Ollama...")
    prompt = f"{instruction}\n\n###KNOWLEDGE: " + chroma_knowledge + "\n###CHAT-HISTORY: " + str(history) + "\n\n###QUESTION: " + message
    result = ollama(prompt) + "\n\n\nReferences:\n" + sources 
    print("Response generated successfully")
    
    return result

print("\nTesting chat function...")
test_response = chat_ollama("Who is Alice?", "")
print("Chat function test completed")

import gradio as gr
print("\nSetting up Gradio web interface...")
gradio_interface = gr.ChatInterface(
        chat_ollama,
        chatbot=gr.Chatbot(),
        textbox=gr.Textbox(placeholder="Example: Who is Alice?", container=False, scale=7),
        title="The Ollama test chatbot",
        description=f"Ask the {OLLAMA_MODEL} chatbot a question!",
        theme='gradio/base', # themes at https://huggingface.co/spaces/gradio/theme-gallery
        retry_btn=None,
        undo_btn="Delete Previous",
        clear_btn="Clear",

)

print("Starting Gradio web server...")
gradio_interface.launch()
print("Gradio server is running!") 