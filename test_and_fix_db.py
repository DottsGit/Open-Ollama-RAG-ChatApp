import os
import shutil
import time
import json
import argparse
from typing import List, Dict, Tuple, Optional
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Configuration
CHROMA_PATH = "./chroma"
DATA_PATH = "./data"
TRACKER_PATH = "document_tracker.json"

# Using a fast, efficient model with GPU acceleration
MODEL_CONFIG = {
    "model_name": "BAAI/bge-large-en-v1.5",  # Larger model with 1024-dim embeddings
    "model_kwargs": {"device": "cuda"},
    "encode_kwargs": {
        "device": "cuda",
        "batch_size": 128,  # Adjusted for larger model
        "show_progress_bar": True,
        "normalize_embeddings": True
    }
}

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

def clean_chroma_db():
    """Remove the existing Chroma database."""
    if os.path.exists(CHROMA_PATH):
        print(f"Removing existing Chroma database at {CHROMA_PATH}")
        try:
            shutil.rmtree(CHROMA_PATH)
            time.sleep(1)  # Give the system time to release resources
        except Exception as e:
            print(f"Error cleaning Chroma DB: {e}")

def get_document_signature(file_path: str) -> str:
    """Get a unique signature for a document based on its path, size, and modification time."""
    stats = os.stat(file_path)
    return f"{file_path}_{stats.st_size}_{stats.st_mtime}"

def load_document_tracker() -> Dict[str, str]:
    """Load the document tracker from disk."""
    if os.path.exists(TRACKER_PATH):
        with open(TRACKER_PATH, 'r') as f:
            return json.load(f)
    return {}

def save_document_tracker(tracker: Dict[str, str]):
    """Save the document tracker to disk."""
    with open(TRACKER_PATH, 'w') as f:
        json.dump(tracker, f)

def process_documents(data_path: str, limit: Optional[int] = None) -> List[str]:
    """Load and process documents from a directory."""
    if not os.path.exists(data_path):
        print(f"Data directory {data_path} does not exist.")
        return []

    files = []
    for root, _, filenames in os.walk(data_path):
        for filename in filenames:
            if filename.endswith('.md'):
                file_path = os.path.join(root, filename)
                files.append(file_path)
                if limit and len(files) >= limit:
                    return files
    return files

def create_vector_db(
    initial_db: bool = True,
    specific_doc: Optional[str] = None,
    batch_size: int = 256,  # Increased batch size for GPU
    max_workers: int = 1,
    doc_limit: Optional[int] = None
) -> Tuple[Chroma, Dict[str, str]]:
    """Create or update the vector database."""
    
    if initial_db:
        clean_chroma_db()
    
    # Ensure the Chroma directory exists
    os.makedirs(CHROMA_PATH, exist_ok=True)
    
    # Initialize embeddings with GPU support
    print("Initializing embeddings with CUDA support...")
    embeddings = HuggingFaceEmbeddings(**MODEL_CONFIG)
    
    # Verify CUDA is being used
    import torch
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("Warning: CUDA not available, falling back to CPU")
    
    chroma_db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )
    
    # Load document tracker
    doc_tracker = load_document_tracker()
    
    # Get list of documents to process
    if specific_doc:
        files = [os.path.join(DATA_PATH, specific_doc)] if os.path.exists(os.path.join(DATA_PATH, specific_doc)) else []
    else:
        files = process_documents(DATA_PATH, doc_limit)
    
    if not files:
        print("No documents to process.")
        return chroma_db, doc_tracker
    
    # Track missing documents
    missing_docs = []
    for file_path in files:
        current_signature = get_document_signature(file_path)
        if file_path not in doc_tracker or doc_tracker[file_path] != current_signature:
            missing_docs.append(file_path)
            print(f"Processing missing file: {os.path.relpath(file_path, DATA_PATH)}")
    
    if not missing_docs:
        print("No missing or modified documents found.")
        return chroma_db, doc_tracker
    
    print(f"Processing {len(missing_docs)} missing documents...")
    
    # Process documents one at a time
    for file_path in missing_docs:
        try:
            print(f"\nProcessing {os.path.relpath(file_path, DATA_PATH)}")
            
            # Load and split the document
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split by headers first
            markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
            header_splits = markdown_splitter.split_text(content)
            
            # Further split by characters
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,  # Smaller chunk size
                chunk_overlap=50  # Smaller overlap
            )
            
            # Process each header section separately
            for doc in header_splits:
                chunks = text_splitter.split_text(doc.page_content)
                
                # Process chunks in small batches
                for i in range(0, len(chunks), batch_size):
                    batch = chunks[i:i + batch_size]
                    try:
                        chroma_db.add_texts(
                            texts=batch,
                            metadatas=[{"source": file_path} for _ in batch]
                        )
                        # Persist after each batch
                        chroma_db.persist()
                        print(f"Processed and persisted batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")
                    except Exception as e:
                        print(f"Error processing batch: {e}")
                        continue
            
            # Update tracker after successful processing of the entire file
            doc_tracker[file_path] = get_document_signature(file_path)
            save_document_tracker(doc_tracker)
            print(f"Successfully processed {os.path.relpath(file_path, DATA_PATH)}")
            
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            continue
    
    return chroma_db, doc_tracker

def validate_vector_db() -> List[str]:
    """Validate that all documents in the data directory have vectors in the database."""
    files = process_documents(DATA_PATH)
    doc_tracker = load_document_tracker()
    
    missing_docs = []
    for file_path in files:
        if file_path not in doc_tracker:
            missing_docs.append(os.path.relpath(file_path, DATA_PATH))
    
    if missing_docs:
        print(f"\nFound {len(missing_docs)} documents missing from the vector database:")
        for doc in missing_docs:
            print(f"- {doc}")
    else:
        print("\nAll documents are present in the vector database.")
    
    return missing_docs

def check_for_duplicates(chroma_db: Chroma) -> List[str]:
    """Check for duplicate vectors in the database."""
    print("\nChecking for duplicate vectors...")
    
    try:
        # Get all documents and their metadata
        collection_data = chroma_db._collection.get(include=['metadatas', 'documents'])
        all_metadatas = collection_data.get('metadatas', [])
        all_documents = collection_data.get('documents', [])
        
        # Count occurrences by source and content
        duplicates = {}
        content_map = {}
        
        for i, (metadata, content) in enumerate(zip(all_metadatas, all_documents)):
            if not metadata or not content:
                continue
                
            source = metadata.get('source', 'unknown')
            content_hash = hash(content.strip())
            
            if content_hash in content_map:
                # This is a duplicate
                if source not in duplicates:
                    duplicates[source] = 0
                duplicates[source] += 1
                
                print(f"Found duplicate in {source}:")
                print(f"  Original: {content[:100]}...")
                print(f"  Duplicate: {all_documents[content_map[content_hash]][:100]}...")
            else:
                content_map[content_hash] = i
        
        total_duplicates = sum(duplicates.values())
        if total_duplicates > 0:
            print(f"\nFound {total_duplicates} total duplicate vectors:")
            for source, count in duplicates.items():
                print(f"  {os.path.basename(source)}: {count} duplicates")
        else:
            print("No duplicate vectors found.")
            
        return list(duplicates.keys())
        
    except Exception as e:
        print(f"Error checking for duplicates: {e}")
        return []

def remove_duplicates(chroma_db: Chroma) -> int:
    """Remove duplicate vectors from the database by preserving the existing database."""
    print("\nRemoving duplicate vectors...")
    
    try:
        # First get the total count
        collection = chroma_db._collection
        total_count = collection.count()
        print(f"Total vectors in database: {total_count}")
        
        # Get a small sample to determine embedding dimension
        sample = collection.get(limit=1, include=['embeddings'])
        if sample and sample['embeddings'] and len(sample['embeddings']) > 0:
            embedding_dim = len(sample['embeddings'][0])
            print(f"Original embedding dimension: {embedding_dim}")
        
        # Track content hashes and IDs globally across all chunks
        content_to_id = {}  # Maps content hash to first ID that contains it
        duplicate_ids = []  # IDs to remove
        
        # First pass: identify all duplicates across all chunks
        print("\nPASS 1: Identifying all duplicates...")
        for offset in range(0, total_count, 1000):
            chunk_size = min(1000, total_count - offset)
            print(f"Scanning chunk {offset//1000 + 1}/{(total_count + 999)//1000} ({chunk_size} items)")
            
            # Get chunk of documents
            chunk = collection.get(
                limit=chunk_size,
                offset=offset,
                include=['documents']
            )
            
            # Get IDs separately
            ids_chunk = collection.get(
                limit=chunk_size,
                offset=offset
            )
            
            chunk_docs = chunk.get('documents', [])
            chunk_ids = ids_chunk.get('ids', [])
            
            # Map document content to IDs
            for i, (doc_id, content) in enumerate(zip(chunk_ids, chunk_docs)):
                if not content:
                    continue
                    
                content_hash = hash(content.strip())
                
                if content_hash in content_to_id:
                    # This is a duplicate, mark for removal
                    duplicate_ids.append(doc_id)
                else:
                    # First time seeing this content, keep it
                    content_to_id[content_hash] = doc_id
        
        if not duplicate_ids:
            print("No duplicates found in database.")
            return 0
            
        print(f"\nFound {len(duplicate_ids)} duplicates across {len(content_to_id)} unique documents")
        
        # Second pass: remove duplicates in batches
        print("\nPASS 2: Removing duplicates...")
        batch_size = 500  # Smaller batches to avoid overloading API
        for i in range(0, len(duplicate_ids), batch_size):
            batch = duplicate_ids[i:i + batch_size]
            print(f"Removing batch {i//batch_size + 1}/{(len(duplicate_ids) + batch_size - 1)//batch_size} ({len(batch)} items)")
            
            try:
                collection.delete(ids=batch)
            except Exception as e:
                print(f"Error removing batch: {e}")
                import traceback
                traceback.print_exc()
        
        # Third pass: verify changes took effect
        print("\nPASS 3: Verifying removal...")
        new_count = collection.count()
        removed = total_count - new_count
        
        print(f"Original count: {total_count}")
        print(f"Current count: {new_count}")
        print(f"Removed count: {removed}")
        
        if removed != len(duplicate_ids):
            print(f"WARNING: Expected to remove {len(duplicate_ids)} but only removed {removed}")
            
        # Check if any duplicates remain
        print("\nChecking for any remaining duplicates...")
        remaining_content = set()
        has_remaining_dupes = False
        
        for offset in range(0, new_count, 1000):
            chunk_size = min(1000, new_count - offset)
            
            # Get chunk of documents
            chunk = collection.get(
                limit=chunk_size,
                offset=offset,
                include=['documents']
            )
            
            chunk_docs = chunk.get('documents', [])
            
            # Check for duplicates
            for content in chunk_docs:
                if not content:
                    continue
                
                content_hash = hash(content.strip())
                
                if content_hash in remaining_content:
                    print("WARNING: Found remaining duplicate after cleanup")
                    has_remaining_dupes = True
                    # Only need to show one example
                    print(f"Example content: {content[:100]}...")
                    break
                else:
                    remaining_content.add(content_hash)
            
            if has_remaining_dupes:
                break
                
        if not has_remaining_dupes:
            print("All duplicates successfully removed!")
            
        # Persist changes
        chroma_db.persist()
        
        return removed
            
    except Exception as e:
        print(f"Error removing duplicates: {e}")
        import traceback
        traceback.print_exc()
        return 0

def main():
    parser = argparse.ArgumentParser(description="Test and fix vector database issues")
    parser.add_argument("--validate", action="store_true", help="Validate the vector database")
    parser.add_argument("--process-document", type=str, help="Process a specific document")
    parser.add_argument("--reprocess-all", action="store_true", help="Reprocess all documents")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for processing documents")
    parser.add_argument("--max-workers", type=int, default=1, help="Maximum number of workers for processing")
    parser.add_argument("--limit", type=int, help="Limit the number of documents to process")
    parser.add_argument("--check-duplicates", action="store_true", help="Check for duplicate vectors")
    parser.add_argument("--remove-duplicates", action="store_true", help="Remove duplicate vectors")
    
    args = parser.parse_args()
    
    # Initialize Chroma DB connection if needed for duplicate operations
    if args.check_duplicates or args.remove_duplicates:
        embeddings = HuggingFaceEmbeddings(**MODEL_CONFIG)
        chroma_db = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings
        )
    
    if args.check_duplicates:
        check_for_duplicates(chroma_db)
    elif args.remove_duplicates:
        remove_duplicates(chroma_db)
    elif args.validate:
        validate_vector_db()
    elif args.process_document:
        create_vector_db(
            initial_db=False,
            specific_doc=args.process_document,
            batch_size=args.batch_size,
            max_workers=args.max_workers
        )
    elif args.reprocess_all:
        create_vector_db(
            initial_db=True,
            batch_size=args.batch_size,
            max_workers=args.max_workers,
            doc_limit=args.limit
        )
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 