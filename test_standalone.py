import os
import time
import shutil
import tempfile
import unittest
import json
from langchain.document_loaders import TextLoader
from langchain.vectorstores import Chroma

# Use a simple mock embedding function for testing
class MockEmbeddings:
    def __init__(self):
        pass
        
    def embed_documents(self, texts):
        # Return fixed but unique embeddings for each text (use hash for uniqueness)
        return [[0.1, 0.2, 0.3, hash(text) % 100 / 100] for text in texts]
        
    def embed_query(self, text):
        # Return a simple fixed embedding for the query
        return [0.1, 0.2, 0.3, hash(text) % 100 / 100]

# Mock document for testing
class MockDocument:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata

# Modified version of create_vector_db for testing with mock embeddings
def create_vector_db_for_test(all_document_chunks, chroma_path, batch_size=5, max_workers=2):
    """Test version that uses MockEmbeddings instead of Ollama"""
    import time
    import concurrent.futures
    import os
    import json
    from tqdm import tqdm
    
    # Get total number of document chunks
    total_chunks = len(all_document_chunks)
    print(f"Processing {total_chunks} document chunks")
    
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
    
    # Create directory if it doesn't exist
    os.makedirs(chroma_path, exist_ok=True)
            
    # Use mock embeddings for testing
    embeddings = MockEmbeddings()
    
    # Create or load Chroma instance
    print(f"Loading existing Chroma vector database at {chroma_path}")
    chroma_db = Chroma(
        embedding_function=embeddings, 
        persist_directory=chroma_path
    )
    
    # Identify documents that need processing (new or modified)
    print("Identifying documents that need processing...")
    source_to_chunks = {}
    
    # Group chunks by source file
    for chunk in all_document_chunks:
        source = chunk.metadata.get("source", "")
        if source not in source_to_chunks:
            source_to_chunks[source] = []
        source_to_chunks[source].append(chunk)
    
    # Check which files need processing
    files_processed = 0
    chunks_to_process = []
    modified_sources = []
    
    for source, chunks in source_to_chunks.items():
        # Skip if source is empty
        if not source:
            chunks_to_process.extend(chunks)
            continue
            
        # Get file modification time
        if os.path.exists(source):
            file_mtime = os.path.getmtime(source)
            file_size = os.path.getsize(source)
            
            # Create a file signature that combines path, size and mtime
            file_signature = f"{source}:{file_size}:{file_mtime}"
            
            # Check if file has been modified since last processing
            if source in document_tracker and document_tracker[source]["signature"] == file_signature:
                print(f"Skipping unchanged file: {source}")
                files_processed += 1
                continue
            
            # File is new or modified, add to modified sources list
            print(f"File new or modified, will process: {source}")
            modified_sources.append(source)
            
            # Update tracker
            document_tracker[source] = {
                "signature": file_signature,
                "last_processed": time.time()
            }
        
        # Add chunks to processing list
        chunks_to_process.extend(chunks)
    
    print(f"Skipped {files_processed} unchanged files")
    print(f"Processing {len(chunks_to_process)} chunks from {len(modified_sources)} new or modified files")
    
    # If there's nothing to process, save tracker and return
    if not chunks_to_process:
        print("No new or modified files to process")
        with open(tracker_file, 'w') as f:
            json.dump(document_tracker, f, indent=2)
        return chroma_db
    
    # Remove old vectors from modified files
    if modified_sources and hasattr(chroma_db._collection, "get") and hasattr(chroma_db._collection, "delete"):
        try:
            print(f"Retrieving existing documents to identify those needing deletion...")
            # Check if collection is empty to avoid errors
            collection_count = chroma_db._collection.count()
            
            if collection_count > 0:
                # Get all documents in the collection
                collection_data = chroma_db._collection.get(include=['metadatas', 'documents', 'embeddings', 'ids'])
                all_metadatas = collection_data.get('metadatas', [])
                all_ids = collection_data.get('ids', [])
                
                # Find documents that match the modified sources
                ids_to_delete = []
                for i, metadata in enumerate(all_metadatas):
                    if metadata and "source" in metadata and metadata["source"] in modified_sources:
                        ids_to_delete.append(all_ids[i])
                
                # Delete the old documents
                if ids_to_delete:
                    print(f"Deleting {len(ids_to_delete)} old document vectors from modified files...")
                    chroma_db._collection.delete(ids=ids_to_delete)
                    print(f"Successfully deleted old vectors")
                else:
                    print(f"No existing vectors found for modified files")
        except Exception as e:
            print(f"Error removing old vectors: {e}")
            print("Will continue with adding new vectors")
    
    # Process all chunks
    print(f"Adding {len(chunks_to_process)} new document chunks...")
    for chunk in chunks_to_process:
        chroma_db.add_documents(documents=[chunk])
    
    # Save updated tracker
    with open(tracker_file, 'w') as f:
        json.dump(document_tracker, f, indent=2)
    
    # Persist database
    chroma_db.persist()
    
    print(f"Vector database updated with {len(chunks_to_process)} document chunks!")
    return chroma_db

def run_test():
    """Run a simple test of file modification detection and vector preservation"""
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    data_dir = os.path.join(temp_dir, "data")
    chroma_dir = os.path.join(temp_dir, "chroma")
    
    try:
        # Create data directory
        os.makedirs(data_dir, exist_ok=True)
        
        # Create test files
        test_file1 = os.path.join(data_dir, "test1.md")
        test_file2 = os.path.join(data_dir, "test2.md")
        
        with open(test_file1, "w") as f:
            f.write("# Test Document 1\n\nThis is test document 1.")
            
        with open(test_file2, "w") as f:
            f.write("# Test Document 2\n\nThis is test document 2.")
        
        # Create documents with metadata
        document1 = MockDocument("Test content 1", {"source": test_file1})
        document2 = MockDocument("Test content 2", {"source": test_file2})
        
        # First run - should process both files
        print("\n=== FIRST RUN (INITIAL PROCESSING) ===")
        db = create_vector_db_for_test([document1, document2], chroma_dir)
        
        # Check document count
        first_run_count = db._collection.count()
        print(f"Documents in DB after first run: {first_run_count}")
        
        # Second run with no changes - should skip both files
        print("\n=== SECOND RUN (NO CHANGES) ===")
        db = create_vector_db_for_test([document1, document2], chroma_dir)
        
        # Check document count - should be the same
        second_run_count = db._collection.count()
        print(f"Documents in DB after second run: {second_run_count}")
        assert first_run_count == second_run_count, "Document count changed despite no file changes"
        
        # Modify one file
        time.sleep(1)  # Ensure modification time is different
        with open(test_file1, "w") as f:
            f.write("# Modified Test Document 1\n\nThis document has been modified.")
        
        # Create updated document with metadata
        document1_modified = MockDocument("Modified content 1", {"source": test_file1})
        
        # Third run with one modified file - should process only the modified file
        print("\n=== THIRD RUN (ONE FILE MODIFIED) ===")
        db = create_vector_db_for_test([document1_modified, document2], chroma_dir)
        
        # Check document count - should be the same (replaced one document)
        third_run_count = db._collection.count()
        print(f"Documents in DB after third run: {third_run_count}")
        assert third_run_count == second_run_count, "Total document count incorrect after update"
        
        # Verify contents of the collection
        collection_data = db._collection.get(include=['metadatas', 'documents'])
        docs = collection_data.get('documents', [])
        metadatas = collection_data.get('metadatas', [])
        
        # Check if both documents exist
        modified_found = False
        original_found = False
        
        for i, doc in enumerate(docs):
            if "Modified content 1" in doc:
                modified_found = True
            if "Test content 2" in doc:
                original_found = True
        
        assert modified_found, "Modified document content not found in database"
        assert original_found, "Unchanged document not preserved in database"
        
        # Add a new file
        test_file3 = os.path.join(data_dir, "test3.md")
        with open(test_file3, "w") as f:
            f.write("# New Test Document 3\n\nThis is a newly added document.")
        
        # Create new document with metadata
        document3 = MockDocument("New content 3", {"source": test_file3})
        
        # Fourth run with a new file - should process only the new file
        print("\n=== FOURTH RUN (NEW FILE ADDED) ===")
        db = create_vector_db_for_test([document1_modified, document2, document3], chroma_dir)
        
        # Check document count - should have increased by one
        fourth_run_count = db._collection.count()
        print(f"Documents in DB after fourth run: {fourth_run_count}")
        assert fourth_run_count == third_run_count + 1, "New document not added correctly"
        
        print("\nTest completed successfully!")
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    run_test() 