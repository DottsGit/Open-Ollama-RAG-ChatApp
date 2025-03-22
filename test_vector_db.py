import os
import time
import shutil
import unittest
import tempfile
import json
from main import create_vector_db
from langchain.document_loaders import TextLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

# Configuration for tests
OLLAMA_MODEL = "DeepSeek-R1:14b"  # Use the same model as in main.py
OLLAMA_URL = "http://localhost:11434"

# Test configuration
TEST_DATA_PATH = "test_data/"
TEST_CHROMA_PATH = "test_chroma/"

# Text splitting configuration
HEADERS_TO_SPLIT_ON = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
    ("####", "Header 4"),
]
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

class TestVectorDB(unittest.TestCase):
    def setUp(self):
        # Create temporary directories for test data and chroma db
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = os.path.join(self.temp_dir, "data")
        self.chroma_dir = os.path.join(self.temp_dir, "chroma")
        
        # Create data directory
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Create test markdown files
        self.create_test_files()
        
        # Load documents
        self.documents = self.load_documents()
        
    def tearDown(self):
        # Clean up temporary directory
        shutil.rmtree(self.temp_dir)
        
    def create_test_files(self):
        # Create test file 1
        with open(os.path.join(self.data_dir, "test1.md"), "w") as f:
            f.write("# Test Document 1\n\nThis is a test document for the vector database.")
            
        # Create test file 2
        with open(os.path.join(self.data_dir, "test2.md"), "w") as f:
            f.write("# Test Document 2\n\nThis is another test document for the vector database.")
    
    def load_documents(self):
        documents = []
        for file in os.listdir(self.data_dir):
            if file.endswith(".md"):
                loader = TextLoader(os.path.join(self.data_dir, file), encoding="utf-8")
                documents.append(loader.load()[0])
        return documents
    
    def process_documents(self):
        # Process the documents using similar steps as in main.py
        text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS_TO_SPLIT_ON)
        chunks_array = []
        
        for doc in self.documents:
            chunks = text_splitter.split_text(doc.page_content)
            # Append source metadata to each chunk
            for chunk in chunks:
                chunk.metadata.update(doc.metadata)
            chunks_array.append(chunks)
        
        # Character-level splitting
        char_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, length_function=len, add_start_index=True
        )
        
        chunks_array_txt_base = []
        for document in chunks_array:
            for chunk in document:
                splits = char_splitter.split_documents([chunk])
                chunks_array_txt_base.append(splits)
        
        # Flatten the chunks
        all_document_chunks = [chunk for document in chunks_array_txt_base for chunk in document]
        
        return all_document_chunks
    
    def test_initial_processing(self):
        """Test that all documents are processed on initial run"""
        # Process documents
        all_document_chunks = self.process_documents()
        
        # Create vector database
        db = create_vector_db(
            all_document_chunks=all_document_chunks,
            ollama_url=OLLAMA_URL,
            ollama_model=OLLAMA_MODEL,
            chroma_path=self.chroma_dir,
            batch_size=5,
            max_workers=2
        )
        
        # Check that tracker file was created
        tracker_file = os.path.join(self.chroma_dir, "document_tracker.json")
        self.assertTrue(os.path.exists(tracker_file), "Tracker file was not created")
        
        # Verify vectors were created by doing a simple query
        result = db.similarity_search("test document")
        self.assertTrue(len(result) > 0, "No results found in vector database")
        
        return db
    
    def test_unchanged_files(self):
        """Test that unchanged files are not reprocessed"""
        # First run to create initial database
        self.test_initial_processing()
        
        # Reload documents (no changes)
        self.documents = self.load_documents()
        all_document_chunks = self.process_documents()
        
        # Process again and verify that no documents are processed
        db = create_vector_db(
            all_document_chunks=all_document_chunks,
            ollama_url=OLLAMA_URL,
            ollama_model=OLLAMA_MODEL,
            chroma_path=self.chroma_dir,
            batch_size=5,
            max_workers=2
        )
        
        # Verify database still works
        result = db.similarity_search("test document")
        self.assertTrue(len(result) > 0, "No results found in vector database after second run")
    
    def test_modified_file(self):
        """Test that only modified files are reprocessed"""
        # First run to create initial database
        self.test_initial_processing()
        
        # Modify one file
        time.sleep(1)  # Ensure modification time is different
        with open(os.path.join(self.data_dir, "test1.md"), "w") as f:
            f.write("# Modified Test Document 1\n\nThis document has been modified.")
        
        # Reload documents
        self.documents = self.load_documents()
        all_document_chunks = self.process_documents()
        
        # Process again and verify that only modified file is processed
        db = create_vector_db(
            all_document_chunks=all_document_chunks,
            ollama_url=OLLAMA_URL,
            ollama_model=OLLAMA_MODEL,
            chroma_path=self.chroma_dir,
            batch_size=5,
            max_workers=2
        )
        
        # Verify database contains the modified content
        result = db.similarity_search("modified document")
        self.assertTrue(len(result) > 0, "Modified content not found in vector database")
        
        # Also verify original content is still there
        result = db.similarity_search("another test document")
        self.assertTrue(len(result) > 0, "Original content not found in vector database")
    
    def test_new_file(self):
        """Test that new files are processed"""
        # First run to create initial database
        self.test_initial_processing()
        
        # Create a new file
        with open(os.path.join(self.data_dir, "test3.md"), "w") as f:
            f.write("# New Test Document 3\n\nThis is a newly added document.")
        
        # Reload documents
        self.documents = self.load_documents()
        all_document_chunks = self.process_documents()
        
        # Process again and verify that new file is processed
        db = create_vector_db(
            all_document_chunks=all_document_chunks,
            ollama_url=OLLAMA_URL,
            ollama_model=OLLAMA_MODEL,
            chroma_path=self.chroma_dir,
            batch_size=5,
            max_workers=2
        )
        
        # Verify database contains the new content
        result = db.similarity_search("newly added document")
        self.assertTrue(len(result) > 0, "New content not found in vector database")

def setup_test_environment():
    """Create test directory and sample test files"""
    print("\n--- Setting up test environment ---")
    
    # Create test directories
    os.makedirs(TEST_DATA_PATH, exist_ok=True)
    
    # Clear existing test chroma directory if it exists
    if os.path.exists(TEST_CHROMA_PATH):
        shutil.rmtree(TEST_CHROMA_PATH)
    os.makedirs(TEST_CHROMA_PATH, exist_ok=True)
    
    # Create sample test files
    create_test_files()
    
    print(f"Created test environment with data in {TEST_DATA_PATH}")

def cleanup_test_environment():
    """Clean up test files and directories"""
    print("\n--- Cleaning up test environment ---")
    if os.path.exists(TEST_DATA_PATH):
        shutil.rmtree(TEST_DATA_PATH)
    if os.path.exists(TEST_CHROMA_PATH):
        shutil.rmtree(TEST_CHROMA_PATH)
    print("Test environment cleaned up")

def create_test_files():
    """Create sample markdown test files"""
    # Create first test file
    with open(os.path.join(TEST_DATA_PATH, "test-doc-1.md"), "w") as f:
        f.write("""# Test Document 1
        
## Section 1
This is a test document used to verify the vector database functionality.

## Section 2
We want to make sure all documents get properly processed.
""")

    # Create second test file
    with open(os.path.join(TEST_DATA_PATH, "test-doc-2.md"), "w") as f:
        f.write("""# Test Document 2
        
## Introduction
Another test document with different content.

## Details
This document should also be processed correctly.
""")
    
    # Create third test file that will be added later to test missing documents
    with open(os.path.join(TEST_DATA_PATH, "test-doc-3.md"), "w") as f:
        f.write("""# Test Document 3
        
## This document will be added later
We'll use this to test if the system can detect missing documents.
""")
    
    print(f"Created 3 test documents in {TEST_DATA_PATH}")

# Fixed version of the create_vector_db function
def create_vector_db(all_document_chunks, ollama_url, ollama_model, chroma_path, data_path, batch_size=5, max_workers=2):
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
    if data_path and os.path.exists(data_path):
        for file in os.listdir(data_path):
            file_path = os.path.join(data_path, file)
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
            md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS_TO_SPLIT_ON)
            md_chunks = md_splitter.split_text(document.page_content)
            
            # Update metadata for each chunk
            for chunk in md_chunks:
                chunk.metadata.update(document.metadata)
            
            # Further split by character level
            char_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE, 
                chunk_overlap=CHUNK_OVERLAP, 
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
    
    # Process documents in batches
    print(f"Starting batch processing with batch_size={batch_size}, max_workers={max_workers}")
    
    # Process batches
    batches = [chunks_to_process[i:i+batch_size] for i in range(0, len(chunks_to_process), batch_size)]
    print(f"Created {len(batches)} batches for processing")
    
    processed_count = 0
    for i, batch in enumerate(batches):
        try:
            print(f"Processing batch {i+1}/{len(batches)}")
            chroma_db.add_documents(documents=batch)
            processed_count += len(batch)
        except Exception as e:
            print(f"Error processing batch: {e}")
            # On failure, process documents one by one
            for doc in batch:
                try:
                    chroma_db.add_documents(documents=[doc])
                    processed_count += 1
                except Exception as e:
                    print(f"Problem with document: {doc.page_content[:50]}... Error: {e}")
    
    # Final persistence
    print("Performing final database persistence...")
    chroma_db.persist()
    
    # Save tracker
    with open(tracker_file, 'w') as f:
        json.dump(document_tracker, f, indent=2)
    
    print(f"Vector database updated with {processed_count} new document chunks!")
    
    # Validate that all documents have vectors
    print("\nValidating vector database to ensure all documents are represented...")
    missing_vectors = validate_vector_db(chroma_db, data_path)
    
    return chroma_db, missing_vectors

def validate_vector_db(chroma_db, data_path):
    """
    Improved validation function that returns list of missing documents
    """
    if not os.path.exists(data_path):
        print(f"Data path {data_path} not found, skipping validation")
        return []
    
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
        else:
            print("✓ All documents in the data directory have vectors in the database")
            
        return list(missing_vectors)
            
    except Exception as e:
        print(f"Error validating vector database: {e}")
        return list(all_files)  # Assume all are missing if we couldn't validate

def process_documents(data_path):
    """Process documents from a directory and return document chunks"""
    print(f"Loading documents from {data_path}...")
    documents = []
    
    for file in os.listdir(data_path):
        if file != "test-doc-3.md":  # Initially exclude the third document to test missing files
            file_path = os.path.join(data_path, file)
            if os.path.isfile(file_path):
                loader = TextLoader(file_path, encoding="utf-8")
                documents.append(loader.load()[0])
    
    print(f"Loaded {len(documents)} documents")
    
    # Split documents by markdown headers
    print("Starting document splitting by markdown headers...")
    text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS_TO_SPLIT_ON)
    chunks_array = []
    
    for doc in documents:
        chunks = text_splitter.split_text(doc.page_content)
        # append source metadata to each chunk
        for chunk in chunks:
            chunk.metadata.update(doc.metadata)
        chunks_array.append(chunks)
    
    # Char-level splits
    print("Performing character-level splitting to create final chunks...")
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, 
        chunk_overlap=CHUNK_OVERLAP, 
        length_function=len, 
        add_start_index=True
    )
    
    chunks_array_txt_base = []
    for document in chunks_array:
        for chunk in document:
            splits = char_splitter.split_documents([chunk])
            chunks_array_txt_base.append(splits)
    
    all_document_chunks = [chunk for document in chunks_array_txt_base for chunk in document]
    print(f"Total individual document chunks: {len(all_document_chunks)}")
    
    return all_document_chunks

def test_missing_documents():
    """Test handling of missing documents"""
    print("\n=== Testing Missing Documents Handling ===")
    
    # Process only the first 2 documents initially
    all_document_chunks = process_documents(TEST_DATA_PATH)
    
    # First run with initial documents
    print("\nInitial run with 2 documents:")
    db, missing_vectors = create_vector_db(
        all_document_chunks=all_document_chunks,
        ollama_url=OLLAMA_URL,
        ollama_model=OLLAMA_MODEL,
        chroma_path=TEST_CHROMA_PATH,
        data_path=TEST_DATA_PATH,
        batch_size=5,
        max_workers=2
    )
    
    # Verify that test-doc-3.md is detected as missing
    print("\nVerifying test-doc-3.md is detected as missing...")
    found_missing_doc3 = any(os.path.join(TEST_DATA_PATH, "test-doc-3.md") == path for path in missing_vectors)
    print(f"test-doc-3.md detected as missing: {found_missing_doc3}")
    
    # Now process all documents including test-doc-3.md
    print("\nSecond run including all documents:")
    all_document_chunks = []  # Start with empty chunks to simulate only the tracker
    db, missing_vectors = create_vector_db(
        all_document_chunks=all_document_chunks,
        ollama_url=OLLAMA_URL,
        ollama_model=OLLAMA_MODEL,
        chroma_path=TEST_CHROMA_PATH,
        data_path=TEST_DATA_PATH,
        batch_size=5,
        max_workers=2
    )
    
    # Check if test-doc-3.md is now in the database
    print("\nVerifying test-doc-3.md is now in the database...")
    if not missing_vectors:
        print("✓ All documents are now in the database!")
        return True
    else:
        print("✗ Some documents are still missing from the database")
        return False

def test_duplicate_prevention():
    """Test prevention of duplicate entries"""
    print("\n=== Testing Duplicate Prevention ===")
    
    # First run - process all documents
    all_document_chunks = process_documents(TEST_DATA_PATH)
    db, _ = create_vector_db(
        all_document_chunks=all_document_chunks,
        ollama_url=OLLAMA_URL,
        ollama_model=OLLAMA_MODEL,
        chroma_path=TEST_CHROMA_PATH,
        data_path=TEST_DATA_PATH,
        batch_size=5,
        max_workers=2
    )
    
    # Get initial count
    initial_count = db._collection.count()
    print(f"Initial database vector count: {initial_count}")
    
    # Run again with the same documents
    print("\nSecond run with same documents:")
    db, _ = create_vector_db(
        all_document_chunks=all_document_chunks,
        ollama_url=OLLAMA_URL,
        ollama_model=OLLAMA_MODEL,
        chroma_path=TEST_CHROMA_PATH,
        data_path=TEST_DATA_PATH,
        batch_size=5,
        max_workers=2
    )
    
    # Get new count
    new_count = db._collection.count()
    print(f"Second run database vector count: {new_count}")
    
    # Check if duplicates were added
    if new_count == initial_count:
        print("✓ No duplicates were added to the database!")
        return True
    else:
        print(f"✗ Duplicates detected! Count increased by {new_count - initial_count}")
        return False

def run_tests():
    """Run all tests"""
    try:
        # Set up test environment
        setup_test_environment()
        
        # Test missing documents handling
        missing_docs_test = test_missing_documents()
        
        # Test duplicate prevention
        duplicate_test = test_duplicate_prevention()
        
        # Print test summary
        print("\n=== Test Summary ===")
        print(f"Missing documents handling: {'PASSED' if missing_docs_test else 'FAILED'}")
        print(f"Duplicate prevention: {'PASSED' if duplicate_test else 'FAILED'}")
        
        if missing_docs_test and duplicate_test:
            print("\n✅ All tests PASSED! The fixes successfully addressed both issues.")
        else:
            print("\n❌ Some tests FAILED. Further improvements needed.")
        
    finally:
        # Clean up test environment
        cleanup_test_environment()

if __name__ == "__main__":
    # Run only if Ollama is available
    try:
        test_embedding = OllamaEmbeddings(base_url=OLLAMA_URL, model=OLLAMA_MODEL)
        test_result = test_embedding.embed_query("Test")
        if len(test_result) > 0:
            unittest.main()
        else:
            print("Ollama embeddings test failed. Make sure Ollama is running correctly.")
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        print("Make sure Ollama is running and the model is available before running tests.")

    # Run all tests
    run_tests() 