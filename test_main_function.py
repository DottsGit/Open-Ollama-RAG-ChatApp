import os
import time
import json
import shutil
import tempfile
from langchain.document_loaders import TextLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from main import create_vector_db
from langchain.embeddings import OllamaEmbeddings

# Define test configuration
class TestConfig:
    def __init__(self):
        # Set up temporary test directories
        self.test_dir = tempfile.mkdtemp()
        self.test_data_dir = os.path.join(self.test_dir, "test_data")
        self.test_chroma_dir = os.path.join(self.test_dir, "test_chroma")
        
        # Create directories
        os.makedirs(self.test_data_dir, exist_ok=True)
        
        # Set up test parameters
        self.ollama_url = "http://localhost:11434"
        self.ollama_model = "DeepSeek-R1:14b"
        
        # Markdown header settings for text splitting
        self.headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
        ]
        
        # Chunk settings
        self.chunk_size = 500
        self.chunk_overlap = 100
        
        print(f"Created test environment at {self.test_dir}")
        
    def cleanup(self):
        print(f"Cleaning up test environment: {self.test_dir}")
        shutil.rmtree(self.test_dir)

def create_test_file(path, content):
    with open(path, 'w') as f:
        f.write(content)
    return path

def process_test_files(files_paths, config):
    """Process test files using similar approach as in main.py"""
    # Load documents
    documents = []
    for file_path in files_paths:
        loader = TextLoader(file_path, encoding="utf-8")
        documents.append(loader.load()[0])
    
    print(f"Loaded {len(documents)} test documents")
    
    # Define splitters
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=config.headers_to_split_on)
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        length_function=len,
        add_start_index=True
    )
    
    # Process documents similar to main.py
    chunks_array = []
    for doc in documents:
        # Split by markdown headers
        chunks = md_splitter.split_text(doc.page_content)
        # Update metadata
        for chunk in chunks:
            chunk.metadata.update(doc.metadata)
        chunks_array.append(chunks)
    
    # Character-level splitting
    chunks_array_txt_base = []
    for document in chunks_array:
        for chunk in document:
            splits = char_splitter.split_documents([chunk])
            chunks_array_txt_base.append(splits)
    
    # Flatten the chunks
    all_document_chunks = [chunk for document in chunks_array_txt_base for chunk in document]
    print(f"Created {len(all_document_chunks)} document chunks")
    
    return all_document_chunks

def test_create_vector_db():
    """Test create_vector_db with a small set of test files"""
    # Create test files
    file1_path = os.path.join(config.test_data_dir, "test1.md")
    create_test_file(file1_path, """# Test Document 1
This is the first test document.

## Section 1
This is a section in the first document.

## Section 2
This is another section in the first document.
""")

    file2_path = os.path.join(config.test_data_dir, "test2.md")
    create_test_file(file2_path, """# Test Document 2
This is the second test document.

## Section A
This is a section in the second document.

## Section B
This is another section in the second document.
""")
    
    print("\n=== TEST 1: Initial Processing ===")
    # Process both files
    all_document_chunks = process_test_files([file1_path, file2_path], config)
    
    # Create vector database
    db = create_vector_db(
        all_document_chunks=all_document_chunks,
        ollama_url=config.ollama_url,
        ollama_model=config.ollama_model,
        chroma_path=config.test_chroma_dir,
        batch_size=5,
        max_workers=2
    )
    
    # Verify tracker file was created
    tracker_file = os.path.join(config.test_chroma_dir, "document_tracker.json")
    assert os.path.exists(tracker_file), "Tracker file was not created"
    
    # Check if tracker contains both files
    with open(tracker_file, 'r') as f:
        tracker_data = json.load(f)
        assert file1_path in tracker_data, "First file not in tracker"
        assert file2_path in tracker_data, "Second file not in tracker"
    
    print("Initial processing completed successfully")
    
    print("\n=== TEST 2: Skip Unchanged Files ===")
    # Process the same files again
    all_document_chunks = process_test_files([file1_path, file2_path], config)
    
    # Create vector database again
    db = create_vector_db(
        all_document_chunks=all_document_chunks,
        ollama_url=config.ollama_url,
        ollama_model=config.ollama_model,
        chroma_path=config.test_chroma_dir,
        batch_size=5,
        max_workers=2
    )
    
    print("Second processing completed - files should have been skipped")
    
    print("\n=== TEST 3: Process Modified File ===")
    # Modify one file
    time.sleep(1)  # Ensure modification time changes
    create_test_file(file1_path, """# Modified Test Document 1
This document has been modified.

## New Section
This is a new section in the modified document.

## Updated Section
This section was updated.
""")
    
    # Process both files again
    all_document_chunks = process_test_files([file1_path, file2_path], config)
    
    # Create vector database again
    db = create_vector_db(
        all_document_chunks=all_document_chunks,
        ollama_url=config.ollama_url,
        ollama_model=config.ollama_model,
        chroma_path=config.test_chroma_dir,
        batch_size=5,
        max_workers=2
    )
    
    print("Third processing completed - only modified file should have been processed")
    
    print("\n=== TEST 4: Process New File ===")
    # Create a new file
    file3_path = os.path.join(config.test_data_dir, "test3.md")
    create_test_file(file3_path, """# Test Document 3
This is a new test document.

## Section X
This is a section in the new document.

## Section Y
This is another section in the new document.
""")
    
    # Process all files
    all_document_chunks = process_test_files([file1_path, file2_path, file3_path], config)
    
    # Create vector database again
    db = create_vector_db(
        all_document_chunks=all_document_chunks,
        ollama_url=config.ollama_url,
        ollama_model=config.ollama_model,
        chroma_path=config.test_chroma_dir,
        batch_size=5,
        max_workers=2
    )
    
    # Check if tracker contains all files
    with open(tracker_file, 'r') as f:
        tracker_data = json.load(f)
        assert file1_path in tracker_data, "First file not in tracker"
        assert file2_path in tracker_data, "Second file not in tracker"
        assert file3_path in tracker_data, "Third file not in tracker"
    
    print("Fourth processing completed - only new file should have been processed")
    print("\nAll tests completed successfully!")

# Run test with try-finally to ensure cleanup
if __name__ == "__main__":
    config = TestConfig()
    try:
        print("Running tests for create_vector_db function")
        test_create_vector_db()
    finally:
        config.cleanup() 