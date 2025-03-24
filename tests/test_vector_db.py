"""
Test the vector database functionality.

This test ensures that the vector database correctly processes and stores documents.
"""

import os
import time
import shutil
import tempfile
import pytest
import json
from unittest.mock import patch, MagicMock

from langchain.schema.document import Document

from src.config import HEADERS_TO_SPLIT_ON, CHUNK_SIZE, CHUNK_OVERLAP
from src.document_processor import process_documents
from src.vector_db import (
    get_embeddings, 
    load_vector_db,
    create_vector_db,
    validate_vector_db
)

class TestVectorDB:
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Set up the test environment before each test and clean up after."""
        # Create temporary directories for test data and chroma db
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = os.path.join(self.temp_dir, "data")
        self.chroma_dir = os.path.join(self.temp_dir, "chroma")
        
        # Create directories
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.chroma_dir, exist_ok=True)
        
        # Create test files
        self.create_test_files()
        
        # Mock the embeddings and Chroma database
        self.mock_embeddings = MagicMock()
        self.mock_chroma = MagicMock()
        
        # Create a patcher for get_embeddings
        self.embeddings_patcher = patch('src.vector_db.get_embeddings', return_value=self.mock_embeddings)
        self.embeddings_patcher.start()
        
        # Create a patcher for load_vector_db
        self.vector_db_patcher = patch('src.vector_db.load_vector_db', return_value=self.mock_chroma)
        self.vector_db_patcher.start()
        
        # Create a patcher for Chroma.from_documents
        self.chroma_patcher = patch('langchain_community.vectorstores.Chroma.from_documents', return_value=self.mock_chroma)
        self.chroma_patcher.start()
        
        yield
        
        # Stop patchers
        self.embeddings_patcher.stop()
        self.vector_db_patcher.stop()
        self.chroma_patcher.stop()
        
        # Clean up the temporary directory
        shutil.rmtree(self.temp_dir)
    
    def create_test_files(self):
        """Create test markdown files."""
        # Create test file 1
        with open(os.path.join(self.data_dir, "test1.md"), "w") as f:
            f.write("# Test Document 1\n\nThis is a test document for the vector database.")
            
        # Create test file 2
        with open(os.path.join(self.data_dir, "test2.md"), "w") as f:
            f.write("# Test Document 2\n\nThis is another test document for the vector database.")
    
    def create_test_documents(self):
        """Create test document objects."""
        documents = []
        
        # Create Document 1
        doc1 = Document(
            page_content="# Test Document 1\n\nThis is a test document for the vector database.",
            metadata={"source": os.path.join(self.data_dir, "test1.md")}
        )
        documents.append(doc1)
        
        # Create Document 2
        doc2 = Document(
            page_content="# Test Document 2\n\nThis is another test document for the vector database.",
            metadata={"source": os.path.join(self.data_dir, "test2.md")}
        )
        documents.append(doc2)
        
        return documents
    
    def test_initial_processing(self):
        """Test that all documents are processed on initial run."""
        # Create test documents
        documents = self.create_test_documents()
        
        # Process documents
        all_document_chunks = process_documents(documents)
        
        # Create a mock tracker for the return value
        mock_tracker = {
            os.path.join(self.data_dir, "test1.md"): {
                "signature": "test_signature1",
                "last_processed": time.time()
            },
            os.path.join(self.data_dir, "test2.md"): {
                "signature": "test_signature2",
                "last_processed": time.time()
            }
        }
        
        # Mock the function that processes files
        with patch('src.vector_db.create_vector_db', return_value=(self.mock_chroma, mock_tracker)):
            # Create vector database
            db, tracker = create_vector_db(
                all_document_chunks=all_document_chunks,
                chroma_path=self.chroma_dir,
                data_path=self.data_dir
            )
        
        # Check that all sources were processed
        assert len(tracker) == 2, "Tracker should have 2 entries"
        
        # Check that tracker contains both files
        assert os.path.join(self.data_dir, "test1.md") in tracker
        assert os.path.join(self.data_dir, "test2.md") in tracker
    
    def test_unchanged_files(self):
        """Test that unchanged files are not reprocessed."""
        # Create test documents
        documents = self.create_test_documents()
        
        # Process documents
        all_document_chunks = process_documents(documents)
        
        # Create tracker file with current file signatures
        tracker = {}
        for doc in documents:
            source = doc.metadata["source"]
            file_mtime = os.path.getmtime(source)
            file_size = os.path.getsize(source)
            file_signature = f"{source}:{file_size}:{file_mtime}"
            
            tracker[source] = {
                "signature": file_signature,
                "last_processed": time.time()
            }
        
        # Save tracker
        tracker_file = os.path.join(self.chroma_dir, "document_tracker.json")
        with open(tracker_file, 'w') as f:
            json.dump(tracker, f, indent=2)
        
        # Mock Chroma.add_texts to just return
        self.mock_chroma.add_texts.return_value = None
        
        # Reset mock
        self.mock_chroma.add_texts.reset_mock()
        
        # Create vector database
        db, tracker = create_vector_db(
            all_document_chunks=all_document_chunks,
            chroma_path=self.chroma_dir,
            data_path=self.data_dir
        )
        
        # Check that add_texts was not called (files unchanged)
        assert not self.mock_chroma.add_texts.called
    
    def test_modified_file(self):
        """Test that only modified files are reprocessed."""
        # Create test documents
        documents = self.create_test_documents()
        
        # Process documents
        all_document_chunks = process_documents(documents)
        
        # Create tracker file with current file signatures
        tracker = {}
        for doc in documents:
            source = doc.metadata["source"]
            file_mtime = os.path.getmtime(source)
            file_size = os.path.getsize(source)
            file_signature = f"{source}:{file_size}:{file_mtime}"
            
            tracker[source] = {
                "signature": file_signature,
                "last_processed": time.time()
            }
        
        # Save tracker
        tracker_file = os.path.join(self.chroma_dir, "document_tracker.json")
        with open(tracker_file, 'w') as f:
            json.dump(tracker, f, indent=2)
        
        # Modify one file
        time.sleep(1)  # Ensure modification time is different
        with open(os.path.join(self.data_dir, "test1.md"), "w") as f:
            f.write("# Modified Test Document 1\n\nThis document has been modified.")
        
        # Mock Chroma.add_texts to just return
        self.mock_chroma.add_texts.return_value = None
        
        # Reset mock
        self.mock_chroma.add_texts.reset_mock()
        
        # Create vector database
        db, new_tracker = create_vector_db(
            all_document_chunks=all_document_chunks,
            chroma_path=self.chroma_dir,
            data_path=self.data_dir
        )
        
        # Check that add_texts was called (one file modified)
        assert self.mock_chroma.add_texts.called
        
        # Check that the tracker was updated
        assert new_tracker[os.path.join(self.data_dir, "test1.md")]["signature"] != tracker[os.path.join(self.data_dir, "test1.md")]["signature"]
        assert new_tracker[os.path.join(self.data_dir, "test2.md")]["signature"] == tracker[os.path.join(self.data_dir, "test2.md")]["signature"]
    
    def test_validate_vector_db(self):
        """Test that validate_vector_db correctly identifies missing files."""
        # Mock Chroma.get to return a list of metadata
        self.mock_chroma.get.return_value = {
            "metadatas": [
                {"source": os.path.join(self.data_dir, "test1.md")}
            ]
        }
        
        # Call validate_vector_db
        missing_files = validate_vector_db(self.mock_chroma, self.data_dir)
        
        # Should find test2.md as missing
        assert len(missing_files) == 1
        assert os.path.join(self.data_dir, "test2.md") in missing_files 