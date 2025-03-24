"""
Test the document processor functionality.

This test ensures that documents are correctly loaded and processed.
"""

import os
import tempfile
import pytest
import shutil
from unittest.mock import patch

from langchain.schema.document import Document

from src.document_processor import (
    load_documents,
    process_documents,
    normalize_document_name
)

class TestDocumentProcessor:
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Set up the test environment before each test and clean up after."""
        # Create a temporary directory for test data
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = os.path.join(self.temp_dir, "data")
        
        # Create the data directory
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Create test files
        self.create_test_files()
        
        yield
        
        # Clean up
        shutil.rmtree(self.temp_dir)
    
    def create_test_files(self):
        """Create test markdown files."""
        # Create a test markdown file
        with open(os.path.join(self.data_dir, "test1.md"), "w") as f:
            f.write("""# Test Document 1
            
## Section 1
This is a test document.

## Section 2
This document is used for testing the document processor.
""")

        # Create a second markdown file
        with open(os.path.join(self.data_dir, "test2.md"), "w") as f:
            f.write("""# Test Document 2
            
## Introduction
Another test document.

## Details
With multiple sections.

### Subsection
And subsections.
""")
    
    def test_load_documents(self):
        """Test that documents are correctly loaded from the data directory."""
        # Load documents
        documents = load_documents(self.data_dir)
        
        # Check that both documents were loaded
        assert len(documents) == 2
        
        # Check that the documents have the correct metadata
        sources = [doc.metadata["source"] for doc in documents]
        assert os.path.join(self.data_dir, "test1.md") in sources
        assert os.path.join(self.data_dir, "test2.md") in sources
        
        # Check that the documents have the correct content
        for doc in documents:
            if "test1.md" in doc.metadata["source"]:
                assert "Test Document 1" in doc.page_content
                assert "Section 1" in doc.page_content
            elif "test2.md" in doc.metadata["source"]:
                assert "Test Document 2" in doc.page_content
                assert "Introduction" in doc.page_content
    
    def test_process_documents(self):
        """Test that documents are correctly processed into chunks."""
        # Create test documents
        doc1 = Document(
            page_content="""# Test Document 1
            
## Section 1
This is a test document.

## Section 2
This document is used for testing the document processor.""",
            metadata={"source": os.path.join(self.data_dir, "test1.md")}
        )
        
        doc2 = Document(
            page_content="""# Test Document 2
            
## Introduction
Another test document.

## Details
With multiple sections.

### Subsection
And subsections.""",
            metadata={"source": os.path.join(self.data_dir, "test2.md")}
        )
        
        documents = [doc1, doc2]
        
        # Process documents
        document_chunks = process_documents(documents)
        
        # Check that chunks were created
        assert len(document_chunks) > 0
        
        # Check that each chunk has the correct metadata
        for chunk in document_chunks:
            assert "source" in chunk.metadata
            assert chunk.metadata["source"] in [os.path.join(self.data_dir, "test1.md"), os.path.join(self.data_dir, "test2.md")]
    
    def test_normalize_document_name(self):
        """Test that document names are correctly normalized."""
        # Test with a normal file path
        file_path = os.path.join("path", "to", "document.md")
        assert normalize_document_name(file_path) == file_path
        
        # Test with a file path containing a suffix
        file_path = os.path.join("path", "to", "document__123456.md")
        normalized_path = normalize_document_name(file_path)
        assert os.path.basename(normalized_path) == "document.md"
        assert os.path.dirname(normalized_path) == os.path.join("path", "to")
        
        # Test with a file path containing multiple underscores
        file_path = os.path.join("path", "to", "document__123__456.md")
        normalized_path = normalize_document_name(file_path)
        assert os.path.basename(normalized_path) == "document.md"
        assert os.path.dirname(normalized_path) == os.path.join("path", "to") 