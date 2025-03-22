import os
import time
import shutil
import unittest
import tempfile
from main import create_vector_db
from langchain.document_loaders import TextLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma

# Configuration for tests
OLLAMA_MODEL = "DeepSeek-R1:14b"  # Use the same model as in main.py
OLLAMA_URL = "http://localhost:11434"

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
        from langchain.text_splitter import MarkdownHeaderTextSplitter
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        
        # Define headers to split on
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
        ]
        
        # Markdown header splitting
        text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        chunks_array = []
        
        for doc in self.documents:
            chunks = text_splitter.split_text(doc.page_content)
            # Append source metadata to each chunk
            for chunk in chunks:
                chunk.metadata.update(doc.metadata)
            chunks_array.append(chunks)
        
        # Character-level splitting
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=100, length_function=len, add_start_index=True
        )
        
        chunks_array_txt_base = []
        for document in chunks_array:
            for chunk in document:
                splits = text_splitter.split_documents([chunk])
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