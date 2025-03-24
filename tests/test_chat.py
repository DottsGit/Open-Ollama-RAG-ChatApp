import pytest
from src.vector_db import load_vector_db
import os

def test_chat_query():
    """
    Test that the vector database can retrieve documents
    """
    # Define a test query
    query = "Test query"
    k = 3
    
    try:
        # Load the vector database
        db = load_vector_db()
        
        # Perform similarity search
        docs = db.similarity_search(query, k=k)
        
        # Assert that we got some documents back
        assert len(docs) > 0, "Expected to get at least one document back"
        
        # Check that each document has the expected structure
        for doc in docs:
            assert hasattr(doc, 'page_content'), "Document should have page_content"
            assert hasattr(doc, 'metadata'), "Document should have metadata"
            assert 'source' in doc.metadata, "Document metadata should have source"
    
    except Exception as e:
        pytest.fail(f"Error during query: {str(e)}")

if __name__ == "__main__":
    # Test more specific queries to see if we can get 104-10326-10014.md
    queries = [
        "What is in document 104-10326-10014.md?",
        "Information about JFK in 104-10326-10014",
        "Tell me about CIA document 104-10326-10014"
    ]
    
    for query in queries:
        print("\n" + "="*80)
        test_chat_query()
        print("="*80 + "\n") 