from src.vector_db import load_vector_db
import os

def test_chat_query(query, k=3):
    """
    Simulate a chat query to test retrieval capabilities
    """
    try:
        # Load the vector database
        db = load_vector_db()
        
        # Perform similarity search
        docs = db.similarity_search(query, k=k)
        
        print(f"Query: {query}\n")
        print("Retrieved Documents:")
        
        found_target = False
        
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "Unknown")
            content = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            
            print(f"\nDocument {i+1} from: {os.path.basename(source)}")
            print(f"Content: {content}\n")
            
            if "104-10326-10014" in source:
                found_target = True
                print("*** TARGET DOCUMENT FOUND ***")
        
        if not found_target:
            print("\n*** TARGET DOCUMENT NOT FOUND IN RESULTS ***")
        
        print("\nSimulated Answer:")
        print("Based on the retrieved documents, I would generate an answer here.")
    
    except Exception as e:
        print(f"Error during query: {str(e)}")

if __name__ == "__main__":
    # Test more specific queries to see if we can get 104-10326-10014.md
    queries = [
        "What is in document 104-10326-10014.md?",
        "Information about JFK in 104-10326-10014",
        "Tell me about CIA document 104-10326-10014"
    ]
    
    for query in queries:
        print("\n" + "="*80)
        test_chat_query(query)
        print("="*80 + "\n") 