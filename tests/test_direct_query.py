from src.vector_db import load_vector_db
import os

def get_document_by_name(target_file, limit=10):
    """
    Retrieve a document directly by name/ID rather than similarity search
    
    Args:
        target_file: Part of the filename to search for
        limit: Maximum number of documents to return
    """
    # Load the vector database
    db = load_vector_db()
    
    # Get all documents
    all_docs = db.get()
    
    # Find documents matching the target file
    found_docs = []
    
    for i, metadata in enumerate(all_docs["metadatas"]):
        if metadata and "source" in metadata and target_file in metadata["source"]:
            source = metadata["source"]
            content = all_docs["documents"][i]
            found_docs.append((source, content))
            
            if len(found_docs) >= limit:
                break
    
    return found_docs

def main():
    # Target file to find
    target_file = "104-10326-10014.md"
    
    print(f"Searching for documents with filename containing: {target_file}\n")
    
    # Get documents
    found_docs = get_document_by_name(target_file)
    
    if not found_docs:
        print(f"No documents found containing {target_file}")
        return
    
    print(f"Found {len(found_docs)} documents")
    
    # Display the documents
    for i, (source, content) in enumerate(found_docs):
        print(f"\nDocument {i+1} from: {os.path.basename(source)}")
        print(f"Content: {content[:200]}..." if len(content) > 200 else content)
    
    print("\nRecommendation:")
    print("To improve retrieval of this document, consider:")
    print("1. Implementing a keyword-based fallback strategy when queries mention document IDs")
    print("2. Adding metadata filtering to the query process to search by document ID")
    print("3. Updating the chat interface to first check for document ID patterns before semantic search")

if __name__ == "__main__":
    main() 