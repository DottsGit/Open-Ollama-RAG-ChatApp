from src.vector_db import load_vector_db

def main():
    # Load the vector database
    db = load_vector_db()
    
    # Get all document chunks
    all_docs = db.get()
    
    # Search for chunks from the specific file
    target_file = "104-10326-10014__c06931192_.md"
    found = False
    
    print(f"Searching for chunks from file containing: {target_file}\n")
    
    for i, metadata in enumerate(all_docs["metadatas"]):
        if metadata and "source" in metadata and target_file in metadata["source"]:
            source = metadata["source"]
            content = all_docs["documents"][i]
            
            print(f"\nSource: {source}")
            print(f"Content: {content[:200]}..." if len(content) > 200 else content)
            found = True
    
    if not found:
        print(f"No chunks found containing {target_file}")

if __name__ == "__main__":
    main() 