import os
import time
import json
import shutil
import tempfile

print("Script started")

# Mock function that simulates the create_vector_db behavior for file change detection
def create_mock_vector_db(files, chroma_path):
    """
    This is a simplified version of create_vector_db that only tests
    the file change detection functionality.
    """
    print(f"create_mock_vector_db called with {len(files)} files")
    
    # Create directory if it doesn't exist
    os.makedirs(chroma_path, exist_ok=True)
    print(f"Directory created: {chroma_path}")
    
    # Define path for document tracking file
    tracker_file = os.path.join(chroma_path, "document_tracker.json")
    print(f"Tracker file path: {tracker_file}")
    
    # Initialize or load document tracker
    document_tracker = {}
    if os.path.exists(tracker_file):
        try:
            print(f"Loading existing tracker file")
            with open(tracker_file, 'r') as f:
                document_tracker = json.load(f)
            print(f"Loaded document tracker with {len(document_tracker)} entries")
        except Exception as e:
            print(f"Error loading document tracker: {e}")
            document_tracker = {}
    else:
        print("No existing tracker file found, creating new tracker")
    
    # Check which files have been modified
    print("Checking for new or modified files...")
    modified_sources = []
    files_skipped = 0
    files_processed = 0
    
    for source in files:
        print(f"Checking file: {source}")
        # Check if file exists and get metadata
        if os.path.exists(source):
            file_mtime = os.path.getmtime(source)
            file_size = os.path.getsize(source)
            
            # Create file signature that combines path, size and mtime
            file_signature = f"{source}:{file_size}:{file_mtime}"
            print(f"File signature: {file_signature}")
            
            # Check if file has been modified since last processing
            if source in document_tracker and document_tracker[source]["signature"] == file_signature:
                print(f"Skipping unchanged file: {source}")
                files_skipped += 1
                continue
            
            # File is new or modified
            print(f"File new or modified, will process: {source}")
            modified_sources.append(source)
            files_processed += 1
            
            # Update tracker for this file
            document_tracker[source] = {
                "signature": file_signature,
                "last_processed": time.time()
            }
            print(f"Updated tracker for: {source}")
        else:
            print(f"Warning: File does not exist: {source}")
    
    print(f"Skipped {files_skipped} unchanged files")
    print(f"Found {len(modified_sources)} new or modified files")
    
    # Save tracker
    print(f"Saving tracker file to {tracker_file}")
    with open(tracker_file, 'w') as f:
        json.dump(document_tracker, f, indent=2)
    print("Tracker saved successfully")
    
    result = {
        "processed": files_processed,
        "skipped": files_skipped,
        "modified_sources": modified_sources
    }
    print(f"Returning result: {result}")
    return result

def create_test_file(path, content):
    print(f"Creating test file: {path}")
    with open(path, 'w') as f:
        f.write(content)
    print(f"Test file created: {path}")
    return path

def test_file_change_detection():
    """Test file change detection functionality"""
    print("Starting test_file_change_detection function")
    
    # Create temporary test directory
    test_dir = tempfile.mkdtemp()
    print(f"Created temporary directory: {test_dir}")
    
    test_data_dir = os.path.join(test_dir, "test_data")
    test_chroma_dir = os.path.join(test_dir, "test_chroma")
    print(f"Data dir: {test_data_dir}, Chroma dir: {test_chroma_dir}")
    
    try:
        print("Entered try block")
        # Create test data directory
        os.makedirs(test_data_dir, exist_ok=True)
        print(f"Created test data directory: {test_data_dir}")
        
        # Create test files
        print("Creating test files...")
        file1_path = os.path.join(test_data_dir, "test1.md")
        create_test_file(file1_path, "# Test Document 1\nThis is the first test document.")
        
        file2_path = os.path.join(test_data_dir, "test2.md")
        create_test_file(file2_path, "# Test Document 2\nThis is the second test document.")
        print("Test files created successfully")
        
        print("\n=== TEST 1: Initial Processing ===")
        # Process both files
        print("Running initial processing test")
        result = create_mock_vector_db([file1_path, file2_path], test_chroma_dir)
        print(f"Initial processing result: {result}")
        
        assert result["processed"] == 2, f"Expected 2 files processed, got {result['processed']}"
        assert result["skipped"] == 0, f"Expected 0 files skipped, got {result['skipped']}"
        print("Initial processing assertions passed")
        
        # Verify tracker file was created
        tracker_file = os.path.join(test_chroma_dir, "document_tracker.json")
        print(f"Checking for tracker file: {tracker_file}")
        assert os.path.exists(tracker_file), "Tracker file was not created"
        print("Tracker file exists")
        
        # Check if tracker contains both files
        print("Checking tracker contents")
        with open(tracker_file, 'r') as f:
            tracker_data = json.load(f)
        
        assert file1_path in tracker_data, "First file not in tracker"
        assert file2_path in tracker_data, "Second file not in tracker"
        print("Tracker contains both files")
        
        print("Initial processing completed successfully")
        
        print("\n=== TEST 2: Skip Unchanged Files ===")
        # Process the same files again
        print("Running unchanged files test")
        result = create_mock_vector_db([file1_path, file2_path], test_chroma_dir)
        print(f"Unchanged files test result: {result}")
        
        assert result["processed"] == 0, f"Expected 0 files processed, got {result['processed']}"
        assert result["skipped"] == 2, f"Expected 2 files skipped, got {result['skipped']}"
        print("Unchanged files assertions passed")
        print("Second processing completed - files correctly skipped")
        
        print("\n=== TEST 3: Process Modified File ===")
        # Modify one file
        print("Starting modified file test")
        print("Waiting 1 second to ensure modification time changes...")
        time.sleep(1)  # Ensure modification time changes
        
        print(f"Modifying file: {file1_path}")
        create_test_file(file1_path, "# Modified Test Document 1\nThis document has been modified.")
        print("File modified successfully")
        
        # Process both files again
        print("Processing files after modification")
        result = create_mock_vector_db([file1_path, file2_path], test_chroma_dir)
        print(f"Modified file test result: {result}")
        
        assert result["processed"] == 1, f"Expected 1 file processed, got {result['processed']}"
        assert result["skipped"] == 1, f"Expected 1 file skipped, got {result['skipped']}"
        assert file1_path in result["modified_sources"], "Modified file not detected"
        print("Modified file assertions passed")
        print("Third processing completed - only modified file processed")
        
        print("\n=== TEST 4: Process New File ===")
        # Create a new file
        print("Starting new file test")
        file3_path = os.path.join(test_data_dir, "test3.md")
        create_test_file(file3_path, "# Test Document 3\nThis is a new test document.")
        print("New file created successfully")
        
        # Process all files
        print("Processing all files including new file")
        result = create_mock_vector_db([file1_path, file2_path, file3_path], test_chroma_dir)
        print(f"New file test result: {result}")
        
        assert result["processed"] == 1, f"Expected 1 file processed, got {result['processed']}"
        assert result["skipped"] == 2, f"Expected 2 files skipped, got {result['skipped']}"
        assert file3_path in result["modified_sources"], "New file not detected"
        print("New file assertions passed")
        
        # Check if tracker contains all files
        print("Checking tracker for all files")
        with open(tracker_file, 'r') as f:
            tracker_data = json.load(f)
        
        assert file1_path in tracker_data, "First file not in tracker"
        assert file2_path in tracker_data, "Second file not in tracker"
        assert file3_path in tracker_data, "Third file not in tracker"
        print("Tracker contains all three files")
        
        print("Fourth processing completed - only new file processed")
        print("\nAll tests completed successfully!")
        print("Test function completed successfully")
        return True
        
    except Exception as e:
        print(f"ERROR in test: {str(e)}")
        print(f"Exception type: {type(e).__name__}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False
        
    finally:
        # Clean up
        print(f"In finally block, cleaning up test environment: {test_dir}")
        try:
            shutil.rmtree(test_dir)
            print(f"Successfully cleaned up test directory: {test_dir}")
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")

print("At bottom of script file")
if __name__ == "__main__":
    print("Running test_file_change_detection from main")
    result = test_file_change_detection()
    print(f"Test result: {'SUCCESS' if result else 'FAILURE'}")
    print("Script execution completed") 