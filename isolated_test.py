import os
import time
import json
import shutil
import tempfile
import sys

# Create a completely separate test that doesn't touch the main data
def test_file_change_detection():
    print("Starting file change detection test...")
    
    # Create a temporary directory for all test files
    temp_dir = tempfile.mkdtemp()
    test_data_dir = os.path.join(temp_dir, "test_data")
    test_chroma_dir = os.path.join(temp_dir, "test_chroma")
    
    # Create our test directories
    os.makedirs(test_data_dir, exist_ok=True)
    os.makedirs(test_chroma_dir, exist_ok=True)
    
    # Path for tracker file
    tracker_file = os.path.join(test_chroma_dir, "document_tracker.json")
    
    try:
        # Create document tracker
        document_tracker = {}
        
        # TEST 1: Initial processing of a new file
        print("\n=== TEST 1: Initial Processing ===")
        
        # Create a test file
        file1_path = os.path.join(test_data_dir, "test1.md")
        with open(file1_path, 'w') as f:
            f.write("# Test Document 1\nThis is a test document.")
        
        # Get file metadata
        file1_mtime = os.path.getmtime(file1_path)
        file1_size = os.path.getsize(file1_path)
        file1_signature = f"{file1_path}:{file1_size}:{file1_mtime}"
        
        # Process the file (in real code, this would compute embeddings)
        print(f"Processing new file: {file1_path}")
        document_tracker[file1_path] = {
            "signature": file1_signature,
            "last_processed": time.time()
        }
        
        # Save tracker
        with open(tracker_file, 'w') as f:
            json.dump(document_tracker, f, indent=2)
        
        print(f"Test 1 completed: File processed and tracker created.")
        
        # TEST 2: Verify unchanged file is skipped
        print("\n=== TEST 2: Skip Unchanged File ===")
        
        # Load tracker
        with open(tracker_file, 'r') as f:
            document_tracker = json.load(f)
        
        # Check if file has changed
        file1_mtime = os.path.getmtime(file1_path)
        file1_size = os.path.getsize(file1_path)
        file1_signature_new = f"{file1_path}:{file1_size}:{file1_mtime}"
        
        if file1_path in document_tracker and document_tracker[file1_path]["signature"] == file1_signature_new:
            print(f"Correctly skipped unchanged file: {file1_path}")
            print("Test 2 PASSED")
        else:
            print(f"ERROR: Test 2 FAILED - File should be skipped but was detected as changed!")
            print(f"  Old signature: {document_tracker[file1_path]['signature']}")
            print(f"  New signature: {file1_signature_new}")
            return False
        
        # TEST 3: Process a modified file
        print("\n=== TEST 3: Process Modified File ===")
        
        # Wait to ensure modification time changes
        time.sleep(1)
        
        # Modify the file
        with open(file1_path, 'w') as f:
            f.write("# Modified Test Document 1\nThis file has been modified.")
        
        # Check if file has changed
        file1_mtime = os.path.getmtime(file1_path)
        file1_size = os.path.getsize(file1_path)
        file1_signature_modified = f"{file1_path}:{file1_size}:{file1_mtime}"
        
        # Verify file is detected as changed
        if file1_path in document_tracker and document_tracker[file1_path]["signature"] == file1_signature_modified:
            print(f"ERROR: Test 3 FAILED - Modified file not detected as changed!")
            return False
        else:
            print(f"Correctly detected modified file: {file1_path}")
            
            # Process the modified file
            print(f"Processing modified file: {file1_path}")
            document_tracker[file1_path] = {
                "signature": file1_signature_modified,
                "last_processed": time.time()
            }
            
            # Save tracker
            with open(tracker_file, 'w') as f:
                json.dump(document_tracker, f, indent=2)
            
            print("Test 3 PASSED")
        
        # TEST 4: Process only new files when multiple files exist
        print("\n=== TEST 4: Process Only New Files ===")
        
        # Create a second test file
        file2_path = os.path.join(test_data_dir, "test2.md")
        with open(file2_path, 'w') as f:
            f.write("# Test Document 2\nThis is another test document.")
        
        # Process both files
        files_processed = 0
        files_skipped = 0
        
        # Check first file
        file1_mtime = os.path.getmtime(file1_path)
        file1_size = os.path.getsize(file1_path)
        file1_signature_current = f"{file1_path}:{file1_size}:{file1_mtime}"
        
        if file1_path in document_tracker and document_tracker[file1_path]["signature"] == file1_signature_current:
            print(f"Correctly skipped unchanged file: {file1_path}")
            files_skipped += 1
        else:
            print(f"ERROR: Modified file incorrectly detected as changed")
            files_processed += 1
            document_tracker[file1_path] = {
                "signature": file1_signature_current,
                "last_processed": time.time()
            }
        
        # Check second file
        file2_mtime = os.path.getmtime(file2_path)
        file2_size = os.path.getsize(file2_path)
        file2_signature = f"{file2_path}:{file2_size}:{file2_mtime}"
        
        if file2_path in document_tracker and document_tracker[file2_path]["signature"] == file2_signature:
            print(f"ERROR: New file incorrectly detected as existing!")
            files_skipped += 1
        else:
            print(f"Correctly processing new file: {file2_path}")
            files_processed += 1
            document_tracker[file2_path] = {
                "signature": file2_signature,
                "last_processed": time.time()
            }
        
        # Save tracker
        with open(tracker_file, 'w') as f:
            json.dump(document_tracker, f, indent=2)
        
        # Verify results
        print(f"Files processed: {files_processed}, Files skipped: {files_skipped}")
        
        if files_processed == 1 and files_skipped == 1:
            print("Test 4 PASSED: Only the new file was processed")
        else:
            print("Test 4 FAILED: Wrong number of files processed")
            return False
        
        print("\nAll tests PASSED!")
        return True
        
    except Exception as e:
        print(f"Error during testing: {e}")
        return False
        
    finally:
        # Clean up
        try:
            shutil.rmtree(temp_dir)
            print(f"Cleaned up temporary test directory: {temp_dir}")
        except Exception as e:
            print(f"Error cleaning up: {e}")

if __name__ == "__main__":
    success = test_file_change_detection()
    sys.exit(0 if success else 1) 