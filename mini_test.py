import os
import time
import json
import shutil
import tempfile

print("Starting minimal file change detection test...")

# Create a temporary test directory
test_dir = tempfile.mkdtemp()
data_dir = os.path.join(test_dir, "data")
db_dir = os.path.join(test_dir, "db")

# Create test directories
os.makedirs(data_dir, exist_ok=True)
os.makedirs(db_dir, exist_ok=True)

# File to track document processing
tracker_file = os.path.join(db_dir, "document_tracker.json")

try:
    # TEST 1: Process a new file
    print("\n=== TEST 1: Process New File ===")
    
    # Create a test markdown file
    file1_path = os.path.join(data_dir, "test1.md")
    with open(file1_path, 'w') as f:
        f.write("# Test Document 1\nThis is a test markdown file.")
    
    # Get file metadata
    file1_mtime = os.path.getmtime(file1_path)
    file1_size = os.path.getsize(file1_path)
    
    # Create a file signature
    file1_signature = f"{file1_path}:{file1_size}:{file1_mtime}"
    
    # Initialize document tracker
    document_tracker = {}
    
    # Add file to tracker (simulate processing)
    document_tracker[file1_path] = {
        "signature": file1_signature,
        "last_processed": time.time()
    }
    
    # Save tracker
    with open(tracker_file, 'w') as f:
        json.dump(document_tracker, f, indent=2)
    
    print(f"Processed new file: {file1_path}")
    print(f"Signature: {file1_signature}")
    print(f"Tracker saved with {len(document_tracker)} entries")
    
    # TEST 2: Skip unchanged file
    print("\n=== TEST 2: Skip Unchanged File ===")
    
    # Load tracker
    with open(tracker_file, 'r') as f:
        document_tracker = json.load(f)
    
    # Check if file needs processing
    file1_mtime = os.path.getmtime(file1_path)
    file1_size = os.path.getsize(file1_path)
    file1_signature_new = f"{file1_path}:{file1_size}:{file1_mtime}"
    
    if file1_path in document_tracker and document_tracker[file1_path]["signature"] == file1_signature_new:
        print(f"Skipping unchanged file: {file1_path}")
    else:
        print(f"ERROR: File should be skipped but was detected as changed!")
        print(f"Old signature: {document_tracker[file1_path]['signature']}")
        print(f"New signature: {file1_signature_new}")
    
    # TEST 3: Process modified file
    print("\n=== TEST 3: Process Modified File ===")
    
    # Wait a moment to ensure modification time will be different
    time.sleep(1)
    
    # Modify the file
    with open(file1_path, 'w') as f:
        f.write("# Updated Test Document 1\nThis file has been modified.")
    
    # Check if file needs processing
    file1_mtime = os.path.getmtime(file1_path)
    file1_size = os.path.getsize(file1_path)
    file1_signature_modified = f"{file1_path}:{file1_size}:{file1_mtime}"
    
    if file1_path in document_tracker and document_tracker[file1_path]["signature"] == file1_signature_modified:
        print(f"ERROR: Modified file not detected as changed!")
    else:
        print(f"Detected modified file: {file1_path}")
        print(f"Old signature: {document_tracker[file1_path]['signature']}")
        print(f"New signature: {file1_signature_modified}")
        
        # Update tracker (simulate processing)
        document_tracker[file1_path] = {
            "signature": file1_signature_modified,
            "last_processed": time.time()
        }
        
        # Save tracker
        with open(tracker_file, 'w') as f:
            json.dump(document_tracker, f, indent=2)
    
    # TEST 4: Process only new file when multiple files exist
    print("\n=== TEST 4: Process Only New Files ===")
    
    # Create a second test file
    file2_path = os.path.join(data_dir, "test2.md")
    with open(file2_path, 'w') as f:
        f.write("# Test Document 2\nThis is a second test markdown file.")
    
    # Check first file
    file1_mtime = os.path.getmtime(file1_path)
    file1_size = os.path.getsize(file1_path)
    file1_signature_current = f"{file1_path}:{file1_size}:{file1_mtime}"
    
    # Check second file
    file2_mtime = os.path.getmtime(file2_path)
    file2_size = os.path.getsize(file2_path)
    file2_signature = f"{file2_path}:{file2_size}:{file2_mtime}"
    
    # Process files
    files_processed = 0
    
    # Check first file
    if file1_path in document_tracker and document_tracker[file1_path]["signature"] == file1_signature_current:
        print(f"Skipping unchanged file: {file1_path}")
    else:
        print(f"Processing changed file: {file1_path}")
        files_processed += 1
        document_tracker[file1_path] = {
            "signature": file1_signature_current,
            "last_processed": time.time()
        }
    
    # Check second file
    if file2_path in document_tracker and document_tracker[file2_path]["signature"] == file2_signature:
        print(f"Skipping unchanged file: {file2_path}")
    else:
        print(f"Processing new file: {file2_path}")
        files_processed += 1
        document_tracker[file2_path] = {
            "signature": file2_signature,
            "last_processed": time.time()
        }
    
    # Save tracker
    with open(tracker_file, 'w') as f:
        json.dump(document_tracker, f, indent=2)
    
    print(f"Processed {files_processed} files")
    print(f"Expected: 1 file processed (only the new file)")
    
    if files_processed == 1:
        print("TEST PASSED: Only the new file was processed")
    else:
        print("TEST FAILED: Incorrect number of files processed")
    
    print("\nAll tests completed!")

except Exception as e:
    print(f"Error during tests: {e}")

finally:
    # Clean up
    try:
        shutil.rmtree(test_dir)
        print(f"Cleaned up test directory: {test_dir}")
    except Exception as e:
        print(f"Error cleaning up: {e}") 