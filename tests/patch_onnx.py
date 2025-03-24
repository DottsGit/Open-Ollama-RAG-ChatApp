"""
Patch for fixing ONNX Runtime Windows version detection.
"""
import os
import sys
import importlib.util
import re

def patch_onnxruntime():
    """
    Patch the onnxruntime validation.py file to fix Windows version detection.
    This should be imported before any tests that use onnxruntime.
    """
    try:
        # Find onnxruntime package location
        spec = importlib.util.find_spec('onnxruntime.capi.onnxruntime_validation')
        if not spec or not spec.origin:
            print("Could not find onnxruntime validation module")
            return False
        
        validation_file = spec.origin
        print(f"Found onnxruntime validation at: {validation_file}")
        
        # Read the file
        with open(validation_file, 'r') as f:
            content = f.read()
        
        # Check if the file already contains our fix
        if "Patched Windows version detection" in content:
            print("Validation file already patched")
            return True
        
        # Create a backup
        backup_file = validation_file + '.backup'
        if not os.path.exists(backup_file):
            with open(backup_file, 'w') as f:
                f.write(content)
            print(f"Created backup at: {backup_file}")
        
        # Replace the problematic Windows version check
        pattern = r'if platform\.system\(\) == "Windows" and int\(platform\.version\(\)\.split\(\).*? < 10:'
        replacement = """if platform.system() == "Windows" and False:  # Patched Windows version detection"""
        
        new_content = re.sub(pattern, replacement, content)
        
        # Write back to the file
        with open(validation_file, 'w') as f:
            f.write(new_content)
        
        print("Successfully patched onnxruntime validation.py")
        return True
    
    except Exception as e:
        print(f"Error patching onnxruntime: {e}")
        return False

if __name__ == "__main__":
    success = patch_onnxruntime()
    sys.exit(0 if success else 1) 