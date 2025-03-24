"""
Direct patch for ONNX Runtime validation file to fix Windows version detection issue.
"""
import os

# Path to the onnxruntime validation file
validation_file = r"C:\Users\lguxr\Documents\Programming\Community\Open-Ollama-RAG-ChatApp\.venv\Lib\site-packages\onnxruntime\capi\onnxruntime_validation.py"

# Read the original content
with open(validation_file, 'r') as f:
    content = f.read()

# Make a backup if it doesn't exist
backup_file = validation_file + '.backup'
if not os.path.exists(backup_file):
    with open(backup_file, 'w') as f:
        f.write(content)
    print(f"Created backup at: {backup_file}")

# Replace the problematic line
original_line = 'if platform.system() == "Windows" and int(platform.version().split()[0]) < 10:'
fixed_line = 'if platform.system() == "Windows" and False:  # Patched to support all Windows versions'

if original_line in content:
    fixed_content = content.replace(original_line, fixed_line)
    
    # Write the fixed content
    with open(validation_file, 'w') as f:
        f.write(fixed_content)
    
    print("Successfully patched onnxruntime validation.py")
else:
    print("Could not find the line to patch. The file may have changed.")
    print("Original line:", original_line)
    print("First 200 characters of file:", content[:200]) 