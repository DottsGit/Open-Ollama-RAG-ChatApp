"""
Runtime monkey patch for ONNX Runtime to prevent Windows version warning.
"""
import warnings
import re
from unittest.mock import patch

# Save the original warnings.warn function
original_warn = warnings.warn

# Define a filter for the warning
def filtered_warn(message, *args, **kwargs):
    # Check if it's the ONNX Windows version warning
    if isinstance(message, str) and "Unsupported Windows version" in message:
        # Don't show this warning
        return
    
    # For all other warnings, use the original function
    return original_warn(message, *args, **kwargs)

# Apply the monkey patch
warnings.warn = filtered_warn

print("Applied monkey patch to suppress ONNX Runtime Windows version warning") 