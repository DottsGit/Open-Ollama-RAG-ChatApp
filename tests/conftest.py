"""
Pytest configuration file.
"""

import pytest
import warnings
import sys
import os

# Add the parent directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Import and apply our monkey patch to suppress ONNX Runtime warnings
from tests.monkey_patch import filtered_warn
warnings.warn = filtered_warn

def pytest_configure(config):
    """
    Configure pytest.
    """
    pass

def pytest_sessionstart(session):
    """
    Called after the Session object has been created and before performing collection
    and entering the run test loop.
    """
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print(f"sys.path: {sys.path}") 