"""
Pytest configuration file.
"""

import pytest
import os
import sys

# Add the parent directory to the path so that we can import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Register test markers
def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line("markers", "integration: mark a test as an integration test")
    config.addinivalue_line("markers", "slow: mark a test as a slow test")
    config.addinivalue_line("markers", "unit: mark a test as a unit test")

def pytest_sessionstart(session):
    """
    Called after the Session object has been created and before performing collection
    and entering the run test loop.
    """
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print(f"sys.path: {sys.path}") 