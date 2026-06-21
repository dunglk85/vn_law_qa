"""Pytest configuration for law-crawler tests."""
import sys
from pathlib import Path

# Add law-crawler root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
