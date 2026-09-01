"""Streamlit Community Cloud Application Entrypoint.

Delegates execution to frontend/app.py while preserving root path context.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Run main Streamlit application
from frontend.app import *
