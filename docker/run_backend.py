#!/usr/bin/env python3
"""
Simple script to start the backend service.
This avoids Python module import issues by directly running the code.
"""

import asyncio
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now import from the backend module
from backend.core.settings import settings
import uvicorn

def run_service():
    """Run the FastAPI service."""
    print(f"Starting service on {settings.HOST}:{settings.PORT}")
    
    # Set Compatible event loop policy on Windows Systems.
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    uvicorn.run(
        "backend.service.service:app", 
        host=settings.HOST, 
        port=settings.PORT, 
        reload=settings.is_dev()
    )

if __name__ == "__main__":
    run_service()