"""Middleware for CORS support and other HTTP-level concerns."""

from fastapi.middleware.cors import CORSMiddleware

def add_cors_middleware(app):
    """Add CORS middleware to the application."""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # For development - replace with specific frontend URL in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return app