#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Print Python and pip information
echo "Python version: $(python --version)"
echo "Pip version: $(pip --version)"

# Print working directory and directory structure
echo "Working directory: $(pwd)"
echo "Directory structure:"
ls -la

# Print Python path
echo "PYTHONPATH: $PYTHONPATH"

# Print module structure
echo "Backend module structure:"
ls -la backend/

# Install the project as a package
echo "Installing project as a package..."
pip install -e .

# Try to import the module to check if it's installed correctly
echo "Testing backend module import..."
python -c "import backend; print('Successfully imported backend module')"

# Run the service with detailed error reporting
echo "Starting the backend service..."
python -m backend.run_service