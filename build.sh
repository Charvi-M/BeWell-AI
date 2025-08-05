#!/usr/bin/env bash

# Make sure pip and build tools are installed before anything else
pip install --upgrade pip setuptools wheel

# Install the rest of your dependencies
pip install --no-cache-dir -r requirements.txt
      
      # Clean up pip cache to save space
pip cache purge
      
      # Remove unnecessary files
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -name "*.pyc" -delete