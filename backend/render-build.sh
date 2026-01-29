#!/bin/bash
# Render build script for backend

echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "Creating weights directory..."
mkdir -p weights

echo "Backend build complete!"