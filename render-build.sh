#!/bin/bash
# Install system dependencies
apt-get update && apt-get install -y \
    libgl1 \
    libgtk2.0-dev

# Install Python packages with clear error handling
pip install --upgrade pip
pip install -r requirements.txt || {
    echo "Failed to install requirements"
    exit 1
}
