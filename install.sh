#!/bin/bash

set -e

echo "Setting up Text-to-Image Generation System..."

if ! command -v python >/dev/null 2>&1; then
	echo "Python is not installed or not available on PATH."
	exit 1
fi

PYTHON_VERSION=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [ "$PYTHON_VERSION" != "3.14" ]; then
	echo "Unsupported Python version: $PYTHON_VERSION"
	echo "This project is pinned to package versions that work with Python 3.14."
	echo "Install Python 3.14, recreate the venv, then rerun this script."
	exit 1
fi

# Create virtual environment
python -m venv venv

# Activate virtual environment
if [ -f "venv/Scripts/activate" ]; then
	# Git Bash on Windows
	source venv/Scripts/activate
else
	source venv/bin/activate
fi

# Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "Setup complete! Run python app.py to start the application."