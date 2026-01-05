#!/bin/bash
set -e

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv is not installed. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
else
    echo "uv is already installed."
fi

# Create virtual environment
echo "Creating virtual environment..."
uv venv --python 3.11

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
uv pip install vllm --torch-backend=auto
uv pip install 'ms-swift[llm]'

echo "Setup complete!"