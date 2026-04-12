#!/bin/bash

# Setup environment for MedForge-Reasoner

echo "Installing core requirements..."
pip install -r requirements.txt

echo "Installing ms-swift from source..."
if [ ! -d "ms-swift" ]; then
    git clone https://github.com/modelscope/ms-swift.git
fi
cd ms-swift
pip install -e .
cd ..

echo "Installing flash-attention..."
# Note: Requires CUDA and appropriate compiler
pip install flash-attn --no-build-isolation

echo "Environment setup complete!"

