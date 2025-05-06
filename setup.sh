#!/bin/bash

# Activate your virtual environment if needed
# source /path/to/venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Downloading spaCy model en_core_web_trf..."
python -m spacy download en_core_web_trf

echo "Setup complete."
