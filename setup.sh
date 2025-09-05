#!/bin/bash
# Simple environment setup for FLAME-ICU
python3 -m venv flameICU
source flameICU/bin/activate
pip install -r requirements.txt
echo "Setup complete. Run 'source flameICU/bin/activate' to use."