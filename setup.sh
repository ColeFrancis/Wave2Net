#!/bin/bash

python3 -m venv workshop_env
source workshop_env/bin/activate
pip install --upgrade pip
pip install torch numpy matplotlib

echo
echo
echo "Setup complete. Run:"
echo
echo "source workshop_env/bin/activate"