#!/bin/bash
apt-get update && apt-get install -y libgl1
pip install --upgrade "tensorflow<2.16"  # Force compatible version
pip install -r requirements.txt
