#!/bin/bash
apt-get update && apt-get install -y libgl1 libgtk2.0-dev
pip install --upgrade pip
pip install -r requirements.txt