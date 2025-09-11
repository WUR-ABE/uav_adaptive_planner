#!/bin/bash
 
# Install Python dependencies
sudo apt-get update && sudo apt-get install -y --no-install-recommends libimage-exiftool-perl ffmpeg python3-tk
python3 -m pip install --upgrade pip
python3 -m pip install -e /workspaces/uav_adaptive_planner
