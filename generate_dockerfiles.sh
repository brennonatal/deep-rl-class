#!/bin/bash

# Loop through each unique unit directory
for unit in $(grep -oP '^\.\/[^\/]+' input.txt | sort -u); do
    # Create a Dockerfile for the unit
    dockerfile="notebooks/${unit}/Dockerfile"
    mkdir -p "$unit"
    echo "Creating Dockerfile for $unit..."

    # Start writing the Dockerfile
    cat <<EOL >"$dockerfile"
# Dockerfile for ${unit}
FROM python:3.8-slim

# Install basic dependencies
RUN apt-get update && apt-get install -y \\
    python3-opengl \\
    ffmpeg \\
    xvfb \\
    swig \\
    cmake \\
    && apt-get clean

# Install Python dependencies
RUN pip install --upgrade pip setuptools

# Install PyVirtualDisplay for headless rendering
RUN pip install pyvirtualdisplay

EOL

    # Add specific commands for each unit
    grep "^${unit}" input.txt | cut -d':' -f2 | sed 's/^!//' | while read -r line; do
        # Handle pip installations
        if [[ "$line" == pip* || "$line" == pip3* ]]; then
            echo "RUN $line" >>"$dockerfile"
        # Handle apt installations
        elif [[ "$line" == apt* || "$line" == sudo\ apt* ]]; then
            echo "RUN $line" >>"$dockerfile"
        # Handle other shell commands
        else
            echo "RUN $line" >>"$dockerfile"
        fi
    done

    echo "Dockerfile created for $unit at $dockerfile"
done
