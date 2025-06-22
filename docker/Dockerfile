# Base image with CUDA and Python
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    git \
    curl \
    unzip \
    nano \
    libgl1-mesa-glx \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set python3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Upgrade pip
RUN python -m pip install --upgrade pip

# Create working directory
WORKDIR /usr/src/app

# Install YOLO + dependencies
RUN pip install ultralytics

# Optional: install extra tools for training visualization
RUN pip install matplotlib seaborn pandas

# Copy your training config and data (optional)
# COPY ./data ./data
# COPY ./train.py ./train.py

# Default command (can be overridden at runtime)
CMD ["yolo", "task=detect", "mode=train", "model=yolov8n.pt", "data=simobj.yaml", "epochs=10"]
