#!/usr/bin/env bash

on_error() {
  # Output Error info
  echo "ERROR: A command on line $BASH_LINENO returned a non-zero exit code."
  echo "      Command: $BASH_COMMAND"
  # Exit
  exit 1
}

# Trap any command returning a non-zero exit code and call our custom handler
trap on_error ERR

if ! command -v docker &> /dev/null
then
    echo "Docker not found. Installing docker.io..."
    sudo apt-get update
    sudo apt-get install -y docker.io
    sudo systemctl enable docker
    sudo systemctl start docker
    echo "Docker installed successfully."
else
    echo "Docker is already installed."
fi