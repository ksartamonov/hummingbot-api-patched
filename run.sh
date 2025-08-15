#!/bin/bash

# Run script for Backend API
# Usage: ./run.sh [--dev]
# --dev: Run API from source using uvicorn
# Without --dev: Run using docker compose

if [[ "$1" == "--dev" ]]; then
    echo "Running API from source..."
    # Activate conda environment and run with uvicorn
    docker compose up emqx postgres -d
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate hummingbot-api
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
else
    echo "Running with Docker Compose..."
    docker compose up -d
fi