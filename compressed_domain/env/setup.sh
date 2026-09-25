#!/bin/bash

set -e

# Default paths
ENV_PATH="/tmp/clip4clip"
REQUIREMENTS_PATH="/notebooks/CLIP4Clip-CompressedDomain/compressed_domain/env/requirements.txt"
COVIAR_PATH="/storage/pytorch-coviar"

# Optional command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --env-path)
            ENV_PATH="$2"
            shift 2
            ;;
        --requirements-path)
            REQUIREMENTS_PATH="$2"
            shift 2
            ;;
        --coviar-path)
            COVIAR_PATH="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

echo "Environment path: $ENV_PATH"
echo "Requirements:     $REQUIREMENTS_PATH"
echo "COVIAR path:      $COVIAR_PATH"

# Create and activate virtual environment
python3 -m venv "$ENV_PATH"
source "$ENV_PATH/bin/activate"

# Install Jupyter kernel
pip install ipykernel

python -m ipykernel install --user \
    --name=clip4clip \
    --display-name "clip4clip"

# Install project requirements
pip install -r "$REQUIREMENTS_PATH"

# Install COVIAR
pip install "$COVIAR_PATH/data_loader/"

echo "Environment setup complete."
echo "Activate with: source $ENV_PATH/bin/activate"