#!/bin/bash

##################### Paths #####################

# Set default paths
DEFAULT_DATA_DIR="${PWD}/dataset"
DEFAULT_LOG_DIR="${PWD}/logs"

# Prompt the user for input, allowing overrides
read -p "Enter the desired data directory [default: ${DEFAULT_DATA_DIR}], leave empty to use default: " DATA_DIR
DIWA_DATA_DIR=${DATA_DIR:-$DEFAULT_DATA_DIR}  # Use user input or default if input is empty

read -p "Enter the desired logging directory [default: ${DEFAULT_LOG_DIR}], leave empty to use default: " LOG_DIR
DIWA_LOG_DIR=${LOG_DIR:-$DEFAULT_LOG_DIR}  # Use user input or default if input is empty

# Export to current session
export DIWA_ROOT_DIR="$PWD"
export DIWA_DATA_DIR="$DIWA_DATA_DIR"
export DIWA_LOG_DIR="$DIWA_LOG_DIR"

# Confirm the paths with the user
echo "DIWA directory set to: $DIWA_ROOT_DIR"
echo "Data directory set to: $DIWA_DATA_DIR"
echo "Log directory set to: $DIWA_LOG_DIR"

# Append environment variables to .bashrc
echo "export DIWA_ROOT_DIR=\"$DIWA_ROOT_DIR\"" >> ~/.bashrc
echo "export DIWA_DATA_DIR=\"$DIWA_DATA_DIR\"" >> ~/.bashrc
echo "export DIWA_LOG_DIR=\"$DIWA_LOG_DIR\"" >> ~/.bashrc

echo "Environment variables DIWA_DATA_DIR and DIWA_LOG_DIR added to .bashrc and applied to the current session."

##################### WandB #####################

# Prompt the user for input, allowing overrides
read -p "Enter your WandB entity (username or team name), leave empty to skip: " ENTITY

# Check if ENTITY is not empty
if [ -n "$ENTITY" ]; then
  # If ENTITY is not empty, set the environment variable
  export DIWA_WANDB_ENTITY="$ENTITY"

  # Confirm the entity with the user
  echo "WandB entity set to: $DIWA_WANDB_ENTITY"

  # Append environment variable to .bashrc
  echo "export DIWA_WANDB_ENTITY=\"$ENTITY\"" >> ~/.bashrc
  
  echo "Environment variable DIWA_WANDB_ENTITY added to .bashrc and applied to the current session."
else
  # If ENTITY is empty, skip setting the environment variable
  echo "No WandB entity provided. Please set wandb=null when running scripts to disable wandb logging and avoid error."
fi
