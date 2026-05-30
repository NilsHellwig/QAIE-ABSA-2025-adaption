import os

# Paths
TOOLKIT_PATH = "/home/hellwig/absa-toolkit"
RESULTS_DIR = "03_results"
AUG_DIR = "01_augmentations"
AUG_FS_DIR = f"{AUG_DIR}/fs_examples"
DATASET_AUG_DIR = "02_dataset_augmentations"

# Task and Dataset configuration
TASKS = ["asqp", "tasd"]
DATASETS = ["rest15", "rest16", "flightabsa", "hotels", "coursera"]
N_SHOTS = [10, 50, 100]
SEEDS = [0, 1, 2, 3, 4]

# Model Names
MODEL_NAME_CHAT = "google/gemma-4-31b-it"
MODEL_NAME_BASE = "google/gemma-4-31b"
T5_MODEL = "google-t5/t5-base"

# vLLM Configuration
MAX_TOKENS = 256
TEMPERATURE = 0.0
