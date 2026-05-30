import subprocess
from itertools import product
import shutil, os
from qaie_const import TASKS, DATASETS, N_SHOTS, SEEDS, MODEL_NAME_CHAT, T5_MODEL, RESULTS_DIR

# Parameter-Optionen
data_cou_values = [str(ns) for ns in N_SHOTS]
tasks = TASKS
datasets = DATASETS
seeds = [str(s) for s in SEEDS]

# Alle Kombinationen durchgehen
for seed, task, dataset, data_cou in product(seeds, tasks, datasets, data_cou_values):
  if os.path.exists("outputs"):
    shutil.rmtree("outputs")
  
  # New path format: 03_results/google_gemma-4-31b-it/synth_training_{dataset}_{data_cou}_{task}_qaie_seed{seed}_nsynth1000.json
  # (Keeping qaie and nsynth1000 to match the user's template exactly)
  log_file_path = f"{RESULTS_DIR}/{MODEL_NAME_CHAT}/synth_training_{dataset}_{data_cou}_{task}_qaie_seed{seed}_nsynth1000.json"
  
  # run only if log file does not exist
  if not(os.path.exists(log_file_path)):

    print(f"Starte Run für task={task}, dataset={dataset}, data_cou={data_cou}, seed={seed}")
    command = [
        "python", "02_train_model.py",
        "--task", task,
        "--dataset", dataset,
        "--seed", seed,
        "--model_name_or_path", T5_MODEL,
        "--n_gpu", "0",
        "--do_train",
        "--train_batch_size", "16",
        "--gradient_accumulation_steps", "1",
        "--eval_batch_size", "16",
        "--learning_rate", "3e-4",
        "--num_train_epochs", "20",
        "--data_cou", data_cou,
        "--log_file_path", log_file_path,
        "--do_direct_eval"
    ]

    subprocess.run(command)
