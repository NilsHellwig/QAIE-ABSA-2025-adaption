import subprocess
from itertools import product
import shutil, os

# Parameter-Optionen
data_cou_values = ["50", "10"]
tasks = ["asqp"]
datasets = ["rest16", "rest15", "flightabsa", "coursera", "hotels"]
seeds = ["0", "1", "2", "3", "4"]



# Alle Kombinationen durchgehen
for seed, task, dataset, data_cou in product(seeds, tasks, datasets, data_cou_values):
  if os.path.exists("outputs"):
    shutil.rmtree("outputs")
  log_file_path = f"03_results/{task}_{dataset}_fs_{data_cou}_{seed}.json"
  # run only if log file does not exist
  if not(os.path.exists(log_file_path)):

    print(f"Starte Run für task={task}, dataset={dataset}, data_cou={data_cou}, seed={seed}")
    command = [
        "python", "02_train_model.py",
        "--task", task,
        "--dataset", dataset,
        "--seed", seed,
        "--model_name_or_path", "google-t5/t5-base",
        "--n_gpu", "0",
        "--do_train",
        "--train_batch_size", "8",
        "--gradient_accumulation_steps", "1",
        "--eval_batch_size", "8",
        "--learning_rate", "3e-4",
        "--num_train_epochs", "20",
        "--data_cou", data_cou,
        "--log_file_path", log_file_path,
        "--do_direct_eval"
    ]

    print(f"Starte Run für task={task}, dataset={dataset}, data_cou={data_cou}")
    subprocess.run(command)
