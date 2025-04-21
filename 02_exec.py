import subprocess

command = [
    "python", "02_train_model.py",
    "--task", "asqp",
    "--dataset", "rest16",
    "--model_name_or_path", "google-t5/t5-base",
    "--n_gpu", "0",
    "--do_train",
    "--train_batch_size", "8",
    "--gradient_accumulation_steps", "1",
    "--eval_batch_size", "16",
    "--learning_rate", "3e-4",
    "--num_train_epochs", "20",
    "--data_cou", "50",
    "--do_direct_eval"
]

subprocess.run(command)
