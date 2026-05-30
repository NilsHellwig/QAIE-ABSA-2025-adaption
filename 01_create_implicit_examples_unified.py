import os
import json
import time
import sys
from tqdm import tqdm
from llm import get_llm
from vllm import SamplingParams
from qaie_const import (
    TASKS, DATASETS, N_SHOTS, SEEDS, TOOLKIT_PATH, RESULTS_DIR, 
    AUG_FS_DIR, DATASET_AUG_DIR, MODEL_NAME_CHAT, MAX_TOKENS, TEMPERATURE
)

# Add toolkit path
sys.path.append(TOOLKIT_PATH)
try:
    from helper import get_fs_examples_new_with_seed, parse_label_string
except ImportError:
    print(f"Warning: Toolkit helper not found in {TOOLKIT_PATH}")

# Model Configuration
llm_wrapper = get_llm(model_name=MODEL_NAME_CHAT)
sampling_params = SamplingParams(
    temperature=TEMPERATURE,
    max_tokens=MAX_TOKENS,
    stop=["\n"]
)

def get_chat_messages(prompt):
    return [{"role": "user", "content": prompt}]

def read_line_examples_from_file_simple(data_path):
    if not os.path.exists(data_path):
        return [], [], []
    reviews, sents, labels = [], [], []
    with open(data_path, "r", encoding="UTF-8") as fp:
        for line in fp:
            line = line.strip()
            if line != "":
                parts = line.split("####")
                if len(parts) == 2:
                    words, tuples = parts
                    reviews.append(words)
                    sents.append(words.split())
                    labels.append(eval(tuples))
    return sents, reviews, labels

def generate_implicit_sentences(reviews, labels, task, dataset, n_shot, seed, suffix):
    if not reviews:
        return [], {}

    local_gpu_stats = {}
    # Step 1: Reasoning
    messages_r1 = []
    meta_r1 = []
    for sentence, label in zip(reviews, labels):
        if not label: continue
        # Handle different label lengths (ASQP=4/5, TASD=3)
        if len(label[0]) >= 3:
            at = label[0][0]
            ac = label[0][1]
            sp = label[0][2]
        else:
            continue
            
        prompt = f"Original text: {sentence}. Sentiment: {sp}. Give only a short reasoning for this sentiment regarding {at} and {ac} in 10-15 words."
        messages_r1.append(get_chat_messages(prompt))
        meta_r1.append({'sentence': sentence, 'label': label, 'sp': sp, 'at': at, 'ac': ac})
    
    if not messages_r1:
        return [], {}
        
    ans_r1, stats_r1 = llm_wrapper.chat(messages_r1, sampling_params=sampling_params, monitor_name=f"im_r1_{task}_{dataset}_{n_shot}_s{seed}_{suffix}")
    local_gpu_stats[f"im_r1_{task}_{dataset}_{n_shot}_s{seed}_{suffix}"] = stats_r1
    
    # Step 2: Implicit Sentence Generation
    messages_r2 = []
    for i, reason in enumerate(ans_r1):
        m = meta_r1[i]
        prompt = f"Generate only a very short sentence for the following reasoning that only includes the aspect term \"{m['at']}\", following information: reasoning: {reason}. The output should NOT mention the sentiment: {m['sp']}."
        messages_r2.append(get_chat_messages(prompt))
    
    ans_r2, stats_r2 = llm_wrapper.chat(messages_r2, sampling_params=sampling_params, monitor_name=f"im_r2_{task}_{dataset}_{n_shot}_s{seed}_{suffix}")
    local_gpu_stats[f"im_r2_{task}_{dataset}_{n_shot}_s{seed}_{suffix}"] = stats_r2
    
    final_ans = [ans.replace("\n", " ").strip() for ans in ans_r2]
    return final_ans, local_gpu_stats

def do_augmentation_batch():
    # 1. Process standard datasets (test) first
    for task in TASKS:
        for dataset in DATASETS:
            for split in ["test"]:
                data_path = f"{TOOLKIT_PATH}/data/datasets/{task}/{dataset}/{split}.txt"
                out_dir = f"{DATASET_AUG_DIR}/{task}/{dataset}"
                os.makedirs(out_dir, exist_ok=True)
                out_path = f"{out_dir}/{split}_im.txt"
                out_gpu_path = f"{out_dir}/{split}_im_gpu.json"
                
                if os.path.exists(data_path) and not os.path.exists(out_path):
                    print(f"Generating implicit for {task}/{dataset}/{split}")
                    sents, reviews, labels = read_line_examples_from_file_simple(data_path)
                    if reviews:
                        ans, stats = generate_implicit_sentences(reviews, labels, task, dataset, 0, 0, split)
                        with open(out_path, "w") as f:
                            for line in ans:
                                f.write(line + "\n")
                        with open(out_gpu_path, "w") as f:
                            json.dump(stats, f, indent=4)

    # 2. Process Shots and Seeds
    for task in TASKS:
        for dataset in DATASETS:
            for n_shot in N_SHOTS:
                for seed in SEEDS:
                    print(f"Implicit Examples - Task: {task}, Dataset: {dataset}, Shot: {n_shot}, Seed: {seed}")
                    
                    seed_dir = f"{AUG_FS_DIR}/{task}/{dataset}/fs_{n_shot}/seed_{seed}"
                    os.makedirs(seed_dir, exist_ok=True)

                    # 1. Process Augmented Examples
                    aug_path = f"{seed_dir}/aug.txt"
                    aug_im_path = f"{seed_dir}/aug_im.txt"
                    aug_im_gpu_path = f"{seed_dir}/aug_im_gpu.json"
                    
                    if os.path.exists(aug_path) and not os.path.exists(aug_im_path):
                        sents, reviews, labels = read_line_examples_from_file_simple(aug_path)
                        if reviews:
                            ans, stats = generate_implicit_sentences(reviews, labels, task, dataset, n_shot, seed, "aug")
                            with open(aug_im_path, "w") as f:
                                for line in ans:
                                    f.write(line + "\n")
                            with open(aug_im_gpu_path, "w") as f:
                                json.dump(stats, f, indent=4)

                    # 2. Process Original Seed Examples
                    fs_im_path = f"{seed_dir}/fs_im.txt"
                    fs_im_gpu_path = f"{seed_dir}/fs_im_gpu.json"
                    if not os.path.exists(fs_im_path):
                        try:
                            fs_examples = get_fs_examples_new_with_seed(dataset, task, n_shot, seed)
                            if fs_examples:
                                reviews_fs = [ex['text'] for ex in fs_examples]
                                labels_fs = [ex['label'] for ex in fs_examples]
                                ans_fs, stats_fs = generate_implicit_sentences(reviews_fs, labels_fs, task, dataset, n_shot, seed, "fs")
                                with open(fs_im_path, "w") as f:
                                    for line in ans_fs:
                                        f.write(line + "\n")
                                with open(fs_im_gpu_path, "w") as f:
                                    json.dump(stats_fs, f, indent=4)
                        except Exception as e:
                            print(f"Error loading original seeds for {task}/{dataset}/s{seed}: {e}")

if __name__ == "__main__":
    do_augmentation_batch()
