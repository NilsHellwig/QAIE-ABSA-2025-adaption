import os
import re
import json
import torch
import time
from tqdm import tqdm
from llm import get_llm
from vllm import SamplingParams
import sys
from qaie_const import (
    TASKS, DATASETS, N_SHOTS, SEEDS, TOOLKIT_PATH, RESULTS_DIR, 
    AUG_FS_DIR, MODEL_NAME_CHAT, MAX_TOKENS, TEMPERATURE
)

# Add toolkit path
sys.path.append(TOOLKIT_PATH)
try:
    from helper import get_fs_examples_new_with_seed, parse_label_string
    from gpu_monitor import GPUMonitor
except ImportError:
    print(f"Warning: Toolkit helper or GPUMonitor not found in {TOOLKIT_PATH}")

# LLM setup
llm_wrapper = get_llm(model_name=MODEL_NAME_CHAT)
sampling_params = SamplingParams(
    temperature=TEMPERATURE,
    max_tokens=MAX_TOKENS,
    stop=["\n", "####"]
)

def get_chat_messages(prompt):
    return [{"role": "user", "content": prompt}]

senot = {"positive": "negative", "negative": "positive"}

def get_at_sm_prompt(at, review):
    return f"""Task Description: You can generate a new smooth text by following the Implementation Details based on the given texts and words. Give you these elements: Original Text, Replaced Word, Implementation Details, An input and output example, Principle, Tip. Finally, output the new text with the replacement.
                            Original Text: "{review}".
                            Replaced Word: "{at}".
                            Implementation Details: Finding a similar word or phrase with ''Replaced Word'' based on its actual meaning in the text ''Original Text'' and replacing the ''Replaced Word'' in the Text ``Original Text'' with this found word. 
                            An input and output example: Task Description: You can generate ..., Original Text: good drink, Replaced Word: drink, Implementation Details: Finding a similar word ..., Principle: Except for the word ..., Tip: If you can't ..., Result: good beverage.
                            Principle: Except for the word in the replaced position in the input text, which can be changed, the word in the other positions in the text remains unchanged.
                            Tip: If you can't choose a suitable word, look up synonyms or related words and check that the replacement text makes sense. Always make sure that the replacement word matches the grammar and context of the original text.
                    """

def get_ot_sm_prompt(ot, review):
    return f"""Task Description: You can generate a new smooth text by following the Implementation Details based on the given texts and words. Give you these elements: Original Text, Replaced Word, Implementation Details, An input and output example, Principle, Tip. Finally, output the new text with the replacement.
                            Original Text: "{review}".
                            Replaced Word: "{ot}".
                            Implementation Details: Finding a similar word or phrase with ''Replaced Word'' based on its actual meaning in the text ''Original Text'' and replacing the ''Replaced Word'' in the Text ``Original Text'' with this found word. 
                            An input and output example: Task Description: You can generate ..., Original Text: good drink, Replaced Word: good, Implementation Details: Finding a similar word ..., Principle: Except for the word ..., Tip: If you can't ..., Result: great drink.
                            Principle: Except for the word in the replaced position in the input text, which can be changed, the word in the other positions in the text remains unchanged.
                            Tip: If you can't choose a suitable word, look up synonyms or related words and check that the replacement text makes sense. Always make sure that the replacement word matches the grammar and context of the original text.
                    """

def get_ot_op_prompt(ot, review):
    return f"""Task Description: You can generate a new smooth text by following the Implementation Details based on the given texts and words. Give you these elements: Original Text, Replaced Word, Implementation Details, An input and output example, Principle, Tip. Finally, output the new text with the replacement.
                            Original Text: "{review}".
                            Replaced Word: "{ot}".
                            Implementation Details: Finding a opposite word or phrase with ''Replaced Word'' based on its actual meaning in the text ''Original Text'' and replacing the ''Replaced Word'' in the Text ``Original Text'' with this found word. 
                            An input and output example: Task Description: You can generate ..., Original Text: good drink, Replaced Word: good, Implementation Details: Finding a opposite word ..., Principle: Except for the word ..., Tip: If you can't ..., Result: bad drink.
                            Principle: Except for the word in the replaced position in the input text, which can be changed, the word in the other positions in the text remains unchanged.
                            Tip: If you can't choose a suitable word, look up synonyms or related words and check that the replacement text makes sense. Always make sure that the replacement word matches the grammar and context of the original text.
                    """

def get_find_label_prompt(sentence, review):
    return f"""Task Description: You can find out the difference between the given two texts by Implementation Details. Give you these elements: Original Text, Augmentation Text, Implementation Details, An input and output example, Principle, Tip. Finally output difference between the Original Text and the Augmentation Text.
                            Original Text: "{review}".
                            Augmented Text: "{sentence}".
                            Implementation Details: The text generated when the original text ''Original Review Text'' is replaced with the specified string is ''Augmented Review Text''. Find the string used to replace the specified string.
                            An input and output example: Task Description: You can find ..., Original Text: good drink, Replaced Word: good beverage, Implementation Details: The text generated..., Principle: Except for the word.., Tip: If you can't..., Result: beverage
                            Principle: The output string must be found in the text ''Augmented Review Text''.
                            Tip: If you can't find this label, you can gradually narrow it down by elimination. You can use context clues to infer the possible string.
                    """

def run_augmentation_batch():
    for task in TASKS:
        for dataset in DATASETS:
            for n_shot in N_SHOTS:
                for seed in SEEDS:
                    all_gpu_stats = {}
                    print(f"Processing Task: {task}, Dataset: {dataset}, Shot: {n_shot}, Seed: {seed}")
                    
                    try:
                        fs_examples = get_fs_examples_new_with_seed(dataset, task, n_shot, seed)
                    except Exception as e:
                        print(f"Error loading {task}/{dataset} (seed {seed}): {e}")
                        continue
                        
                    if not fs_examples:
                        print(f"Skipping {task}/{dataset} - no data found")
                        continue

                    reviews = [ex['text'] for ex in fs_examples]
                    labels = [ex['label'] for ex in fs_examples]

                    # Round 1: Generate initial variations
                    messages_r1 = []
                    meta_r1 = []
                    
                    for i, (review, label) in enumerate(zip(reviews, labels)):
                        # Process each quad in the label
                        for j, quad in enumerate(label):
                                if task == "asqp":
                                    at, ac, sp, ot = quad
                                else: # tasd
                                    at, ac, sp = quad
                                    ot = "none"
                                
                                at = str(at) if at is not None and str(at).lower() != "none" else "none"
                                ot = str(ot) if ot is not None and str(ot).lower() != "none" else "none"

                                if at != "none":
                                    messages_r1.append(get_chat_messages(get_at_sm_prompt(at, review)))
                                    meta_r1.append({'idx': i, 'type': 'at_sm', 'orig_quad': quad, 'quad_idx': j})
                                if ot != "none":
                                    messages_r1.append(get_chat_messages(get_ot_sm_prompt(ot, review)))
                                    meta_r1.append({'idx': i, 'type': 'ot_sm', 'orig_quad': quad, 'quad_idx': j})
                                    if sp in ["positive", "negative"]:
                                        messages_r1.append(get_chat_messages(get_ot_op_prompt(ot, review)))
                                        meta_r1.append({'idx': i, 'type': 'ot_op', 'orig_quad': quad, 'quad_idx': j})

                    if not messages_r1:
                        continue

                    responses_r1, stats_r1 = llm_wrapper.chat(messages_r1, sampling_params=sampling_params, monitor_name=f"aug_r1_{task}_{dataset}_{n_shot}_s{seed}")
                    all_gpu_stats.update({f"aug_r1_{task}_{dataset}_{n_shot}_s{seed}": stats_r1})

                    # Round 2: Label Finding
                    messages_r2 = []
                    meta_r2 = []
                    for resp, meta in zip(responses_r1, meta_r1):
                        review = reviews[meta['idx']]
                        messages_r2.append(get_chat_messages(get_find_label_prompt(resp, review)))
                        meta_r2.append({**meta, 'sentence': resp})

                    responses_r2, stats_r2 = llm_wrapper.chat(messages_r2, sampling_params=sampling_params, monitor_name=f"aug_r2_{task}_{dataset}_{n_shot}_s{seed}")
                    all_gpu_stats.update({f"aug_r2_{task}_{dataset}_{n_shot}_s{seed}": stats_r2})

                    # Collect results
                    final_text = []
                    # Keep originals
                    for review, label in zip(reviews, labels):
                        final_text.append(f"{review}####{str(label)}")

                    # Map results back
                    for resp_label, meta in zip(responses_r2, meta_r2):
                        sentence = meta['sentence']
                        new_word = resp_label.strip()
                        idx = meta['idx']
                        quad_idx = meta['quad_idx']
                        orig_labels = labels[idx]
                        
                        if new_word and new_word in sentence:
                            new_labels = [list(q) for q in orig_labels]
                            
                            if meta['type'] == 'at_sm':
                                new_labels[quad_idx][0] = new_word
                            elif meta['type'] == 'ot_sm':
                                if task == "asqp":
                                    new_labels[quad_idx][3] = new_word
                            elif meta['type'] == 'ot_op':
                                if task == "asqp":
                                    sp = new_labels[quad_idx][2]
                                    new_labels[quad_idx][2] = senot.get(sp, sp)
                                    new_labels[quad_idx][3] = new_word
                            
                            final_labels = [tuple(q) for q in new_labels]
                            final_text.append(f"{sentence}####{str(final_labels)}")

                    # Save files
                    output_dir = f"{AUG_FS_DIR}/{task}/{dataset}/fs_{n_shot}/seed_{seed}"
                    os.makedirs(output_dir, exist_ok=True)
                    
                    with open(f"{output_dir}/aug.txt", "w") as f:
                        for line in final_text:
                            f.write(line + "\n")
                    
                    # Save GPU stats for this specific seed
                    with open(f"{output_dir}/aug_gpu.json", "w") as f:
                        json.dump(all_gpu_stats, f, indent=4)
                    
                    # Round 3 & 4 (Combined transformations) could be added here for full coverage

if __name__ == "__main__":
    run_augmentation_batch()
