import argparse
import os
import json
import logging
import time
import sys
import torch
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer, get_linear_schedule_with_warmup

# Lokale Imports
from data_utils import ABSADataset
from eval_utils import compute_scores
from qaie_const import TOOLKIT_PATH

# Integration des ABSA-Toolkits
sys.path.append(TOOLKIT_PATH)
try:
    from gpu_monitor import GPUMonitor
    from helper import get_dataset as get_toolkit_dataset
except ImportError:
    GPUMonitor = None
    get_toolkit_dataset = None

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data_cou", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--log_file_path", type=str, required=True)
    
    # Flags aus 02_exec.py
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_direct_eval", action='store_true')
    
    # Standard-Parameter
    parser.add_argument("--max_seq_length", default=160, type=int)
    parser.add_argument("--n_gpu", default="0", type=str)
    parser.add_argument("--train_batch_size", default=16, type=int)
    parser.add_argument("--eval_batch_size", default=16, type=int)
    parser.add_argument("--learning_rate", default=3e-4, type=float)
    parser.add_argument("--num_train_epochs", default=20, type=int)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--warmup_steps", default=0, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    
    return parser.parse_args()

def evaluate(test_loader, model, tokenizer, device, args):
    model.eval()
    
    # Lade Originaldaten für das Scoring (Labels und Sents) via Toolkit Helper
    if get_toolkit_dataset:
        test_data = get_toolkit_dataset(args.dataset, "test", args.task, base_path=f"{TOOLKIT_PATH}/data")
        reviews = [ex['text'] for ex in test_data]
        sents = [text.split() for text in reviews]
    else:
        # Fallback (sollte nicht passieren)
        reviews, sents = [], []

    outputs, targets = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            # Generierung
            outs = model.generate(
                input_ids=batch['source_ids'].to(device),
                attention_mask=batch['source_mask'].to(device),
                max_length=128
            )
            dec = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
            target = [tokenizer.decode(ids, skip_special_tokens=True) for ids in batch["target_ids"]]
            outputs.extend(dec)
            targets.extend(target)

    scores, all_labels, all_preds = compute_scores(args.dataset, args.task, outputs, targets, reviews, sents)
    return scores, all_labels, all_preds

def main():
    args = init_args()
    set_seed(args.seed)
    
    # Device setup (using n_gpu from args)
    device = torch.device(f"cuda:{args.n_gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
    model.to(device)

    # 1. TRAINING
    avg_gpu_power_train = 0
    total_time_train = 0
    
    if args.do_train:
        print("\n****** Conduct Training ******")
        
        train_dataset = ABSADataset(
            tokenizer=tokenizer, 
            data_dir=args.dataset, 
            absa_task=args.task, 
            data_count=args.data_cou, 
            data_type="train", 
            seed=args.seed, 
            max_len=args.max_seq_length
        )
        train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, drop_last=True)
        
        # Optimizer Setup
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        
        t_total = len(train_loader) * args.num_train_epochs // args.gradient_accumulation_steps
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
        
        monitor = GPUMonitor() if GPUMonitor else None
        if monitor: monitor.start()

        start_train = time.time()
        model.train()
        
        for epoch in range(args.num_train_epochs):
            epoch_loss = 0
            for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
                input_ids = batch["source_ids"].to(device)
                attention_mask = batch["source_mask"].to(device)
                labels = batch["target_ids"].to(device)
                labels[labels == tokenizer.pad_token_id] = -100
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / args.gradient_accumulation_steps
                loss.backward()
                
                if (i + 1) % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                epoch_loss += loss.item()
            
            print(f"Epoch {epoch+1} Loss: {epoch_loss/len(train_loader):.4f}")

        total_time_train = time.time() - start_train
        if monitor:
            avg_gpu_power_train, _ = monitor.stop()

    # 2. EVALUATION
    avg_gpu_power_eval = 0
    total_time_eval = 0
    scores = {"precision": 0, "recall": 0, "f1": 0}
    all_labels, all_preds = [], []

    if args.do_direct_eval:
        print("\n****** Conduct Evaluation ******")
        test_dataset = ABSADataset(
            tokenizer=tokenizer, 
            data_dir=args.dataset, 
            absa_task=args.task, 
            data_count=args.data_cou, 
            data_type="test", 
            seed=args.seed, 
            max_len=args.max_seq_length
        )
        test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size)

        monitor_eval = GPUMonitor() if GPUMonitor else None
        if monitor_eval: monitor_eval.start()
        
        start_eval = time.time()
        scores, all_labels, all_preds = evaluate(test_loader, model, tokenizer, device, args)
        total_time_eval = time.time() - start_eval
        
        if monitor_eval:
            avg_gpu_power_eval, _ = monitor_eval.stop()

    # 3. LOGGING
    os.makedirs(os.path.dirname(args.log_file_path), exist_ok=True)
    result_data = {
        "total_time_train": total_time_train,
        "avg_gpu_power_train_W": avg_gpu_power_train,
        "total_time_eval": total_time_eval,
        "avg_gpu_power_eval_W": avg_gpu_power_eval,
        "precision": scores['precision'],
        "recall": scores['recall'],
        "f1": scores['f1'],
        "all_labels": all_labels,
        "all_preds": all_preds
    }
    
    with open(args.log_file_path, "w") as f:
        json.dump(result_data, f, indent=4)
    
    print(f"Results saved to {args.log_file_path}")
    print(f"Final F1: {scores['f1']:.4f}")

if __name__ == "__main__":
    main()
