import os
import torch
import time
import json
import sys
from vllm import LLM, SamplingParams
from qaie_const import TOOLKIT_PATH, MODEL_NAME_BASE

# Add toolkit path for GPUMonitor
sys.path.append(TOOLKIT_PATH)
try:
    from gpu_monitor import GPUMonitor
except ImportError:
    print(f"Warning: GPUMonitor not found in {TOOLKIT_PATH}")
    GPUMonitor = None

class VLLMWrapper:
    def __init__(self, model_name=MODEL_NAME_BASE, max_model_len=8192):
        print(f"Initializing vLLM with model: {model_name}")
        self.model_name = model_name
        self.llm = LLM(
            model=model_name,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            max_model_len=max_model_len,
            # max_num_seqs=1024, # Adjust based on GPU memory
            seed=0
        )
        self.sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            max_tokens=512,
            stop=["\n", "####"] 
        )

    def generate(self, prompts, sampling_params=None, use_gpu_monitor=True, monitor_name="llm_generation"):
        if sampling_params is None:
            sampling_params = self.sampling_params
            
        gpu_stats = {}
        if use_gpu_monitor and GPUMonitor:
            monitor = GPUMonitor()
            monitor.start()
            
        outputs = self.llm.generate(prompts, sampling_params)
        
        if use_gpu_monitor and GPUMonitor:
            avg_watt, total_time = monitor.stop()
            gpu_stats = {
                "monitor_name": monitor_name,
                "avg_watt": avg_watt,
                "total_time": total_time,
                "n_prompts": len(prompts)
            }
            
        responses = [output.outputs[0].text.strip() for output in outputs]
        
        return responses, gpu_stats

    def chat(self, messages, sampling_params=None, use_gpu_monitor=True, monitor_name="llm_chat"):
        if sampling_params is None:
            sampling_params = self.sampling_params
            
        gpu_stats = {}
        if use_gpu_monitor and GPUMonitor:
            monitor = GPUMonitor()
            monitor.start()
            
        outputs = self.llm.chat(messages, sampling_params)
        
        if use_gpu_monitor and GPUMonitor:
            avg_watt, total_time = monitor.stop()
            gpu_stats = {
                "monitor_name": monitor_name,
                "avg_watt": avg_watt,
                "total_time": total_time,
                "n_messages": len(messages)
            }
            
        responses = [output.outputs[0].text.strip() for output in outputs]
        
        return responses, gpu_stats

# Global instance to avoid multiple initializations
_llm_instance = None

def get_llm(model_name="google/gemma-4-31b"):
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = VLLMWrapper(model_name=model_name)
    return _llm_instance

class LLM_Old:
    # Dummy class for compatibility if needed during transition
    pass

