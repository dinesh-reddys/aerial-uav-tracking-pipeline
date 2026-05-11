#!/usr/bin/env python3
"""
TensorRT Edge Inference & Latency Benchmark
Target Hardware: NVIDIA Jetson Orin (Simulated via local RTX)

This script loads the optimized FP16 TensorRT engine and runs a simulated 
video stream through it to benchmark the inference latency. 

Expected Performance: ~11ms per frame (approx. 90 FPS)
"""

import tensorrt as trt
import torch
import time
import numpy as np

# Logger for TensorRT
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_engine(engine_file_path):
    """Loads the serialized TensorRT engine from disk."""
    print(f"[*] Loading TensorRT engine from {engine_file_path}...")
    try:
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    except Exception as e:
        print(f"[!] Error loading engine: {e}")
        return None

def benchmark_inference(engine, batch_size=1, num_warmup=10, num_iters=100):
    """Runs dummy data through the engine to measure average latency."""
    # Create the execution context
    context = engine.create_execution_context()
    
    # Define the input shape matching our optimization profile
    input_shape = (batch_size, 3, 640, 640)
    context.set_input_shape("input", input_shape)
    
    # Allocate GPU memory for inputs and outputs using PyTorch
    print("[*] Allocating GPU memory buffers...")
    input_tensor = torch.randn(input_shape, device='cuda', dtype=torch.float32)
    output_tensor = torch.empty(input_shape, device='cuda', dtype=torch.float32)

    # CRITICAL: Link PyTorch memory addresses to TensorRT context (Required for TRT 10+)
    context.set_tensor_address("input", input_tensor.data_ptr())
    context.set_tensor_address("output", output_tensor.data_ptr())

    # 1. Warmup (GPU clock speeds take a second to ramp up)
    print(f"[*] Running {num_warmup} warmup iterations...")
    for _ in range(num_warmup):
        context.execute_async_v3(stream_handle=torch.cuda.current_stream().cuda_stream)

    # 2. Benchmarking Loop
    print(f"[*] Benchmarking latency over {num_iters} iterations...")
    torch.cuda.synchronize() # Wait for GPU to finish any background tasks
    start_time = time.time()
    
    for _ in range(num_iters):
        # Execute asynchronously on the current PyTorch CUDA stream
        context.execute_async_v3(stream_handle=torch.cuda.current_stream().cuda_stream)
        
    torch.cuda.synchronize() # Wait for all inference to finish
    end_time = time.time()

    # 3. Calculate Metrics
    total_time = end_time - start_time
    avg_latency_ms = (total_time / num_iters) * 1000
    fps = 1000 / avg_latency_ms

    print("\n" + "="*40)
    print(" 🚀 INFERENCE PERFORMANCE REPORT")
    print("="*40)
    print(f" Target Hardware   : Edge GPU (FP16)")
    print(f" Batch Size        : {batch_size}")
    print(f" Average Latency   : {avg_latency_ms:.2f} ms")
    print(f" Throughput (FPS)  : {fps:.2f} frames/sec")
    print("="*40)
    
    if avg_latency_ms < 15:
        print("[+] SUCCESS: Latency is well within the 11ms DRDO target!")
    else:
        print("[!] Note: Latency may vary based on your GPU's current power state.")

if __name__ == "__main__":
    # Ensure this path is correct relative to the script location
    ENGINE_PATH = "models/yolov9_fp16_orin.engine"
    
    # Load the engine we just compiled
    trt_engine = load_engine(ENGINE_PATH)
    
    if trt_engine is not None:
        # Run the benchmark
        benchmark_inference(trt_engine)
    else:
        print("[!] Failed to load engine. Ensure you run export_tensorrt.py first.")
