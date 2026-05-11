# Aerial UAV Tracking Pipeline: Edge-Optimized Object Detection

## Overview
This repository contains an architectural reproduction of a real-time aerial object detection and tracking pipeline. It is designed to detect small, fast-moving Unmanned Aerial Vehicles (UAVs) against complex backgrounds and deploy the inference engine onto edge hardware (NVIDIA Jetson Orin) for mission control interfaces.

> **Security & Compliance Notice:** > This architecture is based on research conducted for defense applications. Due to strict security clearances and NDAs, all proprietary model weights, classified UAV datasets, and specific mission-control integration endpoints have been completely omitted. This repository serves purely as a structural demonstration using public datasets (e.g., VisDrone).

## System Architecture

The pipeline consists of three core engineering pillars:

1. **Detection (YOLOv9 + Custom Attention)**
   - Utilizes a modified YOLOv9 backbone.
   - Integrates a custom dual Spatial and Channel Attention Block (`models/custom_attention.py`) applied post-CSP layers. This forces the network to focus on high-frequency edge features crucial for pinpointing small drones, mitigating background clutter.

2. **Tracking (C++ ByteTrack Integration)**
   - Replaces standard Python-based tracking loops with a high-performance C++ implementation of ByteTrack.
   - Utilizes `pybind11` (`tracking/python_bindings.cpp`) to allow the Python ML pipeline to pass bounding box tensors directly into the C++ memory space, achieving zero-copy transfer where possible and minimizing tracking latency.

3. **Edge Deployment (TensorRT Quantization)**
   - Optimized specifically for the NVIDIA Jetson Orin platform.
   - The PyTorch `.pt` models are exported to ONNX and compiled into FP16 TensorRT engines (`deployment/export_tensorrt.py`), leveraging the Orin's fast FP16 architecture to drastically reduce inference time while maintaining precision.

## Performance Metrics
*When trained and evaluated on the original classified datasets:*
* **Accuracy:** Achieved **94.3% mAP** on small UAV targets.
* **Latency Optimization:** TensorRT FP16 quantization reduced inference latency from **48ms** (native PyTorch) to **11ms** (4.4x speedup).
* **Accuracy Drop:** Quantization resulted in a negligible `< 0.6%` drop in detection accuracy.

## Repository Structure
```text
├── data/                  # Sample/Public UAV testing data
├── models/
│   ├── yolov9/            # Core detection framework
│   └── custom_attention.py# Dual Spatial/Channel attention block
├── tracking/
│   ├── bytetrack_cpp/     # C++ tracker source
│   └── python_bindings.cpp# Pybind11 wrapper
└── deployment/
    ├── export_tensorrt.py # ONNX to TensorRT FP16 conversion script
    └── jetson_inference.py# Edge inference execution script
