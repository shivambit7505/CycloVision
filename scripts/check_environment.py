#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CycloVision AI — Rapid Environment & Hardware Acceleration Check
Audits Python, PyTorch, CUDA/GPU, YOLO, Node, npm, and configures optimal training parameters.
"""

import os
import sys
import platform
import subprocess

# Ensure Windows terminal UTF-8 encoding compatibility
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

def get_cli_version(command):
    """Executes a CLI command and returns its standard output stripped."""
    try:
        res = subprocess.run(
            command,
            capture_output=True,
            text=True,
            shell=True,
            timeout=5
        )
        if res.returncode == 0:
            return res.stdout.strip()
    except Exception:
        pass
    return "Not Available"

def audit_environment():
    print("=" * 72)
    print("CYCLOVISION AI -- ENVIRONMENT & HARDWARE ACCELERATION AUDIT")
    print("=" * 72)

    # 1. Python Details
    py_version = sys.version.split()[0]
    py_platform = platform.platform()
    print(f"* Python Version     : {py_version} ({sys.executable})")
    print(f"* OS / Platform      : {py_platform}")

    # 2. PyTorch & CUDA Details
    try:
        import torch
        torch_ver = torch.__version__
        cuda_avail = torch.cuda.is_available()
    except ImportError:
        torch_ver = "NOT INSTALLED"
        cuda_avail = False
        torch = None

    print(f"* PyTorch Version    : {torch_ver}")
    print(f"* CUDA Availability  : {cuda_avail}")

    if cuda_avail and torch is not None:
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        print(f"* GPU Device Name    : {gpu_name} ({gpu_count} device(s), {gpu_mem_gb} GB VRAM)")
        compute_mode = "GPU (CUDA)"
    else:
        gpu_name = "None (CPU Mode)"
        print(f"* GPU Device Name    : {gpu_name}")
        compute_mode = "CPU (Optimized Multithreaded)"

    # 3. Ultralytics / YOLO Details
    try:
        import ultralytics
        yolo_avail = f"Available (v{ultralytics.__version__})"
    except ImportError:
        yolo_avail = "NOT INSTALLED"
    print(f"* YOLO Availability  : {yolo_avail}")

    # 4. Node.js & npm Details
    node_ver = get_cli_version("node -v")
    npm_ver = get_cli_version("npm -v")
    print(f"* Node Version       : {node_ver}")
    print(f"* npm Version        : {npm_ver}")

    # 5. Training & Compute Configuration Engine
    print("\n" + "-" * 72)
    print(f"[COMPUTE PROFILE: {compute_mode}]")
    print("-" * 72)

    cpu_cores = os.cpu_count() or 4
    if cuda_avail and torch is not None:
        recommended_device = "cuda"
        batch_size = 32
        epochs = 30
        num_samples = 2500
        dataloader_workers = 4
        print("  [CUDA ACCELERATION ENABLED]")
        print(f"  * Primary Device     : {recommended_device}:0")
        print(f"  * Batch Size         : {batch_size}")
        print(f"  * Max Epochs         : {epochs}")
        print(f"  * Dataset Samples    : {num_samples}")
        print(f"  * DataLoader Workers : {dataloader_workers}")
        print(f"  * Mixed Precision    : torch.cuda.amp.autocast(enabled=True)")
    else:
        recommended_device = "cpu"
        threads = max(1, cpu_cores - 1)
        if torch is not None:
            torch.set_num_threads(threads)
        batch_size = 16
        epochs = 8
        num_samples = 400
        dataloader_workers = 0
        print("  [FAST CPU PROFILE ACTIVE -- REDUCED TRAINING SIZE/EPOCHS FOR HACKATHON]")
        print(f"  * Primary Device     : {recommended_device}")
        print(f"  * CPU Cores / Threads: {cpu_cores} total -> {threads} active threads")
        print(f"  * Batch Size         : {batch_size} (cache-friendly)")
        print(f"  * Max Epochs         : {epochs} (fast convergence in < 60s)")
        print(f"  * Dataset Samples    : {num_samples} (curated representative subset)")
        print(f"  * DataLoader Workers : {dataloader_workers} (avoids Windows process-spawn overhead)")

    print("=" * 72)
    print("[SUCCESS] ENVIRONMENT AUDIT COMPLETED SUCCESSFULLY\n")

    return {
        "python_version": py_version,
        "pytorch_version": torch_ver,
        "cuda_available": cuda_avail,
        "gpu_name": gpu_name,
        "yolo_available": yolo_avail,
        "node_version": node_ver,
        "npm_version": npm_ver,
        "recommended_device": recommended_device,
        "recommended_batch_size": batch_size,
        "recommended_epochs": epochs
    }

if __name__ == "__main__":
    audit_environment()
