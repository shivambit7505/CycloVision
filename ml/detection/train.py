# -*- coding: utf-8 -*-
"""
CycloVision AI - YOLOv8 Tropical Cyclone Detection Model Training
Applies transfer learning from pretrained weights (yolov8n.pt).
Automatically adapts training configuration based on CUDA vs CPU hardware.
"""

import os
import shutil
import json
import torch
import yaml
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET_YAML = os.path.join(BASE_DIR, 'datasets', 'cyclone_detection', 'data.yaml')
OUTPUT_MODEL_DIR = os.path.join(BASE_DIR, 'models', 'yolo')
OUTPUT_REPORT_DIR = os.path.join(BASE_DIR, 'reports', 'yolo')
PRETRAINED_WEIGHTS = os.path.join(BASE_DIR, 'yolov8n.pt')

def verify_dataset(data_yaml_path: str):
    if not os.path.exists(data_yaml_path):
        raise FileNotFoundError(f"data.yaml not found at {data_yaml_path}")
    with open(data_yaml_path, 'r') as f:
        cfg = yaml.safe_load(f)
    ds_root = cfg.get('path', os.path.dirname(data_yaml_path))
    names = cfg.get('names', {})
    if not names:
        raise ValueError("No classes defined in data.yaml!")
    for split in ['train', 'val']:
        img_dir = os.path.join(ds_root, split, 'images')
        lbl_dir = os.path.join(ds_root, split, 'labels')
        if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
            raise FileNotFoundError(f"Missing {split} image/label directory!")
        img_files = os.listdir(img_dir)
        if len(img_files) == 0:
            raise ValueError(f"Split {split} has 0 images!")
        for ifile in img_files:
            lfile = os.path.splitext(ifile)[0] + '.txt'
            lpath = os.path.join(lbl_dir, lfile)
            if not os.path.exists(lpath):
                raise FileNotFoundError(f"Label file {lpath} missing!")
            with open(lpath, 'r') as lf:
                for line in lf:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    if len(parts) != 5:
                        raise ValueError(f"Malformed label in {lpath}: {line}")
                    cls_id, cx, cy, bw, bh = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    if cls_id not in names:
                        raise ValueError(f"Invalid class index {cls_id} in {lpath}")
                    if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 < bw <= 1.0 and 0.0 < bh <= 1.0):
                        raise ValueError(f"Coordinate out of bounds [0, 1] in {lpath}: {line}")
    print("[VERIFIED] Dataset structure and label bounds passed strict integrity check.")

def get_hardware_config():
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        batch_size = 16 if vram_gb >= 6.0 else 8
        print(f"[HARDWARE] CUDA GPU Detected: {gpu_name} ({vram_gb:.2f} GB VRAM)")
        return {
            'device': 0, 'imgsz': 512, 'batch': batch_size,
            'epochs': 25, 'patience': 10, 'workers': 2,
            'hardware': f"CUDA GPU ({gpu_name})"
        }
    else:
        num_threads = torch.get_num_threads()
        print(f"[HARDWARE] CPU Only Mode ({num_threads} threads). Using optimized lightweight parameters.")
        return {
            'device': 'cpu', 'imgsz': 320, 'batch': 8,
            'epochs': 8, 'patience': 4, 'workers': 0,
            'hardware': f"CPU ({num_threads} threads)"
        }

def train():
    os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_REPORT_DIR, exist_ok=True)
    print("=" * 65)
    print("CYCLOVISION AI — YOLOV8 CYCLONE DETECTION MODEL TRAINING")
    print("=" * 65)
    verify_dataset(DATASET_YAML)
    hw = get_hardware_config()
    weights_path = PRETRAINED_WEIGHTS if os.path.exists(PRETRAINED_WEIGHTS) else 'yolov8n.pt'
    print(f"[MODEL] Loading pretrained weights for transfer learning: {weights_path}")
    model = YOLO(weights_path)
    print(f"[TRAINING] Commencing transfer learning (epochs={hw['epochs']}, imgsz={hw['imgsz']}, batch={hw['batch']})...")
    results = model.train(
        data=DATASET_YAML,
        epochs=hw['epochs'],
        imgsz=hw['imgsz'],
        batch=hw['batch'],
        device=hw['device'],
        patience=hw['patience'],
        workers=hw['workers'],
        project=OUTPUT_REPORT_DIR,
        name='train_run',
        exist_ok=True,
        verbose=True
    )
    save_dir = results.save_dir if hasattr(results, 'save_dir') else os.path.join(OUTPUT_REPORT_DIR, 'train_run')
    best_weights_src = os.path.join(save_dir, 'weights', 'best.pt')
    last_weights_src = os.path.join(save_dir, 'weights', 'last.pt')
    best_weights_dst = os.path.join(OUTPUT_MODEL_DIR, 'best.pt')
    last_weights_dst = os.path.join(OUTPUT_MODEL_DIR, 'last.pt')
    if os.path.exists(best_weights_src):
        shutil.copy(best_weights_src, best_weights_dst)
        print(f"[SAVED] Best model checkpoint saved to: {best_weights_dst}")
    if os.path.exists(last_weights_src):
        shutil.copy(last_weights_src, last_weights_dst)
        print(f"[SAVED] Last model checkpoint saved to: {last_weights_dst}")
    summary = {
        'hardware': hw['hardware'],
        'imgsz': hw['imgsz'],
        'epochs': hw['epochs'],
        'batch': hw['batch'],
        'best_weights': best_weights_dst,
        'save_dir': str(save_dir)
    }
    with open(os.path.join(OUTPUT_REPORT_DIR, 'train_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print("\n[COMPLETED] Training pipeline completed successfully.")
    return summary

if __name__ == '__main__':
    train()
