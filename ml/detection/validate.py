# -*- coding: utf-8 -*-
"""
CycloVision AI - YOLOv8 Model Validation and Benchmarking
Evaluates precision, recall, F1, mAP50, mAP50-95, and inference latency.
"""
import os
import json
import time
import yaml
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET_YAML = os.path.join(BASE_DIR, 'datasets', 'cyclone_detection', 'data.yaml')
WEIGHTS_PATH = os.path.join(BASE_DIR, 'models', 'yolo', 'best.pt')
REPORT_DIR = os.path.join(BASE_DIR, 'reports', 'yolo')

def validate(weights_path=None, data_yaml=None):
    weights = weights_path or WEIGHTS_PATH
    data = data_yaml or DATASET_YAML
    os.makedirs(REPORT_DIR, exist_ok=True)
    
    if not os.path.exists(weights):
        fallback = os.path.join(BASE_DIR, 'models', 'yolo', 'last.pt')
        weights = fallback if os.path.exists(fallback) else os.path.join(BASE_DIR, 'yolov8n.pt')
        
    print(f"[VALIDATING] Loading model weights from: {weights}")
    model = YOLO(weights)
    
    metrics = model.val(data=data, split='val', project=REPORT_DIR, name='val_run', exist_ok=True)
    
    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)
    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)
    f1 = float(2 * (precision * recall) / (precision + recall + 1e-16))
    
    speed = metrics.speed
    preprocess_ms = float(speed.get('preprocess', 1.0))
    inference_ms = float(speed.get('inference', 15.0))
    postprocess_ms = float(speed.get('postprocess', 1.0))
    total_latency_ms = round(preprocess_ms + inference_ms + postprocess_ms, 2)
    
    report = {
        'model_weights': weights,
        'dataset': data,
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1_score': round(f1, 4),
        'mAP50': round(map50, 4),
        'mAP50_95': round(map50_95, 4),
        'inference_speed_ms': {
            'preprocess': preprocess_ms,
            'inference': inference_ms,
            'postprocess': postprocess_ms,
            'total_latency': total_latency_ms
        },
        'classes': {
            0: 'cyclone_eye',
            1: 'cyclone_body'
        }
    }
    
    json_path = os.path.join(REPORT_DIR, 'validation_report.json')
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)
        
    txt_path = os.path.join(REPORT_DIR, 'metrics.txt')
    with open(txt_path, 'w') as f:
        f.write("=== CYCLOVISION AI YOLOV8 DETECTION BENCHMARK ===\n")
        f.write(f"Weights: {weights}\n")
        f.write(f"Precision: {report['precision']:.4f}\n")
        f.write(f"Recall: {report['recall']:.4f}\n")
        f.write(f"F1 Score: {report['f1_score']:.4f}\n")
        f.write(f"mAP@0.50: {report['mAP50']:.4f}\n")
        f.write(f"mAP@0.50:0.95: {report['mAP50_95']:.4f}\n")
        f.write(f"Total Inference Latency: {total_latency_ms:.2f} ms/frame\n")
        
    print("\n" + "=" * 55)
    print("CYCLONE DETECTION MODEL VALIDATION METRICS:")
    print(f"  Precision:       {report['precision']:.4f}")
    print(f"  Recall:          {report['recall']:.4f}")
    print(f"  F1-Score:        {report['f1_score']:.4f}")
    print(f"  mAP@50:          {report['mAP50']:.4f}")
    print(f"  mAP@50-95:       {report['mAP50_95']:.4f}")
    print(f"  Inference Speed: {total_latency_ms:.2f} ms")
    print("=" * 55)
    return report

if __name__ == '__main__':
    validate()
