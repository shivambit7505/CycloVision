# -*- coding: utf-8 -*-
"""
CycloVision AI - YOLOv8 Model Inference Engine
Runs inference on unseen test/validation satellite crops.
Saves annotated bounding box outputs to reports/yolo/predictions/
"""
import os
import glob
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WEIGHTS_PATH = os.path.join(BASE_DIR, 'models', 'yolo', 'best.pt')
TEST_IMAGES_DIR = os.path.join(BASE_DIR, 'datasets', 'cyclone_detection', 'test', 'images')
PRED_OUTPUT_DIR = os.path.join(BASE_DIR, 'reports', 'yolo', 'predictions')

def run_predictions(weights_path=None, source_dir=None, num_images=5):
    weights = weights_path or WEIGHTS_PATH
    if not os.path.exists(weights):
        fallback = os.path.join(BASE_DIR, 'models', 'yolo', 'last.pt')
        weights = fallback if os.path.exists(fallback) else os.path.join(BASE_DIR, 'yolov8n.pt')
        
    src_dir = source_dir or TEST_IMAGES_DIR
    os.makedirs(PRED_OUTPUT_DIR, exist_ok=True)
    
    image_files = sorted(glob.glob(os.path.join(src_dir, '*.png')) + glob.glob(os.path.join(src_dir, '*.jpg')))
    if not image_files:
        raise FileNotFoundError(f"No test images found in {src_dir}")
        
    test_subset = image_files[:max(5, num_images)]
    print(f"[INFERENCE] Loading model from: {weights}")
    model = YOLO(weights)
    
    print(f"[INFERENCE] Running prediction on {len(test_subset)} unseen test images...")
    results = model.predict(
        source=test_subset,
        save=True,
        project=os.path.dirname(PRED_OUTPUT_DIR),
        name='predictions',
        exist_ok=True,
        conf=0.25
    )
    
    predictions_summary = []
    print("\n" + "=" * 65)
    print("CYCLOVISION AI — INFERENCE RESULTS ON UNSEEN SATELLITE CROPS")
    print("=" * 65)
    for r in results:
        fname = os.path.basename(r.path)
        boxes = r.boxes
        box_count = len(boxes)
        details = []
        for b in boxes:
            cls_id = int(b.cls[0])
            conf = float(b.conf[0])
            name = model.names[cls_id]
            xyxy = [round(float(x), 1) for x in b.xyxy[0].tolist()]
            details.append({'class': name, 'confidence': round(conf, 3), 'bbox': xyxy})
            
        predictions_summary.append({'filename': fname, 'detections': details})
        print(f"Image: {fname} | Detections: {box_count}")
        for d in details:
            print(f"   -> Class: {d['class']:<14} | Confidence: {d['confidence']:.2f} | BBox: {d['bbox']}")
            
    print("=" * 65)
    print(f"[SAVED] Prediction visualizations saved to: {PRED_OUTPUT_DIR}")
    return predictions_summary

if __name__ == '__main__':
    run_predictions()
