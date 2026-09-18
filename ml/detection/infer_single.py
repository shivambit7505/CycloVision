"""
CycloVision AI - Single Image Cyclone Detection Inference
Loads fine-tuned YOLO model and runs detection on an input satellite image.
Returns structured JSON with detection results and saves an annotated visualization.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from ultralytics import YOLO
import cv2

DEFAULT_MODEL_PATH = Path("models/yolo/best.pt")
DEFAULT_OUTPUT_DIR = Path("reports/yolo/predictions")


def infer_cyclone(image_path: str, model_path: str = None, conf_threshold: float = 0.25, output_dir: str = None):
    """
    Run cyclone detection inference on a single satellite image.
    
    Args:
        image_path: Path to satellite image (PNG/JPG)
        model_path: Path to YOLO weights (.pt)
        conf_threshold: Minimum detection confidence threshold
        output_dir: Directory to save annotated visualization image
        
    Returns:
        dict with detection findings (cyclone_detected, detections, annotated_image_path)
    """
    img_p = Path(image_path)
    if not img_p.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")
        
    m_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
    if not m_path.exists():
        raise FileNotFoundError(f"Model weights not found: {m_path}. Please train the model first.")
        
    out_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model = YOLO(str(m_path))
    
    # Run prediction
    results = model.predict(
        source=str(img_p),
        conf=conf_threshold,
        save=False,
        verbose=False
    )
    
    result = results[0]
    detections = []
    
    # Extract boxes
    if result.boxes is not None and len(result.boxes) > 0:
        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            class_name = result.names.get(cls_id, f"class_{cls_id}")
            confidence = float(box.conf[0].item())
            # xyxy format
            coords = [round(x, 2) for x in box.xyxy[0].tolist()]
            # normalized xywh
            xywhn = [round(x, 4) for x in box.xywhn[0].tolist()] if box.xywhn is not None else []
            
            detections.append({
                "class_id": cls_id,
                "class_name": class_name,
                "confidence": round(confidence, 4),
                "bbox_xyxy": coords,
                "bbox_xywh_norm": xywhn
            })
            
    cyclone_detected = len(detections) > 0
    
    # Render and save annotated image
    annotated_img = result.plot()
    out_filename = f"infer_{img_p.stem}.jpg"
    out_filepath = out_dir / out_filename
    cv2.imwrite(str(out_filepath), annotated_img)
    
    output = {
        "image_path": str(img_p.resolve()),
        "model_used": str(m_path.resolve()),
        "cyclone_detected": cyclone_detected,
        "detection_count": len(detections),
        "detections": detections,
        "annotated_image_saved_to": str(out_filepath.resolve())
    }
    
    return output


def main():
    parser = argparse.ArgumentParser(description="CycloVision AI - Single Image Cyclone Detection")
    parser.add_argument("--image", type=str, required=True, help="Path to input satellite image")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL_PATH), help="Path to trained YOLO model (.pt)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    parser.add_argument("--outdir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory for annotated image")
    
    args = parser.parse_args()
    
    try:
        res = infer_cyclone(
            image_path=args.image,
            model_path=args.model,
            conf_threshold=args.conf,
            output_dir=args.outdir
        )
        print(json.dumps(res, indent=2))
    except Exception as e:
        print(json.dumps({"error": str(e), "cyclone_detected": False}))
        sys.exit(1)


if __name__ == "__main__":
    main()
