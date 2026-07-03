from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import json
import random

app = FastAPI(
    title="FABRIC-AI API",
    description="Synthetic Indian Road Scene Generator + YOLOv8 Detector",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / \
    "runs/detect/runs/train/fabric_ai_v1/weights/best.pt"

# Load model once at startup
model = None

@app.on_event("startup")
async def load_model():
    global model
    if MODEL_PATH.exists():
        try:
            from ultralytics import YOLO
            model = YOLO(str(MODEL_PATH))
            print("[INFO] YOLOv8 model loaded successfully")
        except Exception as e:
            print(f"[WARN] Could not load model: {e}")
    else:
        print(f"[WARN] Model not found at {MODEL_PATH}")

# ---------- MODELS ----------
class SceneConfig(BaseModel):
    weather: str = "clear"
    time_of_day: str = "midday"
    object_density: float = 0.6
    auto_rickshaw_count: int = 3
    cow: bool = False
    road_type: str = "asphalt"
    buildings: bool = True

class DetectRequest(BaseModel):
    image_path: str
    confidence: float = 0.25

# ---------- ROUTES ----------
@app.get("/")
def root():
    return {
        "name": "FABRIC-AI API",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": model is not None,
        "endpoints": [
            "/generate",
            "/detect",
            "/stats",
            "/docs"
        ]
    }

@app.get("/stats")
def get_stats():
    images_dir = PROJECT_ROOT / "output" / "images"
    image_count = len(list(images_dir.glob("*.png"))) \
        if images_dir.exists() else 0

    ann_path = PROJECT_ROOT / "output" / "latest_annotations.json"
    ann_count = 0
    if ann_path.exists():
        with open(ann_path) as f:
            data = json.load(f)
        ann_count = len(data.get("annotations", []))

    return {
        "total_images": image_count,
        "total_annotations": ann_count,
        "model_loaded": model is not None,
        "model_metrics": {
            "mAP50": 0.758,
            "mAP50_95": 0.574,
            "precision": 0.867,
            "recall": 0.582
        },
        "classes": ["car", "auto", "truck", "bus", "cow"]
    }

@app.post("/generate")
def generate_scene(config: SceneConfig):
    try:
        images_dir = PROJECT_ROOT / "output" / "images"
        if not images_dir.exists():
            return {
                "status": "error",
                "message": "No images found. Run blenderproc first."
            }

        images = sorted(images_dir.glob("*.png"))
        if not images:
            return {
                "status": "error",
                "message": "No PNG images found in output/images/"
            }

        # Return a random existing image for demo
        img = random.choice(images)

        # Get annotation count for this image
        ann_path = PROJECT_ROOT / "output" / "latest_annotations.json"
        ann_count = 0
        if ann_path.exists():
            with open(ann_path) as f:
                data = json.load(f)
            img_id = int(img.stem.split("_")[1])
            ann_count = sum(
                1 for a in data.get("annotations", [])
                if a["image_id"] == img_id
            )

        return {
            "status": "success",
            "image_path": str(img),
            "image_name": img.name,
            "annotation_count": ann_count,
            "config_used": config.dict()
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/detect")
def detect_objects(request: DetectRequest):
    global model

    if model is None:
        return {
            "status": "error",
            "message": "Model not loaded. Check model path."
        }

    try:
        img_path = Path(request.image_path)
        if not img_path.exists():
            return {
                "status": "error",
                "message": f"Image not found: {img_path}"
            }

        results = model(
            str(img_path),
            conf=request.confidence,
            save=True,
            device='cpu',
            project=str(PROJECT_ROOT / "runs/detect"),
            name="api_results",
            exist_ok=True
        )

        names = {
            0: "car", 1: "auto",
            2: "truck", 3: "bus", 4: "cow"
        }
        detections = []
        for r in results:
            for box in r.boxes:
                detections.append({
                    "label": names.get(int(box.cls), "unknown"),
                    "confidence": round(float(box.conf), 3),
                    "bbox": [
                        round(float(x), 2)
                        for x in box.xyxy[0]
                    ]
                })

        result_path = str(
            PROJECT_ROOT / "runs/detect/api_results" / img_path.name
        )

        return {
            "status": "success",
            "total_detections": len(detections),
            "detections": detections,
            "result_path": result_path
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}