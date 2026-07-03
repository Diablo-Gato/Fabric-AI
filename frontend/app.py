"""
FABRIC-AI Streamlit Frontend
Shows generated scenes, YOLO detection outputs, and training stats.
"""

import csv
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "backend"))

OUTPUT_DIR = PROJECT_ROOT / "output"
# Search the direct output folder where BlenderProc logs say it is writing:
IMAGES_DIR = OUTPUT_DIR
ANNOTATIONS_FILE = OUTPUT_DIR / "latest_annotations.json"
YOLO_LABEL_DIR = PROJECT_ROOT / "datasets" / "yolo" / "train" / "labels"
TRAIN_DIR = PROJECT_ROOT / "runs" / "detect" / "runs" / "train" / "fabric_ai_v1"
DETECT_RESULTS_DIR = PROJECT_ROOT / "runs" / "detect" / "streamlit_results"
MODEL_PATH = TRAIN_DIR / "weights" / "best.pt"
FALLBACK_MODEL = PROJECT_ROOT / "yolov8n.pt"
PIPELINE_SCRIPT = PROJECT_ROOT / "run_pipeline.bat"
CONFIGS_PATH = PROJECT_ROOT / "configs" / "scene_configs_100.json"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fabric_ai_frontend")

RAG_AVAILABLE = (PROJECT_ROOT / "backend" / "rag_service.py").exists()

YOLO_AVAILABLE = False
try:
    import ultralytics  # type: ignore
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False


def safe_json_load(path: Path) -> Any:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning(f"Unable to load JSON {path}: {exc}")
        return {}


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}

def list_images(directory: Path) -> List[Path]:
    if not directory.exists():
        return []
    return sorted(
        [
            p
            for p in directory.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )


def parse_training_stats() -> Dict[str, str]:
    stats = {"mAP50": "0.000", "mAP50-95": "0.000", "precision": "0.000", "recall": "0.000"}
    csv_path = TRAIN_DIR / "results.csv"
    if not csv_path.exists():
        return stats
    try:
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            last = None
            for row in reader:
                last = row
            if last is None:
                return stats
            for key, alias in [
                ("metrics/mAP_0.5", "mAP50"),
                ("metrics/mAP_0.5:0.95", "mAP50-95"),
                ("metrics/precision", "precision"),
                ("metrics/recall", "recall"),
            ]:
                if key in last and last[key] != "":
                    stats[alias] = f"{float(last[key]):.3f}"
            if stats["mAP50"] == "0.000" and "mAP_50" in last and last["mAP_50"] != "":
                stats["mAP50"] = f"{float(last['mAP_50']):.3f}"
            if stats["mAP50-95"] == "0.000" and "mAP_50_95" in last and last["mAP_50_95"] != "":
                stats["mAP50-95"] = f"{float(last['mAP_50_95']):.3f}"
            return stats
    except Exception as exc:
        logger.warning(f"Failed to parse training stats: {exc}")
        return stats


def annotation_summary() -> Dict[str, Any]:
    data = safe_json_load(ANNOTATIONS_FILE)
    annotations = data.get("annotations", []) if isinstance(data, dict) else []
    categories = data.get("categories", []) if isinstance(data, dict) else []
    return {
        "scenes": len(list_images(IMAGES_DIR)),
        "annotations": len(annotations),
        "classes": len(categories) if categories else 5,
        "category_counts": {cat.get("name", "unknown"): 0 for cat in categories} if categories else {},
    }


def run_pipeline(prompt: str) -> List[str]:
    if not PIPELINE_SCRIPT.exists():
        raise FileNotFoundError(f"run_pipeline.bat not found at {PIPELINE_SCRIPT}")
    command = ["cmd.exe", "/c", "call", str(PIPELINE_SCRIPT), prompt]
    result = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True)
    logs = []
    if result.stdout:
        logs.extend(result.stdout.splitlines())
    if result.stderr:
        logs.extend(result.stderr.splitlines())
    if result.returncode != 0:
        raise RuntimeError("Pipeline failed. See logs for details.")
    return logs


def run_yolo_detection(image_path: Path, confidence: float, iou: float) -> Dict[str, Any]:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    model_path = MODEL_PATH if MODEL_PATH.exists() else FALLBACK_MODEL
    if not model_path.exists():
        raise FileNotFoundError(f"YOLO model not found: {model_path}")

    YOLO_LABEL_DIR.mkdir(parents=True, exist_ok=True)
    result_json = YOLO_LABEL_DIR / f"{image_path.stem}_yolo_results.json"
    result_json.unlink(missing_ok=True)

    try:
        from run_yolo_stage import run_yolo_stage
        run_yolo_stage(
            image_path=image_path,
            model_path=model_path,
            output_labels_dir=YOLO_LABEL_DIR,
            confidence=confidence,
            iou=iou,
        )
    except Exception:
        if not YOLO_AVAILABLE:
            raise
        from ultralytics import YOLO  # type: ignore
        model = YOLO(str(model_path))
        results = model(
            str(image_path),
            conf=confidence,
            iou=iou,
            save=True,
            device="cpu",
            project=str(PROJECT_ROOT / "runs" / "detect"),
            name="streamlit_results",
            exist_ok=True,
        )
        detections = []
        cls_map = {0: "car", 1: "auto", 2: "truck", 3: "bus", 4: "cow"}
        for r in results:
            for box in r.boxes:
                label = cls_map.get(int(box.cls), "unknown")
                confidence_val = float(box.conf)
                bbox = [round(float(v), 1) for v in box.xyxy[0]]
                detections.append({"label": label, "confidence": round(confidence_val, 3), "bbox": bbox})
        result_json.write_text(
            json.dumps({"image": str(image_path), "detections": detections, "confidence": confidence, "iou": iou}, indent=2),
            encoding="utf-8",
        )

    return safe_json_load(result_json)


def training_charts() -> List[Path]:
    candidates = [
        TRAIN_DIR / "results.png",
        TRAIN_DIR / "confusion_matrix_normalized.png",
        TRAIN_DIR / "BoxPR_curve.png",
        TRAIN_DIR / "BoxF1_curve.png",
        TRAIN_DIR / "train_batch0.jpg",
        TRAIN_DIR / "val_batch0_pred.jpg",
        TRAIN_DIR / "val_batch0_labels.jpg",
    ]
    return [p for p in candidates if p.exists()]


def display_scene_cards(images: List[Path]) -> None:
    if not images:
        st.info("No scene images found. Generate a scene first.")
        return
    cols = st.columns(3)
    for idx, image_path in enumerate(images[::-1]):
        with cols[idx % 3]:
            st.image(str(image_path), caption=image_path.name, width="stretch")
            if st.button(f"Select {image_path.name}", key=f"scene_{idx}"):
                st.session_state.selected_scene = str(image_path)
                st.success(f"Selected {image_path.name} for detection.")


st.set_page_config(page_title="FABRIC-AI", layout="wide", initial_sidebar_state="expanded")

if "prompt" not in st.session_state:
    st.session_state.prompt = ""
if "pipeline_logs" not in st.session_state:
    st.session_state.pipeline_logs = []
if "selected_scene" not in st.session_state:
    st.session_state.selected_scene = None
if "detection_result" not in st.session_state:
    st.session_state.detection_result = None
if "images" not in st.session_state:
    st.session_state.images = list_images(IMAGES_DIR)
if "detect_images" not in st.session_state:
    st.session_state.detect_images = list_images(DETECT_RESULTS_DIR)
if "charts" not in st.session_state:
    st.session_state.charts = training_charts()
if "summary" not in st.session_state:
    st.session_state.summary = annotation_summary()
if "train_stats" not in st.session_state:
    st.session_state.train_stats = parse_training_stats()

train_stats = st.session_state.train_stats
images = st.session_state.images
detect_images = st.session_state.detect_images
summary = st.session_state.summary
charts = st.session_state.charts

with st.sidebar:
    st.header("FABRIC-AI")
    st.markdown("Synthetic Indian road scenes with YOLO analysis.")
    st.markdown("---")
    st.markdown(f"**RAG available:** {'Yes' if RAG_AVAILABLE else 'No'}")
    st.markdown(f"**YOLO available:** {'Yes' if YOLO_AVAILABLE else 'No'}")
    st.markdown(f"**Model path:** {'found' if MODEL_PATH.exists() else 'fallback' if FALLBACK_MODEL.exists() else 'missing'}")
    st.markdown(f"**Scenes:** {len(images)}")
    st.markdown(f"**Detection outputs:** {len(detect_images)}")
    st.markdown(f"**Training charts:** {len(charts)}")
    st.markdown("---")
    st.markdown("### Training stats")
    st.markdown(f"- mAP @ 50: **{train_stats['mAP50']}**")
    st.markdown(f"- mAP @ 50-95: **{train_stats['mAP50-95']}**")
    st.markdown(f"- Precision: **{train_stats['precision']}**")
    st.markdown(f"- Recall: **{train_stats['recall']}**")
    st.markdown("---")
    st.markdown("### Dataset summary")
    st.markdown(f"- Scenes: **{summary['scenes']}**")
    st.markdown(f"- Annotations: **{summary['annotations']}**")
    st.markdown(f"- Classes: **{summary['classes']}**")

st.title("FABRIC-AI Dashboard")

tabs = st.tabs(["Generate", "Detect", "Results", "Dataset"])

with tabs[0]:
    st.subheader("Scene generation")
    prompt_text = st.text_area("Describe your scene", value=st.session_state.prompt, height=120)
    st.session_state.prompt = prompt_text
    if st.button("Run pipeline", type="primary"):
        if not st.session_state.prompt.strip():
            st.warning("Enter a prompt before generating.")
        else:
            with st.spinner("Running the generation pipeline..."):
                try:
                    logs = run_pipeline(st.session_state.prompt.strip())
                    st.session_state.pipeline_logs = logs
                    st.success("Pipeline completed successfully.")
                    st.session_state.images = list_images(IMAGES_DIR)
                    st.session_state.summary = annotation_summary()
                    st.session_state.train_stats = parse_training_stats()
                except Exception as exc:
                    st.error(f"Pipeline failed: {exc}")
    if st.session_state.pipeline_logs:
        with st.expander("Pipeline logs"):
            st.code("\n".join(st.session_state.pipeline_logs[-100:]), language="bash")
    st.markdown("---")
    st.subheader("Generated scenes")
    display_scene_cards(images)

with tabs[1]:
    st.subheader("YOLO detection")
    if not images:
        st.info("No generated scenes available. Generate first.")
    else:
        selected = Path(st.session_state.selected_scene) if st.session_state.selected_scene else images[-1]
        selected = selected if selected.exists() else images[-1]
        choice = st.selectbox("Pick a scene", images, index=images.index(selected), format_func=lambda p: p.name)
        st.session_state.selected_scene = str(choice)
        st.image(str(choice), caption=choice.name, width="stretch")
        conf = st.slider("Confidence", 0.05, 0.95, 0.25, 0.05)
        iou = st.slider("IoU", 0.10, 0.90, 0.45, 0.05)
        if st.button("Run YOLO inference", type="primary"):
            with st.spinner("Running YOLO..."):
                try:
                    result = run_yolo_detection(choice, conf, iou)
                    st.session_state.detection_result = result
                    st.success("YOLO inference completed.")
                    st.session_state.detect_images = list_images(DETECT_RESULTS_DIR)
                except Exception as exc:
                    st.error(f"Detection failed: {exc}")
        if st.session_state.detection_result:
            detections = st.session_state.detection_result.get("detections", [])
            st.metric("Detections", len(detections))
            if detections:
                st.dataframe(pd.DataFrame(detections), width="stretch")
            else:
                st.info("No detections found.")
    st.markdown("---")
    st.subheader("Existing YOLO output images")
    if detect_images:
        cols = st.columns(4)
        for idx, dpath in enumerate(detect_images[::-1]):
            with cols[idx % 4]:
                st.image(str(dpath), caption=dpath.name, width="stretch")
    else:
        st.info("No YOLO output images found yet.")

with tabs[2]:
    st.subheader("Training results")
    st.markdown(f"**mAP @ 50:** {train_stats['mAP50']}  •  **mAP @ 50-95:** {train_stats['mAP50-95']}  •  **Precision:** {train_stats['precision']}  •  **Recall:** {train_stats['recall']}")
    if charts:
        for chart in charts:
            st.image(str(chart), caption=chart.name, width="stretch")
    else:
        st.info("No training charts found in the training directory.")

with tabs[3]:
    st.subheader("Dataset overview")
    st.markdown(f"**Scenes:** {summary['scenes']}  •  **Annotations:** {summary['annotations']}  •  **Classes:** {summary['classes']}")
    if summary['category_counts']:
        df = pd.DataFrame([summary['category_counts']]).T.reset_index()
        df.columns = ["class", "count"]
        st.dataframe(df, width="stretch")
    else:
        st.info("Category metadata not available in latest_annotations.json.")
    if CONFIGS_PATH.exists():
        configs = safe_json_load(CONFIGS_PATH)
        if isinstance(configs, list) and configs:
            st.markdown("---")
            st.subheader("Scene config summary")
            counts = {"weather": {}, "time_of_day": {}, "road_type": {}}
            for cfg in configs:
                for key in counts:
                    value = str(cfg.get(key, "unknown"))
                    counts[key][value] = counts[key].get(value, 0) + 1
            for key, vals in counts.items():
                st.markdown(f"**{key.replace('_', ' ').title()}**")
                df = pd.DataFrame(list(vals.items()), columns=[key, "count"])
                st.dataframe(df, width="stretch")
    else:
        st.info("No scene config file found.")
