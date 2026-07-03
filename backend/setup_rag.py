"""
FABRIC-AI Streamlit Frontend — v2
Fixes: HTML escape bug replaced with st.columns cards
Added: Scene Generation tab, RAG Status tab
"""

import os
import sys
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional
import streamlit as st
import pandas as pd
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_DIR  = PROJECT_ROOT / "output"
BACKEND_URL = "http://localhost:8000"

st.set_page_config(
    page_title="FABRIC-AI",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── STYLES ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main {
    background: #0d0d14 !important;
    color: #dbd8ee !important;
}
section[data-testid="stSidebar"] {
    background: #0a0a10 !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}

/* ── Typography (system stack, no external fonts) ── */
html, body, button, input, select, textarea,
[data-testid="stMarkdownContainer"] {
    font-family: -apple-system, BlinkMacSystemFont,
        "Segoe UI", system-ui, sans-serif !important;
}

/* ── Hero ── */
.fab-hero { text-align:center; padding:2.5rem 0 1.5rem; }
.fab-wordmark {
    font-size: 3.2rem; font-weight:800; letter-spacing:-0.04em;
    background: linear-gradient(130deg,#a78bfa 0%,#7c3aed 45%,#c084fc 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    background-clip:text; display:block; line-height:1.05;
}
.fab-tagline {
    font-size:0.75rem; letter-spacing:0.14em; text-transform:uppercase;
    color:#4e4a6a !important; margin-top:0.4rem;
}
.fab-sub { font-size:0.9rem; color:#7a76a0 !important; margin-top:0.2rem; }
.fab-rule {
    height:1px;
    background:linear-gradient(90deg,transparent,rgba(124,58,237,.35),transparent);
    border:none; margin:1.25rem 0 0.5rem;
}

/* ── KPI card ── */
.kpi {
    background:rgba(255,255,255,0.03);
    border:1px solid rgba(255,255,255,0.08);
    border-radius:14px; padding:1rem 1.2rem;
    position:relative; overflow:hidden;
}
.kpi::after {
    content:''; position:absolute; top:0; left:0; right:0; height:2px;
    background:var(--kc,linear-gradient(90deg,#7c3aed,#a78bfa));
}
.kpi-lbl {
    font-size:0.65rem; font-weight:600; letter-spacing:0.1em;
    text-transform:uppercase; color:#5a5680; margin-bottom:0.3rem;
}
.kpi-val {
    font-size:1.75rem; font-weight:700; letter-spacing:-0.02em;
    color:#f0edff; line-height:1.05;
}
.kpi-tag {
    display:inline-block; margin-top:0.3rem; font-size:0.65rem;
    font-weight:500; padding:2px 8px; border-radius:20px;
}
.tag-purple { background:rgba(167,139,250,.12); color:#a78bfa;
              border:1px solid rgba(167,139,250,.22); }
.tag-green  { background:rgba(52,211,153,.1);   color:#34d399;
              border:1px solid rgba(52,211,153,.2); }
.tag-amber  { background:rgba(251,191,36,.1);   color:#fbbf24;
              border:1px solid rgba(251,191,36,.2); }

/* ── Section heading ── */
.sh {
    font-size:0.68rem; font-weight:700; letter-spacing:0.13em;
    text-transform:uppercase; color:#5a5680;
    display:flex; align-items:center; gap:0.5rem;
    margin:1.4rem 0 0.7rem;
}
.sh::after { content:''; flex:1; height:1px; background:rgba(255,255,255,0.06); }

/* ── Tabs ── */
[data-testid="stTabs"] button[role="tab"] {
    font-size:0.83rem !important; font-weight:500 !important;
    color:#5a5680 !important; padding:0.65rem 1rem !important;
    border-bottom:2px solid transparent !important;
    background:transparent !important;
}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    color:#a78bfa !important; border-bottom-color:#7c3aed !important;
}
[data-testid="stTabs"] [role="tablist"] {
    border-bottom:1px solid rgba(255,255,255,0.06) !important;
    background:transparent !important; gap:0 !important;
}

/* ── Buttons ── */
.stButton > button {
    font-size:0.83rem !important; font-weight:500 !important;
    border-radius:9px !important;
    border:1px solid rgba(124,58,237,0.35) !important;
    background:rgba(124,58,237,0.12) !important;
    color:#c4b5fd !important; transition:all .18s !important;
    padding:0.45rem 1.1rem !important;
}
.stButton > button:hover {
    background:rgba(124,58,237,0.25) !important;
    border-color:rgba(167,139,250,0.55) !important; color:#f0edff !important;
}
.stButton > button[kind="primary"] {
    background:linear-gradient(135deg,#7c3aed,#6d28d9) !important;
    border-color:rgba(167,139,250,0.3) !important; color:#fff !important;
    font-weight:600 !important;
    box-shadow:0 3px 16px rgba(124,58,237,0.28) !important;
}
.stButton > button[kind="primary"]:hover {
    background:linear-gradient(135deg,#8b5cf6,#7c3aed) !important;
    box-shadow:0 5px 22px rgba(124,58,237,0.42) !important;
}

/* ── Inputs / select ── */
[data-baseweb="select"] > div {
    background:rgba(255,255,255,0.04) !important;
    border:1px solid rgba(255,255,255,0.09) !important;
    border-radius:9px !important; color:#dbd8ee !important;
}

/* ── Sliders ── */
[data-baseweb="slider"] [role="slider"] {
    background:#7c3aed !important;
    box-shadow:0 0 0 4px rgba(124,58,237,0.22) !important; border:none !important;
}
[data-baseweb="slider"] > div:first-child > div > div {
    background:linear-gradient(90deg,#7c3aed,#a78bfa) !important;
}

/* ── Native metrics ── */
[data-testid="stMetric"] {
    background:rgba(255,255,255,0.03) !important;
    border:1px solid rgba(255,255,255,0.07) !important;
    border-radius:12px !important; padding:0.9rem 1rem !important;
}
[data-testid="stMetricValue"] { color:#a78bfa !important; font-weight:700 !important; }
[data-testid="stMetricLabel"] { color:#5a5680 !important; font-size:0.72rem !important; }

/* ── Alerts ── */
.stSuccess>div{background:rgba(52,211,153,.07)!important;border:1px solid rgba(52,211,153,.2)!important;border-radius:10px!important;color:#6ee7b7!important;}
.stWarning>div{background:rgba(251,191,36,.07)!important;border:1px solid rgba(251,191,36,.2)!important;border-radius:10px!important;color:#fde68a!important;}
.stError>div{background:rgba(248,113,113,.07)!important;border:1px solid rgba(248,113,113,.2)!important;border-radius:10px!important;color:#fca5a5!important;}
.stInfo>div{background:rgba(96,165,250,.07)!important;border:1px solid rgba(96,165,250,.15)!important;border-radius:10px!important;color:#93c5fd!important;}

/* ── Dataframes ── */
[data-testid="stDataFrame"] {
    border:1px solid rgba(255,255,255,0.07)!important;
    border-radius:12px!important; overflow:hidden!important;
}

/* ── Markdown tables ── */
[data-testid="stMarkdownContainer"] table{width:100%;border-collapse:collapse;font-size:.83rem;}
[data-testid="stMarkdownContainer"] th{background:rgba(124,58,237,.12)!important;color:#c4b5fd!important;font-weight:600!important;padding:7px 13px!important;border:1px solid rgba(255,255,255,.07)!important;font-size:.7rem!important;letter-spacing:.05em;text-transform:uppercase;}
[data-testid="stMarkdownContainer"] td{padding:7px 13px!important;border:1px solid rgba(255,255,255,.05)!important;color:#c4c0d8!important;}
[data-testid="stMarkdownContainer"] tr:nth-child(even) td{background:rgba(255,255,255,.02)!important;}

/* ── Sidebar text ── */
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p { color:#8884aa!important; font-size:.83rem!important; }

/* ── Scrollbar ── */
::-webkit-scrollbar{width:4px;height:4px;}
::-webkit-scrollbar-thumb{background:rgba(124,58,237,.3);border-radius:2px;}
::-webkit-scrollbar-track{background:transparent;}

/* ── Hide Streamlit chrome ── */
#MainMenu,footer,header{visibility:hidden;}
</style>
""", unsafe_allow_html=True)


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def kpi(col, value: str, label: str,
        tag: str = "", tag_cls: str = "tag-purple"):
    """Render a KPI card inside an st column. No HTML escape issue."""
    tag_html = (f'<span class="kpi-tag {tag_cls}">{tag}</span>'
                if tag else "")
    col.markdown(
        f'<div class="kpi">'
        f'<div class="kpi-lbl">{label}</div>'
        f'<div class="kpi-val">{value}</div>'
        f'{tag_html}</div>',
        unsafe_allow_html=True,
    )


def sh(icon: str, text: str):
    st.markdown(
        f'<div class="sh">{icon}&nbsp;{text}</div>',
        unsafe_allow_html=True,
    )


def rule():
    st.markdown('<hr class="fab-rule">', unsafe_allow_html=True)


def backend_alive() -> bool:
    try:
        import requests
        r = requests.get(f"{BACKEND_URL}/health", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def rag_status_api() -> dict:
    try:
        import requests
        r = requests.get(f"{BACKEND_URL}/rag/status", timeout=3)
        return r.json() if r.status_code == 200 else {}
    except Exception:
        return {}


def generate_scenes_api(config: dict) -> dict:
    try:
        import requests
        r = requests.post(f"{BACKEND_URL}/generate",
                          json=config, timeout=120)
        return r.json() if r.status_code == 200 else {"error": r.text}
    except Exception as e:
        return {"error": str(e)}


# ─── APP ──────────────────────────────────────────────────────────────────────

class FABRICAIFrontend:

    def __init__(self):
        self._init_session_state()

    def _init_session_state(self):
        defaults = {
            "detection_result": None,
            "detection_counts": None,
            "detected_image":   None,
            "selected_scene":   None,
            "gen_log":          [],
        }
        for k, v in defaults.items():
            if k not in st.session_state:
                st.session_state[k] = v

    # ── HEADER ────────────────────────────────────────────────────────────────
    def render_header(self):
        st.markdown("""
        <div class="fab-hero">
            <span class="fab-wordmark">FABRIC-AI</span>
            <p class="fab-tagline">
                Fiducial Augmentation &amp; Blender-based Raw Image Creation for AI
            </p>
            <p class="fab-sub">
                Synthetic Indian Road Scene Generator &nbsp;·&nbsp;
                YOLOv8 Object Detector
            </p>
        </div>
        <hr class="fab-rule">
        """, unsafe_allow_html=True)

    # ── SIDEBAR ───────────────────────────────────────────────────────────────
    def render_sidebar(self):
        with st.sidebar:
            st.markdown("## ⚙️ Scene Settings")
            alive = backend_alive()
            st.markdown(
                f"{'🟢' if alive else '🔴'} **Backend** "
                f"{'online' if alive else 'offline'}"
            )
            st.markdown("---")

            st.markdown("### 🌤️ Environment")
            weather     = st.selectbox("Weather", ["clear","rainy","foggy"])
            time_of_day = st.selectbox("Time of Day",
                                       ["morning","midday","evening","night"],
                                       index=1)
            road_type   = st.selectbox("Road Type", ["asphalt","sand"])

            st.markdown("---")
            st.markdown("### 🚗 Traffic")
            object_density = st.slider("Vehicle Density", 0.1, 1.0, 0.6, 0.1)
            auto_count     = st.slider("Auto Rickshaws", 0, 8, 3)

            st.markdown("---")
            st.markdown("### 🐄 Elements")
            cow       = st.checkbox("Cow on Road", value=False,
                                    help="Clear/morning/midday only")
            buildings = st.checkbox("Roadside Buildings", value=True)

            st.markdown("---")
            st.markdown("### 📊 Model")
            st.success("✅ YOLOv8n — FABRIC-AI v1")
            c1, c2 = st.columns(2)
            c1.metric("mAP50",    "0.758")
            c2.metric("Precision","0.867")
            c1.metric("Recall",   "0.582")
            c2.metric("mAP50-95", "0.574")

            st.markdown("---")
            st.markdown("### 📁 Dataset")
            images_dir  = OUTPUT_DIR / "images"
            image_count = (len(list(images_dir.glob("*.png")))
                           if images_dir.exists() else 0)
            ann_path    = OUTPUT_DIR / "latest_annotations.json"
            ann_count   = 0
            if ann_path.exists():
                with open(ann_path) as f:
                    d = json.load(f)
                ann_count = len(d.get("annotations", []))
            c1, c2 = st.columns(2)
            c1.metric("Scenes",      image_count)
            c2.metric("Annotations", ann_count)

        return {
            "weather":            weather,
            "time_of_day":        time_of_day,
            "road_type":          road_type,
            "object_density":     object_density,
            "auto_rickshaw_count":auto_count,
            "cow":                cow,
            "buildings":          buildings,
        }

    # ── TABS ──────────────────────────────────────────────────────────────────
    def render_tabs(self, config):
        t1,t2,t3,t4,t5,t6 = st.tabs([
            "  Scene Gallery  ",
            "  Generate Scenes  ",
            "  Object Detection  ",
            "  Training Results  ",
            "  Dataset Viewer  ",
            "  RAG & Backend  ",
        ])
        with t1: self.render_gallery_tab(config)
        with t2: self.render_generation_tab(config)
        with t3: self.render_detection_tab()
        with t4: self.render_results_tab()
        with t5: self.render_dataset_tab()
        with t6: self.render_rag_tab()

    # ── TAB 1: GALLERY ────────────────────────────────────────────────────────
    def render_gallery_tab(self, config):
        sh("🖼️", "Generated Scene Gallery")
        st.markdown(
            "Browse synthetic Indian road scenes rendered "
            "with Blender Cycles ray-tracing."
        )

        c1,c2,c3,c4 = st.columns(4)
        kpi(c1, config["weather"].capitalize(),          "Weather")
        kpi(c2, config["time_of_day"].capitalize(),      "Time of day")
        kpi(c3, f"{int(config['object_density']*100)}%", "Vehicle density",
            "", "tag-amber")
        kpi(c4, config["road_type"].capitalize(),        "Road type")

        rule()

        images_dir = OUTPUT_DIR / "images"
        if not images_dir.exists() or not list(images_dir.glob("*.png")):
            st.info(
                "No scenes found. Use the **Generate Scenes** tab or run:\n"
                "```\nblenderproc run "
                "blender_scenes/scripts/indian_street_gen.py\n```"
            )
            return

        image_files = sorted(images_dir.glob("*.png"))

        sh("🔎", "Filter")
        fc1, fc2, fc3 = st.columns([2, 2, 1])
        with fc1:
            show_count = st.slider("Scenes to display",
                                   3, min(len(image_files), 30), 9, 3)
        with fc2:
            sort_order = st.radio("Sort",
                                  ["Latest First","Oldest First"],
                                  horizontal=True)
        with fc3:
            st.markdown(
                f"<p style='padding-top:1.7rem;color:#5a5680;"
                f"font-size:.82rem;'>{len(image_files)} total</p>",
                unsafe_allow_html=True,
            )

        display_images = (
            image_files[-show_count:][::-1]
            if sort_order == "Latest First"
            else image_files[:show_count]
        )

        cols = st.columns(3)
        for idx, img_path in enumerate(display_images):
            with cols[idx % 3]:
                st.image(Image.open(img_path),
                         caption=img_path.name,
                         use_container_width=True)
                if st.button("🔍 Detect", key=f"gal_{idx}",
                             use_container_width=True):
                    st.session_state.selected_scene = str(img_path)
                    st.info("Switch to the Object Detection tab.")

        rule()

        config_path = PROJECT_ROOT / "configs/scene_configs_100.json"
        if config_path.exists():
            sh("🌍", "Scene Variety")
            with open(config_path) as f:
                cfgs = json.load(f)
            wc, tc = {}, {}
            for c in cfgs:
                w=c.get("weather","clear");       wc[w]=wc.get(w,0)+1
                t=c.get("time_of_day","midday");  tc[t]=tc.get(t,0)+1
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Weather distribution**")
                for k,v in wc.items():
                    st.markdown(f"- {k.capitalize()}: **{v}** "
                                f"({round(v/len(cfgs)*100)}%)")
            with col2:
                st.markdown("**Time of day distribution**")
                for k,v in tc.items():
                    st.markdown(f"- {k.capitalize()}: **{v}** "
                                f"({round(v/len(cfgs)*100)}%)")

    # ── TAB 2: GENERATE SCENES ────────────────────────────────────────────────
    def render_generation_tab(self, config):
        sh("⚡", "Scene Generation")
        st.markdown(
            "Generate new synthetic Indian road scenes via **BlenderProc**. "
            "Configure parameters in the sidebar then run below."
        )
        rule()

        sh("📋", "Current Configuration")
        c1,c2,c3,c4 = st.columns(4)
        kpi(c1, config["weather"].capitalize(),        "Weather")
        kpi(c2, config["time_of_day"].capitalize(),    "Time of day")
        kpi(c3, str(config["auto_rickshaw_count"]),    "Auto rickshaws",
            "", "tag-amber")
        kpi(c4, "Yes" if config["cow"] else "No",      "Cow on road",
            "Enabled" if config["cow"] else "Disabled",
            "tag-green" if config["cow"] else "tag-amber")

        rule()

        sh("🎛️", "Generation Options")
        g1, g2 = st.columns([2, 3])
        with g1:
            num_scenes = st.number_input(
                "Number of scenes", min_value=1, max_value=50, value=5
            )
        with g2:
            user_prompt = st.text_input(
                "Optional: describe the scene (used by RAG)",
                placeholder="e.g. rainy evening with autos and cows",
            )

        gen_config = {**config,
                      "num_scenes": num_scenes,
                      "prompt":     user_prompt}

        with st.expander("Full config JSON"):
            st.json(gen_config)

        rule()
        sh("🚀", "Run Generation")

        col_btn1, col_btn2, _ = st.columns([2, 2, 3])
        with col_btn1:
            run_via_api = st.button("🚀  Generate via FastAPI",
                                    type="primary",
                                    use_container_width=True)
        with col_btn2:
            run_local = st.button("⚙️  Run BlenderProc Locally",
                                  use_container_width=True)

        log_placeholder = st.empty()

        # Via FastAPI
        if run_via_api:
            if not backend_alive():
                st.error(
                    "Backend is offline. Start it with:\n"
                    "```\nuvicorn backend.main:app --reload --port 8000\n```"
                )
            else:
                with st.spinner("Sending config to FastAPI…"):
                    result = generate_scenes_api(gen_config)
                if "error" in result:
                    st.error(f"Backend error: {result['error']}")
                else:
                    st.success("✓ Generation request accepted.")
                    st.json(result)

        # Local BlenderProc subprocess
        if run_local:
            script = (PROJECT_ROOT /
                      "blender_scenes/scripts/indian_street_gen.py")
            if not script.exists():
                st.error(f"Script not found: {script}")
            else:
                cmd = ["blenderproc", "run", str(script)]
                log_placeholder.info(f"Running: `{' '.join(cmd)}`")
                try:
                    proc = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        cwd=str(PROJECT_ROOT),
                    )
                    log_lines: List[str] = []
                    for line in proc.stdout:        # type: ignore
                        log_lines.append(line.rstrip())
                        st.session_state.gen_log = log_lines.copy()
                        log_placeholder.code(
                            "\n".join(log_lines[-40:]), language=""
                        )
                    proc.wait()
                    if proc.returncode == 0:
                        st.success(
                            "✓ Done. Refresh the Scene Gallery tab."
                        )
                    else:
                        st.error(
                            f"BlenderProc exited with code "
                            f"{proc.returncode}."
                        )
                except FileNotFoundError:
                    st.error(
                        "`blenderproc` not found in PATH. "
                        "Activate your virtual environment."
                    )
                except Exception as e:
                    st.error(f"Error: {e}")

        if st.session_state.gen_log and not run_local:
            with st.expander("Previous generation log"):
                st.code(
                    "\n".join(st.session_state.gen_log[-60:]),
                    language=""
                )

        rule()
        sh("📖", "Command Reference")
        st.markdown("""
| Action | Command |
|--------|---------|
| Generate scenes | `blenderproc run blender_scenes/scripts/indian_street_gen.py` |
| Convert annotations | `python convert_to_yolo.py` |
| Split dataset | `python split_dataset.py` |
| Train YOLOv8 | `python train_yolo.py` |
| Verify annotations | `python verify_annotations.py` |
| Plot training graphs | `python plot_training.py` |
| Start FastAPI | `uvicorn backend.main:app --reload --port 8000` |
| Init / rebuild RAG | `python backend/setup_rag.py` |
        """)

    # ── TAB 3: DETECTION ──────────────────────────────────────────────────────
    def render_detection_tab(self):
        sh("🔍", "Object Detection")
        st.markdown("Run the FABRIC-AI trained YOLOv8n model on generated scenes.")

        model_path = (PROJECT_ROOT /
                      "runs/detect/runs/train/fabric_ai_v1/weights/best.pt")
        if not model_path.exists():
            st.error("Trained model not found. Run `python train_yolo.py` first.")
            return

        st.success(f"✓ Model ready — `{model_path.name}`")

        images_dir = OUTPUT_DIR / "images"
        if not images_dir.exists() or not list(images_dir.glob("*.png")):
            st.warning("No generated scenes found.")
            return

        image_files = sorted(images_dir.glob("*.png"))

        rule()
        sh("⚙️", "Detection Settings")

        col1, col2, col3 = st.columns(3)
        with col1:
            default_idx = len(image_files) - 1
            if st.session_state.selected_scene:
                sel_name   = Path(st.session_state.selected_scene).name
                names_list = [f.name for f in image_files]
                if sel_name in names_list:
                    default_idx = names_list.index(sel_name)
            selected = st.selectbox("Select Scene",
                                    [f.name for f in image_files],
                                    index=default_idx)
        with col2:
            confidence = st.slider("Confidence Threshold",
                                   0.05, 0.95, 0.25, 0.05)
        with col3:
            iou_thresh = st.slider("IoU Threshold",
                                   0.1, 0.9, 0.45, 0.05)

        selected_path = images_dir / selected
        rule()

        col_orig, col_result = st.columns(2)
        with col_orig:
            st.markdown("**Original Scene**")
            st.image(str(selected_path), use_container_width=True)

        if st.button("🚀  Run Detection", type="primary",
                     use_container_width=True):
            with st.spinner("Running YOLOv8 on CPU…"):
                try:
                    from ultralytics import YOLO
                    model   = YOLO(str(model_path))
                    results = model(
                        str(selected_path),
                        conf=confidence, iou=iou_thresh,
                        save=True, device="cpu",
                        project=str(PROJECT_ROOT / "runs/detect"),
                        name="streamlit_results",
                        exist_ok=True,
                    )
                    result_path = (PROJECT_ROOT /
                                   "runs/detect/streamlit_results" / selected)
                    cls_map  = {0:"car",1:"auto",2:"truck",3:"bus",4:"cow"}
                    counts   = {v: 0 for v in cls_map.values()}
                    det_data = []
                    for r in results:
                        for box in r.boxes:
                            lbl      = cls_map.get(int(box.cls), "unknown")
                            conf_val = float(box.conf)
                            if lbl in counts:
                                counts[lbl] += 1
                            bbox = [round(float(v),1) for v in box.xyxy[0]]
                            det_data.append({
                                "Class":      lbl,
                                "Confidence": f"{conf_val:.3f}",
                                "x1":bbox[0],"y1":bbox[1],
                                "x2":bbox[2],"y2":bbox[3],
                            })
                    st.session_state.detection_counts = counts
                    st.session_state.detection_result = det_data
                    if result_path.exists():
                        st.session_state.detected_image = str(result_path)
                except Exception as e:
                    st.error(f"Detection error: {e}")

        if (st.session_state.detected_image and
                Path(st.session_state.detected_image).exists()):
            with col_result:
                st.markdown("**Detection Result**")
                st.image(st.session_state.detected_image,
                         use_container_width=True)

        if st.session_state.detection_counts:
            counts = st.session_state.detection_counts
            total  = sum(counts.values())
            rule()
            st.success(
                f"✓ {total} object{'s' if total != 1 else ''} detected"
            )
            sh("📦", "Detection Summary")
            c1,c2,c3,c4,c5 = st.columns(5)
            kpi(c1, str(counts["car"]),   "Cars")
            kpi(c2, str(counts["auto"]),  "Autos")
            kpi(c3, str(counts["truck"]), "Trucks")
            kpi(c4, str(counts["bus"]),   "Buses")
            kpi(c5, str(counts["cow"]),   "Cows", "", "tag-green")

        if st.session_state.detection_result:
            det_data = st.session_state.detection_result
            if det_data:
                sh("🎯", "Individual Detections")
                st.dataframe(pd.DataFrame(det_data),
                             use_container_width=True)
                sh("📊", "Avg Confidence by Class")
                cdf = pd.DataFrame(det_data)
                cdf["Confidence"] = cdf["Confidence"].astype(float)
                st.bar_chart(
                    cdf.groupby("Class")["Confidence"].mean()
                )

    # ── TAB 4: TRAINING RESULTS ───────────────────────────────────────────────
    def render_results_tab(self):
        sh("📊", "Training Results")
        st.markdown(
            "Model trained on **100 synthetic Indian road scenes** — "
            "zero manual annotation."
        )
        rule()

        sh("🏆", "Key Performance Metrics")
        c1,c2,c3,c4 = st.columns(4)
        kpi(c1, "0.758", "mAP @ 50",    "↑ Excellent", "tag-purple")
        kpi(c2, "0.867", "Precision",   "↑ Excellent", "tag-green")
        kpi(c3, "0.582", "Recall",      "Moderate",     "tag-amber")
        kpi(c4, "0.574", "mAP @ 50-95", "↑ Very Good", "tag-purple")

        rule()

        training_graph = PROJECT_ROOT / "training_results_figure4.png"
        if training_graph.exists():
            sh("📈", "Loss & mAP Curves")
            st.image(str(training_graph), use_container_width=True)
        else:
            st.warning("Training graph not found. Run `python plot_training.py`")

        weather_chart = PROJECT_ROOT / "fig8_weather_analysis.png"
        if weather_chart.exists():
            rule()
            sh("🌤️", "Environmental Conditions Analysis")
            st.image(str(weather_chart), use_container_width=True)
            st.caption(
                "Fig. 8 — mAP@50 across weather conditions, "
                "FABRIC-AI vs baselines"
            )

        results_png = (PROJECT_ROOT /
                       "runs/detect/runs/train/fabric_ai_v1/results.png")
        if results_png.exists():
            rule()
            sh("📉", "YOLOv8 Training Overview")
            st.image(str(results_png), use_container_width=True)

        rule()
        col1, col2 = st.columns(2)
        confusion = (PROJECT_ROOT /
                     "runs/detect/runs/train/fabric_ai_v1/"
                     "confusion_matrix_normalized.png")
        pr_curve  = (PROJECT_ROOT /
                     "runs/detect/runs/train/fabric_ai_v1/"
                     "BoxPR_curve.png")
        with col1:
            sh("🔢", "Confusion Matrix")
            if confusion.exists():
                st.image(str(confusion), use_container_width=True)
            else:
                st.info("Not found.")
        with col2:
            sh("📊", "Precision-Recall Curve")
            if pr_curve.exists():
                st.image(str(pr_curve), use_container_width=True)
            else:
                st.info("Not found.")

        rule()
        sh("🖼️", "Validation — Ground Truth vs Predictions")
        for i in range(3):
            lp = (PROJECT_ROOT /
                  f"runs/detect/runs/train/fabric_ai_v1/"
                  f"val_batch{i}_labels.jpg")
            pp = (PROJECT_ROOT /
                  f"runs/detect/runs/train/fabric_ai_v1/"
                  f"val_batch{i}_pred.jpg")
            if lp.exists() and pp.exists():
                st.markdown(f"**Batch {i+1}**")
                c1, c2 = st.columns(2)
                with c1:
                    st.image(str(lp), caption="Ground Truth",
                             use_container_width=True)
                with c2:
                    st.image(str(pp), caption="Predictions",
                             use_container_width=True)

        rule()
        sh("⚙️", "Training Configuration")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
| Parameter      | Value     |
|----------------|-----------|
| Model          | YOLOv8n   |
| Epochs         | 100       |
| Image Size     | 640×640   |
| Batch Size     | 4         |
| Device         | CPU       |
| Train Images   | 80        |
| Val Images     | 20        |
| Early Stopping | 20 epochs |
| Optimiser      | SGD       |
            """)
        with col2:
            st.markdown("""
| Class         | ID | Type    |
|---------------|----|---------|
| Car           | 0  | Vehicle |
| Auto Rickshaw | 1  | Vehicle |
| Truck         | 2  | Vehicle |
| Bus           | 3  | Vehicle |
| Cow           | 4  | Animal  |
            """)

    # ── TAB 5: DATASET ────────────────────────────────────────────────────────
    def render_dataset_tab(self):
        sh("📁", "Dataset Viewer")
        st.markdown("Explore the synthetic Indian road scene dataset.")

        ann_path   = OUTPUT_DIR / "latest_annotations.json"
        images_dir = OUTPUT_DIR / "images"

        image_count = (len(list(images_dir.glob("*.png")))
                       if images_dir.exists() else 0)
        ann_count, cat_counts, coco_data = 0, {}, {}
        if ann_path.exists():
            with open(ann_path) as f:
                coco_data = json.load(f)
            ann_count = len(coco_data.get("annotations", []))
            nmap = {1:"car",2:"auto",3:"truck",4:"bus",5:"cow"}
            for ann in coco_data.get("annotations", []):
                cat = nmap.get(ann["category_id"], "unknown")
                cat_counts[cat] = cat_counts.get(cat, 0) + 1

        c1,c2,c3,c4 = st.columns(4)
        kpi(c1, str(image_count), "Total Images",
            "COCO format", "tag-purple")
        kpi(c2, str(ann_count),   "Total Annotations", "", "tag-green")
        kpi(c3, "5",              "Object Classes",    "", "tag-purple")
        kpi(c4, "1920×1080",      "Resolution",        "HD","tag-amber")

        rule()

        if cat_counts:
            sh("📊", "Annotation Distribution")
            df_cats = (pd.DataFrame(list(cat_counts.items()),
                                    columns=["Class","Count"])
                       .sort_values("Count", ascending=False))
            col1, col2 = st.columns([3, 1])
            with col1:
                st.bar_chart(df_cats.set_index("Class"), color="#7c3aed")
            with col2:
                st.markdown("**Class counts**")
                st.dataframe(df_cats, use_container_width=True,
                             hide_index=True)
        else:
            st.warning(f"Annotations not found at `{ann_path}`")

        if coco_data.get("categories"):
            rule()
            sh("🏷️", "COCO Categories")
            st.dataframe(pd.DataFrame(coco_data["categories"]),
                         use_container_width=True, hide_index=True)

        if coco_data.get("annotations"):
            rule()
            sh("🔍", "Sample Annotation Inspector")
            nmap   = {1:"car",2:"auto",3:"truck",4:"bus",5:"cow"}
            sample = coco_data["annotations"][:10]
            ann_df = pd.DataFrame([{
                "ann_id":   a["id"],
                "image_id": a["image_id"],
                "class":    nmap.get(a["category_id"],"unknown"),
                "x":  round(a["bbox"][0],1), "y":  round(a["bbox"][1],1),
                "w":  round(a["bbox"][2],1), "h":  round(a["bbox"][3],1),
                "area": round(a["area"],1),
            } for a in sample])
            st.dataframe(ann_df, use_container_width=True, hide_index=True)

        verify_dir = PROJECT_ROOT / "datasets/verify"
        if verify_dir.exists():
            verify_imgs = sorted(verify_dir.glob("*.png"))
            if verify_imgs:
                rule()
                sh("✅", "Verified Annotations")
                show_n = st.slider("Images to show",
                                   3, min(len(verify_imgs),12), 6)
                cols = st.columns(3)
                for idx, ip in enumerate(verify_imgs[:show_n]):
                    with cols[idx % 3]:
                        st.image(Image.open(ip), caption=ip.name,
                                 use_container_width=True)

        config_path = PROJECT_ROOT / "configs/scene_configs_100.json"
        if config_path.exists():
            rule()
            sh("🌍", "Scene Diversity Breakdown")
            with open(config_path) as f:
                cfgs = json.load(f)
            wc, tc, rc = {}, {}, {}
            for c in cfgs:
                w=c.get("weather","clear");      wc[w]=wc.get(w,0)+1
                t=c.get("time_of_day","midday"); tc[t]=tc.get(t,0)+1
                r=c.get("road_type","asphalt");  rc[r]=rc.get(r,0)+1
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**🌤️ Weather**")
                for k,v in wc.items():
                    st.markdown(f"- {k.capitalize()}: **{v}** "
                                f"({round(v/len(cfgs)*100)}%)")
            with col2:
                st.markdown("**🕐 Time of Day**")
                for k,v in tc.items():
                    st.markdown(f"- {k.capitalize()}: **{v}** "
                                f"({round(v/len(cfgs)*100)}%)")
            with col3:
                st.markdown("**🛣️ Road Type**")
                for k,v in rc.items():
                    st.markdown(f"- {k.capitalize()}: **{v}** "
                                f"({round(v/len(cfgs)*100)}%)")

    # ── TAB 6: RAG & BACKEND ──────────────────────────────────────────────────
    def render_rag_tab(self):
        sh("🤖", "RAG & Backend Status")
        st.markdown(
            "Monitor the FastAPI backend and the FAISS-based "
            "RAG scene parameter system."
        )
        rule()

        # Backend
        sh("🌐", "FastAPI Backend")
        alive = backend_alive()
        if alive:
            st.success(
                "✅ Backend **online** at `http://localhost:8000`  "
                "— [Open API docs →](http://localhost:8000/docs)"
            )
        else:
            st.error("❌ Backend **offline**.")
            st.markdown(
                "Start it with:\n"
                "```\nuvicorn backend.main:app --reload --port 8000\n```"
            )

        rule()

        # RAG index info
        sh("🧠", "RAG System")
        faiss_path  = PROJECT_ROOT / "datasets/knowledge_base/faiss_index"
        kb_dir      = PROJECT_ROOT / "datasets/knowledge_base"
        pdf_count   = (len(list(kb_dir.glob("*.pdf")))
                       if kb_dir.exists() else 0)
        index_exists = faiss_path.exists()

        c1,c2,c3 = st.columns(3)
        kpi(c1,
            "✓ Built" if index_exists else "✗ Missing",
            "FAISS Index",
            "Ready" if index_exists else "Run setup_rag.py",
            "tag-green" if index_exists else "tag-amber")
        kpi(c2, str(pdf_count), "PDF Documents", "in knowledge base")
        kpi(c3, "all-MiniLM-L6-v2", "Embedding Model",
            "sentence-transformers")

        if alive:
            status = rag_status_api()
            if status:
                rule()
                st.markdown("**Live RAG status from API:**")
                st.json(status)

        rule()

        # Init / Rebuild
        sh("🔧", "Initialize / Rebuild RAG")
        st.markdown(
            "Builds (or rebuilds) the FAISS vector index from the "
            "knowledge base PDFs and built-in scene knowledge."
        )
        force_rebuild = st.checkbox(
            "Force rebuild (ignore existing index)", value=False
        )
        if st.button("🔨  Initialize RAG", type="primary"):
            with st.spinner("Initializing RAG…"):
                try:
                    sys.path.insert(0, str(PROJECT_ROOT / "backend"))
                    from setup_rag import initialize_rag   # type: ignore
                    _, vs = initialize_rag(force_rebuild=force_rebuild)
                    st.success(
                        f"✓ RAG initialized — "
                        f"**{vs.index.ntotal}** embeddings in vector store."
                    )
                except ImportError:
                    st.error(
                        "Could not import `setup_rag`. "
                        "Check your backend folder."
                    )
                except Exception as e:
                    st.error(f"RAG init error: {e}")

        rule()

        # Test query
        sh("🔬", "Test RAG Query")
        test_q = st.text_input(
            "Enter a scene description to test retrieval:",
            placeholder="Rainy evening in Mumbai with auto-rickshaws",
        )
        if st.button("🔍  Retrieve Context") and test_q:
            with st.spinner("Querying vector store…"):
                try:
                    sys.path.insert(0, str(PROJECT_ROOT / "backend"))
                    from rag_service import get_rag_service  # type: ignore
                    svc     = get_rag_service()
                    context = svc.retrieve_context(test_q, k=3)
                    params  = svc.generate_parameters(test_q)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Retrieved context:**")
                        st.code(context, language="")
                    with col2:
                        st.markdown("**Generated scene parameters:**")
                        st.json(params)
                except ImportError:
                    st.error(
                        "Could not import `rag_service`. "
                        "Check your backend folder."
                    )
                except Exception as e:
                    st.error(f"RAG query error: {e}")

        rule()

        # Architecture table
        sh("📐", "System Architecture")
        st.markdown("""
| Embedding Model | all-MiniLM-L6-v2 | CPU inference |
| Vector Store | FAISS (local) | Local index |
| Backend API | FastAPI + Uvicorn | Port 8000 |
| Frontend | Streamlit | Port 8501 |
| Annotation Format | COCO → YOLO | Auto-converted |
        """)

    # ── RUN ───────────────────────────────────────────────────────────────────
    def run(self):
        self.render_header()
        config = self.render_sidebar()
        self.render_tabs(config)


def main():
    app = FABRICAIFrontend()
    app.run()


if __name__ == "__main__":
    main()
def initialize_rag(force_rebuild=False):
    """Wrapper so main.py can call initialize_rag()"""
    index_path = save_path
    if force_rebuild or not index_path.exists():
        build_brain()
    else:
        print(f"[RAG] Index already exists at {index_path}, skipping rebuild.")