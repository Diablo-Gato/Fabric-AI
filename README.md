# FABRIC-AI

**Fiducial Augmentation & Blender-based Raw Image Creation**

A complete AI system for generating realistic Indian road scene images using Blender and a RAG-based AI pipeline. The system creates synthetic images with COCO-format annotations for AI training.

## Features

- **Natural Language Interface**: Enter scene descriptions in plain English
- **RAG-Powered Parameter Generation**: Uses Retrieval-Augmented Generation to convert prompts into 3D rendering parameters
- **BlenderProc Integration**: Procedural 3D Indian street scenes with realistic vehicles
- **COCO Export**: Standard COCO format annotations for object detection training
- **Domain Randomization**: Camera poses, lighting, and object placement variations
- **GPU Acceleration**: NVIDIA OptiX support for fast rendering

## Architecture

```
User Prompt → RAG Retrieval → Parameter Generation → BlenderProc → COCO Export
```

### Layers

1. **User Interface Layer**: Streamlit web interface
2. **RAG Knowledge System**: FAISS vector database + sentence transformers
3. **Scene Parameter Engine**: LLM-based parameter generation
4. **BlenderProc Rendering Engine**: 3D scene generation and rendering
5. **Dataset Generation**: COCO format export

## Installation

```bash
# Clone and navigate to project
cd fabric-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install BlenderProc (requires Blender)
# See: https://github.com/WPSYS/BlenderProc
```

### Requirements

- Python 3.10+
- Blender 3.0+ with BlenderProc
- FAISS vector database
- SentenceTransformers
- LangChain
- Streamlit

### Environment Variables

```bash
# For OpenAI LLM (optional)
export OPENAI_API_KEY="your-api-key"
```

## Project Structure

```
fabric-ai/
├── backend/
│   ├── rag_service.py        # RAG parameter generation
│   └── setup_rag.py           # RAG system initialization
├── blender_scenes/
│   ├── indian_street.blend   # Base 3D scene
│   └── scripts/
│       └── indian_street_gen.py  # BlenderProc rendering
├── datasets/
│   ├── knowledge_base/       # PDF research papers
│   └── synthetic/            # Generated outputs
│       ├── images/
│       └── coco_annotations.json
├── frontend/
│   └── app.py                # Streamlit app
├── main.py                   # Main controller
├── requirements.txt
└── README.md
```

## Usage

### Command Line

```bash
# Basic usage
python main.py "Rainy evening in Mumbai with auto-rickshaws"

# Multiple scenes
python main.py "Busy Delhi street" --num-scenes 5

# Without RAG
python main.py "Sunny road" --no-rag
```

### Streamlit Web Interface

```bash
cd fabric-ai
streamlit run frontend/app.py
```

### Python API

```python
from main import FABRICAI

fabric = FABRICAI(use_rag=True)
result = fabric.run(
    prompt="Rainy evening in Mumbai with auto-rickshaws",
    num_scenes=5
)

print(f"Generated {result['num_scenes']} scenes")
```

## Scene Parameters

The system generates parameters including:

| Parameter | Range | Description |
|-----------|-------|-------------|
| sun_intensity | 0.0 - 1.5 | Sun brightness |
| rain_intensity | 0.0 - 1.0 | Rain intensity |
| camera_angle | 0° - 90° | Camera elevation |
| camera_distance | 5 - 50m | Distance from scene |
| fov | 25° - 90° | Field of view |
| object_density | 0.0 - 1.0 | Scene complexity |
| occlusion_level | 0.0 - 1.0 | Object occlusion |

## COCO Categories

| ID | Name | Description |
|----|------|-------------|
| 1 | person | Pedestrians |
| 2 | bicycle | Bicycles |
| 3 | car | Cars |
| 4 | motorcycle | Motorcycles |
| 5 | bus | Buses |
| 6 | truck | Trucks |
| 7 | auto_rickshaw | Auto-rickshaws |
| 8 | cow | Cows |
| 9 | traffic_light | Traffic lights |
| 10 | traffic_sign | Traffic signs |
| 11 | street_vendor | Street vendors |

## Example Prompts

- "Rainy evening in Mumbai with auto-rickshaws and pedestrians"
- "Sunny afternoon on a busy Delhi street with cows"
- "Foggy morning in Bangalore with light traffic"
- "Night scene in Chennai with rain and auto-rickshaws"
- "Quiet rural road at dawn with cows and bicycles"

## GPU Acceleration

The system automatically detects and uses NVIDIA GPUs with OptiX:

```python
bpy.context.scene.cycles.device = "GPU"
prefs = bpy.context.preferences.addons["cycles"].preferences
prefs.compute_device_type = "OPTIX"
```

## Knowledge Base

Add research papers (PDF) to `datasets/knowledge_base/` to improve parameter generation:

```bash
# Add papers
cp my_paper.pdf datasets/knowledge_base/

# Rebuild RAG index
python -c "from backend.setup_rag import initialize_rag; initialize_rag(force_rebuild=True)"
```

## Output

Generated datasets in COCO format:

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "scene_0001.png",
      "width": 1920,
      "height": 1080
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 7,
      "bbox": [100, 200, 50, 80],
      "area": 4000,
      "iscrowd": 0
    }
  ],
  "categories": [...]
}
```

## License

MIT License

## Contributing

Contributions welcome! Please open an issue or submit a PR.
