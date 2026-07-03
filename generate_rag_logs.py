import os
from pathlib import Path

# Configuration paths
LABEL_DIR = Path(r"C:\Users\priya\OneDrive\Desktop\fabric-ai-inte\fabric-ai\runs\detect\predict-3\labels")
OUTPUT_DIR = Path(r"C:\Users\priya\OneDrive\Desktop\fabric-ai-inte\fabric-ai\rag_documents")

# Your custom YOLO class mapping
CLASS_MAPPING = {
    '0': 'car',
    '1': 'auto',
    '2': 'truck',
    '3': 'bus',
    '4': 'cow'
}

# Ensure the output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def parse_yolo_labels():
    if not LABEL_DIR.exists():
        print(f"Error: Label directory {LABEL_DIR} does not exist. Check your path!")
        return

    txt_files = list(LABEL_DIR.glob("*.txt"))
    print(f"Found {len(txt_files)} prediction files to convert for RAG...")

    for file_path in txt_files:
        scene_name = file_path.stem  # e.g., 'scene_0002'
        
        # Dictionary to keep count of objects in this specific scene
        counts = {name: 0 for name in CLASS_MAPPING.values()}
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if parts:
                    class_id = parts[0]
                    if class_id in CLASS_MAPPING:
                        class_name = CLASS_MAPPING[class_id]
                        counts[class_name] += 1

        # Build a narrative description that a Vector DB / LLM can easily index
        log_content = f"Traffic Observation Report for {scene_name}.\n"
        log_content += f"Location Context: Main roadway tracking grid.\n"
        log_content += "Detected Objects Summary:\n"
        
        detected_items = [f"{qty} {name}(s)" for name, qty in counts.items() if qty > 0]
        
        if detected_items:
            log_content += f"In this scene, the vision system identified: {', '.join(detected_items)}.\n"
        else:
            log_content += "In this scene, the vision system detected no vehicles or target obstacles.\n"
            
        # Add a specific status flag for the RAG engine to filter queries easily
        if counts['cow'] > 0:
            log_content += "Alert: Obstacle detected on road (cow blocking lane).\n"
        if counts['truck'] > 2 or counts['bus'] > 2:
            log_content += "Traffic Density: Heavy commercial vehicle presence.\n"
        else:
            log_content += "Traffic Density: Normal operating conditions.\n"

        # Save this narrative log file
        output_file = OUTPUT_DIR / f"{scene_name}_log.txt"
        with open(output_file, 'w') as out_f:
            out_f.write(log_content)

    print(f"Success! All structured logs saved as text documents in: {OUTPUT_DIR}")

if __name__ == "__main__":
    parse_yolo_labels()