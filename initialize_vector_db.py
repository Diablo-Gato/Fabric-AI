import os
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions

# Configuration Paths
DOCS_DIR = Path(r"C:\Users\priya\OneDrive\Desktop\fabric-ai-inte\fabric-ai\rag_documents")
DB_DIR = Path(r"C:\Users\priya\OneDrive\Desktop\fabric-ai-inte\fabric-ai\chroma_db")

def index_documents_for_rag():
    if not DOCS_DIR.exists():
        print(f"Error: {DOCS_DIR} does not exist. Run generate_rag_logs.py first!")
        return

    # 1. Initialize Persistent Local Chroma Client
    print("Initializing local Vector Database...")
    chroma_client = chromadb.PersistentClient(path=str(DB_DIR))
    
    # 2. Setup Local Embedding Function (Runs entirely on your machine)
    default_ef = embedding_functions.DefaultEmbeddingFunction()
    
    # 3. Create or get a collection for your scene logs
    collection = chroma_client.get_or_create_collection(
        name="traffic_scenes", 
        embedding_function=default_ef
    )

    # 4. Read files and prepare chunks
    txt_files = list(DOCS_DIR.glob("*_log.txt"))
    print(f"Loading {len(txt_files)} text files into vector space...")

    documents = []
    metadatas = []
    ids = []

    for file_path in txt_files:
        scene_id = file_path.stem.replace("_log", "") # e.g., scene_0002
        
        with open(file_path, 'r') as f:
            content = f.read()
            
        documents.append(content)
        metadatas.append({"scene_source": scene_id})
        ids.append(scene_id)

    # 5. Add elements to Vector Database
    if documents:
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Success! Vector Database built and saved locally at: {DB_DIR}")
        print(f"Indexed {len(documents)} scenes successfully.")
    else:
        print("No documents found to index.")

if __name__ == "__main__":
    index_documents_for_rag()