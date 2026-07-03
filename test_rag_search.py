import chromadb
from chromadb.utils import embedding_functions

# Connect to your new database
chroma_client = chromadb.PersistentClient(path=r"C:\Users\priya\OneDrive\Desktop\fabric-ai-inte\fabric-ai\chroma_db")
default_ef = embedding_functions.DefaultEmbeddingFunction()
collection = chroma_client.get_collection(name="traffic_scenes", embedding_function=default_ef)

# ─── YOUR SEARCH QUERY HERE ───
user_query = "Are there any dangerous hazards or animals blocking the road?"
print(f"Searching database for: '{user_query}'...\n")

# Query the database for the top 2 closest matches
results = collection.query(
    query_texts=[user_query],
    n_results=2
)

# Print out what it found
for i in range(len(results['documents'][0])):
    print(f"--- MATCH #{i+1} (Scene ID: {results['ids'][0][i]}) ---")
    print(results['documents'][0][i])