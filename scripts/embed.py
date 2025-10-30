from sentence_transformers import SentenceTransformer
import os, json, glob

# 🧠 Load a high-quality legal-aware embedding model
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

input_folder = "data/segmented_policies"
output_file = "data/policy_embeddings.json"

embeddings_data = []

for policy_folder in glob.glob(f"{input_folder}/*"):
    policy_name = os.path.basename(policy_folder)
    print(f"📄 Processing: {policy_name}")
    
    for section_file in glob.glob(f"{policy_folder}/*.txt"):
        section_name = os.path.basename(section_file).replace(".txt", "")
        with open(section_file, "r", encoding="utf-8") as f:
            text = f.read().strip()
        
        if len(text) < 50:
            continue
        
        # Generate embeddings (mean pooling)
        embedding = model.encode(text, normalize_embeddings=True).tolist()
        embeddings_data.append({
            "policy": policy_name,
            "section": section_name,
            "text": text,
            "embedding": embedding
        })

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(embeddings_data, f, indent=2)

print("✅ Embeddings generated and saved to data/policy_embeddings.json")
