import json, glob, os


embeddings_path = os.path.join("data", "policy_embeddings.json")

if not os.path.exists(embeddings_path):
    raise FileNotFoundError(f"❌ Could not find {embeddings_path}. Run embed.py first!")

with open(embeddings_path, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"✅ Loaded {len(data)} embedded sections")


segmented_files = glob.glob("data/segmented_policies/*/*.txt")


embedded_files = {f"{d['policy']}/{d['section']}.txt" for d in data}

# --- Compare segmented vs embedded ---
missing = []
for f in segmented_files:
    # Normalize to same format as in JSON (folder/file.txt)
    rel_path = f"{os.path.basename(os.path.dirname(f))}/{os.path.basename(f)}"
    if rel_path not in embedded_files:
        missing.append(rel_path)

# --- Summary ---
if missing:
    print(f"⚠️ Missing {len(missing)} sections (not embedded):")
    for m in missing[:10]:
        print(" -", m)
else:
    print("✅ All segmented sections are embedded successfully!")
