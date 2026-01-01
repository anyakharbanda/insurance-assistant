import json, glob, os

# ✅ Load your saved embeddings first — this defines "data"
with open("data/policy_embeddings.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# List all segmented section files
segmented_files = glob.glob("data/segmented_policies/*/*.txt")

# Get all embedded file identifiers
embedded_files = {f"{d['policy']}/{d['section']}.txt" for d in data}

# Compare both to find missing sections
missing = []
for f in segmented_files:
    rel_path = os.path.join(os.path.basename(os.path.dirname(f)), os.path.basename(f))
    if rel_path not in embedded_files:
        missing.append(rel_path)

print(f"❌ Missing sections: {len(missing)}")
if missing:
    print("\nSome missing files:")
    for m in missing[:10]:
        print(" -", m)
