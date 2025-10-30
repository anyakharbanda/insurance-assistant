# segment_policies_hybrid.py
import os, re, glob

input_folder = "data/txt_policies"
output_folder = "data/segmented_policies"
os.makedirs(output_folder, exist_ok=True)

# Common insurance section headers (from your earlier version)
KNOWN_SECTION_TITLES = [
    "coverage", "scope of cover", "exclusions", "definitions", "conditions",
    "terms and conditions", "renewal", "claim procedure", "how to claim",
    "extensions", "endorsements", "sum insured", "cancellation", "liability",
    "general conditions", "premium payment", "policy schedule", "insuring clause",
    "deductibles", "add-on covers", "loss notification", "geographical limits"
]

# Regex pattern for known titles
KNOWN_PATTERN = r"\b(" + "|".join(KNOWN_SECTION_TITLES) + r")\b[:\s\n]*"

def detect_dynamic_headers(text):
    """Finds uppercase or numbered headers even if not in known list."""
    headers = []
    lines = text.split("\n")
    for i, line in enumerate(lines):
        clean = line.strip()
        if not clean:
            continue

        if (clean.isupper() and 3 < len(clean) < 60) or \
           re.match(r"^\d+[\.\)]\s?[A-Z].+", clean) or \
           re.match(r"^[A-Z][A-Za-z\s]{3,40}$", clean):
            headers.append((i, clean))
    return headers

def segment_policy(text):
    """
    Combines keyword-based and dynamic header detection for robust segmentation.
    """
    sections = {}
    lines = text.split("\n")

    # Find known headers
    known_matches = [(m.start(), m.group(1)) for m in re.finditer(KNOWN_PATTERN, text, flags=re.IGNORECASE)]

    # Find dynamic headers
    dynamic_headers = detect_dynamic_headers(text)

    # Merge both sets of headers and sort by order
    merged = []
    for (pos, hdr) in known_matches:
        merged.append((pos, hdr))
    for (idx, hdr) in dynamic_headers:
        merged.append((text.find(hdr), hdr))

    # Remove duplicates and sort
    seen = set()
    merged_unique = []
    for pos, hdr in sorted(merged, key=lambda x: x[0]):
        key = hdr.lower().strip()
        if key not in seen:
            seen.add(key)
            merged_unique.append((pos, hdr))

    if not merged_unique:
        return {"full_policy": text}

    # Segment text by header positions
    for i, (pos, header) in enumerate(merged_unique):
        start = text.find(header, pos) + len(header)
        end = merged_unique[i + 1][0] if i + 1 < len(merged_unique) else len(text)
        section_text = text[start:end].strip()
        if len(section_text) > 50:  # avoid empty ones
            sections[header.lower().replace(" ", "_")] = section_text

    return sections

# Run on all policies
for txt_file in glob.glob(f"{input_folder}/*.txt"):
    with open(txt_file, "r", encoding="utf-8") as f:
        text = f.read()

    base_name = os.path.basename(txt_file).replace(".txt", "")
    print(f"📄 Processing: {base_name}")

    sections = segment_policy(text)

    # Save each section to its folder
    out_dir = os.path.join(output_folder, base_name)
    os.makedirs(out_dir, exist_ok=True)
    for name, content in sections.items():
        out_path = os.path.join(out_dir, f"{name}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(content)

    print(f"✅ Segmented {base_name} into {len(sections)} hybrid sections\n")

print("🎯 Hybrid segmentation complete! Check data/segmented_policies/")
