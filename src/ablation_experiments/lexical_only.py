import json
from tqdm import tqdm

INPUT_FILE = "all_repos_master_artifacts.jsonl"


weights = {"lexical_score": 1.0, "tfidf_score": 0.0, "embedding_score": 0.0}
OUTPUT_FILE = INPUT_FILE.replace(".jsonl", "_priority_lexical_only.jsonl")


entries = []

with open(INPUT_FILE, "r") as f:
    for line in tqdm(f, desc="Processing embedding-only variant"):
        entry = json.loads(line)
        priority_score = (
            weights["lexical_score"] * entry.get("lexical_score", 0.0) +
            weights["tfidf_score"] * entry.get("tfidf_score", 0.0) +
            weights["embedding_score"] * entry.get("embedding_score", 0.0)
        )
        entry["priority_score_variant"] = float(priority_score)
        entries.append(entry)

with open(OUTPUT_FILE, "w") as f:
    for entry in entries:
        f.write(json.dumps(entry) + "\n")

print(f"Done! Saved: {OUTPUT_FILE}")
