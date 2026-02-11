import json

INPUT_JSONL = "master_artifacts_with_sentiment.jsonl"

none_rows = []

with open(INPUT_JSONL, "r", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        if row.get("predicted_sentiment") is None:
            none_rows.append(row)

print(f"Found {len(none_rows)} rows with None sentiment.\n")

# Optionally, print first 10 for inspection
for r in none_rows[:3]:
    print(json.dumps(r, indent=2))
