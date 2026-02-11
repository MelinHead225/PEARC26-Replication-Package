import json
import numpy as np
from itertools import combinations
from scipy.stats import spearmanr

def extract_text(entry):
    atype = entry.get("artifact_type")

    if atype == "comment":
        return entry.get("comment")

    if atype == "Commit":
        src = entry.get("source_sections", {})
        return src.get("message") or src.get("raw_message")

    if atype in ["PRSection", "IssueSection"]:
        return entry.get("source_sections", {}).get("text")

    return entry.get("comment") or entry.get("message") or entry.get("text")


# Paths to the six ablation outputs
variant_files = {
    "embed_only": "/bsuhome/ericmelin/ORNL/ORNL-Project-2/Multi-Artifact-SATD-Prioritization/data/cass_repos/all_repos_master_artifacts_priority_embed_only.jsonl",
    "lexical_only": "/bsuhome/ericmelin/ORNL/ORNL-Project-2/Multi-Artifact-SATD-Prioritization/data/cass_repos/all_repos_master_artifacts_priority_lexical_only.jsonl",
    "tfidf_only": "/bsuhome/ericmelin/ORNL/ORNL-Project-2/Multi-Artifact-SATD-Prioritization/data/cass_repos/all_repos_master_artifacts_priority_tfidf_only.jsonl",
    "embed_lex": "/bsuhome/ericmelin/ORNL/ORNL-Project-2/Multi-Artifact-SATD-Prioritization/data/cass_repos/all_repos_master_artifacts_priority_embed_lex.jsonl",
    "embed_tfidf": "/bsuhome/ericmelin/ORNL/ORNL-Project-2/Multi-Artifact-SATD-Prioritization/data/cass_repos/all_repos_master_artifacts_priority_embed_tfidf.jsonl",
    "full": "/bsuhome/ericmelin/ORNL/ORNL-Project-2/Multi-Artifact-SATD-Prioritization/data/cass_repos/all_repos_master_artifacts_priority_full.jsonl",
}

TOP_K = 1000 
TOP_DISPLAY = 25  # top texts to show

# Read all variants
data = {}
for vname, fpath in variant_files.items():
    with open(fpath, "r") as f:
        entries = [json.loads(line) for line in f]
        # filter only SATD entries
        satd_entries = [e for e in entries if e.get("satd_label") != "non_debt"]
        # sort by priority_score_variant descending
        satd_entries.sort(key=lambda x: x["priority_score_variant"], reverse=True)
        data[vname] = satd_entries

# Compute top-K sets
topk_sets = {v: set(e["artifact_id"] for e in entries[:TOP_K]) for v, entries in data.items()}

# Compute Jaccard similarities
print("Jaccard similarities between top-K sets:")
for v1, v2 in combinations(topk_sets.keys(), 2):
    A, B = topk_sets[v1], topk_sets[v2]
    jaccard = len(A & B) / len(A | B) if len(A | B) > 0 else 0.0
    print(f"{v1} vs {v2}: {jaccard:.3f}")

# Compute SATD type distributions and print top UNIQUE texts per variant
for v, entries in data.items():
    top_entries = entries[:TOP_K]
    satd_counts = {}

    print(f"\nVariant: {v}")

    for e in top_entries:
        stype = e.get("satd_label", "unknown")
        satd_counts[stype] = satd_counts.get(stype, 0) + 1

    print("Top-K SATD type distribution:", satd_counts)

    # print top UNIQUE prioritized texts
    print(f"Top {TOP_DISPLAY} UNIQUE prioritized SATD texts:")

    seen_texts = set()
    shown = 0

    for e in top_entries:
        raw_text = extract_text(e)
        if not raw_text:
            continue

        # Normalize text for deduplication
        norm_text = " ".join(raw_text.lower().split())

        if norm_text in seen_texts:
            continue

        seen_texts.add(norm_text)
        shown += 1

        score = e.get("priority_score_variant", 0.0)

        display_text = raw_text[:200]
        if len(raw_text) > 200:
            display_text += "..."

        print(f"{shown}. [score: {score:.4f}] {display_text}")

        if shown >= TOP_DISPLAY:
            break

    if shown < TOP_DISPLAY:
        print(f"(Only {shown} unique SATD texts found)")


# compute rank correlation of scores between variants
variant_names = list(data.keys())
for v1, v2 in combinations(variant_names, 2):
    scores1 = [e["priority_score_variant"] for e in data[v1]]
    scores2 = [e["priority_score_variant"] for e in data[v2]]
    # ensure same length
    min_len = min(len(scores1), len(scores2))
    if min_len > 1:
        corr, _ = spearmanr(scores1[:min_len], scores2[:min_len])
        print(f"Spearman rank correlation {v1} vs {v2}: {corr:.3f}")
