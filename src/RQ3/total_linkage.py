import json
import os
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# Config
JSONL_PATH = "master_artifacts_with_section_links.jsonl"
MAX_DEPTH = 4
PLOT_OUTPUT = "satd_priority_by_type.png"

# Artifact storage
artifact_map = {}
comments = {}
commits = {}
prs = {}
issues = {}

# Load artifacts
with open(JSONL_PATH, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Loading artifacts"):
        a = json.loads(line)
        aid = a["artifact_id"]
        artifact_map[aid] = a

        atype = a["artifact_type"]
        if atype == "comment":
            comments[aid] = a
        elif atype == "Commit":
            commits[aid] = a
        elif atype == "PullRequest":
            prs[aid] = a
        elif atype == "Issue":
            issues[aid] = a

print(f"Loaded: {len(comments)} comments, {len(commits)} commits, {len(prs)} PRs, {len(issues)} issues")

# Index PR / Issue sections
pr_sections = defaultdict(list)
issue_sections = defaultdict(list)

for a in artifact_map.values():
    atype = a["artifact_type"]
    parent = a.get("parent_artifact_id")
    if atype == "PRSection" and parent:
        pr_sections[parent].append(a)
    elif atype == "IssueSection" and parent:
        issue_sections[parent].append(a)

# SATD and priority helpers
def is_effective_satd(a):
    """Return True if the artifact or any of its sections is SATD."""
    if not a:
        return False
    atype = a["artifact_type"]

    if atype in {"comment", "Commit", "PRSection", "IssueSection"}:
        return a.get("satd_label") not in (None, "non_debt")

    if atype == "PullRequest":
        return any(s.get("satd_label") not in (None, "non_debt") for s in pr_sections.get(a["artifact_id"], []))

    if atype == "Issue":
        return any(s.get("satd_label") not in (None, "non_debt") for s in issue_sections.get(a["artifact_id"], []))

    return False

def get_effective_priority(a):
    """Return the max priority of an artifact or its SATD sections."""
    if not a:
        return 0.0
    atype = a["artifact_type"]

    if atype in {"comment", "Commit", "PRSection", "IssueSection"}:
        return float(a.get("priority_score", 0.0))

    if atype == "PullRequest":
        scores = [float(s.get("priority_score", 0.0)) for s in pr_sections.get(a["artifact_id"], []) if s.get("satd_label") not in (None, "non_debt")]
        return max(scores) if scores else 0.0

    if atype == "Issue":
        scores = [float(s.get("priority_score", 0.0)) for s in issue_sections.get(a["artifact_id"], []) if s.get("satd_label") not in (None, "non_debt")]
        return max(scores) if scores else 0.0

    return 0.0

# Traversal
def traverse(artifact, path, chains, priorities):
    aid = artifact["artifact_id"]
    # stop chain if non-SATD
    if not is_effective_satd(artifact):
        return

    path = path + (aid,)
    all_satd = all(is_effective_satd(artifact_map[x]) for x in path)

    # collect priority stats
    scores = [get_effective_priority(artifact_map[x]) for x in path]
    mean_priority = sum(scores) / len(scores)
    max_priority = max(scores)

    chains[len(path)].add(path)
    priorities[len(path)].append({"path": path, "all_satd": all_satd, "mean_priority": mean_priority, "max_priority": max_priority})

    if len(path) >= MAX_DEPTH:
        return

    atype = artifact["artifact_type"]
    if atype == "comment":
        next_ids, next_map = artifact.get("linked_commits", []), commits
    elif atype == "Commit":
        next_ids, next_map = artifact.get("linked_prs", []), prs
    elif atype == "PullRequest":
        next_ids, next_map = artifact.get("linked_issues", []), issues
    else:
        return

    for nid in next_ids:
        if nid in next_map:
            traverse(next_map[nid], path, chains, priorities)

# Run traversal
chains = {i: set() for i in range(1, MAX_DEPTH + 1)}
priorities = {i: [] for i in range(1, MAX_DEPTH + 1)}

starting_artifacts = list(comments.values()) + list(commits.values()) + list(prs.values()) + list(issues.values())
for art in tqdm(starting_artifacts, desc="Exhaustive traversal"):
    traverse(art, tuple(), chains, priorities)

# SATD-only Propagation by Chain Length
print("\n SATD-only Propagation by Chain Length ")
for length in sorted(priorities):
    data = [d for d in priorities[length] if d["all_satd"]]
    if not data:
        continue
    values = sorted(d["mean_priority"] for d in data)
    n = len(values)
    mean = sum(values) / n
    median = values[n // 2]
    print(f"\nChain length: {length}")
    print(f"  SATD-only chains: {n:,}")
    print(f"  Mean priority:   {mean:.4f}")
    print(f"  Median priority: {median:.4f}")

# -------------------------------
# Priority by artifact type (sections only)
# -------------------------------
artifact_types_to_plot = ["Comment", "Commit", "PRSection", "IssueSection"]
artifact_types = {atype: [] for atype in artifact_types_to_plot}

for a in artifact_map.values():
    atype = a["artifact_type"]
    if atype in artifact_types_to_plot and is_effective_satd(a):
        artifact_types[atype].append(get_effective_priority(a))

print("\n Priority by Artifact Type (SATD only) ")
for atype, values in artifact_types.items():
    if not values:
        continue
    vals = np.array(values)
    print(f"{atype:<15} n={len(vals):,} mean={np.mean(vals):.4f} median={np.median(vals):.4f}")

# Boxplot: SATD priority by artifact type
plt.figure(figsize=(8,6))
priority_data = [np.array(artifact_types[atype]) for atype in artifact_types_to_plot if artifact_types[atype]]
labels = [atype for atype in artifact_types_to_plot if artifact_types[atype]]
plt.boxplot(priority_data, tick_labels=labels)
plt.ylabel("Priority Score")
plt.xlabel("Artifact Type")
plt.title("SATD Priority by Artifact Type")
plt.tight_layout()

# ensure output directory exists
os.makedirs(os.path.dirname(PLOT_OUTPUT), exist_ok=True)
plt.savefig(PLOT_OUTPUT)
plt.show()
