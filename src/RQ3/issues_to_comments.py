import json
from collections import defaultdict, Counter
from tqdm import tqdm

# Config
JSONL_PATH = "master_artifacts_with_section_links.jsonl"

# Storage
comments = {}
commits = {}
prs = {}
issues = {}

# Load artifacts
with open(JSONL_PATH, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Loading artifacts"):
        a = json.loads(line)
        atype = a["artifact_type"]

        if atype == "comment":
            comments[a["artifact_id"]] = a
        elif atype == "Commit":
            commits[a["artifact_id"]] = a
        elif atype == "PullRequest":
            prs[a["artifact_id"]] = a
        elif atype == "Issue":
            issues[a["artifact_id"]] = a

print("\nArtifacts loaded:")
print(f"  comments: {len(comments)}")
print(f"  commits : {len(commits)}")
print(f"  PRs     : {len(prs)}")
print(f"  Issues  : {len(issues)}")

# SATD aggregation (propagate from sections to parent)
artifact_has_satd = defaultdict(bool)

with open(JSONL_PATH, "r", encoding="utf-8") as f:
    for line in f:
        a = json.loads(line)
        if a.get("satd_label") not in (None, "non_debt"):
            artifact_has_satd[a["artifact_id"]] = True
            parent = a.get("parent_artifact_id")
            if parent:
                artifact_has_satd[parent] = True

# SATD helper function
def is_satd(a):
    if not a:
        return False
    if a.get("satd_label") not in (None, "non_debt"):
        return True
    return artifact_has_satd.get(a["artifact_id"], False)

# Propagation analysis: Comment -> Commit -> PR -> Issue
chain_counts = Counter()
repo_chains = defaultdict(Counter)

for comment_id, comment in tqdm(comments.items(), desc="Analyzing propagation"):
    if not is_satd(comment):
        continue

    repo = comment["repo"]
    depth = 1  # at least the comment itself

    # Comment -> Commit
    linked_commits = comment.get("linked_commits", [])
    commit_found = False
    pr_found = False
    issue_found = False

    for commit_id in linked_commits:
        commit = commits.get(commit_id)
        if commit and is_satd(commit):
            commit_found = True

            #  Commit -> PR 
            linked_prs = commit.get("linked_prs", [])
            for pr_id in linked_prs:
                pr = prs.get(pr_id)
                if pr and is_satd(pr):
                    pr_found = True

                    #  PR -> Issue 
                    linked_issues = pr.get("linked_issues", [])
                    for issue_id in linked_issues:
                        issue = issues.get(issue_id)
                        if issue and is_satd(issue):
                            issue_found = True
                            break  # Depth 4 reached
                    break  # stop at first SATD PR
            break  # stop at first SATD commit

    # Assign depth based on continuous SATD path
    if issue_found:
        depth = 4
    elif pr_found:
        depth = 3
    elif commit_found:
        depth = 2
    else:
        depth = 1

    chain = {
        1: "comment_only",
        2: "comment→commit",
        3: "comment→commit→PR",
        4: "comment→commit→PR→issue",
    }[depth]

    chain_counts[chain] += 1
    repo_chains[repo][chain] += 1

# Results
print("\n SATD PROPAGATION DEPTH (Comment -> Issue) ")
total = sum(chain_counts.values())
for k, v in chain_counts.items():
    print(f"{k:30s} {v:8d}  ({v/total:.2%})")

print("\n PER-REPO SUMMARY (Top repos) ")
for repo, cnt in sorted(repo_chains.items(), key=lambda x: sum(x[1].values()), reverse=True)[:9]:
    print(f"\n{repo}")
    for chain, v in cnt.items():
        print(f"  {chain:30s} {v}")
