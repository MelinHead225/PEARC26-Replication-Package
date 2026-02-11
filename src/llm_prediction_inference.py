# main.py
import os
import json
import tempfile
import shutil
from tqdm import tqdm
from itertools import cycle
from collections import defaultdict
from dotenv import load_dotenv
from git import Repo
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import re

from comment_extractor import CommentExtractor
from commit_extractor import CommitExtractor
from pull_request_extractor import PullRequestExtractor
from issue_extractor import IssueExtractor

load_dotenv()

TOKENS = [t.strip() for t in os.getenv("GITHUB_TOKENS", "").split(",") if t.strip()]
if not TOKENS:
    raise ValueError("No GitHub tokens")
token_cycle = cycle(TOKENS)

REPOS_TO_PROCESS = [    
    # 'ornladios/ADIOS2',
    # 'visit-dav/visit',
    # 'dyninst/dyninst',
    # 'UO-OACISS/tau2',
    # 'hypre-space/hypre',
    'trilinos/Trilinos',
    # 'kokkos/kokkos',
    # 'StanfordLegion/legion',
    # 'spack/spack'
]

DATA_DIR = "/bsuhome/ericmelin/ORNL/ORNL-Project-2/Multi-Artifact-SATD-Prioritization/data/cass_repos/Trilinos"

os.makedirs(DATA_DIR, exist_ok=True)

OUTPUTS = {
    "comments":      {"jsonl": os.path.join(DATA_DIR, "repo_comments.jsonl")},
    "commits":       {"jsonl": os.path.join(DATA_DIR, "repo_commits.jsonl")},
    "pull_requests": {"jsonl": os.path.join(DATA_DIR, "repo_pull_requests.jsonl")},
    "issues":        {"jsonl": os.path.join(DATA_DIR, "repo_issues.jsonl")},
    "master":        {"jsonl": os.path.join(DATA_DIR, "master_artifacts.jsonl")},
}

def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for e in data:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

all_comments = []
all_commits  = []
all_prs      = []
all_issues   = []

for repo_name in REPOS_TO_PROCESS:
    print(f"\nPROCESSING {repo_name}")
    temp_dir = tempfile.mkdtemp()
    try:
        Repo.clone_from(f"https://github.com/{repo_name}.git", temp_dir, depth=None, single_branch=True)
        token = next(token_cycle)

        # Comments from fresh clone
        all_comments.extend(CommentExtractor().extract_from_repo(temp_dir))

        # API artifacts
        all_commits.extend(CommitExtractor().extract_from_repo(repo_name, github_token=token))
        all_prs.extend(PullRequestExtractor(github_token=token).extract_from_repo(repo_name))
        all_issues.extend(IssueExtractor(github_token=token).extract_from_repo(repo_name))

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

# Save raw
save_jsonl(all_comments, OUTPUTS["comments"]["jsonl"])
save_jsonl(all_commits,  OUTPUTS["commits"]["jsonl"])
save_jsonl(all_prs,      OUTPUTS["pull_requests"]["jsonl"])
save_jsonl(all_issues,   OUTPUTS["issues"]["jsonl"])

# Filter main artifacts (for linking) vs sections (for classification)
all_issues_main   = [i for i in all_issues if i["artifact_type"] == "Issue"]
all_prs_main      = [p for p in all_prs     if p["artifact_type"] == "PullRequest"]

# # Bidirectional Linking (only uses main artifacts)
# print("Building commit SHA -> artifact_id map...")
# commit_sha_to_id = {}
# for c in all_commits:
#     sha = c["metadata"].get("full_sha")
#     if sha and len(sha) == 40:
#         commit_sha_to_id[sha] = c["artifact_id"]
#     elif c["artifact_id"].startswith("Commit_"):
#         sha = c["artifact_id"][7:]
#         if len(sha) == 40:
#             commit_sha_to_id[sha] = c["artifact_id"]
# print(f"-> {len(commit_sha_to_id)} commits mapped by SHA")

# # 2. Comment -> Commit
# print("Linking comments -> commits")
# for comment in tqdm(all_comments, desc="Comments -> Commits"):
#     linked = set()
#     for field in ["occurrence_shas", "first_commit", "last_commit", "removed_commit"]:
#         values = comment.get(field, [])
#         if not isinstance(values, list):
#             values = [values] if values else []
#         for sha in values:
#             sha_str = str(sha)
#             if sha_str in commit_sha_to_id:
#                 linked.add(commit_sha_to_id[sha_str])
#     comment["linked_commits"] = sorted(linked)

# # 3. Commit -> Comment
# print("Linking commits -> comments")
# sha_to_comments = defaultdict(set)
# for comment in all_comments:
#     cid = comment["artifact_id"]
#     for field in ["occurrence_shas", "first_commit", "last_commit", "removed_commit"]:
#         values = comment.get(field, [])
#         if not isinstance(values, list):
#             values = [values] if values else []
#         for sha in values:
#             if str(sha) in commit_sha_to_id:
#                 sha_to_comments[str(sha)].add(cid)

# for commit in all_commits:
#     sha = commit["metadata"].get("full_sha")
#     if not sha and commit["artifact_id"].startswith("Commit_"):
#         sha = commit["artifact_id"][7:]
#     commit["linked_comments"] = sorted(sha_to_comments.get(sha, set()))

# # 4. PR <-> Commit
# print("Linking PRs <-> commits")
# pr_to_commits = defaultdict(set)
# commit_to_prs = defaultdict(set)

# for pr in all_prs_main:  # <- only main PRs
#     pr_id = pr["artifact_id"]
#     # commits_in_pr + merge_commit_sha
#     meta = pr.get("metadata") or {}
#     msha = meta.get("merge_commit_sha")
#     if msha and msha in commit_sha_to_id:
#         cid = commit_sha_to_id[msha]
#         pr_to_commits[pr_id].add(cid)
#         commit_to_prs[cid].add(pr_id)

# for pr in all_prs_main:
#     pr["linked_commits"] = sorted(pr_to_commits[pr["artifact_id"]])
# for commit in all_commits:
#     commit["linked_prs"] = sorted(commit_to_prs.get(commit["artifact_id"], set()))

# # 5. Issue <-> PR
# print("Linking issues <-> PRs")
# issue_num_to_id = {
#     (i.get("metadata") or {}).get("number"): i["artifact_id"]
#     for i in all_issues_main
#     if (i.get("metadata") or {}).get("number") is not None
# }

# # PR -> Issue
# for pr in all_prs_main:
#     refs = {str(x).lstrip("#") for x in pr["metadata"].get("issue_references", [])}
#     pr["linked_issues"] = [
#         f"Issue_{n}" for n in refs
#         if n.isdigit() and int(n) in issue_num_to_id
#     ]

# # Issue -> PR
# for issue in all_issues_main:
#     num_str = str(issue["metadata"]["number"])
#     issue["linked_pull_requests"] = [
#         pr["artifact_id"] for pr in all_prs_main
#         if num_str in {str(r).lstrip("#") for r in pr["metadata"].get("issue_references", [])}
#     ]

# # 6. Issue -> Commits (via PRs)
# print("Linking issues -> commits (via PRs)")
# for issue in all_issues_main:
#     commits = set()
#     for pr_id in issue.get("linked_pull_requests", []):
#         pr = next((p for p in all_prs_main if p["artifact_id"] == pr_id), None)
#         if pr:
#             commits.update(pr.get("linked_commits", []))
#     issue["linked_commits"] = sorted(commits)

# LLM SATD Inference (now correctly classifies every section)
print("\nStarting LLM Inference")

MODEL_REPO = "MelinHead225/multitask-falcon-satd2"
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 101

print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

class MultiTaskFalcon(torch.nn.Module):
    def __init__(self, model_name, num_labels=6, num_tasks=4):
        super().__init__()
        self.encoder = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16 if DEVICE.type == "cuda" else torch.float32,
            device_map="auto" if DEVICE.type == "cuda" else None
        )
        hidden_size = self.encoder.config.hidden_size
        self.heads = torch.nn.ModuleList([torch.nn.Linear(hidden_size, num_labels) for _ in range(num_tasks)])

    def forward(self, input_ids, attention_mask, task_ids):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden = outputs.hidden_states[-1][:, -1, :]
        logits = torch.zeros(input_ids.size(0), 6, device=hidden.device)
        for task_idx in range(4):
            mask = (task_ids == task_idx)
            if mask.any():
                logits[mask] = self.heads[task_idx](hidden[mask])
        return logits

model = MultiTaskFalcon(MODEL_REPO).to(DEVICE)
model.eval()

print("Loading classification heads...")
state_dict = torch.hub.load_state_dict_from_url(
    f"https://huggingface.co/{MODEL_REPO}/resolve/main/heads.pt",
    map_location=DEVICE
)
model.heads.load_state_dict(state_dict)

LABELS = ['requirement_debt', 'code/design_debt', 'documentation_debt',
          'test_debt', 'scientific_debt', 'non_debt']

@torch.no_grad()
def classify_artifacts(artifacts):
    texts = []
    task_ids = []
    indices = []

    for i, art in enumerate(artifacts):
        text = ""
        task = None

        artifact_type = art["artifact_type"]

        if artifact_type == "comment":
            text = art.get("comment", "")
            task = 0
        elif artifact_type == "Commit":
            text = art["source_sections"].get("message", "")
            task = 1
        elif artifact_type in ["IssueSection", "PRSection"]:
            text = art["source_sections"].get("text", "")
            task = 2
        else:
            # Skip main Issue/PullRequest — only classify sections
            continue

        if not text.strip():
            continue

        texts.append(text)
        task_ids.append(task)
        indices.append(i)

    if not texts:
        print("No text to classify.")
        return

    print(f"Classifying {len(texts)} text sections...")
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Inference"):
        batch_texts = texts[i:i+BATCH_SIZE]
        batch_task = torch.tensor(task_ids[i:i+BATCH_SIZE], device=DEVICE)
        enc = tokenizer(batch_texts, truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
        logits = model(enc["input_ids"], enc["attention_mask"], batch_task)
        preds = torch.argmax(logits, dim=-1).cpu().tolist()
        for j, pred in enumerate(preds):
            orig_idx = indices[i + j]
            artifacts[orig_idx]["satd_label"] = LABELS[pred]

# Run classification
all_artifacts = all_comments + all_commits + all_prs + all_issues
classify_artifacts(all_artifacts)

master = all_artifacts

# Final save
save_jsonl(master, OUTPUTS["master"]["jsonl"])
print(f"\nMaster dataset with fine-grained SATD labels saved to:")
print(f"  {OUTPUTS['master']['jsonl']}")

# Final stats
print("\nFINAL STATS")
print(f"Total artifacts: {len(master)}")
print(f"Labeled sections: {sum(1 for a in master if a.get('satd_label'))}")
print(f"Code comments: {len(all_comments)}")
print(f"Commits: {len(all_commits)}")
print(f"PRs (main): {len(all_prs_main)} | PR sections: {sum(1 for p in all_prs if p['artifact_type'] == 'PRSection')}")
print(f"Issues (main): {len(all_issues_main)} | Issue sections: {sum(1 for i in all_issues if i['artifact_type'] == 'IssueSection')}")

print("\nLabel distribution (all classified text):")
from collections import Counter
labels = [a.get("satd_label") for a in master if a.get("satd_label")]
print(Counter(labels))

print(f"\nComments with linked commits : {sum(1 for x in all_comments if x.get('linked_commits'))}/{len(all_comments)}")
print(f"Commits with linked comments : {sum(1 for x in all_commits if x.get('linked_comments'))}/{len(all_commits)}")
print(f"PRs with linked commits      : {sum(1 for x in all_prs_main if x.get('linked_commits'))}/{len(all_prs_main)}")
print(f"Issues with linked PRs       : {sum(1 for x in all_issues_main if x.get('linked_pull_requests'))}/{len(all_issues_main)}")