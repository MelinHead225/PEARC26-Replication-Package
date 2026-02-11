import json
import re
from tqdm import tqdm
from github_request_wrapper import github_request

class CommitExtractor:
    def __init__(self, repo_names=None):
        self.repo_names = repo_names or []

    def _clean_message(self, message):
        """
        Keeps ! ? # : / - . _ and letters/numbers, removes everything else.
        """
        if not message:
            return ""

        # Step 1: Remove markdown code blocks
        message = re.sub(r"```[\s\S]*?```", " ", message)
        message = re.sub(r"`[^`]+`", " ", message)

        # Step 2: Remove common commit boilerplate
        lines = []
        for line in message.split("\n"):
            # Skip Co-authored-by, Signed-off-by
            if re.match(r"^(Co-authored-by|Signed-off-by|Reviewed-by):", line.strip(), re.I):
                continue
            # Skip empty lines or diff-style lines
            if re.match(r"^(\+\+\+|\-\-\-|@@|\s*$)", line.strip()):
                continue
            lines.append(line)
        message = "\n".join(lines)

        # Step 3: Final character filtering — keep only meaningful chars
        message = re.sub(r"[^a-zA-Z0-9\s!?]", " ", message).lower()

        # Step 4: Collapse whitespace
        message = re.sub(r"\s+", " ", message).strip()

        return message

    def _extract_commit_metadata(self, commit_json):
        commit_data = commit_json.get("commit", {})
        author_block = commit_data.get("author", {})
        message = commit_data.get("message", "")

        # Extract associated PR number (for merge commits)
        pr_number = None
        if commit_json.get("parents", []):  # has parents -> possibly merge
            # GitHub convention: "Merge pull request #123 from ..."
            m = re.search(r"Merge pull request #(\d+)", message)
            if m:
                pr_number = int(m.group(1))
            # Alternative: sometimes in title of commit
            elif "pull" in message.lower() and "/" in message:
                m = re.search(r"#(\d+)", message)
                if m:
                    pr_number = int(m.group(1))

        author = author_block.get("name")
        created_at = author_block.get("date")

        raw_msg = commit_data.get("message", "")
        issue_refs = re.findall(r"#\d+", raw_msg)
        urls = re.findall(r"https?://\S+", raw_msg)

        return author, created_at, issue_refs, urls, raw_msg, pr_number

    def _fetch_all_commits(self, repo_full_name, github_token=None):
        commits = []
        page = 1

        while True:
            url = f"https://api.github.com/repos/{repo_full_name}/commits?per_page=100&page={page}"
            page_data = github_request(url, github_token=github_token)  # Pass token here

            if not page_data:
                break

            commits.extend(page_data)
            if len(page_data) < 100:
                break

            page += 1

        return commits

    def _fetch_commit_details(self, repo_full_name, sha, github_token=None):
        url = f"https://api.github.com/repos/{repo_full_name}/commits/{sha}"
        return github_request(url, github_token=github_token)

    def extract_single_repo(self, repo_full_name, github_token=None):
        all_results = []
        project_name = repo_full_name.split("/")[1]

        commit_list = self._fetch_all_commits(repo_full_name, github_token)

        for c in tqdm(commit_list, desc=f"Commits {repo_full_name}", leave=False):
            sha = c["sha"]
            commit_details = self._fetch_commit_details(repo_full_name, sha, github_token)
            if not commit_details:
                continue

            author, created_at, issue_refs, urls, raw_msg, pr_number = self._extract_commit_metadata(commit_details)            
            clean_message = self._clean_message(raw_msg)

            parents = commit_details.get("parents", [])
            parent_shas = [p["sha"] for p in parents]
            files = commit_details.get("files", [])
            files_changed = [f["filename"] for f in files]

            artifact = {
                "artifact_type": "Commit",
                "project": project_name,
                "artifact_id": f"Commit_{sha}",
                "source_sections": {
                    "message": clean_message,
                    "raw_message": raw_msg
                },
                "metadata": {
                    "author": author,
                    "created_at": created_at,
                    "full_sha": sha,
                    "issue_references": [r.lstrip("#") for r in issue_refs],
                    "linked_urls": urls,
                    "parent_commit_shas": parent_shas,
                    "files_changed": files_changed,
                    "associated_pr_number": pr_number,
                    "is_merge_commit": len(parents) > 1,
                },
                "source_link": f"https://github.com/{repo_full_name}/commit/{sha}",
                "linked_comments": [],
                "linked_prs": []
            }

            all_results.append(artifact)

        return all_results

    def extract_from_repo(self, repo_full_name, github_token=None):
        return self.extract_single_repo(repo_full_name, github_token=github_token)

    def extract(self, output_file="commits.jsonl", github_token=None):
        with open(output_file, "w", encoding="utf-8") as f:
            for repo_full_name in self.repo_names:
                for commit in self.extract_single_repo(repo_full_name, github_token=github_token):
                    f.write(json.dumps(commit) + "\n")
        print(f"Saved commits to {output_file}")