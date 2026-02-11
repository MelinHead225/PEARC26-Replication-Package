import json
import re
from tqdm import tqdm
from github_request_wrapper import github_request


def safe_user(user_obj):
    """Return login if user exists, else fallback placeholder."""
    if not user_obj or not isinstance(user_obj, dict):
        return "unknown_user"
    return user_obj.get("login") or "unknown_user"


def safe_text(value):
    """Return empty string if None."""
    return value if isinstance(value, str) else ""


def safe_list(value):
    """Return list if valid else empty list."""
    return value if isinstance(value, list) else []


class IssueExtractor:
    def __init__(self, github_token=None, repo_names=None):
        self.github_token = github_token
        self.repo_names = repo_names or []

    def _clean_github_text(self, text):
        if not text:
            return ""
        text = safe_text(text)
        text = re.sub(r"```[\s\S]*?```", "", text)
        text = re.sub(r"`[^`]*`", "", text)
        lines = [l for l in text.split("\n") if not re.match(r"^\s{4,}", l)]
        text = "\n".join(lines)
        lines = [l for l in text.split("\n") if not re.match(r"^\s*(at |File \")", l)]
        return "\n".join(lines).strip()

    def extract_single_repo(self, repo_full_name, github_token=None):
        token = github_token or self.github_token
        if not token:
            raise ValueError("GitHub token required")

        owner, repo = repo_full_name.split("/")
        base_url = f"https://api.github.com/repos/{owner}/{repo}"
        results = []

        page = 1
        while True:
            url = f"{base_url}/issues"
            params = {"state": "all", "per_page": 100, "page": page}

            data = github_request(url, github_token=token, params=params)

            # Defensive against null/broken responses
            if data is None:
                print(f"[WARN] Null response for issues page {page} in {repo_full_name}, retrying...")
                data = github_request(url, github_token=token, params=params)

            if data is None:
                print(f"[ERROR] Persistent null issues response for {repo_full_name} page {page}, skipping.")
                break

            if not isinstance(data, list):
                print(f"[ERROR] Unexpected issues format {type(data)} on {repo_full_name} page {page}")
                break

            if len(data) == 0:
                break

            # Iterate over issues
            for issue in data:
                if not issue or not isinstance(issue, dict):
                    continue

                # Skip PRs
                if "pull_request" in issue:
                    continue

                number = issue.get("number")
                if not number:
                    continue

                # Clean fields 
                title = self._clean_github_text(safe_text(issue.get("title")))
                body = self._clean_github_text(safe_text(issue.get("body")))

                # Metadata
                author = safe_user(issue.get("user"))
                labels = [lbl.get("name") for lbl in safe_list(issue.get("labels")) if isinstance(lbl, dict)]

                main_artifact = {
                    "artifact_type": "Issue",
                    "project": repo,
                    "artifact_id": f"Issue_{number}",
                    "source_sections": {
                        "title": title,
                        "description": body
                    },
                    "metadata": {
                        "author": author,
                        "created_at": issue.get("created_at"),
                        "closed_at": issue.get("closed_at"),
                        "state": issue.get("state"),
                        "number": number,
                        "labels": labels
                    },
                    "source_link": issue.get("html_url"),
                    "linked_pull_requests": [],
                    "linked_commits": [],
                    "satd_label": None
                }
                results.append(main_artifact)

                # Title section
                if title.strip():
                    results.append({
                        "artifact_type": "IssueSection",
                        "project": repo,
                        "artifact_id": f"Issue_{number}_title",
                        "parent_artifact_id": f"Issue_{number}",
                        "section_type": "title",
                        "source_sections": {"text": title},
                        "metadata": {"author": author},
                        "source_link": issue.get("html_url"),
                        "satd_label": None
                    })

                # Description section 
                if body.strip():
                    results.append({
                        "artifact_type": "IssueSection",
                        "project": repo,
                        "artifact_id": f"Issue_{number}_description",
                        "parent_artifact_id": f"Issue_{number}",
                        "section_type": "description",
                        "source_sections": {"text": body},
                        "metadata": {"author": author},
                        "source_link": issue.get("html_url"),
                        "satd_label": None
                    })

                # Comments
                comments_url = issue.get("comments_url")
                if comments_url:
                    comments = github_request(comments_url, github_token=token)
                    if comments is None or not isinstance(comments, list):
                        print(f"[WARN] Invalid comments list for issue {number} in {repo_full_name}")
                        comments = []

                    for idx, c in enumerate(comments):
                        if not c or not isinstance(c, dict):
                            continue

                        comment_text = self._clean_github_text(safe_text(c.get("body")))
                        if not comment_text.strip():
                            continue

                        results.append({
                            "artifact_type": "IssueSection",
                            "project": repo,
                            "artifact_id": f"Issue_{number}_comment_{idx}",
                            "parent_artifact_id": f"Issue_{number}",
                            "section_type": "comment",
                            "source_sections": {"text": comment_text},
                            "metadata": {
                                "author": safe_user(c.get("user")),
                                "created_at": c.get("created_at")
                            },
                            "source_link": c.get("html_url"),
                            "satd_label": None
                        })

            if len(data) < 100:
                break

            page += 1

        return results

    def extract_from_repo(self, repo_full_name, github_token=None):
        return self.extract_single_repo(
            repo_full_name,
            github_token=github_token or self.github_token
        )
