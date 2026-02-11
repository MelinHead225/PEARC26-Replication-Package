import re
from tqdm import tqdm
from src.extractors.helpers.github_request_wrapper import github_request


class PullRequestExtractor:
    """
    Robust Pull Request extractor.

    - Uses github_request(url, params=params) so the wrapper's token cycling is active.
    - Extracts: PR main artifact, title section, description section, PR comments,
      and inline review comments.
    - Defensive against None/malformed responses so it won't crash on corrupted data.
    """

    def __init__(self, github_token=None, repo_names=None):
        # Keep github_token for backward-compatibility, but we won't pass it to github_request
        # so the wrapper can rotate tokens itself. If you want to force a specific token,
        # callers can still pass github_token to extract_from_repo.
        self.github_token = github_token
        self.repo_names = repo_names or []

    def _clean_github_text(self, text):
        if not text:
            return ""
        # remove code fences and inline code, then remove indented code blocks and trace lines
        text = re.sub(r"```[\s\S]*?```", "", text)
        text = re.sub(r"`[^`]*`", "", text)
        lines = [l for l in text.split("\n") if not re.match(r"^\s{4,}", l)]
        text = "\n".join(lines)
        lines = [l for l in text.split("\n") if not re.match(r"^\s*(at |File \")", l)]
        return "\n".join(lines).strip()

    def extract_single_repo(self, repo_full_name, github_token=None, show_progress=False):
        """
        Extract all PR artifacts from a single repo.
        - repo_full_name: "owner/repo"
        - github_token: if provided, passed as github_token to github_request to override global rotation.
                        (By default we call github_request WITHOUT github_token for rotation.)
        - show_progress: if True, wrap PR list in tqdm for progress display.
        """
        token_override = github_token or self.github_token  # optional override
        owner, repo = repo_full_name.split("/")
        base_url = f"https://api.github.com/repos/{owner}/{repo}"
        results = []

        page = 1
        while True:
            url = f"{base_url}/pulls"
            params = {"state": "all", "per_page": 100, "page": page}

            # When token_override is None, github_request will use the rotating global token.
            if token_override:
                data = github_request(url, github_token=token_override, params=params)
            else:
                data = github_request(url, params=params)

            # Defensive checks
            if data is None:
                print(f"[WARN] Null response for PR page {page} in {repo_full_name}. Retrying once...")
                if token_override:
                    data = github_request(url, github_token=token_override, params=params)
                else:
                    data = github_request(url, params=params)

            if data is None:
                print(f"[ERROR] Still null after retry. Skipping PR page {page} for {repo_full_name}.")
                break

            if not isinstance(data, list):
                print(f"[ERROR] Unexpected PR page format for {repo_full_name} page {page}: {type(data)}")
                break

            if len(data) == 0:
                break

            iterator = data
            if show_progress:
                iterator = tqdm(data, desc=f"PRs {repo_full_name} page {page}", unit="pr")

            for pr in iterator:
                try:
                    # Skip malformed entries
                    if pr is None or not isinstance(pr, dict):
                        print(f"[WARN] Skipping malformed PR entry on page {page} in {repo_full_name}")
                        continue

                    number = pr.get("number")
                    if not number:
                        continue

                    # Safely extract title/body
                    title_raw = pr.get("title") or ""
                    body_raw = pr.get("body") or ""
                    title = self._clean_github_text(title_raw)
                    body = self._clean_github_text(body_raw)

                    combined_text = f"{title_raw} {body_raw}"
                    raw_issue_refs = re.findall(r"#(\d+)", combined_text)
                    issue_numbers = [int(x) for x in raw_issue_refs]

                    # Safely extract author/login
                    author = None
                    user_obj = pr.get("user")
                    if isinstance(user_obj, dict):
                        author = user_obj.get("login")

                    # Build main artifact (defensive field access)
                    labels = []
                    if isinstance(pr.get("labels"), list):
                        try:
                            labels = [l.get("name") for l in pr.get("labels") if isinstance(l, dict) and l.get("name")]
                        except Exception:
                            labels = []

                    main_artifact = {
                        "artifact_type": "PullRequest",
                        "project": repo,
                        "artifact_id": f"PullRequest_{number}",
                        "source_sections": {
                            "title": title,
                            "description": body
                        },
                        "metadata": {
                            "author": author,
                            "created_at": pr.get("created_at"),
                            "merged_at": pr.get("merged_at"),
                            "closed_at": pr.get("closed_at"),
                            "state": pr.get("state"),
                            "number": number,
                            "merge_commit_sha": pr.get("merge_commit_sha"),
                            "head_sha": (pr.get("head") or {}).get("sha"),
                            "base_ref": (pr.get("base") or {}).get("ref"),
                            "labels": labels,
                            "commits_in_pr": [],
                            "issue_references": issue_numbers
                        },
                        "source_link": pr.get("html_url"),
                        "linked_commits": [],
                        "linked_issues": [],
                        "satd_label": None
                    }
                    results.append(main_artifact)

                    # Title section
                    if title.strip():
                        results.append({
                            "artifact_type": "PRSection",
                            "project": repo,
                            "artifact_id": f"PullRequest_{number}_title",
                            "parent_artifact_id": f"PullRequest_{number}",
                            "section_type": "title",
                            "source_sections": {"text": title},
                            "metadata": {"author": author},
                            "source_link": pr.get("html_url"),
                            "satd_label": None
                        })

                    # Description section
                    if body.strip():
                        results.append({
                            "artifact_type": "PRSection",
                            "project": repo,
                            "artifact_id": f"PullRequest_{number}_description",
                            "parent_artifact_id": f"PullRequest_{number}",
                            "section_type": "description",
                            "source_sections": {"text": body},
                            "metadata": {"author": author},
                            "source_link": pr.get("html_url"),
                            "satd_label": None
                        })

                    # PR-level comments (/issues/:number/comments for PR)
                    comments_url = pr.get("comments_url")
                    if comments_url:
                        if token_override:
                            comments = github_request(comments_url, github_token=token_override)
                        else:
                            comments = github_request(comments_url)

                        if comments is None:
                            print(f"[WARN] Null comments response for PR {number} in {repo_full_name}")
                            comments = []

                        for idx, c in enumerate(comments):
                            if c is None or not isinstance(c, dict):
                                continue
                            c_body = c.get("body") or ""
                            comment_text = self._clean_github_text(c_body)
                            if not comment_text.strip():
                                continue

                            # Safely get comment author
                            c_author = (c.get("user") or {}).get("login") if isinstance(c.get("user"), dict) else None

                            results.append({
                                "artifact_type": "PRSection",
                                "project": repo,
                                "artifact_id": f"PullRequest_{number}_comment_{idx}",
                                "parent_artifact_id": f"PullRequest_{number}",
                                "section_type": "comment",
                                "source_sections": {"text": comment_text},
                                "metadata": {
                                    "author": c_author,
                                    "created_at": c.get("created_at")
                                },
                                "source_link": c.get("html_url"),
                                "satd_label": None
                            })

                    # Inline review comments (/pulls/:number/comments)
                    review_comments_url = pr.get("review_comments_url")
                    if review_comments_url:
                        if token_override:
                            review_comments = github_request(review_comments_url, github_token=token_override)
                        else:
                            review_comments = github_request(review_comments_url)

                        if review_comments is None:
                            print(f"[WARN] Null review comments for PR {number} in {repo_full_name}")
                            review_comments = []

                        for idx, rc in enumerate(review_comments):
                            if rc is None or not isinstance(rc, dict):
                                continue
                            rc_body = rc.get("body") or ""
                            review_text = self._clean_github_text(rc_body)
                            if not review_text.strip():
                                continue

                            rc_user = rc.get("user") if isinstance(rc.get("user"), dict) else None
                            rc_author = rc_user.get("login") if rc_user else None

                            results.append({
                                "artifact_type": "PRSection",
                                "project": repo,
                                "artifact_id": f"PullRequest_{number}_reviewcomment_{idx}",
                                "parent_artifact_id": f"PullRequest_{number}",
                                "section_type": "review_comment",
                                "source_sections": {"text": review_text},
                                "metadata": {
                                    "author": rc_author,
                                    "created_at": rc.get("created_at"),
                                    "path": rc.get("path"),
                                    "line": rc.get("position")
                                },
                                "source_link": rc.get("html_url"),
                                "satd_label": None
                            })

                except Exception as e:
                    # Log and continue; protect whole extraction from a single bad PR
                    print(f"[ERROR] Unexpected error while processing PR in {repo_full_name}: {e}")
                    continue

            # pagination end condition
            if len(data) < 100:
                break
            page += 1

        return results

    def extract_from_repo(self, repo_full_name, github_token=None, show_progress=False):
        # Keep signature compatible with older code; simply forwards to extract_single_repo
        return self.extract_single_repo(repo_full_name, github_token=github_token or self.github_token, show_progress=show_progress)
