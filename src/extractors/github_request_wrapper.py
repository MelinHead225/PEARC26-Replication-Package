# src/extractors/helpers/github_request_wrapper.py
import time
import requests
from itertools import cycle
from dotenv import load_dotenv
import os

load_dotenv()

TOKENS = [t.strip() for t in os.getenv("GITHUB_TOKENS", "").split(",") if t.strip()]
if not TOKENS:
    raise ValueError("GITHUB_TOKENS environment variable is empty or not set!")

token_cycle = cycle(TOKENS)
current_token = next(token_cycle)


def github_request(url, github_token=None, method="GET", data=None, params=None, retries=5):
    """
    Universal GitHub API wrapper:
    - Supports params= for query strings
    - Token rotation + override
    - Rate limit handling
    - GET/POST
    """
    global current_token
    token = github_token or current_token
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "SATD-MultiArtifact-Linker/1.0"
    }

    for attempt in range(retries):
        try:
            if method == "GET":
                resp = requests.get(url, headers=headers, params=params, json=data, timeout=30)
            elif method == "POST":
                resp = requests.post(url, headers=headers, params=params, json=data, timeout=30)
            else:
                raise ValueError("Only GET and POST supported")

            if resp.status_code == 200:
                # Auto-rotate token only if we used the global one
                if github_token is None:
                    current_token = next(token_cycle)
                return resp.json()

            if resp.status_code == 403:
                reset = int(resp.headers.get("X-RateLimit-Reset", 0))
                sleep_for = max(reset - time.time() + 5, 60)
                print(f"Rate limit hit (token {token[:8]}...). Sleeping {sleep_for:.0f}s")
                time.sleep(sleep_for)
                continue

            if resp.status_code in (404, 410):
                return None

            print(f"HTTP {resp.status_code} on {url} — retrying in {2**attempt}s")
            time.sleep(2 ** attempt)

        except requests.RequestException as e:
            print(f"Request exception: {e} — retry {attempt+1}/{retries}")
            time.sleep(5)

    print(f"Failed after {retries} attempts: {url}")
    return None