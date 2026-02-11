import os
import re
import json
import subprocess
import hashlib
from datetime import datetime
from collections import defaultdict

class CommentExtractorWithLocation:
    """
    Extracts comments from source files and preserves:
      - first_file / first_line
      - last_file / last_line
      - commit SHAs and dates
    """

    def __init__(self):
        self.LANGUAGE_COMMENTS = {
            'python': {'ext': ['.py', '.pyx'], 'single': r'#(.*)', 'multi_start': r'^\s*(?:"""|\'\'\')(.*)', 'multi_end': r'(.*)(?:"""|\'\'\')\s*$'},
            'c_cpp':  {'ext': ['.c', '.cpp', '.h', '.hpp'], 'single': r'//(.*)', 'multi_start': r'^\s*/\*(.*)', 'multi_end': r'(.*)\*/\s*$'},
            'fortran':{'ext': ['.f', '.for', '.f90'], 'single': r'!(.*)', 'multi_start': None, 'multi_end': None},
            'java':   {'ext': ['.java'], 'single': r'//(.*)', 'multi_start': r'^\s*/\*(.*)', 'multi_end': r'(.*)\*/\s*$'},
            'shell':  {'ext': ['.sh'], 'single': r'#(.*)', 'multi_start': None, 'multi_end': None},
            'cmake':  {'ext': ['.cmake'], 'single': r'#(.*)', 'multi_start': None, 'multi_end': None},
            'matlab': {'ext': ['.m'], 'single': r'%(.*)', 'multi_start': None, 'multi_end': None},
            'rouge':  {'ext': ['.rg'], 'single': r'(?:#|--)\s?(.*)', 'multi_start': None, 'multi_end': None},
        }

    def generate_comment_id(self, repo_name, file_path, line_number, comment_text):
        key = f"{repo_name}:{file_path}:{line_number}:{comment_text.strip()}"
        return hashlib.md5(key.encode('utf-8')).hexdigest()

    def clean_comment(self, comment: str) -> str | None:
        comment = re.sub(r'[^a-zA-Z!?\s]', '', comment)
        comment = re.sub(r'\s+', ' ', comment).strip().lower()
        if not comment or any(k in comment for k in ['license', 'copyright', 'spdx']):
            return None
        return comment

    def normalize_comment(self, text: str) -> str:
        return re.sub(r'[^a-z\s]', '', text.lower()).strip()

    def get_language(self, ext: str):
        for lang in self.LANGUAGE_COMMENTS.values():
            if ext in lang['ext']:
                return lang
        return None

    def extract_comments_from_content(self, lines, file_path):
        comments = []
        lang = self.get_language(os.path.splitext(file_path)[1])
        if not lang:
            return []

        single_pat = lang['single']
        start_pat = lang['multi_start']
        end_pat = lang['multi_end']
        inside = False
        buf = ""
        start_line = None

        for i, raw in enumerate(lines, 1):
            line = raw.rstrip('\n')
            if inside:
                buf += " " + line.strip()
                if end_pat and re.search(end_pat, line):
                    inside = False
                    c = self.clean_comment(buf.strip())
                    if c: comments.append((start_line, c))
                    buf = ""
                continue

            if start_pat and re.match(start_pat, line):
                inside = True
                m = re.match(start_pat, line)
                buf = m.group(1).strip()
                start_line = i
                if end_pat and re.search(end_pat, line):
                    inside = False
                    c = self.clean_comment(buf.strip())
                    if c: comments.append((start_line, c))
                    buf = ""
                continue

            m = re.search(single_pat, line)
            if m:
                c = self.clean_comment(m.group(1).strip())
                if c: comments.append((i, c))
        return comments

    def _run_git(self, repo_path, *args):
        git_dir = os.path.join(repo_path, '.git')
        work_tree = repo_path
        cmd = ['git', '--git-dir', git_dir, '--work-tree', work_tree] + list(args)
        return subprocess.run(cmd, capture_output=True, text=True, check=True)

    def _get_head(self, repo_path):
        try:
            res = self._run_git(repo_path, 'rev-parse', 'HEAD')
            return res.stdout.strip()
        except Exception:
            return None

    def extract_from_repo(self, repo_path):
        """
        Main API: extracts comments preserving file and line locations
        """
        repo_path = os.path.abspath(repo_path)
        repo_name = os.path.basename(repo_path)
        print(f"[+] Extracting comments from {repo_name}")

        try:
            res = self._run_git(repo_path, 'ls-files')
            files = res.stdout.splitlines()
        except Exception:
            return []

        exts = {e for lang in self.LANGUAGE_COMMENTS.values() for e in lang['ext']}
        tracked_files = [f for f in files if os.path.splitext(f)[1] in exts]

        history = defaultdict(list)
        head = self._get_head(repo_path) or ''
        head_ts = self._run_git(repo_path, 'show', '-s', '--format=%at', head).stdout.strip() if head else datetime.utcnow().timestamp()
        head_date = datetime.utcfromtimestamp(int(head_ts)).isoformat()

        for rel_path in tracked_files:
            try:
                log = self._run_git(repo_path, 'log', '--reverse', '--format=%H %at', '--', rel_path)
            except:
                continue

            for line in log.stdout.splitlines():
                parts = line.split()
                if len(parts) != 2: continue
                sha, ts = parts
                try:
                    content = self._run_git(repo_path, 'show', f"{sha}:{rel_path}").stdout
                    lines = content.splitlines(keepends=True)
                    date = datetime.utcfromtimestamp(int(ts)).isoformat()
                    for lineno, comment in self.extract_comments_from_content(lines, rel_path):
                        key = (self.normalize_comment(comment), rel_path)
                        history[key].append({'commit': sha, 'date': date, 'file': rel_path, 'line': lineno, 'display_comment': comment})
                except:
                    continue

        results = []
        for (norm_text, path), apps in history.items():
            apps.sort(key=lambda x: x['date'])
            first, last = apps[0], apps[-1]

            removed_sha = removed_date = None
            if last['commit'] != head:
                removed_sha, removed_date = self._find_removal(repo_path, path, norm_text, last['commit'])

            end_date = removed_date or head_date
            try:
                days = (datetime.fromisoformat(end_date) - datetime.fromisoformat(first['date'])).days
            except:
                days = None

            occurrence_shas = [a['commit'].zfill(40) for a in apps if a.get('commit')]
            comment_id = self.generate_comment_id(repo_name, path, first['line'], first['display_comment'])

            results.append({
                'artifact_id': f"{repo_name}:{comment_id}",
                'artifact_type': 'comment',
                'project': repo_name,
                'repo_name': repo_name,
                'comment_id': comment_id,
                'comment': first['display_comment'],
                'first_file': first['file'],
                'first_line': first['line'],
                'first_commit': first.get('commit'),
                'first_date': first['date'],
                'last_file': last['file'],
                'last_line': last['line'],
                'last_commit': last.get('commit'),
                'last_date': last['date'],
                'occurrence_shas': occurrence_shas,
                'removed_commit': removed_sha,
                'removed_date': removed_date,
                'duration_days': days,
                'occurrences': len(apps),
                'file_type_category': self.classify_file_path(path),
                'linked_commits': []
            })
        print(f"[+] Extracted {len(results)} comments with location info")
        return results

    def classify_file_path(self, path: str) -> str:
        lower = path.lower()
        if any(k in lower for k in ["docs/", "doc/", "readme", ".md", ".rst"]):
            return "documentation"
        if any(k in lower for k in ["test/", "tests/", "_test", "test_"]):
            return "test"
        return "source"

    def _find_removal(self, repo_path, rel_path, normalized_comment, last_sha):
        try:
            log = self._run_git(repo_path, 'log', '--format=%H', f"{last_sha}..HEAD", '--', rel_path)
            for sha in log.stdout.strip().splitlines():
                try:
                    content = self._run_git(repo_path, 'show', f"{sha}:{rel_path}").stdout
                    lines = content.splitlines(keepends=True)
                    comments = self.extract_comments_from_content(lines, rel_path)
                    norm_set = {self.normalize_comment(c) for _, c in comments}
                    if normalized_comment not in norm_set:
                        ts = self._run_git(repo_path, 'show', '-s', '--format=%at', sha).stdout.strip()
                        return sha, datetime.utcfromtimestamp(int(ts)).isoformat()
                except subprocess.CalledProcessError:
                    ts = self._run_git(repo_path, 'show', '-s', '--format=%at', sha).stdout.strip()
                    return sha, datetime.utcfromtimestamp(int(ts)).isoformat()
        except:
            pass
        return None, None
