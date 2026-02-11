import sqlite3
import json
import os

DB_PATH = "satd.db"
JSONL_PATH = "all_repos_cleaned_master_artifacts_classified_priority_embed_only.jsonl"

# Remove existing DB if needed
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

# Create table with all relevant fields
cur.execute("""
CREATE TABLE artifacts (
    artifact_id TEXT PRIMARY KEY,
    artifact_type TEXT,
    project TEXT,
    repo TEXT,
    satd_label TEXT,

    priority_score REAL,
    priority_score_variant REAL,
    embedding_score REAL,
    lexical_score REAL,
    tfidf_score REAL,

    predicted_sentiment TEXT,
    predicted_sentiment_confidence REAL,

    source_link TEXT,
    source_sections TEXT,
    parent_artifact_id TEXT,
    section_type TEXT,
    metadata TEXT,

    comment TEXT,
    comment_id TEXT,
    duration_days REAL,
    file_type_category TEXT,
    first_commit TEXT,
    first_date TEXT,
    first_file TEXT,
    first_line INTEGER,
    last_commit TEXT,
    last_date TEXT,
    last_file TEXT,
    last_line INTEGER,
    occurrence_shas TEXT,
    occurrences INTEGER,
    removed_commit TEXT,
    removed_date TEXT,
    repo_name TEXT,

    linked_commits TEXT,
    linked_prs TEXT,
    linked_pull_requests TEXT
);
""")

# Load JSONL
with open(JSONL_PATH, "r", encoding="utf-8") as f:
    for line in f:
        e = json.loads(line)

        # Convert lists to JSON strings for storage
        linked_commits = json.dumps(e.get("linked_commits", []))
        linked_prs = json.dumps(e.get("linked_prs", []))
        linked_pull_requests = json.dumps(e.get("linked_pull_requests", []))
        occurrence_shas = json.dumps(e.get("occurrence_shas", []))
        source_sections = json.dumps(e.get("source_sections", {}))
        metadata = json.dumps(e.get("metadata", {}))

        cur.execute("""
        INSERT OR IGNORE INTO artifacts (
            artifact_id, artifact_type, project, repo, satd_label,
            priority_score, priority_score_variant, embedding_score, lexical_score, tfidf_score,
            predicted_sentiment, predicted_sentiment_confidence,
            source_link, source_sections, parent_artifact_id, section_type, metadata,
            comment, comment_id, duration_days, file_type_category,
            first_commit, first_date, first_file, first_line,
            last_commit, last_date, last_file, last_line,
            occurrence_shas, occurrences, removed_commit, removed_date, repo_name,
            linked_commits, linked_prs, linked_pull_requests
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            e.get("artifact_id"),
            e.get("artifact_type"),
            e.get("project"),
            e.get("repo"),
            e.get("satd_label"),
            e.get("priority_score"),
            e.get("priority_score_variant"),
            e.get("embedding_score"),
            e.get("lexical_score"),
            e.get("tfidf_score"),
            e.get("predicted_sentiment"),
            e.get("predicted_sentiment_confidence"),
            e.get("source_link"),
            source_sections,
            e.get("parent_artifact_id"),
            e.get("section_type"),
            metadata,
            e.get("comment"),
            e.get("comment_id"),
            e.get("duration_days"),
            e.get("file_type_category"),
            e.get("first_commit"),
            e.get("first_date"),
            e.get("first_file"),
            e.get("first_line"),
            e.get("last_commit"),
            e.get("last_date"),
            e.get("last_file"),
            e.get("last_line"),
            occurrence_shas,
            e.get("occurrences"),
            e.get("removed_commit"),
            e.get("removed_date"),
            e.get("repo_name"),
            linked_commits,
            linked_prs,
            linked_pull_requests
        ))

conn.commit()
cur.close()
conn.close()

print("Database successfully recreated with all linking fields.")
