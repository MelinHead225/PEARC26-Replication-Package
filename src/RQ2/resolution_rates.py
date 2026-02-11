import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# CONFIG
DB_PATH = "satd.db"
OUTPUT_DIR = ""
os.makedirs(OUTPUT_DIR, exist_ok=True)

TIME_FREQ = "Q"

SATD_LABELS = (
    "code/design_debt",
    "documentation_debt",
    "requirement_debt",
    "test_debt",
    "scientific_debt",
)

# LOAD DATA
conn = sqlite3.connect(DB_PATH)

query = f"""
SELECT
    repo,
    first_date,
    removed_date
FROM artifacts
WHERE artifact_type = 'comment'
  AND satd_label IN {SATD_LABELS}
  AND first_date IS NOT NULL
"""

df = pd.read_sql_query(query, conn)
conn.close()

df["first_date"] = pd.to_datetime(df["first_date"], errors="coerce")
df["removed_date"] = pd.to_datetime(df["removed_date"], errors="coerce")

# BUILD EVENT DELTAS
df_start = df[["repo", "first_date"]].copy()
df_start["delta"] = 1
df_start.rename(columns={"first_date": "event_date"}, inplace=True)

df_end = df[df["removed_date"].notna()][["repo", "removed_date"]].copy()
df_end["delta"] = -1
df_end.rename(columns={"removed_date": "event_date"}, inplace=True)

events = pd.concat([df_start, df_end], ignore_index=True)

# AGGREGATE BY QUARTER
events_q = (
    events
    .groupby(["repo", pd.Grouper(key="event_date", freq=TIME_FREQ)])["delta"]
    .sum()
    .reset_index()
    .sort_values(["repo", "event_date"])
)

# LIVE SATD
events_q["live_satd"] = (
    events_q
    .groupby("repo")["delta"]
    .cumsum()
)

# NORMALIZATION (PER REPO)
events_q["normalized_live_satd"] = (
    events_q["live_satd"] /
    events_q.groupby("repo")["live_satd"].transform("max")
)

# PLOT 1: RAW VS NORMALIZED
fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharex=True)

sns.lineplot(
    data=events_q,
    x="event_date",
    y="live_satd",
    hue="repo",
    ax=axes[0],
    linewidth=2
)
axes[0].set_title("Raw Live SATD Comments")
axes[0].set_ylabel("Existing SATD Comments")
axes[0].grid(True)

sns.lineplot(
    data=events_q,
    x="event_date",
    y="normalized_live_satd",
    hue="repo",
    ax=axes[1],
    linewidth=2
)
axes[1].set_title("Normalized Live SATD (Per Repo)")
axes[1].set_ylabel("Normalized Existing SATD")
axes[1].grid(True)

for ax in axes:
    ax.legend_.remove()

handles, labels = axes[1].get_legend_handles_labels()
fig.legend(handles, labels, title="Repository", bbox_to_anchor=(1.02, 0.5), loc="center left")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "raw_vs_normalized_live_satd.png"), dpi=300)
plt.close()

# SATD RESOLUTION RATES
resolution = (
    df
    .assign(introduced=1, removed=df["removed_date"].notna().astype(int))
    .groupby("repo")[["introduced", "removed"]]
    .sum()
    .reset_index()
)

resolution["resolution_rate"] = resolution["removed"] / resolution["introduced"]

resolution.to_csv(
    os.path.join(OUTPUT_DIR, "satd_resolution_rates.csv"),
    index=False
)

