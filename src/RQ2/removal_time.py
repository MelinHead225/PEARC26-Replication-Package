import sqlite3
import pandas as pd
import os

# CONFIG
DB_PATH = "satd.db"
OUTPUT_DIR = ""
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
  AND removed_date IS NOT NULL
"""

df = pd.read_sql_query(query, conn)
conn.close()

# Convert to datetime
df["first_date"] = pd.to_datetime(df["first_date"], errors="coerce")
df["removed_date"] = pd.to_datetime(df["removed_date"], errors="coerce")

# CALCULATE REMOVAL DURATIONS
df["time_to_remove_days"] = (df["removed_date"] - df["first_date"]).dt.days

# AGGREGATE: MEAN + MEDIAN
removal_stats = (
    df.groupby("repo")["time_to_remove_days"]
    .agg(
        avg_removal_days="mean",
        median_removal_days="median",
        count_removed="count"
    )
    .reset_index()
)

# Save results
output_path = os.path.join(OUTPUT_DIR, "satd_removal_time_stats.csv")
removal_stats.to_csv(output_path, index=False)

print(f"Saved SATD removal time stats to '{output_path}'")
print(removal_stats)
