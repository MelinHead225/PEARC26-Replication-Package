import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
import re

# CONFIG
CSV_PATH = "satd_sonar_matched.csv" 
OUTPUT_DIR = Path("analysis_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# LOAD DATA
df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} rows")

# Map severity to ordinal
SEVERITY_MAP = {
    "INFO": 1,
    "MINOR": 2,
    "MAJOR": 3,
    "CRITICAL": 4,
    "BLOCKER": 5
}
df["sonar_severity_num"] = df["sonar_severity"].map(SEVERITY_MAP)

# Drop rows with missing critical values
analysis_df = df.dropna(
    subset=[
        "satd_embedding_score",
        "sonar_severity_num"
    ]
)
print(f"Using {len(analysis_df)} rows after cleaning")

# Convert sonar_severity to an ordered categorical for boxplot
analysis_df["sonar_severity"] = pd.Categorical(
    analysis_df["sonar_severity"],
    categories=list(SEVERITY_MAP.keys()),  # INFO -> BLOCKER
    ordered=True
)


# CORRELATION ANALYSIS

# SATD embedding vs Sonar severity
pearson_sev_r, pearson_sev_p = pearsonr(
    analysis_df["satd_embedding_score"],
    analysis_df["sonar_severity_num"]
)
spearman_sev_r, spearman_sev_p = spearmanr(
    analysis_df["satd_embedding_score"],
    analysis_df["sonar_severity_num"]
)

print("\nSATD Embedding vs Sonar Severity (ordinal)")
print(f"  Pearson r = {pearson_sev_r:.3f}, p = {pearson_sev_p:.3e}")
print(f"  Spearman ρ = {spearman_sev_r:.3f}, p = {spearman_sev_p:.3e}")

# BOX PLOT: SATD embedding by severity
plt.figure(figsize=(7, 5))

box = analysis_df.boxplot(
    column="satd_embedding_score",
    by="sonar_severity",
    grid=False,
    patch_artist=False  # black/white
)

plt.title("SATD Priority Score by SonarQube Issue Severity")
plt.suptitle("")
plt.ylabel("SATD Priority Score")
plt.xlabel("SonarQube Issue Severity")
# Annotate n=group_size below the column titles
group_counts = analysis_df.groupby("sonar_severity")["satd_embedding_score"].count()
xticks = list(SEVERITY_MAP.keys())
plt.xticks(ticks=range(1, len(xticks)+1), labels=xticks, rotation=0)

# Push x-axis down slightly by adjusting tick positions
ax = plt.gca()
ax.tick_params(axis='x', pad=15)  # increase padding between x-axis and labels

# Add n= below each x-tick label
for i, sev in enumerate(xticks, start=1):
    n = group_counts.get(sev, 0)
    # Get current label position
    xtick = ax.get_xticks()[i-1]
    y_pos = ax.get_ylim()[0] - 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0])  # slightly below bottom
    ax.text(xtick, y_pos, f"n={n}", ha='center', va='top', fontsize=9)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "satd_priority_by_severity_n.pdf")
plt.show()

