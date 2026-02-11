import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------
# CONFIG
# -----------------------------
ISSUE_CSV = "issue_level_analysis.csv"
PR_CSV = "pr_level_analysis.csv"
OUTPUT_FOLDER = Path("RQ4")
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

MIN_TOKENS = 10
NUM_BINS = 10

# LOAD & PREPARE
def load_and_prepare(csv_path, artifact_type):
    df = pd.read_csv(csv_path)

    if artifact_type == "Issue":
        df = df[df["issue_total_tokens"] >= MIN_TOKENS].copy()
        df = df.rename(columns={
            "issue_total_tokens": "total_tokens",
            "issue_has_satd": "has_satd",
            "issue_num_comments": "num_comments",
            "issue_sentiment_score": "sentiment",
            "issue_priority_score": "priority",
            "issue_id": "artifact_id"
        })
    else:
        df = df[df["pr_total_tokens"] >= MIN_TOKENS].copy()
        df = df.rename(columns={
            "pr_total_tokens": "total_tokens",
            "pr_has_satd": "has_satd",
            "pr_num_comments": "num_comments",
            "pr_sentiment_score": "sentiment",
            "pr_priority_score": "priority",
            "pr_id": "artifact_id"
        })

    df["artifact_type"] = artifact_type
    df["length_bin"] = pd.qcut(df["total_tokens"], q=NUM_BINS, duplicates="drop")
    df["comments_bin"] = pd.qcut(df["num_comments"], q=NUM_BINS, duplicates="drop")

    return df

issues = load_and_prepare(ISSUE_CSV, "Issue")
prs = load_and_prepare(PR_CSV, "PR")
combined_df = pd.concat([issues, prs], ignore_index=True)

# AGGREGATION
length_summary = combined_df.groupby(["artifact_type", "length_bin"]).agg(
    satd_rate=("has_satd", "mean"),
    mean_sentiment=("sentiment", "mean"),
    mean_priority=("priority", "mean")
).reset_index()

comments_summary = combined_df.groupby(["artifact_type", "comments_bin"]).agg(
    satd_rate=("has_satd", "mean"),
    mean_sentiment=("sentiment", "mean"),
    mean_priority=("priority", "mean")
).reset_index()

# PLOTTING FUNCTION (DUAL X)
def dual_x_plot(issue_df, pr_df, x_col, y_col, ylabel, title, outfile,
                issue_xlabel, pr_xlabel, legend_loc="best"):
    fig, ax_issue = plt.subplots(figsize=(7,5))
    ax_pr = ax_issue.twiny()

    # Issue line (blue)
    ax_issue.plot(
        issue_df[x_col].astype(str),
        issue_df[y_col],
        marker="o",
        color="tab:blue",
        label="Issue"
    )
    ax_issue.set_xlabel(issue_xlabel)
    ax_issue.set_ylabel(ylabel)
    ax_issue.tick_params(axis="x", rotation=45)

    # PR line (orange)
    ax_pr.plot(
        pr_df[x_col].astype(str),
        pr_df[y_col],
        marker="s",
        linestyle="--",
        color="tab:orange",
        label="PR"
    )
    ax_pr.set_xlabel(pr_xlabel)
    ax_pr.tick_params(axis="x", rotation=45)

    # Combined legend with configurable location
    lines1, labels1 = ax_issue.get_legend_handles_labels()
    lines2, labels2 = ax_pr.get_legend_handles_labels()
    ax_issue.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc=legend_loc
    )

    plt.title(title)
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()

# LENGTH-BASED PLOTS
dual_x_plot(
    length_summary[length_summary["artifact_type"] == "Issue"],
    length_summary[length_summary["artifact_type"] == "PR"],
    x_col="length_bin",
    y_col="satd_rate",
    ylabel="SATD Rate",
    title="SATD Rate by Length Bin (Issues vs PRs)",
    outfile=OUTPUT_FOLDER / "satd_rate_length_dual_x.png",
    issue_xlabel="Issue Token Length Bins",
    pr_xlabel="PR Token Length Bins"
)

dual_x_plot(
    length_summary[length_summary["artifact_type"] == "Issue"],
    length_summary[length_summary["artifact_type"] == "PR"],
    x_col="length_bin",
    y_col="mean_sentiment",
    ylabel="Mean Sentiment",
    title="Mean Sentiment by Length Bin (Issues vs PRs)",
    outfile=OUTPUT_FOLDER / "sentiment_length_dual_x.png",
    issue_xlabel="Issue Token Length Bins",
    pr_xlabel="PR Token Length Bins"
)

dual_x_plot(
    length_summary[length_summary["artifact_type"] == "Issue"],
    length_summary[length_summary["artifact_type"] == "PR"],
    x_col="length_bin",
    y_col="mean_priority",
    ylabel="Mean Priority",
    title="Mean Priority by Length Bin (Issues vs PRs)",
    outfile=OUTPUT_FOLDER / "priority_length_dual_x.png",
    issue_xlabel="Issue Token Length Bins",
    pr_xlabel="PR Token Length Bins",
    legend_loc="upper left"
)

# Print length-based summary
print("Length-Based Summary")
print(length_summary)

# Print comments-based summary
print("\nComments-Based Summary")
print(comments_summary)
