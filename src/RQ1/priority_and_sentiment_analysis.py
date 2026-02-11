import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import mannwhitneyu

# Config 
DB_PATH = "satd.db"
FIG_DIR = "RQ1/figures"

# Create figures directory if it doesn't exist
os.makedirs(FIG_DIR, exist_ok=True)

# Connect to DB
conn = sqlite3.connect(DB_PATH)

# Average priority per SATD type   
df_priority = pd.read_sql_query("""
SELECT satd_label, AVG(priority_score_variant) AS avg_priority, COUNT(*) AS n
FROM artifacts
WHERE satd_label != 'non_debt'
GROUP BY satd_label
ORDER BY avg_priority DESC;
""", conn)

# Capitalize SATD labels for the x-axis
df_priority['satd_label'] = df_priority['satd_label'].str.replace('_', ' ').str.title()

plt.figure(figsize=(8,5))
sns.barplot(data=df_priority, x='satd_label', y='avg_priority')
plt.title("Average Priority per SATD Type")
plt.ylabel("Average Priority Score")
plt.xlabel("SATD Type")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "avg_priority_per_satd.png"))
plt.close()

# Priority vs Sentiment   
df_sentiment = pd.read_sql_query("""
SELECT predicted_sentiment, priority_score_variant
FROM artifacts
WHERE satd_label != 'non_debt';
""", conn)

# Capitalize sentiment labels
df_sentiment['predicted_sentiment'] = df_sentiment['predicted_sentiment'].str.replace('_', ' ').str.title()

plt.figure(figsize=(6,5))
sns.barplot(
    data=df_sentiment.groupby('predicted_sentiment')['priority_score_variant'].mean().reset_index(),
    x='predicted_sentiment', y='priority_score_variant'
)
plt.title("SATD Average Priority vs Predicted Sentiment")
plt.xlabel("Predicted Sentiment")
plt.ylabel("SATD Average Priority Score")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "avg_priority_vs_sentiment.png"))
plt.close()

# Average priority per SATD type 
print("\n=== Average Priority per SATD Type ===")
print(df_priority.to_string(index=False))

# Priority vs Sentiment summary 
df_summary = df_sentiment.groupby('predicted_sentiment')['priority_score_variant'].agg(['mean','count']).reset_index()
print("\n=== Average Priority vs Predicted Sentiment ===")
print(df_summary.to_string(index=False))

plt.figure(figsize=(6,6))
sns.violinplot(x='predicted_sentiment', y='priority_score_variant', data=df_sentiment, inner='quartile')
plt.title("Distribution of SATD Priority by Predicted Sentiment")
plt.xlabel("Predicted Sentiment")
plt.ylabel("Priority Score")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "priority_distribution_vs_sentiment.png"))
plt.close()

# Statistical Analysis: Sentiment vs Priority 
neg = df_sentiment[df_sentiment['predicted_sentiment'] == 'Negative']['priority_score_variant']
non_neg = df_sentiment[df_sentiment['predicted_sentiment'] != 'Negative']['priority_score_variant']

# Mann-Whitney U test
u_stat, p_value = mannwhitneyu(neg, non_neg, alternative='two-sided')

# Rank-biserial correlation
n1, n2 = len(neg), len(non_neg)
rbc = 1 - (2*u_stat)/(n1*n2)

print("\n=== Statistical Analysis: Sentiment vs Priority ===")
print(f"Mann-Whitney U: {u_stat}")
print(f"p-value: {p_value:.5e}")
print(f"Rank-biserial correlation (effect size): {rbc:.3f}")

# Close DB connection
conn.close()
