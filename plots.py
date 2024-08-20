import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


if not os.path.isdir("results/imgs"):
    os.mkdir("results/imgs")


# Figure configs
cm = 1 / 2.54
evaluation_colors = {
    "silhouette": "#0072B2",
    "fstat": "#E69F00",
    "information gain": "#009E73",
}
evaluation_order = ["silhouette", "information gain", "fstat"]
extraction_colors = {"clustering": "#0072B2", "FSS": "#D55E00"}
extraction_order = ["clustering", "FSS"]


# Extraction time
fig, ax = plt.subplots(figsize=(14 * cm, 6 * cm))
sns.scatterplot(
    pd.read_parquet("results/data/extraction.parquet.gzip"),
    x="numcandidates",
    y="time",
    hue="extraction",
    ax=ax,
    palette=extraction_colors,
)
ax.set_xlabel("Number of possible subsequences")
ax.set_ylabel("Shapelet extraction\nTime (s)")
ax.set_xscale("log")
fig.tight_layout()
plt.savefig("results/imgs/extraction_time.png", dpi=300)


# Number of candidates
fig, ax = plt.subplots(figsize=(14 * cm, 6 * cm))
sns.scatterplot(
    pd.read_parquet("results/data/numcandidates.parquet.gzip"),
    x="numcandidates",
    y="Number of candidates",
    hue="extraction method",
    palette=extraction_colors,
)
ax.set_yscale("log")
ax.set_xscale("log")
ax.set_xlabel("Number of possible candidates")
ax.set_ylabel("Number of candidates\n(log scale)")
ax.legend(loc="lower center", fancybox=True, framealpha=0.9, ncol=2)
fig.tight_layout()
plt.savefig("results/imgs/clustering_fss_extraction.png", dpi=300)


# Distance evaluation
fig, ax = plt.subplots(figsize=(14 * cm, 7 * cm))
sns.scatterplot(
    pd.read_parquet("results/data/distances.parquet.gzip"),
    x="numcandidates",
    y="time",
    hue="extraction",
    ax=ax,
    palette=extraction_colors,
)
ax.set_xlabel("Number of possible subsequences")
ax.set_ylabel("Distance evaluation\nTime (s)")
ax.set_yscale("log")
ax.set_xscale("log")
fig.tight_layout()
plt.savefig("results/imgs/distance_time.png", dpi=300)


# Accuracy plots
accuracy_df = pd.read_parquet("results/data/accuracy.parquet.gzip")

## FSS vs Clustering
fig, ax = plt.subplots(figsize=(14 * cm, 6 * cm))
sns.boxplot(
    accuracy_df[(accuracy_df.evaluation == "silhouette")],
    x="k",
    y="accuracy",
    hue="extraction",
    showfliers=False,
    palette=extraction_colors,
    hue_order=extraction_order,
    ax=ax,
)
ax.legend(loc="lower center", fancybox=True, framealpha=0.9, ncol=3)
ax.set(ylim=(0.36, 1.01))
fig.tight_layout()
plt.savefig("results/imgs/extraction_comparison.png", dpi=300)

## Evaluation methods
fig, axs = plt.subplots(2, 1, figsize=(14 * cm, 8 * cm), sharex=True)
ax = axs[0]
sns.boxplot(
    accuracy_df[(accuracy_df.extraction == "clustering")],
    x="k",
    y="accuracy",
    hue="evaluation",
    showfliers=False,
    palette=evaluation_colors,
    hue_order=evaluation_order,
    ax=ax,
)
ax.get_legend().remove()
ax.set(ylim=(0.36, 1.01))
ax.set_xlabel(None)
ax.set_ylabel("clustering")

ax = axs[1]
sns.boxplot(
    accuracy_df[(accuracy_df.extraction == "FSS")],
    x="k",
    y="accuracy",
    hue="evaluation",
    showfliers=False,
    palette=evaluation_colors,
    hue_order=evaluation_order,
    ax=ax,
)
ax.legend(loc="lower right", fancybox=True, framealpha=0.9, ncol=3)
ax.set(ylim=(0.36, 1.01))
ax.set_ylabel("FSS")
fig.tight_layout()
plt.savefig("results/imgs/evaluation_comparison.png", dpi=300)
