import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine

from src.storage.database import SAME_LENGTH_CLASSIFICATION_TABLE_NAME
from src.storage.database import VARIABLE_CLASSIFICATION_LENGTH_TABLE_NAME


engine = create_engine("postgresql://postgres:pass@localhost:5432/paper")

fixed_length = pd.read_sql(SAME_LENGTH_CLASSIFICATION_TABLE_NAME, engine)
variable_length = pd.read_sql(VARIABLE_CLASSIFICATION_LENGTH_TABLE_NAME, engine)


datasets = set(fixed_length.dataset_id.unique()).intersection(
    variable_length.dataset_id.unique()
)
print(
    f"Number of datasets classified with same length: {fixed_length.dataset_id.unique().shape[0]}"
)
print(
    f"Number of datasets classified with same length: {variable_length.dataset_id.unique().shape[0]}"
)
print(f"Number of datasets in common: {len(datasets)}")
# variable_length = variable_length[variable_length.dataset.isin(datasets)]
# fixed_length = fixed_length[fixed_length.dataset.isin(datasets)]
palette = {"silhouette": "C0", "gain": "C1", "fstat": "C2"}
markers = {"silhouette": "X", "gain": "+", "fstat": "1"}

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=9)  # fontsize of the axes title
plt.rc("axes", labelsize=SMALL_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)

cm = 1 / 2.54


def distribution_of_difference_between_accuracies():
    # Distribution of difference between the accuracy of variable length and fixed length
    fixed_view = fixed_length.groupby(
        ["dataset_id", "method", "K_shapelets"]
    ).mean()  # mean of classification using different windows
    vl_view = variable_length.set_index(
        ["dataset_id", "method", "K_shapelets"]
    ).reindex(fixed_view.index)

    fig, ax = plt.subplots(figsize=(14 * cm, 5 * cm))
    tmp_view = vl_view.reset_index()[["method", "K_shapelets", "Logistic Regression"]]
    diff = np.array(
        [
            tmp_view.K_shapelets.values,
            tmp_view.method,
            tmp_view["Logistic Regression"].values
            - fixed_view.reset_index()["Logistic Regression"].values,
        ]
    ).T
    diff = pd.DataFrame(diff, columns=["N. shapelets", "method", "accuracy"]).astype(
        {"N. shapelets": "int8"}
    )
    sns.boxplot(
        diff,
        x="N. shapelets",
        y="accuracy",
        hue="method",
        palette=palette,
        showfliers=False,
        ax=ax,
    )
    ax.axhline(y=0, color="black", linestyle="-")
    ax.legend(fancybox=True, framealpha=0.9, ncol=3)
    ax.set_ylabel("Difference of accuracy")
    fig.tight_layout()
    fig.savefig("/results/dist_diff_accuracies.png")
    fig.savefig("/code/results/dist_diff_accuracies.png")


def accuracy_plot(K_shapelets=5):
    # Comparison between methods for a set of datasets
    fig, ax = plt.subplots(1, 1, figsize=(14 * cm, 10 * cm))
    sns.scatterplot(
        variable_length[variable_length.K_shapelets == K_shapelets],
        x="dataset_id",
        y="Logistic Regression",
        hue="method",
        ax=ax,
        palette=palette,
        style="method",
        # markers=markers
    )
    ax.legend(loc="lower center", fancybox=True, framealpha=0.9, ncol=3)
    ax.set_ylabel("Accuracy")
    ax.set_title(
        f"Accuracy using logistic regression and {K_shapelets} shapelets per class"
    )
    plt.xticks(rotation=90)
    plt.grid()
    fig.tight_layout()
    fig.savefig(f"/results/{K_shapelets}_shapelets_accuracy.png")
    fig.savefig(f"/code/results/{K_shapelets}_shapelets_accuracy.png")


def distribution_accuracy():
    # Distribution of accuracy for the 3 methods with both fixed vs variable length
    fig, axs = plt.subplots(1, 2, figsize=(14 * cm, 5 * cm))
    fixed_view = fixed_length.groupby(
        ["dataset_id", "method", "K_shapelets"]
    ).mean()  # mean of classification using different windows
    vl_view = variable_length.set_index(
        ["dataset_id", "method", "K_shapelets"]
    ).reindex(fixed_view.index)
    sns.boxplot(
        fixed_view.reset_index(),
        x="K_shapelets",
        y="Logistic Regression",
        hue="method",
        showfliers=False,
        palette=palette,
        ax=axs[0],
    )
    sns.boxplot(
        vl_view.reset_index(),
        x="K_shapelets",
        y="Logistic Regression",
        hue="method",
        showfliers=False,
        palette=palette,
        ax=axs[1],
    )
    axs[0].get_legend().remove()
    axs[0].set_title("Fixed window length")
    axs[0].set_xlabel("N. shapelets per class")
    axs[0].set_ylabel("Average accuracy")
    axs[1].get_legend().remove()
    axs[1].set_title("Variable window length")
    axs[1].set_ylabel(None)
    axs[1].set_xlabel(None)
    # fig.suptitle("Average accuracy over datasets and different lengths")
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower right", fancybox=True, framealpha=0.9, ncol=3
    )
    fig.tight_layout()
    fig.savefig("/results/accuracy.png")
    fig.savefig("/code/results/accuracy.png")


def runtime_comparisons():
    # Latex table summarizing some stats about the running time of the three approaches
    windows_evals = pd.read_sql("windowevaluation", engine)
    windows_evals_view = windows_evals[
        ["silhouette_time", "fstat_time", "infogain_time"]
    ]

    df = (
        pd.DataFrame(
            [
                windows_evals_view.mean().values,
                windows_evals_view.std().values,
                windows_evals_view.min().values,
                windows_evals_view.quantile(0.25).values,
                windows_evals_view.quantile(0.5).values,
                windows_evals_view.quantile(0.75).values,
                windows_evals_view.max().values,
            ],
            columns=list(
                map(lambda x: " ".join(x.split("_")), windows_evals_view.columns)
            ),
            index=["mean", "std", "min", "25%", "50%", "75%", "max"],
        )
        * 1000
    )  # To milliseconds
    df = df.style.format(decimal=",", thousands=".", precision=2)
    with open("/results/runtimes.tex", "w") as text_file:
        text_file.write(df.to_latex())
    with open("/code/results/runtimes.tex", "w") as text_file:
        text_file.write(df.to_latex())


def variable_fixed_centroids_accuracy_comparison(df, method):
    fig, ax = plt.subplots(figsize=(14 * cm, 5 * cm))
    sns.boxplot(
        df[df.method == method],
        x="top_K",
        y="accuracy",
        hue="selection",
        showfliers=False,
        ax=ax,
    )
    ax.legend(loc="lower center", fancybox=True, framealpha=0.9, ncol=3)
    fig.tight_layout()
    fig.savefig("selection_methods_comparison.png")


if __name__ == "__main__":
    if not os.path.exists("/results/"):
        os.makedirs("/results/")
    if not os.path.exists("/code/results/"):
        os.makedirs("/code/results/")
    distribution_accuracy()
    accuracy_plot()
    distribution_of_difference_between_accuracies()
    runtime_comparisons()
