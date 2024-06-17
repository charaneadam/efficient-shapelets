import streamlit as st
import pandas as pd
from src.presentation.demo import Demo
from src.storage.database import Dataset

"""
# Extension for Information Systems.

## DOLAP summary
Silhouette score is a good measure to evaluate time series shapelets. It has
similar performance to alternatives (Information gain and F-statistic) but
superior when the number of shapelets is small (Figure 3 in the paper shows an
example where the accuracy of classification increases while the number of
shapelets used to transform the data is increasing, whereas the accuracy when
the data is transformed by shapelets selected using silhouette, is higher with
small number of shapelets, but stays the same while the number of shapelets is
increasing).

## Challenge
Silhouette, as well as the other approaches, suffer from a computational
problem in order to evaluate a subsequence:

The score $s(S_l^c)$ for one subsequence $S$ of length $l$ and representing
class $c$ requires the distance to all time series.
For instance, the silhouette score for a shapelet $S_l$ is computed as

$$s(S_l^c) = \\frac{b-a}{\max(a, b)}$$, where:
- $b$: the average distance between the $S_l^c$ and time series
from classes different than $c$
- $a$: the average distance between $S_l^c$ and time series from class $c$.

The distance between a subsequence $S_l$ and a time series is the minimum
distance between $S_l$ and all subsequences of length $l$ from the 
time series).

## Idea/Content of the extension
Instead of directly evaluating every subsequence independently, which requires
comparing it to all other subsequences, we will cluster similar subsequences
into many small groups, and from each group we select a representative. Then,
we restrict the space of candidates to the representatives only. We show that
this reduces the computational cost significantly, while still providing 
shapelets with the same quality in terms of the silhouette score.

## Demo
This demo implments the idea above. In a first step the user chooses a
dataset from the UCR archive, and selects a window size for the shapelet size,
then all the subsequences will be evaluated using silhouette score by brute
force (exact solution).

In the second step, the user choose a number of groups (centroids) to cluster 
the subsequences. After the clustering is done, the silhouette score for 
every centroid is reported.
"""

state = st.session_state


if "evaluated" not in state:
    state.evaluated = False
if "clustered" not in state:
    state.clustered = False


def ucr_info():
    query = Dataset.select().where(
        (Dataset.n_classes <= 10)
        & (Dataset.length < 700)
        & (Dataset.missing_values == False)
    )
    cols = ["data_type", "name", "train", "test", "length", "n_classes"]

    state.ucr_info = pd.DataFrame(list(query.dicts()))[cols]
    state.dataset_name = state.ucr_info.loc[0, "name"]
    demo()


def demo():
    state.demo = Demo(st.session_state.dataset_name)
    state.ts_length = state.demo.data._data.X_train.shape[1]
    state.n_ts = state.demo.data._n_samples * state.demo.data._n_labels
    state.window_size = int(0.1 * state.ts_length)
    resample()


def resample():
    state.demo.data.sample()
    state.evaluated = False
    state.clustered = False


def evaluate():
    evaluation_time = state.demo.evaluate_windows(state.window_size)
    state.evaluation_msg = f"""It took {evaluation_time:.2f}(s) to evaluate all
    {state.n_ts * (state.ts_length - state.window_size + 1)} candidates of size
    {state.window_size} in the {state.n_ts} samples (of length {state.ts_length}).
    """
    state.evaluated = True


def cluster():
    state.demo.run_pca_kmeans(state.n_centroids)
    state.centroids_df = state.demo.run_kmeans()
    state.clustered = True


if "ucr_info" not in state:
    ucr_info()

with st.sidebar:
    st.write("#### Table of contents")

"""### Dataset
>We restrict the archive to datastes where the number of classes less than 7
for convenient plots, and datasets where the length of the time series is less
than 600 for ~~real~~ *reasonable* time interaction."""
if st.checkbox("Show UCR archive datasets summary", value=True):
    st.dataframe(state.ucr_info)

st.selectbox(
    "Select dataset for demo", state.ucr_info.name, on_change=demo, key="dataset_name"
)


col1, col2 = st.columns([0.8, 0.2])
with col1:
    st.write(
        f"""The rest of the demo will be using the {state.n_ts} below.
    You can reandomly select different ones by clicking the resample buttom 
    on the right."""
    )
with col2:
    st.button("Resample", on_click=resample, key="resample_btn")

st.pyplot(state.demo.data.plot())


"""### Evaluation
Here, we extract all the possible candidates of some fixed length, and
evaluate them using the silhouette score."""
with st.form("window_size_form"):
    min_size = int(0.05 * state.ts_length)
    max_size = int(0.7 * state.ts_length)
    default_size = int(0.1 * state.ts_length)
    st.slider(
        "Select a window size (between 5% and 70% of the time series length)\
        to extract candidates",
        min_size,
        max_size,
        default_size,
        key="window_size",
    )
    submitted = st.form_submit_button("Evaluate")
    if submitted:
        state.clustered = False
        evaluate()

if state.evaluated:
    st.write(state.evaluation_msg)
    """The figure below shows for every time series the best candidate
    (highest silhouette score), and highlights it with black."""

    st.pyplot(state.demo.plot_data(plot_shapelets=True))

    """Notice that, depending on the dataset and the samples, time series might
    have more than one representative ***pattern***."""

    f"""The table below shows for each time series the position of the
    shapelet and its corresponding silhouette score. The time series are 
    labeled from 1 to {state.n_ts}, where TS1 is the time series at the bottom
    left in blue, TS2 is the bottom blue time series in the second plot, TS{4}
    is the time series colored with orange on the left plot..."""

    st.table(
        pd.DataFrame(
            [
                (
                    state.demo.data._labels[i // state.demo.data._n_samples],
                    state.demo.silhouette.evaluations_df(i).values[0][0],
                    state.demo.silhouette.evaluations_df(i).index[0],
                )
                for i in range(state.n_ts)
            ],
            columns=["Label", "Silhouette score", "Start position"],
            index=[f"TS {i+1}" for i in range(state.n_ts)],
        )
    )

    "### Clustering"

    
    """This section is the core contribution for Information systems extension.
    """
    
    """
    We want to find subsequences, that have high silhouette score to consider
    as shapelets but without the need to compare each candidate with all other
    candidates. The main idea is to create small, non-overlapping, groups of
    similar candidates, and from each group evaluate one representative.

    More formally, in the brute force approach, in order to evaluate a
    candidate, it has to be compared with all other $N-1$ candidates, which
    results in $\mathcal{O}(N^2)$ comparisons. In our approach, we find $K$
    representatives, then we evaluate only the $K$ representatives, which
    results in $\mathcal{O}(NK)$ comparisons, where $K << N$.

    Concretely, we will cluster the subsequences using KMeans algorithm into
    $K$ clusters. The $K$ centroids will be our candidates, then we evaluate
    every centroid(candidate).
    """
    with st.form("cluster_form"):
        min_size = state.demo.data._n_labels * 3
        max_size = state.demo.data._n_samples * (
            state.ts_length - state.window_size + 1
        )
        max_size = max_size
        default_size = min(state.demo.data._n_labels * 25, max_size//10)
        st.slider(
            "Select number of centroids",
            min_size,
            max_size,
            default_size,
            key="n_centroids",
        )
        submitted = st.form_submit_button("Cluster")
        if submitted:
            cluster()

    if state.clustered:
        """
        The figure below shows the first two principal components of the
        subseuqnces (in grey) as well as the centroids.
        """
        st.checkbox("Show labels of the windows", value=False, key="show_labels")

        st.pyplot(state.demo.pca_kmeans.plot(with_labels=state.show_labels))
        st.dataframe(state.centroids_df)
