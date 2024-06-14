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
Silhouette, as well as the other approaches suffer from a computational
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

After, the user choose a number of groups (centroids) to cluster the
subsequences. After the clustering is done, we compute the silhouette score
for every centroid.

>We restrict the archive to datastes where the number of classes less than 7
for convenient plots, and datasets where the length of the time series is less
than 600 for ~~real~~ *reasonable* time interaction.

"""

state = st.session_state


if "evaluated" not in state:
    state.evaluated = False


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
    state.ts_length = state.demo._data.X_train.shape[1]
    state.n_ts = state.demo._n_samples * state.demo._n_labels
    state.window_size = int(0.1 * state.ts_length)
    resample()


def evaluate():
    evaluation_time = state.demo.evaluate_windows(state.window_size)
    state.evaluation_msg = f"""It took {evaluation_time:.2f}(s) to evaluate all
    {state.n_ts * (state.ts_length - state.window_size + 1)} candidates of size
    {state.window_size} in the {state.n_ts} samples (of length {state.ts_length})."""
    state.evaluated = True


def resample():
    state.demo.sample()
    state.evaluated = False


if "ucr_info" not in state:
    ucr_info()

with st.sidebar:
    st.write("#### Table of contents")

st.checkbox("Show UCR archive datasets summary", key="show_info", value=True)
if state.show_info:
    st.dataframe(state.ucr_info)

st.selectbox(
    "Select dataset for demo", state.ucr_info.name, on_change=demo, key="dataset_name"
)

if "demo" not in state:
    demo()
st.button("Resample", on_click=resample, key="resample_btn")
st.pyplot(state.demo.plot_data())

with st.form("slider_form"):
    min_size = int(0.05 * state.ts_length)
    max_size = int(0.7 * state.ts_length)
    default_size = int(0.1 * state.ts_length)
    st.slider(
        "Select a window size (between 5% and 70% of the time series length)",
        min_size,
        max_size,
        default_size,
        key="window_size",
    )
    submitted = st.form_submit_button("Evaluate")
    if submitted:
        evaluate()



if state.evaluated:
    st.write(state.evaluation_msg)
    st.pyplot(state.demo.plot_data(plot_shapelets=True))

    st.write(
        "The table below shows for each time series the position of the\
        shapelet and its corresponding silhouette score:"
    )

    st.table(
        pd.DataFrame(
            [
                (
                    state.demo.evaluations_df(i).index[0],
                    state.demo.evaluations_df(i).values[0][0],
                )
                for i in range(state.n_ts)
            ],
            columns=["Start position", "Silhouette score"],
            index=[f"TS {i+1}" for i in range(state.n_ts)],
        )
    )
