import streamlit as st
import numpy as np
from io import BytesIO
from streamlit_lottie import st_lottie
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import requests
import lightgbm as lgb
import time
from river import optim

from river import tree, neighbors, naive_bayes, ensemble, linear_model, forest
from river.drift import ADWIN
from river.drift.binary import DDM
import datetime as dt

from river import datasets
from river import evaluate
from river import metrics
from river import preprocessing  # we are going to use that later
from river.datasets import synth  # we are going to use some synthetic datasets too
from river import tree
from river.drift import ADWIN, KSWIN, PageHinkley
from river.drift.binary import DDM, EDDM, HDDM_A, HDDM_W

from river import preprocessing
from river import optim
from river import compose


def plot_performance(dataset, metric, models):
    metric_name = metric.__class__.__name__

    # To make the generated data reusable
    dataset = list(dataset)
    fig, ax = plt.subplots(figsize=(10, 5), nrows=3, dpi=300)
    for model_name, model in models.items():
        step = []
        error = []
        r_time = []
        memory = []
        aa = []

        for checkpoint in evaluate.iter_progressive_val_score(
            dataset, model, metric, measure_time=True, measure_memory=True, step=1
        ):
            step.append(checkpoint["Step"])
            error.append(checkpoint[metric_name].get())

            # Convert timedelta object into seconds
            r_time.append(checkpoint["Time"].total_seconds())
            # Make sure the memory measurements are in MB
            raw_memory = checkpoint["Memory"]
            memory.append(raw_memory * 2**-20)
        aa.append(checkpoint[metric_name].get())

        ax[0].plot(step, error, label=model_name)
        ax[1].plot(step, r_time, label=model_name)
        ax[2].plot(step, memory, label=model_name)

    ax[0].set_ylabel(metric_name)
    ax[1].set_ylabel("Time (seconds)")
    ax[2].set_ylabel("Memory (MB)")
    ax[2].set_xlabel("Instances")

    ax[0].grid(True)
    ax[1].grid(True)
    ax[2].grid(True)

    ax[0].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.25),
        ncol=3,
        fancybox=True,
        shadow=True,
    )
    plt.tight_layout()
    plt.close()

    return fig, aa


def get_dataset(name):
    data = None
    if name == "Elec2":
        data = datasets.Elec2()
    elif name == "Phishing":
        data = datasets.Phishing()
    else:
        data = datasets.SMTP()

    return data


def get_Regressor_dataset(name):
    data = None
    if name == "ChickWeights":
        data = datasets.ChickWeights()
    elif name == "Friedman":
        data == datasets.synth.Friedman(seed=42).take(10_000)
    else:
        data = datasets.TrumpApproval()

    return data


def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == "LogisticRegression":
        optimizer = st.sidebar.radio("optimizer:", ("SGD", "FTRL", "none"))
        params["optimizer"] = optimizer
        l1 = st.sidebar.radio("L1", (0, 1))
        l2 = st.sidebar.radio("L2:", (0, 1))

        params["l1"] = l1
        params["l2"] = l2
    elif clf_name == "HoeffdingTreeClassifier":
        grace_period = st.sidebar.slider("grace_period", 50, 200)
        max_depth = st.sidebar.slider("max_depth", 5, 15)
        params["grace_period"] = grace_period
        params["max_depth"] = max_depth
    elif clf_name == "Naive Bayes" or clf_name == "Perceptron":
        params["max_depth"] = 0
    elif clf_name == "SRPClassifier":
        n_models = st.sidebar.slider("n_models", 2, 12)
        params["n_model"] = n_models
    else:
        n_models = st.sidebar.slider("n_models", 2, 12)
        params["n_models"] = n_models

    return params


def add_Regressor_parameter_ui(clf_name):
    params = dict()
    if clf_name == "HoeffdingTreeRegressor":
        grace_period = st.sidebar.slider("grace_period", 50, 200)
        max_depth = st.sidebar.slider("max_depth", 5, 15)
        params["grace_period"] = grace_period
        params["max_depth "] = max_depth
    else:
        n_models = st.sidebar.slider("n_models", 3, 12)
        params["n_models"] = n_models

    return params


def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


def get_classifier(clf_name, params):
    clf = None
    if clf_name == "Naive Bayes":
        clf = preprocessing.StandardScaler() | naive_bayes.GaussianNB()

    elif clf_name == "Perceptron":
        clf = preprocessing.StandardScaler() | linear_model.Perceptron()

    elif clf_name == "HoeffdingTreeClassifier":
        clf = preprocessing.StandardScaler() | tree.HoeffdingTreeClassifier(
            grace_period=params["grace_period"], max_depth=params["max_depth"]
        )
    elif clf_name == "LogisticRegression":
        if params["optimizer"] == "SGD":
            clf = preprocessing.StandardScaler() | linear_model.LogisticRegression(
                optimizer=optim.SGD(0.1), l1=params["l1"], l2=params["l2"]
            )
        elif params["optimizer"] == "FTRL":
            clf = preprocessing.StandardScaler() | linear_model.LogisticRegression(
                optimizer=optim.FTRLProximal(), l1=params["l1"], l2=params["l2"]
            )
        else:
            clf = preprocessing.StandardScaler() | linear_model.LogisticRegression(
                l1=params["l1"], l2=params["l2"]
            )
    elif clf_name == "SRPClassifier":
        clf = preprocessing.StandardScaler() | ensemble.SRPClassifier(
            n_models=params["n_model"], drift_detector=DDM()
        )

    else:
        clf = preprocessing.StandardScaler() | forest.ARFClassifier(
            n_models=params["n_models"], drift_detector=DDM()
        )

    return clf


def get_Regressor(clf_name, params):
    clf = None
    if clf_name == "HoeffdingTreeRegressor":
        clf = preprocessing.StandardScaler() | tree.HoeffdingTreeRegressor(
            grace_period=params["grace_period"]
        )
    elif clf_name == "SRPRegressor":
        clf = preprocessing.StandardScaler() | ensemble.SRPRegressor(
            n_models=params["n_models"], drift_detector=EDDM(), warning_detector=EDDM()
        )

    else:
        clf = preprocessing.StandardScaler() | forest.ARFRegressor(
            n_models=params["n_models"], drift_detector=EDDM(), warning_detector=EDDM()
        )

    return clf


def app():
    st.write("----------------")
    algor_name = st.sidebar.radio(
        "Classifier or Regressor", ("Classifier", "Regressor")
    )
    # st.title("Online machine learning :")
    if algor_name == "Classifier":
        lottie_coding3 = load_lottieurl(
            "https://assets10.lottiefiles.com/packages/lf20_byi5mrjr.json"
        )
        # st.write("Classifier")
        dataset_name = st.sidebar.selectbox(
            "Select Dataset", ("Elec2", "Phishing", "SMTP")
        )
        classifier_name = st.sidebar.selectbox(
            "Select algorithm",
            (
                "LogisticRegression",
                "Naive Bayes",
                "Perceptron",
                "HoeffdingTreeClassifier",
                "ARFClassifier",
                "SRPClassifier",
            ),
        )
        # Get dataset
        data = get_dataset(dataset_name)
        params = add_parameter_ui(classifier_name)

        # Model
        clf = get_classifier(classifier_name, params)

        left_column, right_column = st.columns([5, 1])
        with left_column:
            st.write(" Dataset :", dataset_name)
            st.write(" Algorithm:", classifier_name)

        with right_column:
            st_lottie(lottie_coding3, height=100, key="coding3")
        # Hyper parameters
        len = 0
        for x, y in data:
            len = len + 1
        k = st.sidebar.slider("number of the dataset", 500, len)
        metricss = ["Accuracy", "Precision", "Recall", "F1-score", "CohenKappa"]
        chosen_metric = st.selectbox("Choose the Target metric", metricss)
        if st.button("Run your Model"):
            if chosen_metric == "Accuracy":
                fig, aa = plot_performance(
                    data.take(k), metrics.Accuracy(), {"Metrics": clf}
                )
            elif chosen_metric == "Precision":
                fig, aa = plot_performance(
                    data.take(k), metrics.Precision(), {"Metrics": clf}
                )
            elif chosen_metric == "Recall":
                fig, aa = plot_performance(
                    data.take(k), metrics.Recall(), {"Metrics": clf}
                )
            elif chosen_metric == "CohenKappa":
                fig, aa = plot_performance(
                    data.take(k), metrics.CohenKappa(), {"Metrics": clf}
                )
            else:
                fig, aa = plot_performance(data.take(k), metrics.F1(), {"Metrics": clf})

            st.write(chosen_metric, ":", aa[0])
            # Resize image
            buf = BytesIO()
            fig.savefig(buf, format="png")
            st.image(buf)

    else:
        st.title("Regressor")
        dataset_name = st.sidebar.selectbox(
            "Select Dataset", ("ChickWeights", " TrumpApproval", "Friedman")
        )

        Regressor_name = st.sidebar.selectbox(
            "Select algorithm",
            ("HoeffdingTreeRegressor", "ARFRegressor", "SRPregressor"),
        )
        # Get dataset
        data = get_Regressor_dataset(dataset_name)
        params = add_Regressor_parameter_ui(Regressor_name)

        # Model
        clf = get_Regressor(Regressor_name, params)
        left_column, right_column = st.columns([5, 1])
        with left_column:
            st.write("### Dataset :", dataset_name)
            st.write("### Algorithm:", Regressor_name)
        if dataset_name == "Friedman":
            k = st.sidebar.slider("number of the dataset", 1000, 10000)

        else:
            len = 0
            for x, y in data:
                len = len + 1
            k = st.sidebar.slider("number of the dataset", 500, len)
            # Model
            metricss = ["MAE", "MSE", "R2", "RMSE"]
            chosen_metric = st.selectbox("Choose the Target metric", metricss)
            if st.button("Run your Model"):
                if chosen_metric == "MAE":
                    fig, aa = plot_performance(
                        data.take(k), metrics.MAE(), {"Metrics": clf}
                    )
                elif chosen_metric == "MSE":
                    fig, aa = plot_performance(
                        data.take(k), metrics.MSE(), {"Metrics": clf}
                    )
                elif chosen_metric == "R2":
                    fig, aa = plot_performance(
                        data.take(k), metrics.R2(), {"Metrics": clf}
                    )
                else:
                    fig, aa = plot_performance(
                        data.take(k), metrics.RMSE(), {"Metrics": clf}
                    )
                st.write(chosen_metric, ":", aa[0])
                # Resize image
                buf = BytesIO()
                fig.savefig(buf, format="png")
                st.image(buf)
