import streamlit as st
import numpy as np
from io import BytesIO
from streamlit_lottie import st_lottie
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import requests


from river.preprocessing import MinMaxScaler
from river.utils import Rolling
from river.tree import HoeffdingTreeClassifier
import matplotlib.pyplot as plt
import matplotlib
import time
import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

from river import forest
from river import (
    datasets,
    tree,
    evaluate,
    neighbors,
    naive_bayes,
    ensemble,
    linear_model,
    preprocessing,
    compose,
    stream,
)
from river.drift import ADWIN, KSWIN, PageHinkley
from river.drift.binary import DDM, EDDM, HDDM_A, HDDM_W
import memory_profiler as mem_profile

from river import metrics, preprocessing, compose, drift

# from river.metrics import Accuracy, F1, Recall, Precision
from river import stream, feature_selection, stats


from operator import index
import streamlit as st
import pandas as pd
import os


def make_WPWPE(bl1, bl2, bl3, bl4):
    clf1 = None
    clf2 = None
    clf3 = None
    clf4 = None

    if bl1 == "HoeffdingTreeClassifier":
        clf1 = tree.HoeffdingTreeClassifier()
    elif bl1 == "ARF-HDDM_A":
        clf1 = forest.ARFClassifier(
            n_models=3, drift_detector=HDDM_A(), warning_detector=HDDM_A()
        )
    elif bl1 == "ARF-HDDM_W":
        clf1 = forest.ARFClassifier(
            n_models=3, drift_detector=HDDM_W(), warning_detector=HDDM_W()
        )
    elif bl1 == "ARF-EDDM":
        clf1 = forest.ARFClassifier(
            n_models=3, drift_detector=EDDM(), warning_detector=EDDM()
        )
    elif bl1 == "ARF-KSWIN":
        clf1 = forest.ARFClassifier(
            n_models=3, drift_detector=KSWIN(), warning_detector=KSWIN()
        )
    elif bl1 == "ARF-ADWIN":
        clf1 = forest.ARFClassifier(
            n_models=3, drift_detector=ADWIN(), warning_detector=ADWIN()
        )
    elif bl1 == "ARF-PageHinkley":
        clf1 = forest.ARFClassifier(
            n_models=3, drift_detector=PageHinkley(), warning_detector=PageHinkley()
        )
    elif bl1 == "SRP-HDDM_A":
        clf1 = ensemble.SRPClassifier(
            n_models=3, drift_detector=HDDM_A(), warning_detector=HDDM_A()
        )
    elif bl1 == "SRP-HDDM_W":
        clf1 = ensemble.SRPClassifier(
            n_models=3, drift_detector=HDDM_W(), warning_detector=HDDM_W()
        )
    elif bl1 == "SRP-EDDM":
        clf1 = ensemble.SRPClassifier(
            n_models=3, drift_detector=EDDM(), warning_detector=EDDM()
        )
    elif bl1 == "SRP-KSWIN":
        clf1 = ensemble.SRPClassifier(
            n_models=3, drift_detector=KSWIN(), warning_detector=KSWIN()
        )
    elif bl1 == "SRP-ADWIN":
        clf1 = ensemble.SRPClassifier(
            n_models=3, drift_detector=ADWIN(), warning_detector=ADWIN()
        )
    elif bl1 == "SRP-PageHinkley":
        clf1 = ensemble.SRPClassifier(
            n_models=3, drift_detector=PageHinkley(), warning_detector=PageHinkley()
        )
    elif bl1 == "GaussianNB":
        clf1 = naive_bayes.GaussianNB()

    if bl1 == "HoeffdingTreeClassifier":
        clf1 = tree.HoeffdingTreeClassifier()
    elif bl1 == "ARF-HDDM_A":
        clf1 = forest.ARFClassifier(
            n_models=3, drift_detector=HDDM_A(), warning_detector=HDDM_A()
        )
    elif bl1 == "ARF-HDDM_W":
        clf1 = forest.ARFClassifier(
            n_models=3, drift_detector=HDDM_W(), warning_detector=HDDM_W()
        )
    elif bl1 == "ARF-EDDM":
        clf1 = forest.ARFClassifier(
            n_models=3, drift_detector=EDDM(), warning_detector=EDDM()
        )
    elif bl1 == "ARF-KSWIN":
        clf1 = forest.ARFClassifier(
            n_models=3, drift_detector=KSWIN(), warning_detector=KSWIN()
        )
    elif bl1 == "ARF-ADWIN":
        clf1 = forest.ARFClassifier(
            n_models=3, drift_detector=ADWIN(), warning_detector=ADWIN()
        )
    elif bl1 == "ARF-PageHinkley":
        clf1 = forest.ARFClassifier(
            n_models=3, drift_detector=PageHinkley(), warning_detector=PageHinkley()
        )
    elif bl1 == "SRP-HDDM_A":
        clf1 = ensemble.SRPClassifier(
            n_models=3, drift_detector=HDDM_A(), warning_detector=HDDM_A()
        )
    elif bl1 == "SRP-HDDM_W":
        clf1 = ensemble.SRPClassifier(
            n_models=3, drift_detector=HDDM_W(), warning_detector=HDDM_W()
        )
    elif bl1 == "SRP-EDDM":
        clf1 = ensemble.SRPClassifier(
            n_models=3, drift_detector=EDDM(), warning_detector=EDDM()
        )
    elif bl1 == "SRP-KSWIN":
        clf1 = ensemble.SRPClassifier(
            n_models=3, drift_detector=KSWIN(), warning_detector=KSWIN()
        )
    elif bl1 == "SRP-ADWIN":
        clf1 = ensemble.SRPClassifier(
            n_models=3, drift_detector=ADWIN(), warning_detector=ADWIN()
        )
    elif bl1 == "SRP-PageHinkley":
        clf1 = ensemble.SRPClassifier(
            n_models=3, drift_detector=PageHinkley(), warning_detector=PageHinkley()
        )
    elif bl1 == "GaussianNB":
        clf1 = naive_bayes.GaussianNB()

    if bl2 == "HoeffdingTreeClassifier":
        clf2 = tree.HoeffdingTreeClassifier()
    elif bl2 == "ARF-HDDM_A":
        clf2 = forest.ARFClassifier(
            n_models=3, drift_detector=HDDM_A(), warning_detector=HDDM_A()
        )
    elif bl2 == "ARF-HDDM_W":
        clf2 = forest.ARFClassifier(
            n_models=3, drift_detector=HDDM_W(), warning_detector=HDDM_W()
        )
    elif bl2 == "ARF-EDDM":
        clf2 = forest.ARFClassifier(
            n_models=3, drift_detector=EDDM(), warning_detector=EDDM()
        )
    elif bl2 == "ARF-KSWIN":
        clf2 = forest.ARFClassifier(
            n_models=3, drift_detector=KSWIN(), warning_detector=KSWIN()
        )
    elif bl2 == "ARF-ADWIN":
        clf2 = forest.ARFClassifier(
            n_models=3, drift_detector=ADWIN(), warning_detector=ADWIN()
        )
    elif bl2 == "ARF-PageHinkley":
        clf2 = forest.ARFClassifier(
            n_models=3, drift_detector=PageHinkley(), warning_detector=PageHinkley()
        )
    elif bl2 == "SRP-HDDM_A":
        clf2 = ensemble.SRPClassifier(
            n_models=3, drift_detector=HDDM_A(), warning_detector=HDDM_A()
        )
    elif bl2 == "SRP-HDDM_W":
        clf2 = ensemble.SRPClassifier(
            n_models=3, drift_detector=HDDM_W(), warning_detector=HDDM_W()
        )
    elif bl2 == "SRP-EDDM":
        clf2 = ensemble.SRPClassifier(
            n_models=3, drift_detector=EDDM(), warning_detector=EDDM()
        )
    elif bl2 == "SRP-KSWIN":
        clf2 = ensemble.SRPClassifier(
            n_models=3, drift_detector=KSWIN(), warning_detector=KSWIN()
        )
    elif bl2 == "SRP-ADWIN":
        clf2 = ensemble.SRPClassifier(
            n_models=3, drift_detector=ADWIN(), warning_detector=ADWIN()
        )
    elif bl2 == "SRP-PageHinkley":
        clf2 = ensemble.SRPClassifier(
            n_models=3, drift_detector=PageHinkley(), warning_detector=PageHinkley()
        )
    elif bl2 == "GaussianNB":
        clf2 = naive_bayes.GaussianNB()

    if bl3 == "HoeffdingTreeClassifier":
        clf3 = tree.HoeffdingTreeClassifier()
    elif bl3 == "ARF-HDDM_A":
        clf3 = forest.ARFClassifier(
            n_models=3, drift_detector=HDDM_A(), warning_detector=HDDM_A()
        )
    elif bl3 == "ARF-HDDM_W":
        clf3 = forest.ARFClassifier(
            n_models=3, drift_detector=HDDM_W(), warning_detector=HDDM_W()
        )
    elif bl3 == "ARF-EDDM":
        clf3 = forest.ARFClassifier(
            n_models=3, drift_detector=EDDM(), warning_detector=EDDM()
        )
    elif bl3 == "ARF-KSWIN":
        clf3 = forest.ARFClassifier(
            n_models=3, drift_detector=KSWIN(), warning_detector=KSWIN()
        )
    elif bl3 == "ARF-ADWIN":
        clf3 = forest.ARFClassifier(
            n_models=3, drift_detector=ADWIN(), warning_detector=ADWIN()
        )
    elif bl3 == "ARF-PageHinkley":
        clf3 = forest.ARFClassifier(
            n_models=3, drift_detector=PageHinkley(), warning_detector=PageHinkley()
        )
    elif bl3 == "SRP-HDDM_A":
        clf3 = ensemble.SRPClassifier(
            n_models=3, drift_detector=HDDM_A(), warning_detector=HDDM_A()
        )
    elif bl3 == "SRP-HDDM_W":
        clf3 = ensemble.SRPClassifier(
            n_models=3, drift_detector=HDDM_W(), warning_detector=HDDM_W()
        )
    elif bl3 == "SRP-EDDM":
        clf3 = ensemble.SRPClassifier(
            n_models=3, drift_detector=EDDM(), warning_detector=EDDM()
        )
    elif bl3 == "SRP-KSWIN":
        clf3 = ensemble.SRPClassifier(
            n_models=3, drift_detector=KSWIN(), warning_detector=KSWIN()
        )
    elif bl3 == "SRP-ADWIN":
        clf3 = ensemble.SRPClassifier(
            n_models=3, drift_detector=ADWIN(), warning_detector=ADWIN()
        )
    elif bl3 == "SRP-PageHinkley":
        clf3 = ensemble.SRPClassifier(
            n_models=3, drift_detector=PageHinkley(), warning_detector=PageHinkley()
        )
    elif bl3 == "GaussianNB":
        clf3 = naive_bayes.GaussianNB()

    if bl4 == "HoeffdingTreeClassifier":
        clf4 = tree.HoeffdingTreeClassifier()
    elif bl4 == "ARF-HDDM_A":
        clf4 = forest.ARFClassifier(
            n_models=3, drift_detector=HDDM_A(), warning_detector=HDDM_A()
        )
    elif bl4 == "ARF-HDDM_W":
        clf4 = forest.ARFClassifier(
            n_models=3, drift_detector=HDDM_W(), warning_detector=HDDM_W()
        )
    elif bl4 == "ARF-EDDM":
        clf4 = forest.ARFClassifier(
            n_models=3, drift_detector=EDDM(), warning_detector=EDDM()
        )
    elif bl4 == "ARF-KSWIN":
        clf4 = forest.ARFClassifier(
            n_models=3, drift_detector=KSWIN(), warning_detector=KSWIN()
        )
    elif bl4 == "ARF-ADWIN":
        clf4 = forest.ARFClassifier(
            n_models=3, drift_detector=ADWIN(), warning_detector=ADWIN()
        )
    elif bl4 == "ARF-PageHinkley":
        clf4 = forest.ARFClassifier(
            n_models=3, drift_detector=PageHinkley(), warning_detector=PageHinkley()
        )
    elif bl4 == "SRP-HDDM_A":
        clf4 = ensemble.SRPClassifier(
            n_models=3, drift_detector=HDDM_A(), warning_detector=HDDM_A()
        )
    elif bl4 == "SRP-HDDM_W":
        clf4 = ensemble.SRPClassifier(
            n_models=3, drift_detector=HDDM_W(), warning_detector=HDDM_W()
        )
    elif bl4 == "SRP-EDDM":
        clf4 = ensemble.SRPClassifier(
            n_models=3, drift_detector=EDDM(), warning_detector=EDDM()
        )
    elif bl4 == "SRP-KSWIN":
        clf4 = ensemble.SRPClassifier(
            n_models=3, drift_detector=KSWIN(), warning_detector=KSWIN()
        )
    elif bl4 == "SRP-ADWIN":
        clf4 = ensemble.SRPClassifier(
            n_models=3, drift_detector=ADWIN(), warning_detector=ADWIN()
        )
    elif bl4 == "SRP-PageHinkley":
        clf4 = ensemble.SRPClassifier(
            n_models=3, drift_detector=PageHinkley(), warning_detector=PageHinkley()
        )
    elif bl4 == "GaussianNB":
        clf4 = naive_bayes.GaussianNB()

    return clf1, clf2, clf3, clf4


def performance(dataset, metric, models):
    metric_name = metric.__class__.__name__
    aa = {}

    # To make the generated data reusable
    dataset = list(dataset)
    fig, ax = plt.subplots(figsize=(20, 10), nrows=3, dpi=300)
    ax[0].set_ylim(0, 1)
    i = 0
    for model_name, model in models.items():
        step = []
        error = []
        r_time = []
        memory = []

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

        aa[model_name] = error[-1]

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


def get_metric(mm="Accuracy"):
    if mm == "Accuracy":
        metriccc = metrics.Accuracy()
    elif mm == "CohenKappa":
        metriccc = metrics.CohenKappa()
    elif mm == "F1":
        metriccc = metrics.F1()
    elif mm == "Recall":
        metriccc = metrics.Recall()
    elif mm == "ROCAUC":
        metriccc = metrics.ROCAUC()
    else:
        metriccc = metrics.Precision()
    return metriccc


def get_dataset(name):
    data = None
    if name == "Elec2":
        data = datasets.Elec2()
    elif name == "Phishing":
        data = datasets.Phishing()

    else:
        data = datasets.SMTP()

    return data


def app():
    st.write("----------------")
    st.title("SETTINGS:")
    if os.path.exists("./dataset2.csv"):
        df = pd.read_csv("dataset2.csv", index_col=None)

    with st.sidebar:
        choice = st.radio("Navigation", ["Upload dataset", "Start experiment"])
    if choice == "Upload dataset":
        st.sidebar.info("This project application helps you build an experiment.")
        st.title("Upload Your Dataset")
        files = st.file_uploader("Upload Your Dataset")
        if files:
            df = pd.read_csv(files, index_col=None)
            df = df.iloc[:1000]
            df.to_csv("dataset2.csv", index=None)
            st.dataframe(df)
            cols = list(df.columns)
            cols.reverse()
            chosen_target = st.selectbox("Choose the Target Column", cols)
            dataset = stream.iter_csv("dataset2.csv", target=chosen_target)

    if choice == "Start experiment":
        dataset_name = st.selectbox(
            "Select Dataset", ("uploaded dataset", "Elec2", "Phishing", "SMTP")
        )
        if dataset_name == "uploaded dataset":
            cols = list(df.columns)
            cols.reverse()
            chosen_target = st.selectbox("Choose the Target Column", cols)
            dataset = stream.iter_csv("dataset2.csv", target=chosen_target)
            k = st.slider("number of the dataset", 100, len(df))

        else:
            dataset = get_dataset(dataset_name)
            l = 0
            for x, y in dataset:
                l = l + 1
            k = st.slider("number of the dataset", 100, l)
            dataset = dataset.take(k)

        mm = st.selectbox(
            "Choose  an metric:",
            (
                "Accuracy",
                "CohenKappa",
                "F1",
                "Recall",
                "ROCAUC",
                "Precision",
            ),
        )
        me = get_metric(mm)

        a1 = st.selectbox(
            "Choose  base learner 1:",
            (
                "ARF-HDDM_A",
                "ARF-HDDM_W",
                "ARF-EDDM",
                "ARF-KSWIN",
                "ARF-ADWIN",
                "ARF-PageHinkley",
                "SRP-HDDM_A",
                "SRP-HDDM_W",
                "SRP-EDDM",
                "SRP-KSWIN",
                "SRP-ADWIN",
                "SRP-PageHinkley",
                "HoeffdingTreeClassifier",
                "GaussianNB",
            ),
        )
        a2 = st.selectbox(
            "Choose  base learner 2:",
            (
                "ARF-HDDM_A",
                "ARF-HDDM_W",
                "ARF-EDDM",
                "ARF-KSWIN",
                "ARF-ADWIN",
                "ARF-PageHinkley",
                "SRP-HDDM_A",
                "SRP-HDDM_W",
                "SRP-EDDM",
                "SRP-KSWIN",
                "SRP-ADWIN",
                "SRP-PageHinkley",
                "HoeffdingTreeClassifier",
                "GaussianNB",
            ),
        )
        a3 = st.selectbox(
            "Choose  base learner 3:",
            (
                "ARF-HDDM_A",
                "ARF-HDDM_W",
                "ARF-EDDM",
                "ARF-KSWIN",
                "ARF-ADWIN",
                "ARF-PageHinkley",
                "SRP-HDDM_A",
                "SRP-HDDM_W",
                "SRP-EDDM",
                "SRP-KSWIN",
                "SRP-ADWIN",
                "SRP-PageHinkley",
                "HoeffdingTreeClassifier",
                "GaussianNB",
            ),
        )
        a4 = st.selectbox(
            "Choose  base learner 4:",
            (
                "ARF-HDDM_A",
                "ARF-HDDM_W",
                "ARF-EDDM",
                "ARF-KSWIN",
                "ARF-ADWIN",
                "ARF-PageHinkley",
                "SRP-HDDM_A",
                "SRP-HDDM_W",
                "SRP-EDDM",
                "SRP-KSWIN",
                "SRP-ADWIN",
                "SRP-PageHinkley",
                "HoeffdingTreeClassifier",
                "GaussianNB",
            ),
        )

        A1, A2, A3, A4 = make_WPWPE(a1, a2, a3, a4)

        if st.button("Run experiment:"):
            fig, aaaa = performance(dataset, me, {a1: A1, a2: A2, a3: A3, a4: A4})

            st.write(a1, "  ", mm, ":", aaaa[a1])
            st.write(a2, "  ", mm, ":", aaaa[a2])
            st.write(a3, "  ", mm, ":", aaaa[a3])
            st.write(a4, "  ", mm, ":", aaaa[a4])

            buf = BytesIO()
            fig.savefig(buf, format="png")
            st.image(buf)

    st.button("Rerun")
