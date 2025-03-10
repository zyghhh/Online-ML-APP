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


def plot_performance(step, error, r_time, memory, name):
    # To make the generated data reusable

    fig, ax = plt.subplots(figsize=(20, 10), nrows=3, dpi=300)
    ax[0].set_ylim(0.50, 1)

    ax[0].plot(step, error, label=name)
    ax[1].plot(step, r_time, label=name)
    ax[2].plot(step, memory, label=name)

    ax[0].set_ylabel("Accuracy")
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

    return fig


def WPWPE_eval(BL1, BL2, BL3, BL4, X, y, window_sizes, dis=20):
    # Record the real-time accuracy of PWPAE and 4 base learners
    metric = metrics.Accuracy()
    metric1 = Rolling(metrics.Accuracy(), window_size=window_sizes)
    metric2 = Rolling(metrics.Accuracy(), window_size=window_sizes)
    metric3 = Rolling(metrics.Accuracy(), window_size=window_sizes)
    metric4 = Rolling(metrics.Accuracy(), window_size=window_sizes)

    Acc = []
    r_time = []
    memory = []
    step = []

    i = 0
    t = []
    m = []
    m1 = []
    m2 = []
    m3 = []
    m4 = []
    yt = []
    yp = []
    num_categories = len(set(y))

    # four base learners
    scaler = preprocessing.StandardScaler()
    # four base learners
    hat1 = compose.Pipeline(scaler, BL1)
    hat2 = compose.Pipeline(scaler, BL2)
    hat3 = compose.Pipeline(scaler, BL3)
    hat4 = compose.Pipeline(scaler, BL4)

    # The four base learners learn the training set
    # Define the two feature selections methods: Variance Threshold and Select-K-Best
    selector1 = feature_selection.VarianceThreshold(threshold=0.001)
    selector2 = feature_selection.SelectKBest(similarity=stats.PearsonCorr(), k=100)
    X_train = X[:500]
    y_train = y[:500]
    # Initial feature selection on the training set
    for xi1, yi1 in stream.iter_pandas(X_train, y_train):
        selector1.learn_one(xi1)

    for xi1, yi1 in stream.iter_pandas(X_train, y_train):
        xi1 = selector1.transform_one(xi1)
        selector2.learn_one(xi1, yi1)
    #    X_train = X_train[:100]
    #    y_train = y_train[:100]
    #    X_test = X_test[:2000]
    #    y_test = y_test[:2000]

    start_time = time.time()
    # get memory usage before operation
    mb = mem_profile.memory_usage()
    # Predict the test set
    for xi, yi in stream.iter_pandas(X, y):
        xi1 = selector1.transform_one(xi1)
        xi1 = selector2.transform_one(xi1)
        y_pred1 = hat1.predict_one(xi)
        y_prob1 = hat1.predict_proba_one(xi)
        hat1.learn_one(xi, yi)

        y_pred2 = hat2.predict_one(xi)
        y_prob2 = hat2.predict_proba_one(xi)
        hat2.learn_one(xi, yi)

        y_pred3 = hat3.predict_one(xi)
        y_prob3 = hat3.predict_proba_one(xi)
        hat3.learn_one(xi, yi)

        y_pred4 = hat4.predict_one(xi)
        y_prob4 = hat4.predict_proba_one(xi)
        hat4.learn_one(xi, yi)

        # Record their real-time accuracy
        metric1 = metric1.update(yi, y_pred1)
        metric2 = metric2.update(yi, y_pred2)
        metric3 = metric3.update(yi, y_pred3)
        metric4 = metric4.update(yi, y_pred4)

        # Calculate the real-time error rates of four base learners

        e1 = 1 - metric1.get()
        e2 = 1 - metric2.get()
        e3 = 1 - metric3.get()
        e4 = 1 - metric4.get()

        ep = 0.001  # The epsilon used to avoid dividing by 0
        # Calculate the weight of each base learner by the reciprocal of its real-time error rate
        ea = 1 / (e1 + ep) + 1 / (e2 + ep) + 1 / (e3 + ep) + 1 / (e4 + ep)
        w1 = 1 / (e1 + ep) / ea
        w2 = 1 / (e2 + ep) / ea
        w3 = 1 / (e3 + ep) / ea
        w4 = 1 / (e4 + ep) / ea
        # w4 = 0

        # Make ensemble predictions by the classification probabilities

        # Calculate the final probabilities of classes 0 & 1 to make predictions
        y_prob = {}
        j = 0

        for j in range(6):
            if y_prob1.get(j, 0) == 0:
                y_prob1[j] = 0
            if y_prob2.get(j, 0) == 0:
                y_prob2[j] = 0
            if y_prob3.get(j, 0) == 0:
                y_prob3[j] = 0
            if y_prob4.get(j, 0) == 0:
                y_prob4[j] = 0

            y_prob[j] = (
                w1 * y_prob1[j] + w2 * y_prob2[j] + w3 * y_prob3[j] + w4 * y_prob4[j]
            )

        y_pred = 0
        val = y_prob[0]
        for p in range(6):
            if val < y_prob[p]:
                val = y_prob[p]
                y_pred = p

        # Update the real-time accuracy of the ensemble model
        metric = metric.update(yi, y_pred)

        if i % dis == 0:
            step.append(i)

            Acc.append(metric.get())
            r_time.append(time.time() - start_time)
            # get memory usage after operation
            mn = mem_profile.memory_usage()
            memory.append((mn[0] - mb[0]) * 2**-20)

        t.append(i)
        m.append(metric.get() * 100)
        yt.append(yi)
        yp.append(y_pred)

        i = i + 1

    return step, Acc, r_time, memory


def adaptive_learning(model, X, y, dis):
    metric = metrics.Accuracy()  # Use accuracy as the metric

    Ac = []
    r_time = []
    memory = []
    step = []
    i = 0  # count the number of evaluated data points
    t = []  # record the number of evaluated data points
    m = []  # record the real-time accuracy
    yt = []  # record all the true labels of the test set
    yp = []  # record all the predicted labels of the test set
    num_categories = len(set(y))
    # Learn the training set

    # Predict the test set
    start_time = time.time()
    # get memory usage before operation
    mb = mem_profile.memory_usage()
    for xi, yi in stream.iter_pandas(X, y):
        y_pred = model.predict_one(xi)  # Predict the test sample
        model.learn_one(xi, yi)  # Learn the test sample
        metric = metric.update(yi, y_pred)  # Update the real-time accuracy
        t.append(i)
        m.append(metric.get() * 100)
        yt.append(yi)
        yp.append(y_pred)
        if i % dis == 0:
            step.append(i)

            Ac.append(metric.get())
            r_time.append(time.time() - start_time)
            # get memory usage after operation
            mn = mem_profile.memory_usage()
            memory.append((mn[0] - mb[0]) * 2**-20)

        i = i + 1

    return step, Ac, r_time, memory


def make_model(chosen_ensemble, base_model, base_num):
    clf = None
    if chosen_ensemble == "LB":
        if base_model == "HoeffdingTreeClassifier":
            clf = ensemble.LeveragingBaggingClassifier(
                model=tree.HoeffdingTreeClassifier(), n_models=base_num
            )
        elif base_model == "ARFClassifier":
            clf = ensemble.LeveragingBaggingClassifier(
                model=forest.ARFClassifier(n_models=3), n_models=base_num
            )
        elif base_model == "SRPClassifier":
            clf = ensemble.LeveragingBaggingClassifier(
                model=ensemble.SRPClassifier(n_models=3), n_models=base_num
            )
        elif base_model == "GaussianNB":
            clf = ensemble.LeveragingBaggingClassifier(
                model=naive_bayes.GaussianNB(), n_models=base_num
            )
    if chosen_ensemble == "BOLE":
        if base_model == "HoeffdingTreeClassifier":
            clf = ensemble.BOLEClassifier(
                model=ensemble.BOLEClassifier(
                    model=drift.DriftRetrainingClassifier(
                        tree.HoeffdingTreeClassifier(), drift_detector=EDDM()
                    ),
                    n_models=base_num,
                    seed=42,
                )
            )

        elif base_model == "ARFClassifier":
            clf = ensemble.BOLEClassifier(
                model=drift.DriftRetrainingClassifier(
                    forest.ARFClassifier(n_models=3), drift_detector=EDDM()
                ),
                n_models=4,
                seed=42,
            )

        elif base_model == "SRPClassifier":
            clf = ensemble.BOLEClassifier(
                model=drift.DriftRetrainingClassifier(
                    ensemble.SRPClassifier(n_models=3), drift_detector=EDDM()
                ),
                n_models=4,
                seed=42,
            )
        elif base_model == "GaussianNB":
            clf = ensemble.LeveragingBaggingClassifier(
                model=naive_bayes.GaussianNB(), n_models=base_num
            )
    return clf


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


def plot_performance2(dataset, metric, models):
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


def app():
    st.write("----------------")
    st.title("集成学习")

    st.write("----------------")
    if os.path.exists("./dataset.csv"):
        df = pd.read_csv("dataset.csv", index_col=None)

    with st.sidebar:
        choice = st.radio("Navigation", ["Upload", "Modelling"])

    if choice == "Upload":
        st.sidebar.info(
            "This project application helps you build and explore your data."
        )
        st.title("Upload Your Dataset")
        files = st.file_uploader("Upload Your Dataset")
        if files:
            df = pd.read_csv(files, index_col=None)
            df = df.iloc[:1000]
            df.to_csv("dataset.csv", index=None)
            st.dataframe(df)

    if choice == "Modelling":
        cols = list(df.columns)
        cols.reverse()
        chosen_target = st.selectbox("Choose the Target Column", cols)
        df2 = pd.read_csv("dataset.csv")
        X = df2.drop([chosen_target], axis=1)
        y = df2[chosen_target]
        X = X[:1000]
        y = y[:1000]

        dataset = stream.iter_csv("dataset.csv", target=chosen_target)

        chosen_ensemble = st.selectbox(
            "Choose an ensemble framework： ", ("LB", "BOLE", "WPWPE", "WPWPE(personal)")
        )
        if chosen_ensemble == "LB":
            base_model = st.selectbox(
                "Choose the base model",
                (
                    "HoeffdingTreeClassifier",
                    "ARFClassifier",
                    "SRPClassifier",
                    "GaussianNB",
                ),
            )
            base_num = st.slider("number of the base learner:", 3, 30)
            dis = st.slider("distance between recorded metrics:", 20, 100)
            # metricss = ["Accuracy", "Precision", "Recall", "F1-score", "CohenKappa"]
            # chosen_metric = st.selectbox("Choose the Target metric", metricss)

            model = make_model(chosen_ensemble, base_model, base_num)

            name = "LB model"
            if st.button("Run LB"):
                step, Ac, r_time, memory = adaptive_learning(model, X, y, dis)

                fig = plot_performance(step, Ac, r_time, memory, name)
                st.write("Samples = ", step[-1] + dis)
                st.write("Accuracy = ", Ac[-1])
                st.write("r_time = ", r_time[-1])
                st.write("memory = ", memory[-1])

                buf = BytesIO()
                fig.savefig(buf, format="png")
                st.image(buf)

        if chosen_ensemble == "BOLE":
            base_model = st.selectbox(
                "Choose the base model",
                (
                    "HoeffdingTreeClassifier",
                    "ARFClassifier",
                    "SRPClassifier",
                    "GaussianNB",
                ),
            )
            base_num = st.slider("number of the base learner:", 3, 30)
            dis = st.slider("distance between recorded metrics:", 20, 100)
            model = make_model(chosen_ensemble, base_model, base_num)
            name = "BOLE model"
            # dataset = stream.iter_csv("dataset.csv", target=chosen_target)
            # metricss = ["Accuracy", "Precision", "Recall", "F1-score"]
            # chosen_metric = st.selectbox("Choose the Target metric", metricss)
            if st.button("Run BOLE"):
                step, Ac, r_time, memory = adaptive_learning(model, X, y, dis)

                fig = plot_performance(step, Ac, r_time, memory, name)
                st.write("Samples = ", step[-1] + dis)
                st.write("Accuracy = ", Ac[-1])
                st.write("r_time = ", r_time[-1])
                st.write("memory = ", memory[-1])

                buf = BytesIO()
                fig.savefig(buf, format="png")
                st.image(buf)

        if chosen_ensemble == "WPWPE(personal)":
            bl1 = st.selectbox(
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
            bl2 = st.selectbox(
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
            bl3 = st.selectbox(
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
            bl4 = st.selectbox(
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
            BL1, BL2, BL3, BL4 = make_WPWPE(bl1, bl2, bl3, bl4)
            window = st.slider("number of the sliding window:", 10, 1000)
            dis = st.slider("distance between recorded metrics:", 20, 100)

            name = "WPWPE(personal) model"
            if st.button("Run WPWPE"):
                step, Ac, r_time, memory = WPWPE_eval(
                    BL1, BL2, BL3, BL4, X, y, window, dis
                )  # Learn the model on the dataset

                fig = plot_performance(step, Ac, r_time, memory, name)
                st.write("Samples = ", step[-1] + dis)
                st.write("Accuracy = ", Ac[-1])
                st.write("r_time = ", r_time[-1])
                st.write("memory = ", memory[-1])
                # Resize image
                buf = BytesIO()
                fig.savefig(buf, format="png")
                st.image(buf)
        if chosen_ensemble == "WPWPE":
            BL1 = forest.ARFClassifier(
                n_models=3, drift_detector=HDDM_A(), warning_detector=HDDM_A()
            )
            BL2 = ensemble.SRPClassifier(
                n_models=3, drift_detector=HDDM_W(), warning_detector=HDDM_W()
            )
            BL3 = forest.ARFClassifier(
                n_models=3, drift_detector=KSWIN(), warning_detector=KSWIN()
            )
            BL4 = ensemble.SRPClassifier(
                n_models=3, drift_detector=ADWIN(), warning_detector=ADWIN()
            )
            window = st.slider("number of the sliding window:", 10, 1000)
            dis = st.slider("distance between recorded metrics:", 20, 100)

            name = "WPWPE(personal) model"
            if st.button("Run WPWPE"):
                step, Ac, r_time, memory = WPWPE_eval(
                    BL1, BL2, BL3, BL4, X, y, window, dis
                )  # Learn the model on the dataset

                st.write("Samples = ", step[-1] + dis)
                st.write("Accuracy = ", Ac[-1])
                st.write("r_time = ", r_time[-1])
                st.write("memory = ", memory[-1])

                fig = plot_performance(step, Ac, r_time, memory, name)
                # Resize image
                buf = BytesIO()
                fig.savefig(buf, format="png")
                st.image(buf)
