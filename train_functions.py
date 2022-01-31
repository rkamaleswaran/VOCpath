# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 08:06:50 2021

@author: mehak
"""

from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn import ensemble
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import math


def get_metrics(X_test, y_test, model):
    
    probs = model.predict_proba(X_test)
    
    prob_bacteria = probs[:, 0]
    prob_fungi = probs[:,1]
    
    tpr_, fpr_, threshold = metrics.roc_curve(y_test, prob_fungi)
    precision, recall, thresholds = metrics.precision_recall_curve(y_test, prob_fungi )
    
    auc_roc = metrics.roc_auc_score(y_test, prob_fungi)
    
    cm = confusion_matrix(y_test, (prob_fungi > 0.45) * 1)

    TP = cm[1][1]
    TN = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN)
    # Specificity or true negative rate
    TNR = TN / (TN + FP)
    # Precision or positive predictive value
    PPV = TP / (TP + FP)
    # Negative predictive value
    NPV = TN / (TN + FN)
    # Fall out or false positive rate
    FPR = FP / (FP + TN)
    # False negative rate
    FNR = FN / (TP + FN)
    # False discovery rate
    FDR = FP / (TP + FP)

    print("Test Sensitivity:", TPR)
    print("Test Specificity:", TNR)
    print("Test Precision:", PPV)
    print("confusion matrix:\n", cm)
    
    acc = model.score(X_test, y_test)
    
    print("Accuracy: ", acc)
    if(math.isnan(PPV)):
        PPV = 0
    
    if(math.isnan(NPV)):
        NPV = 0
        
    result = {
        'tpr_roc':tpr_,
        'fpr_roc':fpr_,
        'precision_prc':precision,
        'recall_prc':recall,
        'auroc':auc_roc,
        'sensitivity':TPR,
        'specificity':TNR,
        'precision':PPV,
        'confusion_matrix':cm,
        'accuracy':acc,
        'npv': NPV
    }
    return result
    

def randomForest(X_up, y_up):
    


    pca = PCA()
    clf = ensemble.RandomForestClassifier() # defining decision tree classifier
    rv = Pipeline(steps=[("pca", pca), ("clf", clf)])
    
    rv.fit(X_up, y_up) # train data on new data and new target
    
    return rv

def logisticRegression(X_up, y_up):
    
    pca = PCA()
    # set the tolerance to a large value to make the example faster
    logistic = LogisticRegression(max_iter=10000, tol=0.1)
    logistic_model = Pipeline(steps=[("pca", pca), ("logistic", logistic)])
    
    X_digits = X_up
    y_digits = y_up
    
    # Parameters of pipelines can be set using ‘__’ separated parameter names:
    param_grid = {
        "pca__n_components": [5,10, 15, 20, 25, 30, 35, 45, 64],
        "logistic__C": np.logspace(-4, 4, 4),
    }
    search = GridSearchCV(logistic_model, param_grid, n_jobs=-1)
    search.fit(X_digits, y_digits)
    print("Best parameter (CV score=%0.3f):" % search.best_score_)
    print(search.best_params_)
    
    # Plot the PCA spectrum
    pca.fit(X_digits)
    
    fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True, figsize=(6, 6))
    ax0.plot(
        np.arange(1, pca.n_components_ + 1), pca.explained_variance_ratio_, "+", linewidth=2
    )
    ax0.set_ylabel("PCA explained variance ratio")
    ax0.set_xlabel("n_components")
    
    
    ax0.axvline(
        search.best_estimator_.named_steps["pca"].n_components,
        linestyle=":",
        label="n_components chosen",
    )
    ax0.legend(prop=dict(size=12))
    
    # For each number of components, find the best classifier results
    results = pd.DataFrame(search.cv_results_)
    components_col = "param_pca__n_components"
    best_clfs = results.groupby(components_col).apply(
        lambda g: g.nlargest(1, "mean_test_score")
    )
    
    best_clfs.plot(
        x=components_col, y="mean_test_score", yerr="std_test_score", legend=False, ax=ax1
    )
    ax1.set_ylabel("Classification accuracy (val)")
    ax1.set_xlabel("n_components")
    
    plt.xlim(-1, 70)
    
    plt.tight_layout()
    plt.show()
    
    logistic_model = search.best_estimator_
    
    return logistic_model

def lasso(X_up, y_up):
    
    pca = PCA()
    # set the tolerance to a large value to make the example faster
    logistic = LogisticRegression(penalty = 'l1', solver = 'saga', max_iter=10000, tol=0.1)
    pipe = Pipeline(steps=[("pca", pca), ("logistic", logistic)])
    
    X_digits = X_up
    y_digits = y_up
    
    # Parameters of pipelines can be set using ‘__’ separated parameter names:
    param_grid = {
        "pca__n_components": [5,10, 15, 20, 25, 30, 35, 45, 64],
        "logistic__C": [1,2,4,5, 10, 50,100,500,1000],
    }
    search = GridSearchCV(pipe, param_grid, n_jobs=-1)
    search.fit(X_digits, y_digits)
    print("Best parameter (CV score=%0.3f):" % search.best_score_)
    print(search.best_params_)
    
    # Plot the PCA spectrum
    pca.fit(X_digits)
    
    fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True, figsize=(6, 6))
    ax0.plot(
        np.arange(1, pca.n_components_ + 1), pca.explained_variance_ratio_, "+", linewidth=2
    )
    ax0.set_ylabel("PCA explained variance ratio")
    ax0.set_xlabel("n_components")
    
    
    ax0.axvline(
        search.best_estimator_.named_steps["pca"].n_components,
        linestyle=":",
        label="n_components chosen",
    )
    ax0.legend(prop=dict(size=12))
    
    # For each number of components, find the best classifier results
    results = pd.DataFrame(search.cv_results_)
    components_col = "param_pca__n_components"
    best_clfs = results.groupby(components_col).apply(
        lambda g: g.nlargest(1, "mean_test_score")
    )
    
    best_clfs.plot(
        x=components_col, y="mean_test_score", yerr="std_test_score", legend=False, ax=ax1
    )
    ax1.set_ylabel("Classification accuracy (val)")
    ax1.set_xlabel("n_components")
    
    plt.xlim(-1, 70)
    
    plt.tight_layout()
    plt.show()
    lasso = search.best_estimator_
    
    return lasso

def svm(X_up, y_up):


    # Define a pipeline to search for the best combination of PCA truncation
    # and classifier regularization.
    pca = PCA()
    # set the tolerance to a large value to make the example faster
    svm = SVC(kernel = 'rbf', probability = True, C = 50, gamma = 0.005)
    pipe = Pipeline(steps=[("pca", pca), ("svm", svm)])
    
    X_digits = X_up
    y_digits = y_up
    
    # Parameters of pipelines can be set using ‘__’ separated parameter names:
    param_grid = {
        "pca__n_components": [5,10, 15, 20, 25, 30, 35, 45, 64],
        "svm__C": np.logspace(-4, 4, 4),
        "svm__gamma": [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1],
        "svm__kernel": ['rbf', 'sigmoid', 'poly', 'linear']
    }
    search = GridSearchCV(pipe, param_grid, n_jobs=-1, scoring = 'f1')
    search.fit(X_digits, y_digits)
    print("Best parameter (CV score=%0.3f):" % search.best_score_)
    print(search.best_params_)
    
    # Plot the PCA spectrum
    pca.fit(X_digits)
    
    fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True, figsize=(6, 6))
    ax0.plot(
        np.arange(1, pca.n_components_ + 1), pca.explained_variance_ratio_, "+", linewidth=2
    )
    ax0.set_ylabel("PCA explained variance ratio")
    ax0.set_xlabel("n_components")
    
    
    ax0.axvline(
        search.best_estimator_.named_steps["pca"].n_components,
        linestyle=":",
        label="n_components chosen",
    )
    ax0.legend(prop=dict(size=12))
    
    # For each number of components, find the best classifier results
    results = pd.DataFrame(search.cv_results_)
    components_col = "param_pca__n_components"
    best_clfs = results.groupby(components_col).apply(
        lambda g: g.nlargest(1, "mean_test_score")
    )
    
    best_clfs.plot(
        x=components_col, y="mean_test_score", yerr="std_test_score", legend=False, ax=ax1
    )
    ax1.set_ylabel("Classification accuracy (val)")
    ax1.set_xlabel("n_components")
    
    plt.xlim(-1, 70)
    
    plt.tight_layout()
    plt.show()
    svm = search.best_estimator_
    
    return svm


def knn(X_up, y_up):



    # Define a pipeline to search for the best combination of PCA truncation
    # and classifier regularization.
    pca = PCA()
    # set the tolerance to a large value to make the example faster
    knn = KNeighborsClassifier(n_neighbors = 15)
    pipe = Pipeline(steps=[("pca", pca), ("knn", knn)])
    
    X_digits = X_up
    y_digits = y_up
    
    # Parameters of pipelines can be set using ‘__’ separated parameter names:
    param_grid = {
        "pca__n_components": [5,10, 15, 20, 25, 30, 35, 45, 64],
        "knn__n_neighbors": [5,6,7,8,9,10] 
    }
    search = GridSearchCV(pipe, param_grid, n_jobs=-1)
    search.fit(X_digits, y_digits)
    print("Best parameter (CV score=%0.3f):" % search.best_score_)
    print(search.best_params_)
    
    # Plot the PCA spectrum
    pca.fit(X_digits)
    
    fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True, figsize=(6, 6))
    ax0.plot(
        np.arange(1, pca.n_components_ + 1), pca.explained_variance_ratio_, "+", linewidth=2
    )
    ax0.set_ylabel("PCA explained variance ratio")
    ax0.set_xlabel("n_components")
    
    
    ax0.axvline(
        search.best_estimator_.named_steps["pca"].n_components,
        linestyle=":",
        label="n_components chosen",
    )
    ax0.legend(prop=dict(size=12))
    
    # For each number of components, find the best classifier results
    results = pd.DataFrame(search.cv_results_)
    components_col = "param_pca__n_components"
    best_clfs = results.groupby(components_col).apply(
        lambda g: g.nlargest(1, "mean_test_score")
    )
    
    best_clfs.plot(
        x=components_col, y="mean_test_score", yerr="std_test_score", legend=False, ax=ax1
    )
    ax1.set_ylabel("Classification accuracy (val)")
    ax1.set_xlabel("n_components")
    
    plt.xlim(-1, 70)
    
    plt.tight_layout()
    plt.show()
    
    knn  = search.best_estimator_
    
    return knn

