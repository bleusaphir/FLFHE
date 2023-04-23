import os

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import shutil

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
import seaborn as sn
import pandas as pd


def choice_device(device):
    if torch.cuda.is_available() and device != "cpu":
        # on Windows, "cuda:0" if torch.cuda.is_available()
        device = "cuda:0"

    elif torch.backends.mps.is_available() and torch.backends.mps.is_built() and device != "cpu":
        """
        on Mac : 
        - torch.backends.mps.is_available() ensures that the current MacOS version is at least 12.3+
        - torch.backends.mps.is_built() ensures that the current current PyTorch installation was built with MPS activated.
        """
        device = "mps"

    else:
        device = "cpu"

    return device


def classes_string(name_dataset):
    if name_dataset == "cifar":
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    elif name_dataset == "animaux":
        classes = ('cat', 'dog')

    else:
        print("Warning problem : unspecified dataset")
        return ()

    return classes


def supp_ds_store(path):
    for i in os.listdir(path):
        if i == ".DS_Store":
            print("Deleting of the hidden file '.DS_Store'")
            os.remove(path + "/" + i)


def folder_len(path_init):
    return len([path_init + classe + "/" + i for classe in os.listdir(path_init) for i in os.listdir(path_init + "/" + classe)])


def move_files(path_init, path_final):
    # Move a file from rep1 to rep2
    print(f"Before : initial folder ({folder_len(path_init)}) and final folder ({folder_len(path_final)})")
    for classe in os.listdir(path_init):
        for i in os.listdir(path_init + "/" + classe):
            shutil.move(path_init + classe + "/" + i, path_final + classe + "/" + i)
    print(f"After : initial folder ({folder_len(path_init)}) et final folder({folder_len(path_final)})")


def save_matrix(y_true, y_pred, path, classes):
    # To get the confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)

    # To normalize the confusion matrix
    cf_matrix_normalized = cf_matrix / np.sum(cf_matrix) * 10

    # To round up the values in the matrix
    cf_matrix_round = np.round(cf_matrix_normalized, 2)

    # To plot the matrix
    df_cm = pd.DataFrame(cf_matrix_round, index=[i for i in classes], columns=[i for i in classes])
    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True)
    plt.xlabel("Predicted label", fontsize=13)
    plt.ylabel("True label", fontsize=13)
    plt.title("Confusion Matrix", fontsize=15)

    plt.savefig(path)


def save_roc(targets, y_proba, path, nbr_classes):
    y_true = np.zeros(shape=(len(targets), nbr_classes))  # array-like of shape (n_samples, n_classes)
    for i in range(len(targets)):
        y_true[i, targets[i]] = 1

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(nbr_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_proba.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(nbr_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(nbr_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= nbr_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
        color="navy",
        linestyle=":",
        linewidth=4,
    )

    lw = 2
    for i in range(nbr_classes):
        plt.plot(
            fpr[i],
            tpr[i],
            lw=lw,
            label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]),
        )

    plt.plot([0, 1], [0, 1], "k--", lw=lw, label='Worst case')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic (ROC) Curve OvR")  # One vs Rest
    plt.legend(loc="lower right")  # loc="best"

    plt.savefig(path)


def plot_graph(list_xplot, list_yplot, x_label, y_label, curve_labels, title, path=None):
    """
    list_xplot: abscissa list with one line per curve
    list_yplot: ordinate list with one line per curve
    curve_labels : list of curve names
    """
    lw = 2

    plt.figure()
    for i in range(len(curve_labels)):
        plt.plot(list_xplot[i], list_yplot[i], lw=lw, label=curve_labels[i])

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    if curve_labels:
        plt.legend(loc="lower right")

    if path:
        plt.savefig(path)
