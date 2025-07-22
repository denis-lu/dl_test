import os
import numpy as np
import datetime
from sklearn.metrics import make_scorer, f1_score, recall_score, precision_score, accuracy_score, confusion_matrix, roc_curve, auc, \
    roc_auc_score


def flat_accuracy(preds, labels):
    pred_flat = preds.flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # round to the nearest second
    elapsed_rounded = int(round(elapsed))
    # format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def cal_metrics(test_y, y_pred):
    acc_res = accuracy_score(test_y, y_pred)
    recall_res = recall_score(test_y, y_pred, average=None)
    precision_res = precision_score(test_y, y_pred, average=None)
    f1_res = f1_score(test_y, y_pred, average=None)
    weighted_precision = precision_score(test_y, y_pred, average="weighted")
    weighted_recall = recall_score(test_y, y_pred, average="weighted")
    weighted_f1 = f1_score(test_y, y_pred, average="weighted")
    return (acc_res, precision_res, recall_res, f1_res, weighted_precision, weighted_recall, weighted_f1)