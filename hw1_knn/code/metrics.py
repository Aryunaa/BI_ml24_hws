import numpy as np


def binary_classification_metrics(y_pred, y_true):
    """
    Computes metrics for binary classification
    Arguments:
    y_pred, np array (num_samples) - model predictions
    y_true, np array (num_samples) - true labels
    Returns:
    precision, recall, f1, accuracy - classification metrics
    """

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score


    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))


    precision = tp / (tp + fp) if tp + fp > 0 else 0 # специфичность
    recall = tp / (tp + fn) if tp + fn > 0 else 0 # чувствительность
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    accuracy = (tp + tn) / (tp + fp + tn + fn)

    return precision, recall, f1, accuracy


def multiclass_accuracy(y_pred, y_true):
    """
    Computes metrics for multiclass classification
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true labels
    Returns:
    accuracy - ratio of accurate predictions to total samples
    """

    true_val = np.sum(y_pred == y_true )
    false_val = np.sum(y_pred != y_true )
    accuracy = true_val / (true_val + false_val) if true_val + false_val > 0 else 0

    return accuracy
    


def r_squared(y_pred, y_true):
    """
    Computes r-squared for regression
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    r2 - r-squared value
    """

    
    sum_sq_residuals = np.sum((y_true - y_pred)**2)
    mean_y = np.mean(y_true)
    sum_sq_total = np.sum((y_true - mean_y)**2)
    r_sq = 1 - (sum_sq_residuals/sum_sq_total)
    return r_sq



def mse(y_pred, y_true):
    """
    Computes mean squared error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mse - mean squared error
    """

    sum_sq_residuals = np.sum((y_true - y_pred)**2)
    meanse = sum_sq_residuals/y_true.shape[0]
    return meanse



def mae(y_pred, y_true):
    """
    Computes mean absolut error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mae - mean absolut error
    """

    abs_er = np.sum(np.abs(y_true - y_pred))
    mean_abser = abs_er/y_true.shape[0]
    return mean_abser
    