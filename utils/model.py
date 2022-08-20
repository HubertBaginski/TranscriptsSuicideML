# to ensure as much reproducibility as possible we set all python and package seeds
# BERT results still vary form run to run with fixed seeds and parameters due to internal segmentation
import os
import random

import numpy as np
import tensorflow as tf
import torch
from ktrain import text
import ktrain
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix


def set_seeds(seed: int = 1):
    """
    Set all random seeds to ensure reproducibility between runs.
    :param seed:
    :return:
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def train_learner(X_train, y_train, X_val: list, y_val: list,
                  lr: float, epoch: int, seed: int, text_length: int, model_name: str, checkpoint_folder: str=None,
                  batch_size=8):
    """
    Trains a model.
    :param X_train: list of training texts
    :param y_train: list of training labels
    :param X_val: list of validation texts
    :param y_val: list of validation labels
    :param lr: learning rate
    :param epoch: number of epochs to train
    :param seed: random seed for reproducibility
    :param text_length: maxmimum length of sequences
    :param model_name: name of transformer model
    :param checkpoint_folder: path to checkpoint folder
    :return:
    """
    class_names = list(set(y_train))
    set_seeds(seed)

    t = text.Transformer(model_name, maxlen=text_length,
                         class_names=class_names)
    trn = t.preprocess_train(X_train, y_train)
    val = t.preprocess_test(X_val, y_val)
    model = t.get_classifier()
    learner = ktrain.get_learner(model, train_data=trn, val_data=val, batch_size=batch_size)
    learner.fit_onecycle(lr, epoch, checkpoint_folder=checkpoint_folder)
    return learner, t, trn, val, model


def get_performance(labels, pred, class_names, score="macro"):
    """
    Evaluates models performance by comparing predictions vs gold labels.
    :param labels: gold labels
    :param pred: predictions
    :param class_names: class names
    :param score: evaluation method
    :return:
    """
    f1 = f1_score(labels, pred, average=score)
    prec = precision_score(labels, pred, average=score)
    rec = recall_score(labels, pred, average=score)
    acc = accuracy_score(labels, pred)
    print(f"Precision: {prec} | Recall: {rec} | F1: {f1} | Accuracy: {acc}")
    print(confusion_matrix(labels, pred))
    mat = confusion_matrix(labels, pred, labels=class_names)
    return mat


def predict_test(X_test, learner, t):
    """
    Predict the labels for a list of texts.
    :param X_test: list of texts
    :param learner: trainer learner
    :param t: training preprocessor
    :return:
    """
    predictor = ktrain.get_predictor(learner.model, preproc=t)
    pred = predictor.predict(X_test)
    return (np.squeeze(pred), predictor)
