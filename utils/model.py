# to ensure as much reproducibility as possible we set all python and package seeds
# BERT results still vary form run to run with fixed seeds and parameters due to internal segmentation
import os
import random
from typing import Tuple, List

import numpy as np
import tensorflow as tf
import torch
from ktrain import text, BERTTextClassLearner
import ktrain
from ktrain.text import Transformer
from ktrain.text.predictor import TextPredictor
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix


def set_seeds(seed: int = 1):
    """
    Set all random seeds to ensure reproducibility between runs.
    Args:
        seed: random seed for reproducibility
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def train_learner(X_train: List[str],
                  y_train: List[str],
                  X_val: List[str],
                  y_val: List[str],
                  lr: float,
                  epoch: int,
                  seed: int,
                  text_length: int,
                  model_name: str,
                  checkpoint_folder: str = None,
                  batch_size=8
    ) -> Tuple[BERTTextClassLearner, Transformer, tf.data.Dataset, tf.data.Dataset]:
    """
    Trains a 'model_name' transformer model.
    Args:
        X_train: list of training texts
        y_train: list of training labels
        X_val: list of validation texts
        y_val: list of validation labels
        lr: learning rate
        epoch: maximum number of epochs to train
        seed: random seed for reproducibility
        text_length: maximum length of sequences
        model_name: name of transformer model
        checkpoint_folder: path to checkpoint folder
        batch_size: number of samples passed to training in one batch

    Returns:
        tuple of learner, transformer, training dataset, validation dataset, model
    """
    class_names = list(set(y_train))
    set_seeds(seed)
    t = text.Transformer(model_name, maxlen=text_length, class_names=class_names)
    trn = t.preprocess_train(X_train, y_train)
    val = t.preprocess_test(X_val, y_val)
    model = t.get_classifier()
    learner = ktrain.get_learner(model, train_data=trn, val_data=val, batch_size=batch_size)
    learner.fit_onecycle(lr, epoch, checkpoint_folder=checkpoint_folder)
    return learner, t, trn, val, model


def get_performance(labels, pred, class_names, score="macro") -> np.array:
    """
    Evaluates models performance by comparing predictions vs gold labels.
    Args:
        labels: gold labels
        pred: predicted labels
        class_names: unique class names
        score: evaluation method

    Returns:
        confusion matrix for gold labels and predicted labels
    """
    f1 = f1_score(labels, pred, average=score)
    prec = precision_score(labels, pred, average=score)
    rec = recall_score(labels, pred, average=score)
    acc = accuracy_score(labels, pred)
    print(f"Precision: {prec} | Recall: {rec} | F1: {f1} | Accuracy: {acc}")
    print(confusion_matrix(labels, pred))
    mat = confusion_matrix(labels, pred, labels=class_names)
    return mat

def predict_test(X_test, learner, t) -> Tuple[np.array, TextPredictor]:
    """
    Predicts labels for test set using a trained model.
    Args:
        X_test: list of texts
        learner: trainer learner
        t: training preprocessor

    Returns:
        predicted labels and predictor
    """
    predictor = ktrain.get_predictor(learner.model, preproc=t)
    pred = predictor.predict(X_test)
    return np.squeeze(pred), predictor
