import argparse
import pathlib
from typing import Union

import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split

from utils.data import process_transcripts
from utils.model import train_learner, set_seeds

plt.rcdefaults()
from ktrain import text

config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.95))
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)


def train_bert_model(variable_code: str,
                     model_name: str = "bert-base-uncased",
                     maxlen: int = 512,
                     checkpoint_folder: Union[pathlib.Path, str] = "../checkpoints",
                     lr: float = 1.5e-5,
                     epochs: int = 1,
                     data_folder: Union[pathlib.Path, str] = "../data/new"):
    """
    Trains a 'model_name' transformer model for with the provided data and parameters.
    Args:
        variable_code: Variable code used for training, see /utils/variable_codes.py
        model_name: transformer model name
        maxlen: maximum number of tokens in one sequence
        checkpoint_folder: path to check point folder
        lr: learning rate
        epochs: maximum number of epochs for training
        data_folder: folder to training data
    """
    set_seeds(1)
    df_var = process_transcripts(variable_code, data_folder)
    X_train, X_test, y_train, y_test = train_test_split(df_var.text,
                                                        df_var[variable_code],
                                                        test_size=0.2,
                                                        random_state=1,
                                                        stratify=df_var[variable_code])

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1, stratify=y_train)

    class_names = y_train.unique()
    _ = text.Transformer(model_name,
                         maxlen=maxlen,
                         class_names=class_names)

    _ = train_learner(
        X_train.values,
        y_train.values,
        X_val.values,
        y_val.values,
        lr=lr,
        epoch=epochs,
        seed=1,
        text_length=maxlen,
        model_name=model_name,
        checkpoint_folder=checkpoint_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run BERT training.")
    parser.add_argument('--variable_code', help='Variable code which should be used for training, '
                                                'check /utils/variable_codes.py', default='MF02_01')
    parser.add_argument('--model_name', help='Which transformers model should be used for training classifier',
                        default='bert-base-uncased')
    parser.add_argument('--batch_size',
                        help='Batch size, if sequence length is large set this lower. The bigger the fast model '
                             'training is finished but more memory is required.',
                        type=int, default=8)
    parser.add_argument('--lr', help='Learning rate for training.', default=1.5e-5)
    parser.add_argument('--epochs', help='Number of epochs for training.', type=int, default=1)
    parser.add_argument('--text_length', help='Maximum number of tokens for each sequence.', type=int, default=512)
    parser.add_argument('--checkpoint_folder', help='Path to checkpoint folder, epoch weights are stored there.',
                        default="../checkpoints")
    parser.add_argument('--data_folder', help='Path to training documents.',
                        default="../data/new")
    args = parser.parse_args()
    train_bert_model(variable_code=args.variable_code, model_name=args.model_name,
                     maxlen=args.text_length, checkpoint_folder=args.checkpoint_folder,
                     lr=args.lr, epochs=args.epochs, data_folder=args.data_folder)
