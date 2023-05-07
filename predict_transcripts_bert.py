import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split

from utils.data import process_transcripts
from utils.model import train_learner, set_seeds, get_performance, predict_test

plt.rcdefaults()
from ktrain import text, get_learner

config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.95))
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)


def train_bert_model(variable_code, model_name, maxlen, checkpoint_folder, lr, epochs, data_folder, weights_file=None):
    set_seeds(1)
    df_var = process_transcripts(variable_code, data_folder)
    X_train, X_test, y_train, y_test = train_test_split(df_var.text,
                                                        df_var[variable_code],
                                                        test_size=0.2,
                                                        random_state=1,
                                                        stratify=df_var[variable_code])

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1, stratify=y_train)

    class_names = y_train.unique()
    t = text.Transformer(model_name,
                         maxlen=maxlen,
                         class_names=class_names)

    original_learner = train_learner(
        X_train.values, y_train.values,
        X_val.values, y_val.values,
        lr=lr, epoch=epochs, seed=1, text_length=maxlen,
        model_name=model_name, checkpoint_folder=checkpoint_folder)

    if weights_file:
        original_learner[4].load_weights(Path(checkpoint_folder) / weights_file)

        learner_reloaded = get_learner(original_learner[4], train_data=original_learner[2],
                                              val_data=original_learner[3], batch_size=2)

        model_ = learner_reloaded
        t_ = original_learner[1]

        set_seeds(1)
        ## PREDICT ON VALIDATION SET
        pred = predict_test(X_val.values, model_,
                            t=t_)
        val = t_.preprocess_test(X_val.values, y_val.values)
        model_.validate(val_data=val)
        mat = get_performance(y_val.values, pred[0], class_names)

        ## PREDICT ON TEST SET
        set_seeds(1)
        pred = predict_test(X_test.values, model_, t=t_)
        predictor = pred[1]
        test = t_.preprocess_test(X_test.values, y_test.values)
        model_.validate(val_data=test, class_names=list(class_names))
        mat = get_performance(y_test.values, pred[0], class_names)

        # saving the ktrain model to disk ~500MB
        predictor.save(Path(checkpoint_folder) / variable_code)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run BERT training.")
    parser.add_argument('--variable_code', help='Variable code which should be used for training, '
                                                'check /utils/variable_codes.py', default='MF02_01')
    parser.add_argument('--weights_file', help='Name of the weights file, if provided do eval.')
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
    args = parser.parse_args()
    train_bert_model(variable_code=args.variable_code, model_name=args.model_name,
                     maxlen=args.text_length, checkpoint_folder=args.checkpoint_folder,
                     lr=args.lr, epochs=args.epochs,weights_file=args.weights_file)
