import argparse

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC

from src.data import Cohort, encode_delta_continuous, encode_delta_discrete
from src.scripts.cantrip import make_train_devel_test_split, print_feature_weights

# noinspection PyProtectedMember

np.random.seed(1337)

parser = argparse.ArgumentParser(description='invoke CANTRIP')
parser.add_argument('--chronology-path', required=True, help='path to cohort chronologies')
parser.add_argument('--vocabulary-path', required=True, help='path to cohort vocabulary')
parser.add_argument('--tdt-ratio', default='8:1:1', help='training:development:testing ratio')
parser.add_argument('--vocabulary-size', type=int, default=50_000, metavar='V',
                    help='maximum vocabulary size, only the top V occurring terms will be used')
parser.add_argument('--final-only', default=False, action='store_true',
                    help='only consider the final prediction in each chronology')
parser.add_argument('--discrete-deltas', dest='delta_encoder', action='store_const',
                    const=encode_delta_discrete, default=encode_delta_continuous,
                    help='encode deltas as discrete time intervals')


def evaluate_classifier(x, y_true, model):
    y_pred = model.predict(x)

    return {
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'Accuracy': accuracy_score(y_true, y_pred),
        'F1': f1_score(y_true, y_pred),
        'AUROC': roc_auc_score(y_true, y_pred)
    }


def train_model(model, args):
    cohort = Cohort.from_chronologies(args.chronology_path, args.vocabulary_path, args.vocabulary_size)
    cohort = cohort.balance_chronologies()
    train, devel, test = make_train_devel_test_split(cohort.patients(), args.tdt_ratio)

    train = cohort[train].make_simple_classification(**vars(args))
    devel = cohort[devel].make_simple_classification(**vars(args))
    test = cohort[test].make_simple_classification(**vars(args))

    from collections import Counter

    print('Train:', Counter(train[1]))
    print('Devel:', Counter(devel[1]))
    print('Test:', Counter(test[1]))

    def evaluate_model(model):
        print('Training SVM...')
        model.fit(*train)

        def evaluate(dataset):
            metrics = evaluate_classifier(*dataset, model)
            return [metrics['Accuracy'], metrics['Precision'], metrics['Recall'], metrics['F1'], metrics['AUROC']]

        from tabulate import tabulate

        table = tabulate([['train'] + evaluate(train),
                          ['devel'] + evaluate(devel),
                          ['test'] + evaluate(test)],
                         headers=['Acc.', 'P', 'R', 'F1', 'AUC'],
                         tablefmt='latex_booktabs')

        print(table)

    print('---SVM---')
    evaluate_model(model)

    print_feature_weights(cohort.vocabulary.terms, np.square(model.coef_[0, :]))

    for baseline in ['Stratified', 'Most Frequent', 'Prior', 'Uniform']:
        print('---%s---' % baseline)
        basemodel = DummyClassifier(strategy=baseline.lower().replace(' ', '_'))
        evaluate_model(basemodel)


def main(argv):
    args = parser.parse_args(argv[1:])

    model = SVC(verbose=1, kernel='linear')

    train_model(model, args)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.DEBUG)
    tf.app.run(main)
