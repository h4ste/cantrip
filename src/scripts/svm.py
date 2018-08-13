import argparse

import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from sklearn import svm

from src.data import Cohort
from src.scripts.cantrip import make_train_devel_test_split

np.random.seed(1337)

parser = argparse.ArgumentParser(description='train and evaluate SVM on the given chronologies and vocabulary')
parser.add_argument('--chronology-path', required=True, help='path to cohort chronologies')
parser.add_argument('--vocabulary-path', required=True, help='path to cohort vocabulary')
parser.add_argument('--tdt-ratio', default='8:1:1', help='training:development:testing ratio')
parser.add_argument('--vocabulary-size', type=int, default=50_000, metavar='V',
                    help='maximum vocabulary size, only the top V occurring terms will be used')
parser.add_argument('--final-only', default=False, action='store_true',
                    help='only consider the final prediction in each chronology')
parser.add_argument('--discrete-deltas', dest='delta_encoder', action='store_const',
                    help='rather than encoding deltas as tanh(log(delta)), '
                         'discretize them into buckets: > 1 day, > 2 days, > 1 week, etc.'
                         '(we don\'t have enough data for this be useful)')
parser.add_argument('--kernel', default='linear', help='SVM kernel to evaluate')


def evaluate_classifier(x, y_true, clf):
    """
    Evaluate the given classifier on x using y_true
    :param x: data to evaluate on
    :param y_true: true labels
    :param clf: classifier to evaluate
    :return: dictionary of metrics and their values
    """
    y_pred = clf.predict(x)

    return {
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'Accuracy': accuracy_score(y_true, y_pred),
        'F1': f1_score(y_true, y_pred),
        'AUROC': roc_auc_score(y_true, y_pred)
    }


def print_model_evaluation(model, train, devel, test):
    print('Training model...')
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


def leaky_rule(x, a=0.001):
    return max(x, a)


def run_model(model, args):
    """
    Train and run SVM
    :param model: classifier to train/evaluate
    :param args: commandline arguments
    :return: nothing
    """

    # Load cohort
    cohort = Cohort.from_disk(args.chronology_path, args.vocabulary_path, args.vocabulary_size)
    # Balance training data
    cohort = cohort.balance_chronologies()
    # Create training:development:testing split
    train, devel, test = make_train_devel_test_split(cohort.patients(), args.tdt_ratio)

    # Convert chronologies into single-step binary classification examples given
    # O_i and delta_(i+1) predict pnemonia in O_(i+1)
    train = cohort[train].make_simple_classification(**vars(args))
    devel = cohort[devel].make_simple_classification(**vars(args))
    test = cohort[test].make_simple_classification(**vars(args))

    print_model_evaluation(model, train, devel, test)


def main():
    """
    Main method for the script. Parses arguments and calls run_model.
    """
    args = parser.parse_args()
    model = svm.SVC(verbose=1, kernel=args.kernel)
    # noinspection PyTypeChecker
    run_model(model, args)


if __name__ == '__main__':
    main()
