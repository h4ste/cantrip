import argparse

import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, fbeta_score, \
    average_precision_score

import preprocess
import run_cantrip

np.random.seed(1337)

parser = argparse.ArgumentParser(description='train and evaluate SVM on the given chronologies and vocabulary')
parser.add_argument('--chronology-path', required=True, help='path to cohort chronologies')
parser.add_argument('--vocabulary-path', required=True, help='path to cohort vocabulary')
parser.add_argument('--tdt-ratio', default='8:1:1', help='training:development:testing ratio')
parser.add_argument('--vocabulary-size', type=int, default=50000, metavar='V',
                    help='maximum vocabulary size, only the top V occurring terms will be used')
parser.add_argument('--final-only', default=False, action='store_true',
                    help='only consider the final prediction in each chronology')
parser.add_argument('--discrete-deltas', dest='delta_encoder', action='store_const',
                    const=preprocess.DiscreteDeltaEncoder(), default=preprocess.TanhLogDeltaEncoder(),
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
        'Accuracy': accuracy_score(y_true, y_pred),
        'AUROC': roc_auc_score(y_true, y_pred),
        'AUPRC': average_precision_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1': f1_score(y_true, y_pred),
        'F2': fbeta_score(y_true, y_pred, beta=2)
    }


def run_model(model, args):
    """
    Train and run SVM
    :param model: classifier to train/evaluate
    :param args: commandline arguments
    :return: nothing
    """

    # Load cohort
    cohort = preprocess.Cohort.from_disk(args.chronology_path, args.vocabulary_path, args.vocabulary_size)
    # Balance training data
    cohort = cohort.balance_chronologies()
    # Create training:development:testing split
    train, devel, test = run_cantrip.make_train_devel_test_split(cohort.patients(), args.tdt_ratio)

    # Convert chronologies into single-step binary classification examples given
    # O_i and delta_(i+1) predict pneumonia in O_(i+1)
    train = cohort[train].make_simple_classification(delta_encoder=args.delta_encoder, final_only=args.final_only)
    devel = cohort[devel].make_simple_classification(delta_encoder=args.delta_encoder, final_only=True)
    test = cohort[test].make_simple_classification(delta_encoder=args.delta_encoder, final_only=True)

    model = model.fit(train[0], train[1])

    train_eval = evaluate_classifier(train[0], train[1], model)
    devel_eval = evaluate_classifier(devel[0], devel[1], model)
    test_eval = evaluate_classifier(test[0], test[1], model)

    # run_cantrip.print_table_results(train_eval, devel_eval, test_eval, tablefmt='simple')
    run_cantrip.print_table_results(train_eval, devel_eval, test_eval, tablefmt='latex_booktabs')


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
