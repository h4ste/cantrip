import argparse

from sklearn.dummy import DummyClassifier

from src.data import encode_delta_discrete, encode_delta_continuous
from run_svm import run_model

parser = argparse.ArgumentParser(description='train and evaluate SVM on the given chronologies and vocabulary')
parser.add_argument('--chronology-path', required=True, help='path to cohort chronologies')
parser.add_argument('--vocabulary-path', required=True, help='path to cohort vocabulary')
parser.add_argument('--tdt-ratio', default='8:1:1', help='training:development:testing ratio')
parser.add_argument('--vocabulary-size', type=int, default=50000, metavar='V',
                    help='maximum vocabulary size, only the top V occurring terms will be used')
parser.add_argument('--final-only', default=False, action='store_true',
                    help='only consider the final prediction in each chronology')
parser.add_argument('--discrete-deltas', dest='delta_encoder', action='store_const',
                    const=encode_delta_discrete, default=encode_delta_continuous,
                    help='rather than encoding deltas as tanh(log(delta)), '
                         'discretize them into buckets: > 1 day, > 2 days, > 1 week, etc.'
                         '(we don\'t have enough data for this be useful)')


def main():
    args = parser.parse_args()
    for baseline in ['Stratified', 'Most Frequent', 'Prior', 'Uniform']:
        print('---%s---' % baseline)
        model = DummyClassifier(strategy=baseline.lower().replace(' ', '_'))
        run_model(model, args)


if __name__ == '__main__':
    main()
