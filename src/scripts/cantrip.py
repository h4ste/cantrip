import argparse
import os
import sys
from collections import Iterable, namedtuple

import numpy as np

import tensorflow as tf
from tensorflow.python import debug as tf_debug
from tensorflow.python.platform import gfile

from sklearn.model_selection import train_test_split

try:
    from tqdm import trange, tqdm
except ImportError:
    print('Package \'tqdm\' not installed. Falling back to simple progress display.')
    from src.models.mock_tqdm import trange, tqdm

from src.data import Cohort, encode_delta_discrete, encode_delta_continuous
from src.models import CANTRIPModel, CANTRIPOptimizer, CANTRIPSummarizer
from src.models.cantrip_model import CELL_TYPES
from src.models.encoder import rnn_encoder, cnn_encoder, bag_encoder, dan_encoder, dense_encoder
from src.models.util import make_dirs_quiet, delete_dir_quiet

np.random.seed(1337)

parser = argparse.ArgumentParser(description='train and evaluate CANTRIP using the given chronologies and observation '
                                             'vocabulary')

# Data parameters
parser.add_argument('--chronology-path', required=True, help='path to cohort chronologies')
parser.add_argument('--vocabulary-path', required=True, help='path to cohort vocabulary')

# Chronology data structure parameters
parser.add_argument('--max-chron-len', type=int, default=7, metavar='L',
                    help='maximum number of snapshots per chronology')
parser.add_argument('--max-snapshot-size', type=int, default=200, metavar='N',
                    help='maximum number of observations to consider per snapshot')
parser.add_argument('--vocabulary-size', type=int, default=50000, metavar='V',
                    help='maximum vocabulary size, only the top V occurring terms will be used')
parser.add_argument('--discrete-deltas', dest='delta_encoder', action='store_const',
                    const=encode_delta_discrete, default=encode_delta_continuous,
                    help='rather than encoding deltas as tanh(log(delta)), '
                         'discretize them into buckets: > 1 day, > 2 days, > 1 week, etc.'
                         '(we don\'t have enough data for this be useful)')

# CANTRIP: General parameters
parser.add_argument('--dropout', type=float, default=0.7, help='dropout used for all dropout layers')
parser.add_argument('--vocab-dropout', type=float, default=0.7,
                    help='dropout used for vocabulary-level dropout (overrides --dropout)')

# CANTRIP: Clinical Snapshot Encoder parameters
parser.add_argument('--observation-embedding-size', type=int, default=200,
                    help="dimensions of observation embedding vectors")
parser.add_argument('--snapshot-embedding-size', type=int, default=200,
                    help="dimensions of clinical snapshot encoding vectors")
parser.add_argument('--snapshot-encoder', choices=['RNN', 'CNN', 'BAG', 'DAN', 'DENSE'],
                    default='DAN', help='type of clinical snapshot encoder to use')

doc_encoder_rnn = parser.add_argument_group(title='Snapshot Encoder: RNN Flags')
doc_encoder_rnn.add_argument('--snapshot-rnn-num-hidden', type=int, nargs='+', default=[200],
                             help='size of hidden layer(s) used for combining clinical obserations to produce the '
                                  'clinical snapshot encoding; multiple arguments result in multiple hidden layers')
doc_encoder_rnn.add_argument('--snapshot-rnn-cell-type', choices=['LSTM', 'LSTM-LN', 'GRU', 'GRU-LN', 'RAN', 'RAN-LN'],
                             nargs='+', default=[200],
                             help='size of hidden layer(s) used for combining clinical observations to produce the '
                                  'clinical snapshot encoding; multiple arguments result in multiple hidden layers')

doc_encoder_cnn = parser.add_argument_group(title='Snapshot Encoder: CNN Flags')
doc_encoder_cnn.add_argument('--snapshot-cnn-windows', type=int, nargs='?', default=[3, 4, 5],
                             help='length of convolution window(s) for CNN-based snapshot encoder; '
                                  'multiple arguments results in multiple convolution windows')
doc_encoder_cnn.add_argument('--snapshot-cnn-kernels', type=int, default=1000, help='number of filters used in CNN')


doc_encoder_dan = parser.add_argument_group(title='Snapshot Encoder: DAN Flags')
doc_encoder_cnn.add_argument('--snapshot-dan-num-hidden-avg', type=int, nargs='+', default=[200, 200],
                             help='number of hidden units to use when refining the DAN average layer; '
                                  'multiple arguments results in multiple dense layers')
doc_encoder_cnn.add_argument('--snapshot-dan-num-hidden-obs', type=int, nargs='+', default=[200, 200],
                             help='number of hidden units to use when refining clinical observation embeddings; '
                                  'multiple arguments results in multiple dense layers')

# CANTRIP: Clinical Picture Inference parameters
parser.add_argument('--rnn-num-hidden', type=int, nargs='+', default=[100],
                    help='size of hidden layer(s) used for inferring the clinical picture; '
                         'multiple arguments result in multiple hidden layers')
parser.add_argument('--rnn-cell-type', choices=CELL_TYPES, default='RAN',
                    help='type of RNN cell to use for inferring the clinical picture')

# Experimental setting parameters
parser.add_argument('--batch-size', type=int, default=40, help='batch size')
parser.add_argument('--num-epochs', type=int, default=30, help='number of training epochs')
parser.add_argument('--tdt-ratio', default='8:1:1', help='training:development:testing ratio')
parser.add_argument('--early-term', default=False, action='store_true', help='stop when F1 on dev set decreases; '
                                                                             'this is pretty much always a bad idea')

# TensorFlow-specific settings
parser.add_argument('--summary-dir', default=os.path.join('data', 'working', 'summaries'))
parser.add_argument('--checkpoint-dir', default=os.path.join('models', 'checkpoints'))
parser.add_argument('--clear', default=False, action='store_true',
                    help='remove previous summary/checkpoints before starting this run')
parser.add_argument('--debug', default=None, help='hostname:port of TensorBoard debug server')

parser.add_argument('--print-performance', default=False, action='store_true')
parser.add_argument('--print-latex-results', default=False, action='store_true')
parser.add_argument('--print-tabbed-results', default=False, action='store_true')


# TODO: break the file into separate training/testing/inference phases
parser.add_argument('--mode', choices=['TRAIN'], default='TRAIN', help=argparse.SUPPRESS)


def make_train_devel_test_split(data: Iterable, ratio: str) -> (Iterable, Iterable, Iterable):
    """
    Split the given dataset into training, development, and testing sets using the given ratio
    e.g., split(data, \'8:1:1\') splits 80% as training, 10% as development, and 10% as testing
    :param data: dataset to split
    :param ratio: ratio encoded as a string, specified as train:devel:test
    :return: a triple containing the training, development, and testing sets
    """
    # Parse the splitting ratio
    train, devel, test = [int(x) for x in ratio.split(':')]

    # First split into training+development and test
    train_devel, _test = train_test_split(data, test_size=(test / (train + devel + test)))
    # Then split training+development into training and development
    _train, _devel = train_test_split(train_devel, test_size=(devel / (train + devel)))

    return _train, _devel, _test


def run_model(model: CANTRIPModel, cohort: Cohort, args):
    """
    Run the given model using the given cohort and experimental settings contained in args.

    This function:
    (1) balanced the dataset
    (2) splits the cohort intro training:development:testing sets at the patient-level
    (3) trains CANTRIP and saves checkpoint/summaries for TensorBoard
    (4) evaluates CANTRIP on the development and testing set
    :param model: an instantiated CANTRIP model
    :param cohort: the cohort to use for this experimental run
    :param args: command-line arguments specifying model/experiment parameters
    :return: nothing
    """

    # Balance the cohort to have an even number of positive/negative chronologies for each patient
    cohort = cohort.balance_chronologies()

    # Split into training:development:testing
    train, devel, test = make_train_devel_test_split(cohort.patients(), args.tdt_ratio)

    # Save summaries and checkpoints into the directories passed to the script
    model_summaries_dir = os.path.join(args.summary_dir, args.rnn_cell_type, args.snapshot_encoder)
    model_checkpoint_dir = os.path.join(args.checkpoint_dir, args.rnn_cell_type, args.snapshot_encoder)

    # Clear any previous summaries/checkpoints if asked
    if args.clear:
        delete_dir_quiet(model_summaries_dir)
        delete_dir_quiet(model_checkpoint_dir)
        print('Deleted previous model summaries/checkpoints')

    # Make output directories so we don't blow up when saving
    make_dirs_quiet(model_checkpoint_dir)

    # Instantiate CANTRIP optimizer and summarizer classes
    optimizer = CANTRIPOptimizer(model)
    summarizer = CANTRIPSummarizer(model, optimizer)

    # Now that everything has been defined in TensorFlow's computation graph, initialize our model saver
    saver = tf.train.Saver(tf.global_variables())

    # Tell TensorFlow to wake up and get ready to rumble
    with tf.Session() as sess:

        # If we specified a TensorBoard debug server, connect to it
        # (this is actually pretty sweet but you have to manually step through your model's flow so 99% of the time
        # you shouldn't need it)
        if args.debug is not None:
            sess = tf_debug.TensorBoardDebugWrapperSession(sess, args.debug)

        # Create our summary writer (used by TensorBoard)
        summary_writer = tf.summary.FileWriter(model_summaries_dir, sess.graph)

        # Restore model if it exists (and we didn't clear it), otherwise create a shiny new one
        checkpoint = tf.train.get_checkpoint_state(model_checkpoint_dir)
        if checkpoint and gfile.Exists(checkpoint.model_checkpoint_path + '.index'):
            print("Reading model parameters from '%s'...", checkpoint.model_checkpoint_path)
            saver.restore(sess, checkpoint.model_checkpoint_path)
        else:
            print("Creating model with fresh parameters...")
            sess.run(tf.global_variables_initializer())

        # Initialize local variables (these are just used for computing average metrics)
        sess.run(tf.local_variables_initializer())

        # Create a progress logger to monitor training (this is a wrapped version of range()
        with trange(args.num_epochs, desc='Training') as train_log:
            # Save the training, development, and testing metrics for our best model (as measured by devel F1)
            # I'm lazy so I initialize best_devel_metrics with a zero F1 so I can compare the first iteration to it
            best_train_metrics, best_devel_metrics, best_test_metrics = {}, {'F1': 0}, {}
            # Iterate over training epochs
            for i in train_log:
                # Get global step and reset training metrics
                global_step, _ = sess.run([optimizer.global_step, summarizer.train.reset_op])
                # Log our progress on the current epoch using tqdm cohort.make_epoch_batches shuffles the order of
                # chronologies and prepares them  into mini-batches with zero-padding if needed
                with tqdm(cohort[train].make_epoch_batches(**vars(args)), desc='Epoch %d' % (i + 1)) as batch_log:
                    # Iterate over each batch
                    for j, batch in enumerate(batch_log):
                        # We train the model by evaluating the optimizer's training op. At the same time we update the
                        # training metrics and get metrics/summaries for the current batch and request the new global
                        # step number (used by TensorBoard to coordinate metrics across different runs
                        _, batch_summary, batch_metrics, global_step = sess.run(
                            [[optimizer.train_op, summarizer.train.metric_ops],  # All fetches we aren't going to read
                             summarizer.batch_summary, summarizer.batch_metrics,
                             optimizer.global_step],
                            batch.feed(model, training=True))

                        # Update tqdm progress indicator with current training metrics on this batch
                        batch_log.set_postfix(batch_metrics)

                        # Save batch-level summaries
                        summary_writer.add_summary(batch_summary, global_step=global_step)

                # Save epoch-level training metrics and summaries
                train_metrics, train_summary = sess.run([summarizer.train.metrics, summarizer.train.summary])
                summary_writer.add_summary(train_summary, global_step=global_step)

                # Evaluate development performance
                sess.run(summarizer.devel.reset_op)
                # Update local variables used to compute development metrics as we process each batch
                for devel_batch in cohort[devel].make_epoch_batches(**vars(args)):
                    sess.run([summarizer.devel.metric_ops], devel_batch.feed(model, training=False))
                # Compute the development metrics
                devel_metrics, devel_summary = sess.run([summarizer.devel.metrics, summarizer.devel.summary])
                # Update training progress bar to indicate current performance on development set
                train_log.set_postfix(devel_metrics)
                # Save TensorBoard summary
                summary_writer.add_summary(devel_summary, global_step=global_step)

                # Evaluate testing performance exactly as described above for development
                sess.run(summarizer.test.reset_op)
                for batch in cohort[test].make_epoch_batches(**vars(args)):
                    sess.run([summarizer.test.metrics, summarizer.test.metric_ops], batch.feed(model, training=False))
                test_metrics, test_summary = sess.run([summarizer.test.metrics, summarizer.test.summary])
                summary_writer.add_summary(test_summary, global_step=global_step)

                # If this run did better on the dev set, save it as the new best model
                if devel_metrics['F1'] > best_devel_metrics['F1']:
                    best_devel_metrics = devel_metrics
                    best_train_metrics = train_metrics
                    best_test_metrics = test_metrics
                    # Save the model
                    saver.save(sess, model_checkpoint_dir, global_step=global_step)
                elif args.early_term:
                    tqdm.write('Early termination!')
                    break

        print('Training complete!')

        if args.print_performance:
            print('Train: %s' % str(best_train_metrics))
            print('Devel: %s' % str(best_devel_metrics))
            print('Test: %s' % str(best_test_metrics))

        if args.print_tabbed_results:
            print_table_results(best_train_metrics, best_devel_metrics, best_test_metrics, 'simple')

        if args.print_latex_results:
            print_table_results(best_train_metrics, best_devel_metrics, best_test_metrics, 'latex_booktabs')



# Facilitates lazy loading of tabulate module
tabulate = None

def print_table_results(train: dict, devel: dict, test: dict, tablefmt):
    """Prints results in a table to the console

    :param train: training metrics
    :param devel: development metrics
    :param test: testing metrics
    :param tablefmt: table format for use with tabular
    :return: nothing
    """

    # Lazy load tabulate
    global tabulate
    if tabulate is None:
        try:
            from tabulate import tabulate
        except ImportError:
            print('Printing latex results requires the `tabulate` package. Tabulate can be installed by running: \n'
                  '$pip install tabulate')
            sys.exit(1)

    def _evaluate(dataset: dict, name: str, metrics=None):
        """
        Fetch the given metrics from the given dataset metric dictionary in the order they were given
        :param dataset: dictionary containing metrics for a specific dataset
        :param metrics: list of metric names to fetch
        :return: list of metric values
        """
        if metrics is None:
            metrics = ['Accuracy', 'AUROC', 'Precision', 'Recall', 'F1', 'F2']
        measures = [dataset[metric] for metric in metrics]
        measures.insert(0, name)
        return measures

    # Create a LaTeX table using tabulate
    table = tabulate([_evaluate(train, 'train'),
                      _evaluate(devel, 'devel'),
                      _evaluate(test, 'test')],
                     headers=['Data', 'Acc.', 'AUC', 'P', 'R', 'F1', 'F2'],
                     tablefmt=tablefmt)

    print(table)


def main(argv):
    """
    Main method for the script. Parses arguments and calls run_model.
    :param argv: commandline arguments
    """
    args = parser.parse_args(argv[1:])

    # Load cohort
    cohort = Cohort.from_disk(args.chronology_path, args.vocabulary_path, args.vocabulary_size)

    # Compute vocabulary size (it may be smaller than args.vocabulary_size)
    vocabulary_size = len(cohort.vocabulary)

    # The embedding size is the same as the word embedding size unless using the BAG encoder
    observation_embedding_size = args.observation_embedding_size

    # Parse snapshot-encoder-specific arguments
    if args.snapshot_encoder == 'RNN':
        snapshot_encoder = rnn_encoder(args.snapshot_rnn_num_hidden)
    elif args.snapshot_encoder == 'CNN':
        snapshot_encoder = cnn_encoder(args.snapshot_cnn_windows, args.snapshot_cnn_num_kernels, args.dropout)
    elif args.snapshot_encoder == 'BAG':
        snapshot_encoder = bag_encoder
        observation_embedding_size = vocabulary_size
    elif args.snapshot_encoder == 'DENSE':
        snapshot_encoder = dense_encoder
    elif args.snapshot_encoder == 'DAN':
        snapshot_encoder = dan_encoder(args.snapshot_dan_num_hidden_obs, args.snapshot_dan_num_hidden_avg)
    else:
        raise ValueError('Given illegal snapshot encoder %s' % args.doc_encoder)

    model = CANTRIPModel(max_seq_len=args.max_chron_len,
                         max_snapshot_size=args.max_snapshot_size,
                         vocabulary_size=vocabulary_size,
                         observation_embedding_size=observation_embedding_size,
                         num_hidden=args.rnn_num_hidden,
                         cell_type=args.rnn_cell_type,
                         batch_size=args.batch_size,
                         snapshot_encoder=snapshot_encoder,
                         dropout=args.dropout,
                         num_classes=2,
                         delta_encoding_size=args.delta_encoder.size)

    if args.mode == 'TRAIN':
        run_model(model, cohort, args)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.DEBUG)
    sys.exit(main(sys.argv))
