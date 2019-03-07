import os
import sys
from collections import Iterable

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.python import debug as tf_debug
from tensorflow.python.platform import gfile

try:
    from tqdm import trange, tqdm
except ImportError:
    print('Package \'tqdm\' not installed. Falling back to simple progress display.')
    from mock_tqdm import trange, tqdm

import nio
import modeling
import preprocess
import optimization
import summarization
import encoding

# Facilitates lazy loading of tabulate module
tabulate = None

np.random.seed(1337)
tf.set_random_seed(1337)

flags = tf.flags

FLAGS = flags.FLAGS

# Data parameters
flags.DEFINE_string('data_dir', None,
                    help='The input data directory. Should contain the chronology files (or other data files).')

flags.DEFINE_string('vocab_file', None, help='The vocabulary file that chronologies were created with.')

# Chronology data structure parameters
flags.DEFINE_integer('max_chrono_length', default=7, lower_bound=1,
                     help='The maximum number of snapshots per chronology.')
flags.DEFINE_integer('max_snapshot_size', default=200, lower_bound=1,
                     help='The maximum number of observations to consider per snapshot.')
flags.DEFINE_integer('max_vocab_size', default=50000, lower_bound=1,
                     help='The maximum vocabulary size, only the top max_vocab_size most-frequent observations will be '
                          'used to encode clinical snapshots. Any remaining observations will be ignored.')

flags.DEFINE_boolean('use_discrete_deltas', default=False,
                     help='Rather than encoding deltas as tanh(log(delta)), they will be discretized into buckets: > '
                          '1 day, > 2 days, > 1 week, etc.')

# CANTRIP: General parameters
flags.DEFINE_float('dropout', default=0.7, lower_bound=0.,
                   help='Dropout used for all dropout layers (except vocabulary)')
flags.DEFINE_float('vocab_dropout', default=0.7, lower_bound=0.,
                   help='Dropout used for vocabulary-level dropout (supersedes --dropout)')

# CANTRIP: Clinical Snapshot Encoder parameters
flags.DEFINE_integer('observation_embedding_size', default=200, lower_bound=1,
                     help='The dimensions of observation embedding vectors.')
flags.DEFINE_integer('snapshot_embedding_size', default=200, lower_bound=1,
                     help='The dimensions of clinical snapshot encoding vectors.')
flags.DEFINE_enum('snapshot_encoder', default='DAN', enum_values=['RNN', 'CNN', 'SPARSE', 'DAN', 'DENSE'],
                  help='The type of clinical snapshot encoder to use')

# RNN
flags.DEFINE_multi_integer('snapshot_rnn_num_hidden', default=[200], lower_bound=1,
                           help='The size of hidden layer(s) used for combining clinical observations to produce the '
                                'clinical snapshot encoding; multiple arguments result in multiple hidden layers')
flags.DEFINE_enum('snapshot_rnn_cell_type', default='RAN', enum_values=['LSTM', 'GRU', 'RAN'],
                  help='The type of RNN cell to use when encoding snapshots')
flags.DEFINE_bool('snap_rnn_layer_norm', default=False,
                  help='Enable layer normalization in the RNN used for snapshot encoding.')

# CNN
flags.DEFINE_multi_integer('snapshot_cnn_windows', default=[3, 4, 5], lower_bound=1,
                           help='The length of convolution window(s) for CNN-based snapshot encoder; '
                                'multiple arguments results in multiple convolution windows.')
flags.DEFINE_integer('snapshot_cnn_kernels', default=1000, lower_bound=1,
                     help='The number of filters used in CNN')

# DAN
flags.DEFINE_multi_integer('snapshot_dan_num_hidden_avg', default=[200, 200], lower_bound=1,
                           help='The number of hidden units to use when refining the DAN average layer; '
                                'multiple arguments results in multiple dense layers.')
flags.DEFINE_multi_integer('snapshot_dan_num_hidden_obs', default=[200, 200], lower_bound=1,
                           help='The number of hidden units to use when refining clinical observation embeddings; '
                                'multiple arguments results in multiple dense layers.')

# CANTRIP: Clinical Picture Inference parameters
flags.DEFINE_multi_integer('rnn_num_hidden', default=[100], lower_bound=1,
                           help='The size of hidden layer(s) used for inferring the clinical picture; '
                                'multiple arguments result in multiple hidden layers.')
flags.DEFINE_enum('rnn_cell_type', enum_values=['RAN', 'LRAN', 'GRU', 'LSTM'], default='RAN',
                  help='The type of RNN cell to use for inferring the clinical picture.')
flags.DEFINE_boolean('rnn_layer_norm', default=True,
                     help='Whether to use layer normalization in RNN used for inferring the clinical picture.')

# Experimental setting parameters
flags.DEFINE_integer('batch_size', default=40, lower_bound=1, help='The batch size.')
flags.DEFINE_integer('num_epochs', default=30, lower_bound=1, help='The number of training epochs.')
flags.DEFINE_string('tdt_ratio', default='8:1:1', help='The training:development:testing ratio.')
flags.DEFINE_boolean('early_term', default=False, help='Stop when F2 on dev set decreases; '
                                                       'this is pretty much always a bad idea.')

# TensorFlow-specific settings
flags.DEFINE_string('output_dir', default=None,
                    help='The output directory where model checkpoints and summaries will be written.')
flags.DEFINE_boolean('clear_prev', default=False,
                     help='Whether to remove previous summary/checkpoints before starting this run.')
flags.DEFINE_string('debug', default=None,
                    help='The hostname:port of TensorBoard debug server; debugging will be enabled if this flag is '
                         'specified.')

flags.DEFINE_boolean('print_performance', default=False, help='Whether to print performance to the console.')
flags.DEFINE_boolean('print_latex_results', default=False,
                     help='Whether to print performance in a LaTeX-friendly table.')
flags.DEFINE_boolean('print_tabbed_results', default=False,
                     help='Whether to print performance in a tab-separated table.')

flags.DEFINE_enum('optimizer', enum_values=['CANTRIP', 'BERT'], default='CANTRIPq',
                  help='The type of optimizer to use when training CANTRIP.')

flags.DEFINE_float('learning_rate', default=1e-4, lower_bound=np.nextafter(np.float32(0), np.float32(1)),
                   help='The initial learning rate.')

flags.DEFINE_boolean('do_train', default=False, help='Whether to train on training data.')
flags.DEFINE_boolean('do_train_eval', default=False,
                     help='Whether to train on training data while repeatedly evaluating on development data.')
flags.DEFINE_boolean('do_eval', default=False, help='Whether to evaluate on development data.')
flags.DEFINE_boolean('do_predict', default=False, help='Whether to run predictions on test data.')


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


def run_model(model, raw_cohort, delta_encoder):
    """
    Run the given model using the given cohort and experimental settings contained in args.

    This function:
    (1) balanced the dataset
    (2) splits the cohort intro training:development:testing sets at the patient-level
    (3) trains CANTRIP and saves checkpoint/summaries for TensorBoard
    (4) evaluates CANTRIP on the development and testing set
    :param model: an instantiated CANTRIP model
    :type model: modeling.CANTRIPModel
    :param raw_cohort: the cohort to use for this experimental run
    :type raw_cohort: preprocess.Cohort
    :param delta_encoder: encoder used to represented elapsed time deltas
    :type delta_encoder: preprocess.DeltaEncoder
    :return: nothing
    """

    # Balance the cohort to have an even number of positive/negative chronologies for each patient
    cohort = raw_cohort.balance_chronologies()

    # Split into training:development:testing
    train, devel, test = make_train_devel_test_split(cohort.patients(), FLAGS.tdt_ratio)

    # Save summaries and checkpoints into the directories passed to the script
    model_summaries_dir = os.path.join(FLAGS.output_dir, 'summaries', FLAGS.optimizer, FLAGS.rnn_cell_type,
                                       FLAGS.snapshot_encoder)
    model_checkpoint_dir = os.path.join(FLAGS.output_dir, 'checkpoints', FLAGS.optimizer, FLAGS.rnn_cell_type,
                                        FLAGS.snapshot_encoder)

    # Clear any previous summaries/checkpoints if asked
    if FLAGS.clear_prev:
        nio.delete_dir_quiet(model_summaries_dir)
        nio.delete_dir_quiet(model_checkpoint_dir)
        print('Deleted previous model summaries/checkpoints')

    # Make output directories so we don't blow up when saving
    nio.make_dirs_quiet(model_checkpoint_dir)

    # Instantiate CANTRIP optimizer and summarizer classes
    if FLAGS.optimizer == 'cantrip':
        optimizer = optimization.CANTRIPOptimizer(model, learning_rate=FLAGS.learning_rate, sparse=True)
    elif FLAGS.optimizer == 'bert':
        epoch_steps = len(cohort[train].make_epoch_batches(batch_size=FLAGS.batch_size,
                                                           max_snapshot_size=FLAGS.max_snapshot_size,
                                                           max_chrono_length=FLAGS.max_chrono_length,
                                                           delta_encoder=delta_encoder))
        optimizer = optimization.BERTOptimizer(model,
                                               num_train_steps=epoch_steps * FLAGS.num_epochs,
                                               num_warmup_steps=epoch_steps * 3,
                                               init_lr=FLAGS.learning_rate)
        print('Created BERT-like optimizer with initial learning rate of %f' % FLAGS.learning_rate)
    else:
        raise NotImplementedError('No optimizer available for %s' % FLAGS.optimizer)

    # noinspection PyUnboundLocalVariable
    summarizer = summarization.CANTRIPSummarizer(model, optimizer)

    # Now that everything has been defined in TensorFlow's computation graph, initialize our model saver
    saver = tf.train.Saver(tf.global_variables())

    first_cohort = cohort

    # Tell TensorFlow to wake up and get ready to rumble
    with tf.Session() as sess:

        # If we specified a TensorBoard debug server, connect to it
        # (this is actually pretty sweet but you have to manually step through your model's flow so 99% of the time
        # you shouldn't need it)
        if FLAGS.debug is not None:
            sess = tf_debug.TensorBoardDebugWrapperSession(sess, FLAGS.debug)

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
        with trange(FLAGS.num_epochs, desc='Training') as train_log:
            # Save the training, development, and testing metrics for our best model (as measured by devel F1)
            # I'm lazy so I initialize best_devel_metrics with a zero F1 so I can compare the first iteration to it
            best_train_metrics, best_devel_metrics, best_test_metrics = {}, {'F2': 0}, {}
            # Iterate over training epochs
            for i in train_log:
                # Get global step and reset training metrics
                global_step, _ = sess.run([optimizer.global_step, summarizer.train.reset_op])
                # Log our progress on the current epoch using tqdm cohort.make_epoch_batches shuffles the order of
                # chronologies and prepares them  into mini-batches with zero-padding if needed
                total_loss = 0.
                batches = cohort[train].make_epoch_batches(batch_size=FLAGS.batch_size,
                                                           max_snapshot_size=FLAGS.max_snapshot_size,
                                                           max_chrono_length=FLAGS.max_chrono_length,
                                                           delta_encoder=delta_encoder)
                num_batches = len(batches)
                with tqdm(batches, desc='Epoch %d' % (i + 1)) as batch_log:
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

                        total_loss += batch_metrics['Loss']

                # Save epoch-level training metrics and summaries
                train_metrics, train_summary = sess.run([summarizer.train.metrics, summarizer.train.summary])
                train_metrics['Loss'] = total_loss / num_batches
                summary_writer.add_summary(train_summary, global_step=global_step)

                # Re-sample chronologies in cohort
                cohort = raw_cohort.balance_chronologies()

                # Evaluate development performance
                sess.run(summarizer.devel.reset_op)
                # Update local variables used to compute development metrics as we process each batch
                for devel_batch in first_cohort[devel].make_epoch_batches(batch_size=FLAGS.batch_size,
                                                                          max_snapshot_size=FLAGS.max_snapshot_size,
                                                                          max_chrono_length=FLAGS.max_chrono_length,
                                                                          delta_encoder=delta_encoder):
                    sess.run([summarizer.devel.metric_ops], devel_batch.feed(model, training=False))
                # Compute the development metrics
                devel_metrics, devel_summary = sess.run([summarizer.devel.metrics, summarizer.devel.summary])
                # Update training progress bar to indicate current performance on development set
                train_log.set_postfix(devel_metrics)
                # Save TensorBoard summary
                summary_writer.add_summary(devel_summary, global_step=global_step)

                def format_metrics(metrics: dict):
                    return dict((key, '%6.4f' % value) for key, value in metrics.items())

                train_log.write('Epoch %d. Train: %s | Devel: %s' % (i + 1,
                                                                     format_metrics(train_metrics),
                                                                     format_metrics(devel_metrics)))

                # Evaluate testing performance exactly as described above for development
                sess.run(summarizer.test.reset_op)
                for batch in first_cohort[test].make_epoch_batches(batch_size=FLAGS.batch_size,
                                                                   max_snapshot_size=FLAGS.max_snapshot_size,
                                                                   max_chrono_length=FLAGS.max_chrono_length,
                                                                   delta_encoder=delta_encoder):
                    sess.run([summarizer.test.metrics, summarizer.test.metric_ops], batch.feed(model, training=False))
                test_metrics, test_summary = sess.run([summarizer.test.metrics, summarizer.test.summary])
                summary_writer.add_summary(test_summary, global_step=global_step)

                # If this run did better on the dev set, save it as the new best model
                if devel_metrics['F2'] > best_devel_metrics['F2']:
                    best_devel_metrics = devel_metrics
                    best_train_metrics = train_metrics
                    best_test_metrics = test_metrics
                    # Save the model
                    saver.save(sess, model_checkpoint_dir, global_step=global_step)
                elif FLAGS.early_term:
                    tqdm.write('Early termination!')
                    break

        print('Training complete!')

        if FLAGS.print_performance:
            print('Train: %s' % str(best_train_metrics))
            print('Devel: %s' % str(best_devel_metrics))
            print('Test: %s' % str(best_test_metrics))

        if FLAGS.print_tabbed_results:
            print_table_results(best_train_metrics, best_devel_metrics, best_test_metrics, 'simple')

        if FLAGS.print_latex_results:
            print_table_results(best_train_metrics, best_devel_metrics, best_test_metrics, 'latex_booktabs')


def print_table_results(train, devel, test, tablefmt):
    """Prints results in a table to the console
    :param train: training metrics
    :type train: dict,
    :param devel: development metrics
    :type devel: dict,
    :param test: testing metrics
    :type test: dict,
    :param tablefmt: table format for use with tabular
    :type tablefmt: str,
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
            metrics = ['Accuracy', 'AUROC', 'AUPRC', 'Precision', 'Recall', 'F1', 'F2']
        measures = [dataset[metric] for metric in metrics]
        measures.insert(0, name)
        return measures

    # Create a LaTeX table using tabulate
    table = tabulate([_evaluate(train, 'train'),
                      _evaluate(devel, 'devel'),
                      _evaluate(test, 'test')],
                     headers=['Data', 'Acc.', 'AUROC', 'AUPRC', 'P', 'R', 'F1', 'F2'],
                     tablefmt=tablefmt)

    print(table)


def main(argv):
    """
    Main method for the script. Parses arguments and calls run_model.
    :param argv: commandline arguments, unused.
    """
    del argv

    # Load cohort
    cohort = preprocess.Cohort.from_disk(FLAGS.data_dir, FLAGS.vocab_file, FLAGS.max_vocab_size)

    # Compute vocabulary size (it may be smaller than args.vocabulary_size)
    vocabulary_size = len(cohort.vocabulary)

    # The embedding size is the same as the word embedding size unless using the BAG encoder
    observation_embedding_size = FLAGS.observation_embedding_size

    # Parse snapshot-encoder-specific arguments
    if FLAGS.snapshot_encoder == 'RNN':
        snapshot_encoder = encoding.rnn_encoder(FLAGS.snapshot_rnn_num_hidden)
    elif FLAGS.snapshot_encoder == 'CNN':
        snapshot_encoder = encoding.cnn_encoder(FLAGS.snapshot_cnn_windows, FLAGS.snapshot_cnn_num_kernels,
                                                FLAGS.dropout)
    elif FLAGS.snapshot_encoder == 'BAG':
        snapshot_encoder = encoding.bag_encoder
        observation_embedding_size = vocabulary_size
    elif FLAGS.snapshot_encoder == 'DENSE':
        snapshot_encoder = encoding.dense_encoder
    elif FLAGS.snapshot_encoder == 'DAN':
        snapshot_encoder = encoding.dan_encoder(FLAGS.snapshot_dan_num_hidden_obs, FLAGS.snapshot_dan_num_hidden_avg)
    else:
        raise ValueError('Given illegal snapshot encoder %s' % FLAGS.doc_encoder)

    cell_type = FLAGS.rnn_cell_type
    if FLAGS.rnn_layer_norm:
        cell_type += '-LN'

    delta_encoder = preprocess.TanhLogDeltaEncoder() if FLAGS.use_discrete_deltas else preprocess.DiscreteDeltaEncoder()

    model = modeling.CANTRIPModel(max_seq_len=FLAGS.max_chrono_length,
                                  max_snapshot_size=FLAGS.max_snapshot_size,
                                  vocabulary_size=vocabulary_size,
                                  observation_embedding_size=observation_embedding_size,
                                  num_hidden=FLAGS.rnn_num_hidden,
                                  cell_type=cell_type,
                                  batch_size=FLAGS.batch_size,
                                  snapshot_encoder=snapshot_encoder,
                                  dropout=FLAGS.dropout,
                                  num_classes=2,
                                  delta_encoding_size=delta_encoder.size)

    if FLAGS.do_train:
        run_model(model, cohort, delta_encoder)


if __name__ == '__main__':
    flags.mark_flag_as_required('data_dir')
    flags.mark_flag_as_required('vocab_file')
    flags.mark_flag_as_required('output_dir')
    tf.logging.set_verbosity(tf.logging.DEBUG)
    tf.app.run()
