
import os
import pickle
import sys
import typing


import numpy as np

# Hide TensorFlow INFO messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow.compat.v1 as tf
from tensorflow.python import debug as tf_debug
from tensorflow.python.platform import gfile

try:
    from tqdm import trange, tqdm
except ImportError:
    print('Package \'tqdm\' not installed. Falling back to simple progress display.')
    from mock_tqdm import trange, tqdm

import nio
import modeling
import optimization
import summarization
import encoding
from preprocessing import Cohort, Chronology

# Facilitates lazy loading of tabulate module
tabulate = None

np.random.seed(1337)
tf.random.set_random_seed(1337)

from absl import logging, flags

FLAGS = flags.FLAGS

# Data parameters

flags.DEFINE_boolean('restore_cohorts', default=True,
                     help='Restore cohorts from .pickle files '
                          '(if changing parameters for chronologies this should be set to False)')

flags.DEFINE_string('data_dir', None,
                    help='Location of folder containing features, labels, & start times for '
                         'training, development, and testing cohorts')


flags.DEFINE_bool('embed_delta', default=False,
                  help='Embed delta vectors to same size as observation embeddings (used with --delta_combine=add)')



# Model parameters
flags.DEFINE_boolean('use_l2_reg', default=False,
                     help='Use l2 regularization')

flags.DEFINE_boolean('use_l1_reg', default=False,
                     help='Use l1 regularization')

flags.DEFINE_boolean('snapshot_l1_reg', default=False,
                     help='Use L1 regularization when encoding clinical snapshots')
flags.DEFINE_boolean('snapshot_l2_reg', default=False,
                     help='Use L2 regularization when encoding clinical snapshots')

flags.DEFINE_boolean('use_weight_decay', default=True,
                     help='Use weight decay')

flags.DEFINE_float('dropout', default=0.0, lower_bound=0.,
                   help='Dropout used for all dropout layers (except vocabulary)')
flags.DEFINE_float('observational_dropout', default=0.0, lower_bound=0.,
                   help='Dropout used for vocabulary-level dropout (supersedes --dropout)')

# CANTRIP: Clinical Snapshot Encoder parameters
flags.DEFINE_integer('observation_embedding_size', default=200, lower_bound=1,
                     help='The dimensions of observation embedding vectors.')
flags.DEFINE_integer('sinusoidal_embedding_size', default=32, lower_bound=1,
                     help='The dimensions of sinusoidal delta encoding vectors.')
flags.DEFINE_integer('snapshot_embedding_size', default=200, lower_bound=1,
                     help='The dimensions of clinical snapshot encoding vectors.')
flags.DEFINE_enum('snapshot_encoder', default='DAN', enum_values=['RNN', 'CNN', 'SPARSE', 'DAN', 'DENSE', 'RMLP', "VHN"],
                  help='The type of clinical snapshot encoder to use')

flags.DEFINE_float('augment_negatives', default=0., lower_bound=0., upper_bound=1.,
                   help='Augment negative examples by randomly truncating the given percent of positive examples to '
                        'end early')
flags.DEFINE_multi_enum('only_augmented_negatives', default=[], enum_values=["train", "devel", "test"],
                        short_name='oan',
                        help='Use ignore negative examples in the train/dev/test data, and evaluate/train on only '
                             'augmented negative examples (legacy behavior)')

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
flags.DEFINE_enum('snapshot_dan_activation', default='tanh', enum_values=['tanh', 'gelu', 'relu', 'sigmoid'],
                  help='The type of activation to use for DAN hidden layers')


flags.DEFINE_enum('snapshot_rmlp_activation', default='gelu', enum_values=['tanh', 'gelu', 'relu', 'sigmoid'],
                  help='The type of activation to use for RMLP hidden layers')
flags.DEFINE_integer('snapshot_rmlp_layers', default=5,
                     help='Number of hidden layers in snapshot RMLP.')


flags.DEFINE_enum('snapshot_vhn_activation', default='gelu', enum_values=['tanh', 'gelu', 'relu', 'sigmoid'],
                  help='The type of activation to use for VHN hidden layers')
flags.DEFINE_integer('snapshot_vhn_layers', default=10,
                     help='Number of hidden layers in snapshot VHN.')
flags.DEFINE_float('snapshot_vhn_noise', default=.5,
                     help='Strength of variational noise.')
flags.DEFINE_integer('snapshot_vhn_depth', default=6,
                     help='Depth of residual connections (i.e., number of layers) in snapshot VHN.')


# CANTRIP: Clinical Picture Inference parameters
flags.DEFINE_multi_integer('rnn_num_hidden', default=[100], lower_bound=1,
                           help='The size of hidden layer(s) used for inferring the clinical picture; '
                                'multiple arguments result in multiple hidden layers.')
flags.DEFINE_enum('rnn_cell_type', enum_values=['RAN', 'GRU', 'LSTM', 'VHRAN', 'RHN'], default='RAN',
                  help='The type of RNN cell to use for inferring the clinical picture.')
flags.DEFINE_boolean('rnn_layer_norm', default=True,
                     help='Whether to use layer normalization in RNN used for inferring the clinical picture.')
flags.DEFINE_integer('rnn_highway_depth', default=3,
                     help='Depth of residual connections in VHRAN/RHN.')
flags.DEFINE_enum('rnn_direction', enum_values=['forward', 'bidirectional'], default='bidirectional',
                  help='Direction for inferring the clinical picture with an RNN.')

# Experimental setting parameters
flags.DEFINE_integer('batch_size', default=40, lower_bound=1, help='The batch size.')
flags.DEFINE_integer('num_epochs', default=30, lower_bound=1, help='The number of training epochs.')

# TensorFlow-specific settings
flags.DEFINE_string('output_dir', default=None,
                    help='The output directory where model checkpoints and summaries will be written.')
flags.DEFINE_integer('max_to_keep', default=3,
                     help='The number of model checkpoints to save.')
flags.DEFINE_boolean('clear_prev', default=False,
                     help='Whether to remove previous summary/checkpoints before starting this run.')
flags.DEFINE_string('debug', default=None,
                    help='The hostname:port of TensorBoard debug server; debugging will be enabled if this flag is '
                         'specified.')

flags.DEFINE_enum('model', default='CANTRIP', enum_values=['CANTRIP', 'LR', 'SVM'],
                  help='model to train and/or evaluate')

flags.DEFINE_boolean('print_performance', default=False, help='Whether to print performance to the console.')
flags.DEFINE_boolean('save_latex_results', default=False,
                     help='Whether to save performance in a LaTeX-friendly table.')
flags.DEFINE_boolean('save_tabbed_results', default=False,
                     help='Whether to save performance in a tab-separated table.')

flags.DEFINE_float('learning_rate', default=1e-4, lower_bound=np.nextafter(np.float32(0), np.float32(1)),
                   help='The initial learning rate.')

flags.DEFINE_enum('correct_imbalance', default='weighted', enum_values=['none', 'weighted', 'downsample', 'upsample'],
                  help='How to correct class imbalance in the training set.')

flags.DEFINE_boolean('do_train', default=False, help='Whether to train on training data.')
flags.DEFINE_boolean('do_test', default=False, help='Whether to evaluate on test data.')
flags.DEFINE_boolean('do_predict', default=False, help='Whether to run predictions on test data.')

def print_cohort_stats(cohort: Cohort) -> None:
    from scipy import stats

    chronologies = cohort.to_list()

    snapshot_sizes = []
    for chronology in chronologies:
        for snapshot in chronology.snapshots:
            snapshot_size = len(snapshot.raw_observations)
            snapshot_sizes.append(snapshot_size)
    print('Statistics on (raw) snapshot sizes:', stats.describe(snapshot_sizes))

    days_til_onset = []
    for chronology in chronologies:
        days_til_onset.append((chronology.label.timestamp - chronology.start_time).total_seconds() / 60. / 60. / 24.)
    print('Statistics on days until event (days):', stats.describe(days_til_onset))

    days_til_first_snapshot = []
    for chronology in chronologies:
        days_til_first_snapshot.append(
            chronology.deltas_start[0].astype('timedelta64[s]').astype(np.int64) / 60. / 60. / 24.)
    print('Statistics on days from admission to first snapshot:', stats.describe(days_til_first_snapshot))

    days_til_snapshot = []
    for chronology in chronologies:
        for delta in chronology.deltas_start:
            days_til_snapshot.append(delta.astype('timedelta64[s]').astype(np.int64) / 60. / 60. / 24.)
    print('Statistics on days from admission to snapshot:', stats.describe(days_til_snapshot))

    elapsed_times = []
    for chronology in chronologies:
        for delta in chronology.deltas_prev:
            elapsed_times.append(delta.astype('timedelta64[s]').astype(np.int64) / 60. / 60. / 24.)
    print('Statistics on successive snapshot gaps (days):', stats.describe(elapsed_times))

    lengths = []
    for chronology in chronologies:
        lengths.append(len(chronology))
    print('Statistics on number of snapshots per chronology:', stats.describe(lengths))

    prediction_windows = []
    for chronology in chronologies:
        prediction_windows.append(
            (chronology.label.timestamp - chronology.snapshots[-1].timestamp).total_seconds() / 60. / 60.)
    print('Statistics on prediction window length (hours):', stats.describe(prediction_windows))

    num_positive, num_negative = 0, 0
    for chronology in chronologies:
        if chronology.class_label == 1:
            num_positive += 1
        else:
            num_negative += 1
    num_chronologies = float(len(chronologies))
    print('Number of positive chronologies:', num_positive, '%05.3f' % (num_positive / num_chronologies))
    print('Number of negative chronologies:', num_negative, '%05.3f' % (num_negative / num_chronologies))


class Experiment(object):

    def __init__(self, prediction_file='predictions.tsv'):
        self.prediction_file = prediction_file

    def run_model(self, model, train: Cohort, devel: Cohort, test: Cohort,
                  weights: typing.Union[float, int, typing.Sequence[typing.Union[float, int]]] = 1):
        pass

    def test_predict(self, model, test: Cohort):
        pass


class TfExperiment(Experiment):

    def get_model_file(self):
        regularizers = []
        if FLAGS.use_l1_reg:
            regularizers.append("l1")
        if FLAGS.use_l2_reg:
            regularizers.append("l2")
        if FLAGS.use_weight_decay:
            regularizers.append("wd")

        if not regularizers:
            regularizer = "none"
        else:
            regularizer = ":".join(regularizers)

        model_file = 'fl=%d_cb=%s_an=%.2f_ln=%d_delta=%s:%s:%s_d=%.2f_vd=%.2f_r=%s_lr=%g_bs=%d' % (
            1 if FLAGS.use_focal_loss else 0,
            FLAGS.correct_imbalance,
            FLAGS.augment_negatives,
            1 if FLAGS.rnn_layer_norm else 0,
            FLAGS.time_repr,
            FLAGS.delta_enc,
            FLAGS.delta_combine,
            FLAGS.dropout,
            FLAGS.observational_dropout,
            regularizer,
            FLAGS.learning_rate,
            FLAGS.batch_size,
        )

        model_summaries_dir = os.path.join(FLAGS.output_dir, FLAGS.rnn_cell_type,
                                           FLAGS.snapshot_encoder, model_file)
        model_checkpoint_dir = os.path.join(FLAGS.output_dir, FLAGS.rnn_cell_type,
                                            FLAGS.snapshot_encoder, model_file, 'checkpoints', 'cantrip')
        return model_summaries_dir, model_checkpoint_dir

    def run_model(self,
                  model: modeling.CANTRIPModel,
                  train: Cohort,
                  devel: Cohort,
                  test: Cohort,
                  weights: typing.Union[float, int, typing.Sequence[typing.Union[float, int]]] = 1):
        """
        Run the given model using the given cohort and experimental settings contained in args.

        This function:
        (1) balanced the dataset
        (2) splits the cohort intro training:development:testing sets at the patient-level
        (3) trains CANTRIP and saves checkpoint/summaries for TensorBoard
        (4) evaluates CANTRIP on the development and testing set
        :param model: an instantiated CANTRIP model
        :param train: the cohort to use for training this experimental run
        :param devel: the cohort to use for validating this experimental run
        :param test: the cohort to use for testing this experimental run
        :param weights: sample weights
        :return: nothing
        """
        # Save summaries and checkpoints into the directories passed to the script
        model_summaries_dir, model_checkpoint_path = self.get_model_file()

        # Clear any previous summaries/checkpoints if asked
        if FLAGS.clear_prev:
            nio.delete_dir_quiet(model_summaries_dir)
            nio.delete_dir_quiet(os.path.dirname(model_checkpoint_path))
            print('Deleted previous model summaries/checkpoints')

        # Make output directories so we don't blow up when saving
        nio.make_dirs_quiet(os.path.dirname(model_checkpoint_path))

        devel_batches = devel.batched(batch_size=FLAGS.batch_size, permute=False)
        test_batches = test.batched(batch_size=FLAGS.batch_size, permute=False)

        epoch_steps = len(train.to_list()) // FLAGS.batch_size

        optimizer = optimization.BERTOptimizer(model,
                                               lr_decay=True,
                                               l1_reg=FLAGS.use_l1_reg,
                                               l2_reg=FLAGS.use_l2_reg,
                                               num_train_steps=epoch_steps * 10,
                                               steps_per_epoch=epoch_steps,
                                               num_warmup_steps=epoch_steps * min(3, FLAGS.num_epochs - 1),
                                               init_lr=FLAGS.learning_rate,
                                               weights=weights,
                                               normalize_weights=FLAGS.use_focal_loss,
                                               focal_loss=FLAGS.use_focal_loss)

        summarizer = summarization.CANTRIPSummarizer(model, optimizer)

        # Now that everything has been defined in TensorFlow's computation graph, initialize our model saver
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.max_to_keep)

        batch_width = int(np.log10(FLAGS.batch_size)) + 1
        count_format = '%0' + str(batch_width) + 'd'
        score_format = '%5.3f'

        # noinspection PyCompatibility
        metric_format = {
            'TP': count_format,
            'TN': count_format,
            'FP': count_format,
            'FN': count_format,
            'Precision': score_format,
            'Recall': score_format,
            'Accuracy': score_format,
            'Specificity': score_format,
            'DOR': '%5.1f',
            'F1': score_format,
            'F2': score_format,
            'F.5': score_format,
            'AUROC': score_format,
            'AUPRC': score_format,
            'Loss': score_format,
            'MCC': score_format,
        }

        log_metrics = {"Accuracy": "Acc",
                       "AUROC": "AUROC",
                       "AUPRC": "AUPRC",
                       "Precision": "Prec",
                       "Recall": "Sens",
                       "Specificity": "Spec",
                       "DOR": "OR",
                       "F1": "F1",
                       "MCC": "MCC",
                       "Loss": "Loss"
                       }

        def format_(results, metrics=None):
            if not metrics:
                metrics = {k: k for k in metric_format.keys()}
            return {metrics[metric]: (metric_format[metric] % value)
                    for metric, value in results.items() if metric in metrics}

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
            checkpoint = tf.train.get_checkpoint_state(model_checkpoint_path)
            if checkpoint and gfile.Exists(checkpoint.model_checkpoint_path + '.index'):
                print("Reading model parameters from '%s'...", checkpoint.model_checkpoint_path)
                saver.restore(sess, checkpoint.model_checkpoint_path)
            else:
                print("Creating model with fresh parameters...")
                sess.run(tf.global_variables_initializer())

            # Initialize local variables (these are just used for computing average metrics)
            sess.run(tf.local_variables_initializer())

            # Create a progress logger to monitor training (this is a wrapped version of range()
            epoch_width = int(np.log10(FLAGS.num_epochs)) + 1
            with trange(FLAGS.num_epochs, desc='Training') as train_log:
                # Save the training, development, and testing metrics for our best model (as measured by devel F1)
                # I'm lazy so I initialize best_devel_metrics with a zero F1 so I can compare the first iteration to it
                best_train_metrics, best_devel_metrics = {}, {'MCC': 0}

                # Iterate over training epochs
                for i in train_log:
                    # Get global step and reset training metrics
                    global_step, _ = sess.run([optimizer.global_step, summarizer.train.reset_op])
                    total_loss = 0.

                    if FLAGS.correct_imbalance == "downsample" or FLAGS.correct_imbalance == "upsample":
                        train_ = train.balance_classes(method=FLAGS.correct_imbalance)
                    else:
                        train_ = train

                    batches = train_.batched(batch_size=FLAGS.batch_size)
                    num_batches = len(batches)
                    with tqdm(batches, desc=('Epoch %0' + str(epoch_width) + 'd') % (i + 1)) as batch_log:
                        # Iterate over each batch
                        for j, batch in enumerate(batch_log):
                            # We train the model by evaluating the optimizer's training op. At the same time we update
                            # the training metrics and get metrics/summaries for the current batch and request the new
                            # global step number (used by TensorBoard to coordinate metrics across different runs
                            _, batch_summary, batch_metrics, global_step = sess.run(
                                [[optimizer.train_op, summarizer.train.metric_ops],
                                 # All fetches we aren't going to read
                                 summarizer.batch_summary, summarizer.batch_metrics,
                                 optimizer.global_step],
                                batch.feed(model, training=True))

                            # Update tqdm progress indicator with current training metrics on this batch
                            batch_log.set_postfix(format_(batch_metrics))

                            # Save batch-level summaries
                            summary_writer.add_summary(batch_summary, global_step=global_step)

                            total_loss += batch_metrics['Loss']

                    # Save epoch-level training metrics and summaries
                    train_metrics, train_summary = sess.run([summarizer.train.metrics, summarizer.train.summary])
                    train_metrics['Loss'] = total_loss / num_batches
                    summary_writer.add_summary(train_summary, global_step=global_step)

                    # Evaluate development performance
                    sess.run(summarizer.devel.reset_op)
                    # Update local variables used to compute development metrics as we process each batch
                    for devel_batch in devel_batches:
                        sess.run([summarizer.devel.metric_ops], devel_batch.feed(model, training=False))
                    # Compute the development metrics
                    devel_metrics, devel_summary = sess.run([summarizer.devel.metrics, summarizer.devel.summary])
                    # Update training progress bar to indicate current performance on development set
                    train_log.set_postfix(format_(devel_metrics))
                    # Save TensorBoard summary
                    summary_writer.add_summary(devel_summary, global_step=global_step)

                    # def format_metrics(metrics: dict):
                    #     return dict((key, '%6.4f' % value) for key, value in metrics.items())
                    train_log.write(
                        ('Epoch %0' + str(epoch_width) + 'd. Train: %s | Devel: %s') %
                        (i + 1,
                         "; ".join("{}: {}".format(k, v) for k, v in format_(train_metrics, log_metrics).items()),
                         "; ".join("{}: {}".format(k, v) for k, v in format_(devel_metrics, log_metrics).items()))
                    )

                    sess.run(summarizer.test.reset_op)
                    for batch in test_batches:
                        sess.run([summarizer.test.metrics, summarizer.test.metric_ops],
                                 batch.feed(model, training=False))
                    test_metrics, test_summary = sess.run([summarizer.test.metrics, summarizer.test.summary])
                    summary_writer.add_summary(test_summary, global_step=global_step)

                    # If this run did better on the dev set, save it as the new best model
                    if devel_metrics['MCC'] > best_devel_metrics['MCC']:
                        best_devel_metrics = devel_metrics
                        best_train_metrics = train_metrics
                        best_test_metrics = test_metrics

                        # Save the model
                        saver.save(sess, model_checkpoint_path, global_step=global_step)
            print('Training complete!')
            return model, best_train_metrics, best_devel_metrics, best_test_metrics

    def test_predict(self, model, test):
        model_summaries_dir, model_checkpoint_path = self.get_model_file()
        model_checkpoint_dir = os.path.dirname(model_checkpoint_path)

        summarizer = summarization.CANTRIPSummarizer(model, None)

        test_batches = test.batched(batch_size=FLAGS.batch_size, permute=False)

        # Now that everything has been defined in TensorFlow's computation graph, initialize our model saver
        saver = tf.train.Saver(tf.global_variables())

        # Tell TensorFlow to wake up and get ready to rumble
        with tf.Session() as sess:

            print('Restoring model with highest development MCC...')
            checkpoint = tf.train.get_checkpoint_state(model_checkpoint_dir)
            print('Looking for checkpoint', checkpoint, 'at', model_checkpoint_dir)
            if checkpoint and gfile.Exists(checkpoint.model_checkpoint_path + '.index'):
                print("Reading model parameters from '%s'...", checkpoint.model_checkpoint_path)
                saver.restore(sess, checkpoint.model_checkpoint_path)
            else:
                raise ValueError('No checkpoint found for model %s' % model_checkpoint_dir)
            print('Evaluating test set performance...')
            sess.run(summarizer.test.reset_op)
            header = ['subject_id', 'hadm_id', 'y_true', 'y_pred', 'logits']
            results = []
            for batch in test_batches:
                batch_predictions, batch_logits, _, _ = sess.run(
                    [model.y, model.logits, summarizer.test.metrics, summarizer.test.metric_ops],
                    batch.feed(model, training=False)
                )
                for c, y_true, y_pred, logits in zip(batch.chronologies, batch.labels, batch_predictions, batch_logits):
                    if c:
                        logits = ' '.join(['%g' % logit for logit in logits])
                        if np.argmax(logits) != y_pred:
                            logging.log_first_n(logging.DEBUG,
                                                'Predict class %d did not match logits %s', 30, y_pred, logits)
                        results.append([c.hadm.patient_id, c.hadm.hadm_id, y_true, y_pred, logits])

            if FLAGS.do_predict:
                import csv
                with open(os.path.join(model_summaries_dir, self.prediction_file), 'w', newline='') as csv_file:
                    writer = csv.writer(csv_file, delimiter='\t')
                    writer.writerow(header)
                    for result in results:
                        writer.writerow(result)

            if FLAGS.do_test:
                test_metrics = sess.run(summarizer.test.metrics)
                return test_metrics


class SkLearnExperiment(Experiment):

    @classmethod
    def evaluate(cls, y_pred, y_score, y_true):
        from sklearn import metrics
        acc = metrics.accuracy_score(y_true=y_true, y_pred=y_pred)
        auroc = metrics.roc_auc_score(y_true=y_true, y_score=y_score)
        auprc = metrics.average_precision_score(y_true=y_true, y_score=y_score)
        p, r, f1, _ = metrics.precision_recall_fscore_support(y_true=y_true, y_pred=y_pred, average='binary')
        tn, fp, fn, tp = metrics.confusion_matrix(y_true=y_true, y_pred=y_pred).ravel()

        f2 = metrics.fbeta_score(y_true=y_true, y_pred=y_pred, beta=2)
        fhalf = metrics.fbeta_score(y_true=y_true, y_pred=y_pred, beta=.5)

        specificity = tn / (tn + fp)
        diagnostic_odds_ratio = (tp * tn) / (fp * fn)
        mcc = metrics.matthews_corrcoef(y_true=y_true, y_pred=y_pred)

        return {
            'Accuracy': acc,
            'AUROC': auroc,
            'AUPRC': auprc,
            'Precision': p,
            'Recall': r,
            'Specificity': specificity,
            'DOR': diagnostic_odds_ratio,
            'F1': f1,
            'F2': f2,
            'F.5': fhalf,
            'MCC': mcc,
            'TP': tp,
            'FP': fp,
            'FN': fn,
            'TN': tn
        }

    # noinspection PyPep8Naming
    def run_model(self, model, train: Cohort, devel: Cohort, test: Cohort,
                  weights: typing.Union[float, int, typing.Sequence[typing.Union[float, int]]] = 1):
        X_train, y_train, _ = train.make_classification()
        print('Created X:', X_train.shape, '& y:', y_train.shape)

        if weights != 1:
            if isinstance(weights, typing.Sequence):
                sample_weights = [weights[y] for y in y_train]
            else:
                raise ValueError
        else:
            sample_weights = None

        model = model.fit(X_train, y_train, sample_weight=sample_weights)

        y_pred = model.predict(X_train)
        y_score = model.predict_proba(X_train)[:, 1]
        train_metrics = SkLearnExperiment.evaluate(y_pred=y_pred, y_score=y_score, y_true=y_train)

        X_devel, y_devel, _ = devel.make_classification()
        y_pred = model.predict(X_devel)
        y_score = model.predict_proba(X_devel)[:, 1]
        devel_metrics = SkLearnExperiment.evaluate(y_pred=y_pred, y_score=y_score, y_true=y_devel)

        return model, train_metrics, devel_metrics

    # noinspection PyPep8Naming
    def test_predict(self, model, test: Cohort):
        X_test, y_test, meta_test = test.make_classification()

        y_pred = model.predict(X_test)
        y_score = model.predict_proba(X_test)

        nio.make_dirs_quiet(FLAGS.output_dir)

        if FLAGS.do_predict:
            import csv
            header = ['subject_id', 'hadm_id', 'y_true', 'y_pred', 'logits']
            with open(os.path.join(FLAGS.output_dir, self.prediction_file), 'w', newline='') as csv_file:
                writer = csv.writer(csv_file, delimiter='\t')
                writer.writerow(header)
                for meta, y_true, y_pred_, y_score_ in zip(meta_test, y_test, y_pred, y_score):
                    writer.writerow(list(meta) + [y_true, y_pred_, ' '.join(['%g' % logit for logit in y_score_])])

        if FLAGS.do_test:
            print('y_test:', y_test.shape, 'y_pred', y_pred.shape, 'y_score', y_score.shape)
            return SkLearnExperiment.evaluate(y_pred=y_pred, y_score=y_score[:, 1], y_true=y_test)


def print_table_results(results: typing.Sequence[typing.Tuple[typing.Dict[str, float], str]],
                        tablefmt, file: typing.TextIO = sys.stdout):
    """Prints results in a table to the console
    :param results: list of (metric dict, name)
    :param tablefmt: table format for use with tabular
    :type tablefmt: str,
    :param file: location to print results to
    :return: nothing
    """

    # Lazy load tabulate
    global tabulate
    if tabulate is None:
        try:
            from tabulate import tabulate
        except ImportError:
            print('Printing table results requires the `tabulate` package. Tabulate can be installed by running:\n'
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
            metrics = ['Accuracy', 'AUROC', 'Recall', 'Specificity', 'DOR', 'AUPRC', 'Precision', 'F1', 'F2', 'F.5',
                       'MCC', 'TP', 'FP', 'FN', 'TN']
        measures = [dataset[metric] for metric in metrics]
        measures.insert(0, name)
        return measures

    # Create a LaTeX table using tabulate
    table = tabulate([_evaluate(data, name) for data, name in results],
                     headers=['Data', 'Acc.', 'AUROC', 'Sens.', 'Spec.', 'DOR', 'AUPRC', 'Prec.', 'F1', 'F2', 'F.5',
                              'MCC', 'TP', 'FP', 'FN', 'TN'],
                     tablefmt=tablefmt)
    print(table, file=file)


def main(argv):
    """
    Main method for the script. Parses arguments and calls run_model.
    :param argv: commandline arguments, unused.
    """
    del argv

    def load_cohort(name: str, directory: str = FLAGS.data_dir, vocab=None) -> Cohort:
        pickle_filename = os.path.join(directory, name + '.pickle')
        if FLAGS.restore_cohorts and os.path.isfile(pickle_filename):
            try:
                with open(pickle_filename, 'rb') as pickle_file:
                    cohort = pickle.load(pickle_file)  # type: Cohort
                print('Restored cohort with %d patients and %d chronologies from %s' %
                      (len(cohort.admissions), len(cohort.chronologies), pickle_filename))
                load_from_files = False
            except (pickle.UnpicklingError, TypeError, AttributeError) as error:
                print(error)
                load_from_files = True
        else:
            load_from_files = True

        if load_from_files:
            cohort = Cohort.from_csv_files(
                feature_csv=os.path.join(directory, name + '.chronologies.csv'),
                admission_csv=os.path.join(directory, name + '.admittimes.csv'),
                label_csv=os.path.join(directory, name + '.labels.csv'),
                vocab=vocab,
                lock_vocab=vocab is not None
            )
            print('Saving %s' % pickle_filename)
            with open(pickle_filename, 'wb') as pickle_file:
                pickle.dump(cohort, pickle_file, protocol=-1)

        # noinspection PyUnboundLocalVariable
        return cohort

    print('Loading training cohort & building vocabulary...')
    train = load_cohort('train')
    vocabulary = train.vocabulary

    print()
    print_cohort_stats(train)

    if FLAGS.do_train:
        print('\nLoading development/validation cohort...')
        devel = load_cohort('devel', vocab=vocabulary)
    else:
        devel = None

    print('\nLoading test/evaluation cohort...')
    test = load_cohort('test', vocab=vocabulary)

    # Compute vocabulary size (it may be smaller than args.vocabulary_size)
    vocabulary_size = len(vocabulary)

    # The embedding size is the same as the word embedding size unless using the BAG encoder
    observation_embedding_size = FLAGS.observation_embedding_size

    # Parse snapshot-encoder-specific arguments
    if FLAGS.snapshot_encoder == 'RNN':
        snapshot_encoder = encoding.rnn_encoder(FLAGS.snapshot_rnn_num_hidden)
    elif FLAGS.snapshot_encoder == 'CNN':
        snapshot_encoder = encoding.cnn_encoder(FLAGS.snapshot_cnn_windows,
                                                FLAGS.snapshot_cnn_kernels,
                                                FLAGS.dropout)
    elif FLAGS.snapshot_encoder == 'SPARSE':
        snapshot_encoder = encoding.bag_encoder
        observation_embedding_size = vocabulary_size
    elif FLAGS.snapshot_encoder == 'DENSE':
        snapshot_encoder = encoding.dense_encoder
    elif FLAGS.snapshot_encoder == 'DAN':
        snapshot_encoder = encoding.dan_encoder(FLAGS.snapshot_dan_num_hidden_obs,
                                                FLAGS.snapshot_dan_num_hidden_avg,
                                                FLAGS.snapshot_dan_activation)
    elif FLAGS.snapshot_encoder == 'RMLP':
        snapshot_encoder = encoding.rmlp_encoder(FLAGS.snapshot_rmlp_activation,
                                                 FLAGS.snapshot_rmlp_layers)
    elif FLAGS.snapshot_encoder == 'VHN':
        snapshot_encoder = encoding.vhn_encoder(FLAGS.snapshot_vhn_activation,
                                                FLAGS.snapshot_vhn_noise,
                                                FLAGS.snapshot_vhn_layers,
                                                FLAGS.snapshot_vhn_depth)
    else:
        raise ValueError('Given illegal snapshot encoder %s' % FLAGS.doc_encoder)

    if FLAGS.only_augmented_negatives and FLAGS.augment_negatives == 0:
        print('WARNING: --only_augmented_negatives given but --augment_negatives not specified, assuming value of 1.0')
        FLAGS.augment_negatives = 1.

    cell_type = FLAGS.rnn_cell_type
    if FLAGS.rnn_layer_norm:
        cell_type += '-LN'

    delta_encoding_size = 0  # type: int
    for chronology in train.chronologies.values():
        delta_encoding_size = chronology.delta_matrix.shape[-1]

    if FLAGS.only_augmented_negatives:
        def keep_positives(subject_id: str, hadm_id: str, chronology_: Chronology) -> bool:
            del subject_id
            del hadm_id
            return chronology_.label.value == 1

        if "train" in FLAGS.only_augmented_negatives and FLAGS.do_train:
            train = train.filter(keep_positives).infer_negatives_from_positives(FLAGS.augment_negatives)
        if "devel" in FLAGS.only_augmented_negatives and FLAGS.do_train:
            assert devel
            devel = devel.filter(keep_positives).infer_negatives_from_positives(FLAGS.augment_negatives)
        if "test" in FLAGS.only_augmented_negatives:
            test = test.filter(keep_positives).infer_negatives_from_positives(FLAGS.augment_negatives)

    elif FLAGS.augment_negatives > 0.:
        train = train.infer_negatives_from_positives(FLAGS.augment_negatives)

    if FLAGS.correct_imbalance == 'weighted':
        train_chronologies = train.to_list()
        num_positive, num_negative = 0, 0
        for chronology in train_chronologies:
            if chronology.class_label == 1:
                num_positive += 1
            elif chronology.class_label == 0:
                num_negative += 1
            else:
                raise ValueError('Found class label %d for chronology %s' % (chronology.class_label, chronology))
        denominator = float(min(num_positive, num_negative))
        # Weights for label 0 (negative): 1 - prior(negative) = prior(positive)
        # Weights for label 1 (positive): 1 - prior(positive) = prior(negative)
        weights = [num_positive / denominator, num_negative / denominator]
        print('Setting class weights as ', weights)
    else:
        weights = 1.

    if FLAGS.correct_imbalance == "downsample" or FLAGS.correct_imbalance == "upsample":
        print(("Down" if FLAGS.correct_imbalance == "downsample" else "Up") +
              "sampling chronologies...")

    output_dir = FLAGS.output_dir
    if FLAGS.model == 'CANTRIP':
        assert delta_encoding_size > 0
        model = modeling.CANTRIPModel(max_seq_len=FLAGS.max_chrono_length,
                                      max_snapshot_size=FLAGS.max_snapshot_size,
                                      vocabulary_size=vocabulary_size,
                                      observation_embedding_size=observation_embedding_size,
                                      num_hidden=FLAGS.rnn_num_hidden,
                                      cell_type=cell_type,
                                      batch_size=FLAGS.batch_size,
                                      snapshot_encoder=snapshot_encoder,
                                      dropout=FLAGS.dropout,
                                      vocab_dropout=FLAGS.observational_dropout,
                                      num_classes=2,
                                      delta_encoding_size=delta_encoding_size,
                                      embed_delta=FLAGS.embed_delta,
                                      delta_combine=FLAGS.delta_combine,
                                      rnn_highway_depth=FLAGS.rnn_highway_depth,
                                      rnn_direction=FLAGS.rnn_direction)
        experiment = TfExperiment()
        output_dir, _ = experiment.get_model_file()
    elif FLAGS.model == 'SVM':
        from sklearn import svm
        model = svm.SVC(probability=True, gamma='scale', verbose=1, cache_size=4000)
        experiment = SkLearnExperiment()
    elif FLAGS.model == 'LR':
        from sklearn import linear_model
        model = linear_model.LogisticRegression(verbose=1)
        experiment = SkLearnExperiment()
    else:
        raise ValueError('Unrecognized model: %s' % FLAGS.model)

    results = []
    if FLAGS.do_train:
        fetches = experiment.run_model(model, train, devel, test, weights=weights)
        model = fetches[0]
        results.extend(zip(fetches[1:], ['Train', 'Devel', 'Test']))

    nio.make_dirs_quiet(output_dir)
    # Evaluate testing performance exactly as described above for development
    if FLAGS.do_test or FLAGS.do_predict:
        test_metrics = experiment.test_predict(model, test)
        if test_metrics:
            results.append((test_metrics, 'Test'))

    if FLAGS.print_performance:
        print_table_results(results, 'fancy_grid', file=sys.stdout)

    if FLAGS.save_tabbed_results:
        with open(os.path.join(output_dir, 'results.tsv'), 'w') as outfile:
            print_table_results(results, 'simple', file=outfile)

    if FLAGS.save_latex_results:
        with open(os.path.join(output_dir, 'results.tex'), 'w') as outfile:
            print_table_results(results, 'latex_booktabs', file=outfile)


if __name__ == '__main__':
    flags.mark_flag_as_required('data_dir')
    flags.mark_flag_as_required('output_dir')
    tf.logging.set_verbosity(tf.logging.DEBUG)
    tf.app.run()
