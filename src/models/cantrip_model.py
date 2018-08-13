from typing import Union, List, Callable

import tensorflow as tf
from tensorflow.contrib.rnn import LayerNormBasicLSTMCell
from tensorflow.nn.rnn_cell import BasicLSTMCell, GRUCell

from src.models.layers import rnn_layer
from src.models.rnn_cell import SimpleRANCell, RANCell, LayerNormGRUCell

CELL_TYPES = ['SRAN', 'RAN', 'RAN-LN', 'LSTM', 'LSTM-LN', 'GRU', 'GRU-LN']


class CANTRIPModel(object):

    def __init__(self,
                 max_seq_len: int,
                 max_snapshot_size: int,
                 vocabulary_size: int,
                 observation_embedding_size: int,
                 delta_encoding_size: int,
                 num_hidden: Union[int, List[int]],
                 cell_type: str,
                 batch_size: int,
                 snapshot_encoder: Callable[['CANTRIPModel'], tf.Tensor],
                 dropout: float = 0.,
                 num_classes: int = 2):
        self.max_seq_len = max_seq_len
        self.max_snapshot_size = max_snapshot_size
        self.vocabulary_size = vocabulary_size
        self.embedding_size = observation_embedding_size
        self.num_hidden = num_hidden
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.delta_encoding_size = delta_encoding_size
        self.dropout = dropout
        self.cell_type = cell_type

        _cell_types = {
            # Original RAN from https://arxiv.org/abs/1705.07393
            'RAN': RANCell,
            'RAN-LN': lambda num_cells: RANCell(num_cells, normalize=True),
            # Super secret simplified RAN variant from https://arxiv.org/abs/1705.07393
            'SRAN': SimpleRANCell,
            'LSTM': BasicLSTMCell,
            'LSTM-LN': LayerNormBasicLSTMCell,
            'GRU': GRUCell,
            'GRU-LN': LayerNormGRUCell
        }

        if cell_type not in _cell_types:
            raise ValueError('unsupported cell type %s', cell_type)

        self.cell_fn = _cell_types[cell_type]

        # Build graph
        self._add_placeholders()
        with tf.variable_scope('snapshot_encoder'):
            self.snapshot_encodings = snapshot_encoder(self)
        self._add_seq_rnn()
        self._add_postprocessing()

    def _add_placeholders(self):
        # Word IDs
        self.observations = tf.placeholder(tf.int32, [self.batch_size, self.max_seq_len, self.max_snapshot_size],
                                           name="observations")

        # Elapsed time deltas
        self.deltas = tf.placeholder(tf.float32, [self.batch_size, self.max_seq_len, self.delta_encoding_size],
                                     name="deltas")

        # Snapshot sizes
        self.snapshot_sizes = tf.placeholder(tf.int32, [self.batch_size, self.max_seq_len], name="snapshot_sizes")

        # Chronology lengths
        self.seq_lengths = tf.placeholder(tf.int32, [self.batch_size], name="seq_lengths")

        # Label
        self.labels = tf.placeholder(tf.int32, [self.batch_size], name="labels")

    def _add_seq_rnn(self):
        with tf.variable_scope('sequence'):
            # Add dropout on deltas
            if self.dropout > 0:
                self.deltas = tf.nn.dropout(self.deltas, keep_prob=self.dropout)

            # Concat observation_t and delta_t (deltas are already shifted by one)
            self.x = tf.concat([self.snapshot_encodings, self.deltas], axis=-1, name='rnn_input_concat')

            # Add dropout on concatenated inputs
            if self.dropout > 0:
                self.x = tf.nn.dropout(self.x, 1 - self.dropout)

            # Compute weights AND final RNN output if looking at RANv2 variants
            self.seq_final_output = rnn_layer(self.cell_fn, self.num_hidden, self.x, self.seq_lengths)

            # Even more fun dropout
            if self.dropout > 0:
                self.seq_final_output = tf.nn.dropout(self.seq_final_output, 1 - self.dropout)

        # Convert to sexy logits
        self.logits = tf.layers.dense(self.seq_final_output, units=self.num_classes,
                                      activation=None, name='class_logits')

    def _add_postprocessing(self):
        # Class labels (used mainly for metrics)
        self.y = tf.argmax(self.logits, axis=-1, output_type=tf.int32, name='class_predictions')


class CANTRIPOptimizer(object):

    def __init__(self, model: CANTRIPModel, sparse: bool = False, learning_rate: float = 1e-3):
        """
        Creates a new CANTRIPOptimizer responsible for optimizing CANTRIP. Allegedly, some day I will get around to
        looking at other optimization strategies (e.g., sequence optimization).
        :param model: a CANTRIPModel object
        :param sparse: whether to use sparse softmax or not (I never actually tested this)
        :param learning_rate: float, learning rate of the optimizer
        """
        self.model = model

        # If sparse calculate sparsesoftmax directly from integer labels
        if sparse:
            self.loss = tf.losses.sparse_softmax_cross_entropy(model.labels, model.logits)
        # If not sparse, convert labels to one hots and do softmax
        else:
            y_true = tf.one_hot(model.labels, model.num_classes, name='labels_onehot')
            self.loss = tf.losses.softmax_cross_entropy(y_true, model.logits)

        # Global step used for coordinating summarizes and checkpointing
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

        # Training operation: fetch this to run a step of the adam optimizer!
        self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss, self.global_step)


class CANTRIPSummarizer(object):

    def __init__(self, model: CANTRIPModel, optimizer: CANTRIPOptimizer):
        self.model = model
        self.optimizer = optimizer

        # Batch-level confusion matrix
        self.tp = tf.count_nonzero(model.y * model.labels, dtype=tf.int32)
        self.tn = tf.count_nonzero((model.y - 1) * (model.labels - 1), dtype=tf.int32)
        self.fp = tf.count_nonzero(model.y * (model.labels - 1), dtype=tf.int32)
        self.fn = tf.count_nonzero((model.y - 1) * model.labels, dtype=tf.int32)

        # Batch-level binary classification metrics
        self.precision = self.tp / (self.tp + self.fp)
        self.recall = self.tp / (self.tp + self.fn)
        self.accuracy = (self.tp + self.tn) / model.batch_size
        self.specificity = self.tn / (self.tn + self.fp)
        self.f1 = 2 * self.precision * self.recall / (self.precision + self.recall)

        # Dict of all metrics to make fetching more convenient
        self.batch_metrics = {
            'TP': self.tp,
            'TN': self.tn,
            'FP': self.fp,
            'FN': self.fn,
            'Precision': self.precision,
            'Recall': self.recall,
            'Accuracy': self.accuracy,
            'Specificity': self.specificity,
            'F1': self.f1,
            'Loss': optimizer.loss
        }

        # Group all batch-level metrics in the same pane in TensorBoard using a name scope
        with tf.name_scope('Batch'):
            self.batch_summary = tf.summary.merge([
                tf.summary.scalar('Accuracy', self.accuracy),
                tf.summary.scalar('Precision', self.precision),
                tf.summary.scalar('Recall', self.recall),
                tf.summary.scalar('F1', self.f1),
                tf.summary.scalar('Specificity', self.specificity),
                tf.summary.scalar('Loss', optimizer.loss)
            ])

        # Specific training/development/testing summarizers
        self.train = _CANTRIPModeSummarizer('train', model)
        self.devel = _CANTRIPModeSummarizer('devel', model)
        self.test = _CANTRIPModeSummarizer('test', model)


class _CANTRIPModeSummarizer(object):

    def __init__(self, mode: str, model: CANTRIPModel):
        self.mode = mode
        with tf.name_scope(mode) as scope:
            # Streaming, epoch-level metrics
            acc, acc_op = tf.metrics.accuracy(model.labels, model.y)
            auroc, auroc_op = tf.metrics.auc(model.labels, model.y, summation_method='careful_interpolation')
            auprc, auprc_op = tf.metrics.auc(model.labels, model.y, curve='PR',
                                             summation_method='careful_interpolation')
            p, p_op = tf.metrics.precision(model.labels, model.y)
            r, r_op = tf.metrics.recall(model.labels, model.y)
            f1 = 2 * p * r / (p + r)

            # Streaming, epoch-level confusion matrix information
            with tf.name_scope('confusion_matrix'):
                tp, tp_op = tf.metrics.true_positives(model.labels, model.y)
                tn, tn_op = tf.metrics.true_negatives(model.labels, model.y)
                fp, fp_op = tf.metrics.false_positives(model.labels, model.y)
                fn, fn_op = tf.metrics.false_negatives(model.labels, model.y)
                # Group these so they all show up together in TensorBoard
                confusion_matrix = tf.summary.merge([
                    tf.summary.scalar('TP', tp),
                    tf.summary.scalar('TN', tn),
                    tf.summary.scalar('FP', fp),
                    tf.summary.scalar('FN', fn),
                ])

            # Summary containing all epoch-level metrics computed for the current mode
            self.summary = tf.summary.merge([
                tf.summary.scalar('Accuracy', acc),
                tf.summary.scalar('AUROC', auroc),
                tf.summary.scalar('AUPRC', auprc),
                tf.summary.scalar('Precision', p),
                tf.summary.scalar('Recall', r),
                tf.summary.scalar('F1', f1),
                confusion_matrix
            ])

            # Dictionary of epoch-level metrics for each fetching
            self.metrics = {
                'Accuracy': acc,
                'AUROC': auroc,
                'AUPRC': auprc,
                'Precision': p,
                'Recall': r,
                'F1': f1
            }

            # TensorFlow operations that need to be run to update the epoch-level metrics on each batch
            self.metric_ops = [acc_op, auroc_op, auprc_op, p_op, r_op,
                               [tp_op, tn_op, fp_op, fn_op]]

            # Operation to reset metrics after each epoch
            metric_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope=scope)
            self.reset_op = tf.variables_initializer(var_list=metric_vars)
