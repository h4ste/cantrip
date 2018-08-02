import tensorflow as tf

from tensorflow.contrib.rnn import LayerNormBasicLSTMCell, LSTMStateTuple
from tensorflow.nn.rnn_cell import BasicLSTMCell, GRUCell, MultiRNNCell

from src.models.ran.ran_cell import RANCell, RANCellv2
from src.models.trip import TRIPModel
from src.models.doc.doc_encoder import rnn_encoder
from src.models.layers import rnn_layer, dense_to_sparse
from src.data.scribe_data import  _DELTA_BUCKETS

_cell_types = {
    'RAN': RANCell,
    'RANv2': RANCellv2,
    'LSTM': BasicLSTMCell,
    'LSTM-LN': LayerNormBasicLSTMCell,
    'GRU': GRUCell,
    'RAN-LN': lambda num_cells: RANCell(num_cells, normalize=True),
    'RANv2-LN': lambda num_cells: RANCellv2(num_cells, normalize=True)
}


def CANTRIP(**params):
    model = CANTRIPModel(**params)
    optimizer = CANTRIPOptimizer(model)
    summarizer = CANTRIPSummarizer(model, optimizer)
    return model, optimizer, summarizer


# DENDRON	  DEep Neural Disease Risk predictiOn Network
# DART	  Deep Additive Risk predicTion
# DARN	  Deep Additive Risk predictioN
# TARP	  recurrenT Additive Risk Prediction
# CARP	  reCurrent Additive Risk Prediction
# DENTuRE	  DEep Neural Temporal Risk prEdiction
# CANTRIP	  reCurrent Additive Network Temporal RIsk Prediction
# RAINDRoP	  Recurrent AddItive Network Disease Risk Prediction

class CANTRIPModel(TRIPModel):

    def __init__(self, max_seq_len, max_doc_len,
                 vocabulary_size, embedding_size,
                 num_hidden,
                 cell_type,
                 batch_size,
                 doc_embedding=rnn_encoder(1),
                 num_classes=2,
                 delta_buckets=len(_DELTA_BUCKETS)):
        self.max_seq_len = max_seq_len
        self.max_doc_len = max_doc_len
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.num_hidden = num_hidden
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.delta_buckets = delta_buckets

        if cell_type not in _cell_types:
            raise ValueError('unsupported cell type %s', cell_type)

        self.cell_fn = _cell_types[cell_type]

        # Build graph
        self._add_placeholders()
        with tf.variable_scope('doc_embedding'):
            self.doc_embeddings = doc_embedding(self)
        self._add_seq_rnn()
        self._add_postprocessing()

    def _add_placeholders(self):
        # Word IDs
        self.words = tf.placeholder(tf.int32, [self.batch_size, self.max_seq_len, self.max_doc_len], name="words")

        # Elapsed time deltas
        self.deltas = tf.placeholder(tf.float32, [self.batch_size, self.max_seq_len, self.delta_buckets], name="deltas")

        # Document lengths
        self.doc_lengths = tf.placeholder(tf.int32, [self.batch_size, self.max_seq_len], name="doc_lengths")

        # Sequence lengths
        self.seq_lengths = tf.placeholder(tf.int32, [self.batch_size], name="seq_lengths")

        # Label
        self.labels = tf.placeholder(tf.int32, [self.batch_size], name="labels")

        # Global training step (used for saving/loading)
        self._global_step = tf.Variable(0, name="global_step", trainable=False)

    def _add_seq_rnn(self):
        print('Doc Embedding:', self.doc_embeddings)
        print('Deltas: ', self.deltas)
        # deltas = tf.expand_dims(self.deltas, axis=-1)
        with tf.variable_scope('sequence'):
            # if isinstance(self.doc_embeddings, tf.SparseTensor):
            #     print('Using sparse concat')
            #     sparse_deltas = dense_to_sparse(deltas)
            #     print(sparse_deltas.get_shape())
            #     x = tf.sparse_concat(axis=-1, sp_inputs=[self.doc_embeddings, sparse_deltas])
            #
            #     des = self.doc_embeddings.get_shape().as_list()
            #     sds = sparse_deltas.get_shape().as_list()
            #     des[-1] += sds[-1]
            #
            #     self.x = tf.SparseTensor(x.indices, x.values, dense_shape=des)
            # else:
            self.deltas = tf.nn.dropout(self.deltas, keep_prob=0.5)
            self.x = tf.concat([self.doc_embeddings, self.deltas], axis=-1, name='rnn_input_concat')
            self.seq_final_output = rnn_layer(self.cell_fn, self.num_hidden, self.x, self.seq_lengths)
            self.seq_final_output = tf.nn.dropout(self.seq_final_output, keep_prob=0.5)

            self.output = tf.layers.dense(self.seq_final_output, units=256, activation=tf.nn.tanh,
                                          name='final_state_dense')

        # Convert to fun logits
        self.logits = tf.layers.dense(self.output, units=self.num_classes, activation=None, name='class_logits')

    def _add_postprocessing(self):
        self.y = tf.argmax(self.logits, axis=-1, output_type=tf.int32, name='class_predictions')


class CANTRIPOptimizer(object):

    def __init__(self, model, sparse=False, learning_rate=1e-3):
        """
        OptimizerVAE initializer

        :param model: a model object
        :param learning_rate: float, learning rate of the optimizer
        """
        self.model = model

        if sparse:
            self.loss = tf.losses.sparse_softmax_cross_entropy(model.labels, model.logits)
        else:
            y_true = tf.one_hot(model.labels, model.num_classes, name='labels_onehot')
            self.loss = tf.losses.softmax_cross_entropy(y_true, model.logits)

        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

        self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss, self.global_step)


class CANTRIPSummarizer(object):

    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

        print('Y:', model.y)
        print('Labels:', model.labels)

        self.tp = tf.count_nonzero(model.y * model.labels, dtype=tf.int32)
        self.tn = tf.count_nonzero((model.y - 1) * (model.labels - 1), dtype=tf.int32)
        self.fp = tf.count_nonzero(model.y * (model.labels - 1), dtype=tf.int32)
        self.fn = tf.count_nonzero((model.y - 1) * model.labels, dtype=tf.int32)

        self.precision = self.tp / (self.tp + self.fp)
        self.recall = self.tp / (self.tp + self.fn)
        self.accuracy = (self.tp + self.tn) / model.batch_size
        self.specificity = self.tn / (self.tn + self.fp)

        self.f1 = 2 * self.precision * self.recall / (self.precision + self.recall)

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

        with tf.name_scope('Batch'):
            self.batch_summary = tf.summary.merge([
                tf.summary.scalar('Accuracy', self.accuracy),
                tf.summary.scalar('Precision', self.precision),
                tf.summary.scalar('Recall', self.recall),
                tf.summary.scalar('F1', self.f1),
                tf.summary.scalar('Specificity', self.specificity),
                tf.summary.scalar('Loss', optimizer.loss)
            ])

        self.train = _CANTRIPModeSummarizer('train', model)
        self.devel = _CANTRIPModeSummarizer('devel', model)
        self.test = _CANTRIPModeSummarizer('test', model)


class _CANTRIPModeSummarizer(object):

    def __init__(self, mode, model):
        self.mode = mode
        with tf.name_scope(mode) as scope:
            acc, acc_op = tf.metrics.accuracy(model.labels, model.y)
            auroc, auroc_op = tf.metrics.auc(model.labels, model.y, summation_method='careful_interpolation')
            auprc, auprc_op = tf.metrics.auc(model.labels, model.y, curve='PR', summation_method='careful_interpolation')
            p, p_op = tf.metrics.precision(model.labels, model.y)
            r, r_op = tf.metrics.recall(model.labels, model.y)
            f1 = 2 * p * r / (p + r)

            with tf.name_scope('confusion_matrix'):
                tp, tp_op = tf.metrics.true_positives(model.labels, model.y)
                tn, tn_op = tf.metrics.true_negatives(model.labels, model.y)
                fp, fp_op = tf.metrics.false_positives(model.labels, model.y)
                fn, fn_op = tf.metrics.false_negatives(model.labels, model.y)
                confusion_matrix = tf.summary.merge([
                    tf.summary.scalar('TP', tp),
                    tf.summary.scalar('TN', tn),
                    tf.summary.scalar('FP', fp),
                    tf.summary.scalar('FN', fn),
                ])

            metric_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope=scope)
            self.reset_op = tf.variables_initializer(var_list=metric_vars)

            self.summary = tf.summary.merge([
                tf.summary.scalar('Accuracy', acc),
                tf.summary.scalar('AUROC', auroc),
                tf.summary.scalar('AUPRC', auprc),
                tf.summary.scalar('Precision', p),
                tf.summary.scalar('Recall', r),
                tf.summary.scalar('F1', f1),
                confusion_matrix
            ])

        self.metrics = {
            'Accuracy': acc,
            'AUROC': auroc,
            'AUPRC': auprc,
            'Precision': p,
            'Recall': r,
            'F1': f1
        }

        self.metric_ops = [acc_op, auroc_op, auprc_op, p_op, r_op,
                           [tp_op, tn_op, fp_op, fn_op]]
