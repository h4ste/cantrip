from typing import Union, List, Callable, Sequence

import tensorflow as tf

import layers
import rnn_cell


class Cantrip(tf.keras.Model):

    def __init__(self,
                 observation_encoder,
                 snapshot_encoder,
                 delta_encoder,
                 num_classes: int = 2,
                 name="Cantrip",
                 **kwargs):
        super(Cantrip, self).__init__(name=name, **kwargs)

        self.observation_encoder = observation_encoder
        self.snapshot_encoder = snapshot_encoder
        self.delta_encoder = delta_encoder
        self.rnn = tf.keras.layers.RNN(cell=rnn_cell)
        self.decoder = tf.keras.layers.Dense(num_classes)

    def call(self, inputs):
        observations = self.observation_encoder(inputs[0])
        snapshots = self.snapshot_encoder(observations)
        deltas = self.delta_encoder(inputs[1])

        x = snapshots + deltas
        x = self.rnn(x)
        return self.decoder(x)


class DAN(tf.keras.Model):

    def __init__(self,
                 units,
                 transform: Union[Sequence[tf.keras.layers.Layer], tf.keras.layers.Layer] = None,
                 name="DAN",
                 **kwargs):
        super(DAN, self).__init__(name=name, **kwargs)

        if not transform:
            self.transform = tf.keras.Sequential([tf.keras.layers.Dense(units, activation=cantrip.gelu)] * 2)
        elif isinstance(transform, Sequence):
            self.transform = tf.keras.Sequential(layers)
        else:
            assert isinstance(transform, tf.keras.layers.Layer)
            self.transform = transform

    def call(self, inputs):

        x = tf.reduce_mean(inputs)
        return self.transform(x)


CELL_TYPES = ['LRAN', 'RAN', 'LSTM', 'GRU']

class CantripModel(object):
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
                 vocab_dropout: float = None,
                 num_classes: int = 2,
                 delta_combine: str = "concat",
                 embed_delta: bool = False,
                 rnn_highway_depth: int = 3):
        """Initializes a new CANTRIP model with the given model parameters
        :param max_seq_len: the maximum number of clinical snapshots used in any mini-batch
        :param max_snapshot_size: the maximum number of observations documented in any clinical snapshot
        :param vocabulary_size:  the number of unique observations
        :param observation_embedding_size: the dimensionality of embedded clinical observations
        :param delta_encoding_size: the dimensionality of delta encodings (typically 1)
        :param num_hidden: if scalar, the number of hidden units used in the single-layer clinical picture
            inference RNN; if list, the number of hidden units used in a multi-layer stacked clinical picture inference
            RNN
        :param cell_type: the type of RNN cell to use for the clinical picture inference RNN
        :param batch_size: the size of all mini-batches
        :param snapshot_encoder: a callable function which adds clinical snapshot encoding operations to the
            TensorFlow graph; see src.models.encoder for options
        :param dropout: the dropout rate used in all dropout layers
        :param num_classes: num_classes (int): the number of classes -- this should be two but will work for
            multivariate (i.e., finer-grained) labels
        """
        self.max_seq_len = max_seq_len
        self.max_snapshot_size = max_snapshot_size
        self.vocabulary_size = vocabulary_size
        self.embedding_size = observation_embedding_size
        self.num_hidden = num_hidden
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.delta_encoding_size = delta_encoding_size
        self.dropout = dropout
        self.vocab_dropout = vocab_dropout or dropout
        self.cell_type = cell_type
        self.delta_combine = delta_combine
        self.embed_delta = embed_delta
        self.rnn_highway_depth = rnn_highway_depth

        if delta_combine == 'add' and not self.embed_delta and self.embedding_size != self.delta_encoding_size:
            raise ValueError("Cannot add delta embeddings of size %d to observation encodings of size %d, "
                             "try setting embed_delta=True" %
                             (self.delta_encoding_size, self.embedding_size))

        # Build computation graph
        # self.regularizer = tfcontrib.layers.l1_regularizer(0.05)
        with tf.variable_scope('cantrip'):
            self._add_placeholders()
            with tf.variable_scope('snapshot_encoder'):  # , regularizer=self.regularizer):
                self.snapshot_encodings = snapshot_encoder(self)

            if self.embed_delta:
                with tf.variable_scope('delta_encoder'):
                    self.deltas = tf.keras.layers.Dense(units=self.embedding_size,
                                                        activation=None,
                                                        name='delta_embeddings')(self.deltas, )

            self._add_seq_rnn(cell_type)
            # Convert to sexy logits
            self.logits = tf.keras.layers.Dense(units=self.num_classes,
                                                activation=None,
                                                name='class_logits')(self.seq_final_output)
        self._add_postprocessing()

    def _add_placeholders(self):
        """Add TensorFlow placeholders/feeds which are used as inputs to the model for each mini-batch"""
        # Observation IDs
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

        # Training
        self.training = tf.placeholder(tf.bool, name="training")

    def _add_seq_rnn(self, cell_type: str):
        """Add the clinical picture inference module; implemented in as an RNN. """
        with tf.variable_scope('sequence'):
            # Add dropout on deltas
            if self.dropout > 0:
                self.deltas = tf.keras.layers.Dropout(rate=self.dropout)(self.deltas, training=self.training)

            # Concat observation_t and delta_t (deltas are already shifted by one)
            if self.delta_combine == 'concat':
                self.x = tf.concat([self.snapshot_encodings, self.deltas], axis=-1, name='rnn_input_concat')
            elif self.delta_combine == 'add':
                self.x = self.snapshot_encodings + self.deltas
            else:
                raise ValueError("Invalid delta combination method: %s" % self.delta_combine)

            # Add dropout on concatenated inputs
            if self.dropout > 0:
                self.x = tf.keras.layers.Dropout(rate=self.dropout)(self.x, training=self.training)

            _cell_types = {
                # Original RAN from https://arxiv.org/abs/1705.07393
                'RAN': rnn_cell.RANCell,
                'RAN-LN': lambda num_cells: rnn_cell.RANCell(num_cells, normalize=True),
                'VHRAN': lambda units: rnn_cell.VHRANCell(units, self.x.shape[-1], depth=self.rnn_highway_depth),
                'VHRAN-LN': lambda units: rnn_cell.VHRANCell(units, self.x.shape[-1], depth=self.rnn_highway_depth, normalize=True),
                'RHN': lambda units: rnn_cell.RHNCell(units, self.x.shape[-1],
                                                      depth=self.rnn_highway_depth,
                                                      is_training=self.training),
                'RHN-LN': lambda units: rnn_cell.RHNCell(units, self.x.shape[-1],
                                                      depth=self.rnn_highway_depth,
                                                      is_training=self.training,
                                                      normalize=True),
                # Super secret simplified RAN variant from Eq. group (2) in https://arxiv.org/abs/1705.07393
                # 'LRAN': lambda num_cells: rnn_cell.SimpleRANCell(self.x.shape[-1]),
                # 'LRAN-LN': lambda num_cells: rnn_cell.SimpleRANCell(self.x.shape[-1], normalize=True),
                'LSTM': tf.nn.rnn_cell.BasicLSTMCell,
                'LSTM-LN': tf.contrib.rnn.LayerNormBasicLSTMCell,
                'GRU': tf.nn.rnn_cell.GRUCell,
                'GRU-LN': rnn_cell.LayerNormGRUCell
            }

            if cell_type not in _cell_types:
                raise ValueError('unsupported cell type %s', cell_type)

            self.cell_fn = _cell_types[cell_type]

            self.seq_final_output = layers.rnn_layer(self.cell_fn, self.num_hidden, self.x, self.seq_lengths)

            print('Final output:', self.seq_final_output)

            # Even more fun dropout
            if self.dropout > 0:
                self.seq_final_output = \
                    tf.keras.layers.Dropout(rate=self.dropout)(self.seq_final_output, training=self.training)

    def _add_postprocessing(self):
        """Categorical arg-max prediction for disease-risk"""
        # Class labels (used mainly for metrics)
        self.y = tf.argmax(self.logits, axis=-1, output_type=tf.int32, name='class_predictions')
