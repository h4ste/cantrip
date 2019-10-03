""" Clinical snapshot encoders for use with CANTRIP Model.

CANTRIPModel expects a clinical snapshot encoder function which takes as input the CANTRIPModel and adds
clinical snapshot encoding ops to the graph returning the final clinical snapshot encoding as
[batch x max_seq_len x embedding_size]

"""

import tensorflow.compat.v1 as tf

import layers
import rnn_cell


def rnn_encoder(num_hidden, cell_fn=rnn_cell.RANCell):
    """
    Creates an RNN encoder with the given number of hidden layers. If
    :param num_hidden: number of hidden (memory) units use; num_hidden is iterable, a multi-layer
    rnn cell will be creating using each number of hidden units
    :param cell_fn: rnn_cell constructor to use
    :return: rnn_encoder function
    """

    def _rnn_encoder(model):
        """

        :type model: modeling.BERTModel
        """
        with tf.variable_scope('rnn_encoder'):
            # Embed clinical observations
            embedded_observations = layers.embedding_layer(model.observations, model.vocabulary_size,
                                                           model.embedding_size,
                                                           model.vocab_dropout,
                                                           training=model.training)

            # Reshape to (batch * seq_len) x doc_len x embedding
            flattened_embedded_obs = tf.reshape(embedded_observations,
                                                [model.batch_size * model.max_seq_len,
                                                 model.max_snapshot_size,
                                                 model.embedding_size],
                                                name='flat_emb_obs')
            flattened_snapshot_sizes = tf.reshape(model.snapshot_sizes, [model.batch_size * model.max_seq_len],
                                                  name='flat_snapshot_sizes')

            # Apply RNN to all documents in all batches
            flattened_snapshot_encodings = layers.rnn_layer(cell_fn=cell_fn,
                                                            num_hidden=num_hidden,
                                                            inputs=flattened_embedded_obs,
                                                            lengths=flattened_snapshot_sizes,
                                                            return_interpretable_weights=False)

            # Reshape back to (batch x seq_len x encoding_size)
            return tf.reshape(flattened_snapshot_encodings,
                              [model.batch_size, model.max_seq_len, flattened_snapshot_encodings.shape[-1]],
                              name='rnn_snapshot_encoding')

    return _rnn_encoder


def cnn_encoder(windows=None, kernels=1000, dropout=0.):
    """
    Creates a CNN encoder with the given number of windows, kernels, and dropout
    :param windows: number of consecutive observations to consider; defaults to [3, 4, 5]
    :param kernels: number of convolutional kernels; defaults to 1,000
    :param dropout: dropout probability; defaults to 0.0 (no dropout)
    :return: cnn_encoder function
    """
    if windows is None:
        windows = [3, 4, 5]

    def _cnn_encoder(model):
        """

        :type model: BERTModel
        """
        with tf.variable_scope('cnn_encoder'):
            # Embed observations
            embedded_observations = layers.embedding_layer(model.observations, model.vocabulary_size,
                                                           model.embedding_size,
                                                           model.vocab_dropout,
                                                           training=model.training)

            # Reshape to (batch * seq_len) x snapshot_size x embedding
            flattened_embedded_obs = tf.reshape(embedded_observations,
                                                [model.batch_size * model.max_seq_len,
                                                 model.max_snapshot_size,
                                                 model.embedding_size])

            # Apply parallel convolutional and pooling layers
            outputs = []
            for n in windows:
                if dropout > 0:
                    flattened_embedded_obs = \
                        tf.keras.layers.Dropout(rate=model.dropout)(flattened_embedded_obs, training=model.training)
                conv_layer = tf.keras.layers.Convolution1D(filters=kernels,
                                                           kernel_size=n,
                                                           activation=tf.nn.leaky_relu,
                                                           name="conv_%dgram" % n)(flattened_embedded_obs)
                pool_layer = tf.keras.layers.MaxPooling1D(pool_size=1,
                                                          strides=model.max_snapshot_size - n + 1,
                                                          name="maxpool_%dgram" % n)(conv_layer)
                outputs.append(pool_layer)

            # Concatenate pooled outputs
            output = tf.concat(outputs, axis=-1)

            # Embed concat output with leaky ReLU
            embeddings = tf.keras.layers.Dense(units=model.embedding_size, activation=tf.nn.relu)(output)

            # Reshape back to [batch_size x max_seq_len x encoding_size]
            return tf.reshape(embeddings, [model.batch_size, model.max_seq_len, model.embedding_size])

    return _cnn_encoder


def get_bag_vectors(model):
    """
    Represents snapshots as a bag of clinical observations. Specifically, returns a V-length
    binary vector such that the v-th index is 1 iff the v-th observation occurs in the given snapshot
    :param model: CANTRIP model
    :type model: modeling.CANTRIPModel
    :return: clinical snapshot encoding
    """
    # 1. Evaluate which entries in model.observations are non-zero
    mask = tf.not_equal(model.observations, 0)
    where = tf.where(mask)

    # 2. Get the vocabulary indices for non-zero observations
    vocab_indices = tf.boolean_mask(model.observations, mask)
    vocab_indices = tf.expand_dims(vocab_indices[:], axis=-1)
    vocab_indices = tf.cast(vocab_indices, dtype=tf.int64)

    # 3. Get batch and sequence indices for non-zero observations
    tensor_indices = where[:, :-1]

    # Concat batch, sequence, and vocabulary indices
    indices = tf.concat([tensor_indices, vocab_indices], axis=-1)

    # Our sparse tensor will be 1 for observed observations, 0, otherwise
    ones = tf.ones_like(indices[:, 0], dtype=tf.float32)

    # The dense shape will be the same as model.observations, but using the entire vocabulary as the final dimension
    dense_shape = model.observations.get_shape().as_list()
    dense_shape[2] = model.vocabulary_size

    # Store as a sparse tensor because they're neat
    st = tf.SparseTensor(indices=indices, values=ones, dense_shape=dense_shape)
    return tf.sparse.reorder(st)


def dense_encoder(model):
    """
    Represents documents as an embedded bag of clinical observations. Specifically, returns an embedded of the V-length
    binary vector encoding all clinical observations included in a snapshot
    :param model: CANTRIP model
    :type model: modeling.CANTRIPModel
    :return: clinical snapshot encoding
    """
    with tf.variable_scope('dense_encoder'):
        # Use the CPU cause things are about to weird (i.e., too big to fit in GPU memory)
        with tf.device("/cpu:0"):
            # Add bag-of-observation vector transformations to the model
            bags = get_bag_vectors(model)

            # Embed bag-of-observation vectors
            embedded_observations = layers.create_embeddings(model.vocabulary_size, model.embedding_size,
                                                             model.vocab_dropout,
                                                             training=model.training)

            # Reshape them so we use the same projection weights for every bag
            flat_emb_bags = tf.sparse.reshape(bags, [model.batch_size * model.max_seq_len,
                                                     model.vocabulary_size],
                                              name='flat_emb_obs')
            # Dropout for fun
            # if model.dropout > 0:
            #     flat_emb_bags = tf.layers.dropout(flat_emb_bags, rate=model.dropout, training=model.training)

            # Sparse to dense projection
            flat_doc_embeddings = tf.sparse_tensor_dense_matmul(flat_emb_bags, embedded_observations,
                                                                name='flat_doc_embeddings')

            # More dropout for fun
            flat_doc_embeddings = tf.keras.layers.Dropout(rate=model.dropout)(flat_doc_embeddings,
                                                                              training=model.training)

            # Reshape back to [batch_size x max_seq_len x encoding_size]

    return tf.reshape(flat_doc_embeddings, [model.batch_size, model.max_seq_len, model.embedding_size],
                      name='doc_embeddings')


def bag_encoder(model):
    """
    Represents snapshots as a bag of clinical observations. Specifically, returns a V-length
    binary vector such that the v-th index is 1 iff the v-th observation occurs in the given snapshot
    :param model: CANTRIP model
    :type model: modeling.CANTRIPModel
    :return: clinical snapshot encoding
    """
    with tf.variable_scope('bow_encoder'):
        # Use the CPU cause everything will be vocabulary-length
        with tf.device("/cpu:0"):
            return tf.sparse.to_dense(get_bag_vectors(model))


class SparseDenseLayer(tf.keras.layers.Dense):

    def __init__(self,
                 units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(SparseDenseLayer, self).__init__(units=units,
                                               activation=activation,
                                               use_bias=use_bias,
                                               kernel_initializer=kernel_initializer,
                                               bias_initializer=bias_initializer,
                                               kernel_regularizer=kernel_regularizer,
                                               bias_regularizer=bias_regularizer,
                                               activity_regularizer=activity_regularizer,
                                               kernel_constraint=kernel_constraint,
                                               bias_constraint=bias_constraint,
                                               **kwargs)

    def call(self, inputs):
        if not isinstance(inputs, tf.SparseTensor):
            return super(SparseDenseLayer, self).call(inputs)

        outputs = tf.sparse.sparse_dense_matmul(inputs, self.kernel)
        outputs = tf.debugging.check_numerics(outputs, "SparseDenseLayer had NaN product")

        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)
            outputs = tf.debugging.check_numerics(outputs, "SparseDenseLayer had NaN bias sum")

        if self.activation is not None:
            outputs = self.activation(outputs)
            outputs = tf.debugging.check_numerics(outputs, "SparseDenseLayer had NaN activation")

        outputs = tf.debugging.check_numerics(outputs, "SparseDenseLayer output had NaNs")
        return outputs


def dan_encoder(obs_hidden_units, avg_hidden_units, activation='gelu'):
    """Represents snapshots as a modified element-wise averages of embedded clinical observations.

    :param obs_hidden_units: number of hidden units in dense layers between observation embeddings and average;
        if iterable multiple dense layers will be added using the respective hidden units
    :param avg_hidden_units: number of hidden units in dense layers between average embeddings and snapshot encoding;
        if iterable multiple dense layers will be added using the respective hidden units
    :param activation: type of activation function to use between layers
    :return: clinical snapshot encoding
    """

    activation_fn = None
    if activation == 'gelu':
        activation_fn = layers.gelu
    elif activation == 'relu':
        activation_fn = tf.nn.relu
    elif activation == 'tanh':
        activation_fn = tf.nn.tanh
    elif activation == 'sigmoid':
        activation_fn = tf.nn.sigmoid
    else:
        raise KeyError('Unsupported activation function: %s' % activation)

    def _dan_encoder(model):
        """
        :param model:
        :type model: modeling.CANTRIPModel
        :return:
        """
        with tf.variable_scope('dan_encoder'):
            embedded_observations = layers.embedding_layer(model.observations, model.vocabulary_size,
                                                           model.embedding_size, model.vocab_dropout,
                                                           training=model.training)

            # Reshape to (batch * seq_len * doc_len) x embedding
            flattened_embedded_observations = tf.reshape(
                embedded_observations,
                [model.batch_size * model.max_seq_len * model.max_snapshot_size,
                 model.embedding_size]
            )
            # Add dense observation layers
            obs_layer = flattened_embedded_observations
            for num_hidden in obs_hidden_units:
                obs_layer = tf.keras.layers.Dense(units=num_hidden, activation=activation_fn)(obs_layer)

            # Reshape final output by grouping observations in the same snapshot together
            obs_layer = tf.reshape(obs_layer, [model.batch_size * model.max_seq_len,
                                               model.max_snapshot_size,
                                               obs_hidden_units[-1]])

            # Divide by active number of observations rather than the padded snapshot size; requires reshaping to
            # (batch x seq_len) x 1 so we can divide by this
            flattened_snapshot_sizes = tf.reshape(model.snapshot_sizes, [model.batch_size * model.max_seq_len, 1])

            mask = tf.sequence_mask(model.snapshot_sizes, maxlen=model.max_snapshot_size, dtype=tf.float32)
            mask = tf.reshape(mask, [model.batch_size * model.max_seq_len, model.max_snapshot_size, 1])

            # Compute dynamic-size element-wise average
            avg_layer = tf.reduce_sum(obs_layer * mask, axis=1)
            avg_layer = avg_layer / tf.cast(tf.maximum(flattened_snapshot_sizes, 1), dtype=tf.float32)

            # More fun dense layers
            for num_hidden in avg_hidden_units:
                avg_layer = tf.keras.layers.Dense(num_hidden, activation_fn)(avg_layer)

            # Final output of the model
            output = tf.keras.layers.Dense(model.embedding_size, activation_fn)(avg_layer)

            # Reshape to [batch_size x seq_len x encoding_size]
            return tf.reshape(output, [model.batch_size, model.max_seq_len, model.embedding_size])

    return _dan_encoder


def rmlp_encoder(activation='gelu', num_layers=10, num_hidden=2048):
    activation_fn = None
    if activation == 'gelu':
        activation_fn = layers.gelu
    elif activation == 'relu':
        activation_fn = tf.nn.relu
    elif activation == 'tanh':
        activation_fn = tf.nn.tanh
    elif activation == 'sigmoid':
        activation_fn = tf.nn.sigmoid
    else:
        raise KeyError('Unsupported activation function: %s' % activation)

    def residual_unit(inputs, i, units):
        with tf.variable_scope("residual_unit%d" % i):
            x = tf.keras.layers.Dense(units=units, activation=activation_fn)(inputs)
            x = tf.keras.layers.BatchNormalization()(x)
            x = activation_fn(x)
            return x + inputs

    def _rmlp_encoder(model):
        # Convert batch x seq_len x doc_len tensor of obs IDs to batch x seq_len x vocab_size bag-of-observation vectors
        with tf.variable_scope("RMLP"):
            bags = get_bag_vectors(model)
            flat_bags = tf.sparse.reshape(bags, [model.batch_size * model.max_seq_len, model.vocabulary_size])
            x = SparseDenseLayer(units=num_hidden, activation=None)(flat_bags)

            # Convert to Dense to debug NaNs
            # flat_bags = tf.sparse.to_dense(flat_bags)
            # flat_bags = tf.debugging.assert_all_finite(flat_bags, 'flat bags had nans')
            # x = tf.keras.layers.Dense(units=num_hidden, activation=None)(flat_bags)

            for i in range(num_layers):
                x = residual_unit(x, i, num_hidden)

            x = tf.keras.layers.Dense(units=model.embedding_size, activation=activation_fn)(x)
            x = tf.debugging.assert_all_finite(x, 'dense had nans')
            x = tf.reshape(x, [model.batch_size, model.max_seq_len, model.embedding_size])
            x = tf.debugging.assert_all_finite(x, 'reshape had nans')
        return x

    return _rmlp_encoder


def vhn_encoder(activation='gelu', noise_weight=0.75, num_layers=10, depth=6, num_hidden=2048):
    activation_fn = None
    if activation == 'gelu':
        activation_fn = layers.gelu
    elif activation == 'relu':
        activation_fn = tf.nn.relu
    elif activation == 'tanh':
        activation_fn = tf.nn.tanh
    elif activation == 'sigmoid':
        activation_fn = tf.nn.sigmoid
    else:
        raise KeyError('Unsupported activation function: %s' % activation)

    def vhn_layer(inputs, units, residuals):
        noise = tf.random.uniform(shape=inputs.shape, dtype=tf.float32) / noise_weight
        out = tf.keras.layers.Dense(units=units, activation=activation_fn)(inputs + noise)
        return tf.math.add_n([out, inputs] + residuals)

    def _vhn_encoder(model):
        # Convert batch x seq_len x doc_len tensor of obs IDs to batch x seq_len x vocab_size bag-of-observation vectors
        bags = get_bag_vectors(model)
        flat_bags = tf.sparse.reshape(bags, [model.batch_size * model.max_seq_len, model.vocabulary_size])
        x = SparseDenseLayer(units=num_hidden, activation=None)(flat_bags)

        residuals = []
        for i in range(num_layers):
            slice_ = min(i + 1, depth)
            x = vhn_layer(x, units=num_hidden, residuals=residuals[-slice_:])
            residuals.append(x)

        x = tf.keras.layers.Dense(units=model.embedding_size, activation=activation_fn)(x)
        x = tf.reshape(x, [model.batch_size, model.max_seq_len, model.embedding_size])
        return x

    return _vhn_encoder
