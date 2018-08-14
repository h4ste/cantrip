""" Clinical snapshot encoders for use with CANTRIP Model.

CANTRIPModel expects a clinical snapshot encoder function which takes as input the CANTRIPModel and adds
clinical snapshot encoding ops to the graph returning the final clinical snapshot encoding as
[batch x max_seq_len x embedding_size]

"""

import tensorflow as tf
from tensorflow.nn.rnn_cell import GRUCell

from src.models import CANTRIPModel
from src.models.layers import rnn_layer, embedding_layer, create_embeddings


def rnn_encoder(num_hidden, cell_fn=GRUCell):
    """
    Creates an RNN encoder with the given number of hidden layers. If
    :param num_hidden: number of hidden (memory) units use; num_hidden is iterable, a multi-layer
    rnn cell will be creating using each number of hidden units
    :param cell_fn: rnn_cell constructor to use
    :return: rnn_encoder function
    """

    def _rnn_encoder(model):
        with tf.variable_scope('rnn_encoder'):
            # Embed clinical observations
            embedded_observations = embedding_layer(model.words, model.vocabulary_size, model.embedding_size)

            # Reshape to (batch * seq_len) x doc_len x embedding
            flattened_embedded_obs = tf.reshape(embedded_observations,
                                                [model.batch_size * model.max_seq_len,
                                                 model.max_doc_len,
                                                 model.embedding_size],
                                                name='flat_emb_obs')
            flattened_snapshot_sizes = tf.reshape(model.doc_lengths, [model.batch_size * model.max_seq_len],
                                                  name='flat_snapshot_sizes')

            # Apply RNN to all documents in all batches
            flattened_snapshot_encodings = rnn_layer(cell_fn=cell_fn,
                                                     num_hidden=num_hidden,
                                                     inputs=flattened_embedded_obs,
                                                     lengths=flattened_snapshot_sizes)

            # Reshape back to (batch x seq_len x encoding_size)
            return tf.reshape(flattened_snapshot_encodings,
                              [model.batch_size, model.max_seq_len, flattened_snapshot_encodings.shape[-1]],
                              name='rnn_snapshot_encoding')

    return _rnn_encoder


def cnn_encoder(windows=None, kernels=1000, dropout=0.0):
    """
    Creates a CNN encoder with the given number of windows, kernels, and dropout
    :param windows: number of consecutive observations to consider; defaults to [3, 4, 5]
    :param kernels: number of convolutional kernels; defaults to 1,000
    :param dropout: dropout probability; defaults to 0.0 (no dropout)
    :return: cnn_encoder function
    """
    if windows is None:
        windows = [3, 4, 5]

    def _cnn_encoder(model: CANTRIPModel):
        with tf.variable_scope('cnn_encoder'):
            # Embed observations
            embedded_observations = embedding_layer(model.observations, model.vocabulary_size, model.embedding_size)

            # Reshape to (batch * seq_len) x snapshot_size x embedding
            flattened_embedded_obs = tf.reshape(embedded_observations,
                                                [model.batch_size * model.max_seq_len,
                                                 model.max_snapshot_size,
                                                 model.embedding_size])

            # Apply parallel convolutional and pooling layers
            outputs = []
            for n in windows:
                dropout_layer = tf.nn.dropout(flattened_embedded_obs, 1. - dropout)
                conv_layer = tf.layers.conv1d(dropout_layer, kernels,
                                              kernel_size=n,
                                              activation=tf.nn.leaky_relu,
                                              name="conv_%dgram" % n)
                pool_layer = tf.layers.max_pooling1d(conv_layer, 1, model.max_snapshot_size - n + 1,
                                                     name="maxpool_%dgram" % n)
                outputs.append(pool_layer)

            # Concatenate pooled outputs
            output = tf.concat(outputs, axis=-1)

            # Embed concat output with leaky relu
            embeddings = tf.layers.dense(output, model.embedding_size, activation=tf.nn.leaky_relu)

            # Reshape back to [batch_size x max_seq_len x encoding_size]
            return tf.reshape(embeddings, [model.batch_size, model.max_seq_len, model.embedding_size])

    return _cnn_encoder


def get_bag_vectors(model: CANTRIPModel):
    """
    Represents snapshots as a bag of clinical observations. Specifically, returns a V-length
    binary vector such that the v-th index is 1 iff the v-th observation occurs in the given snapshot
    :param model: CANTRIP model
    :return: clinical snapshot encoding
    """
    # 1. Evaluate which entries in model.observations are non-zero
    mask = tf.not_equal(model.observations, 0)
    where = tf.where(mask)

    # 2. Get the vocabulary indices for non-zero observations
    vocab_indices = tf.boolean_mask(model.observations, mask)
    vocab_indices = tf.expand_dims(vocab_indices[:], axis=-1)
    vocab_indices = tf.to_int64(vocab_indices)

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
    return tf.sparse_reorder(st)


def dense_encoder(model: CANTRIPModel):
    """
    Represents documents as an embedded bag of clinical observations. Specifically, returns an embedded of the V-length
    binary vector encoding all clinical observations included in a snapshot
    :param model: CANTRIP model
    :return: clinical snapshot encoding
    """
    with tf.variable_scope('dense_encoder'):
        # Use the CPU cause things are about to weird (i.e., too big to fit in GPU memory)
        with tf.device("/cpu:0"):
            # Add bag-of-observation vector transformations to the model
            bags = get_bag_vectors(model)

            # Embed bag-of-observation vectors
            embedded_observations = create_embeddings(model.vocabulary_size, model.embedding_size, model.dropout)

            # Reshape them so we use the same projection weights for every bag
            flat_emb_bags = tf.sparse_reshape(bags, [model.batch_size * model.max_seq_len,
                                                     model.vocabulary_size],
                                              name='flat_emb_obs')
            # Dropout for fun
            flat_emb_bags = tf.nn.dropout(flat_emb_bags, model.dropout)

            # Sparse to dense projection
            flat_doc_embeddings = tf.sparse_tensor_dense_matmul(flat_emb_bags, embedded_observations,
                                                                name='flat_doc_embeddings')

            # More dropout for fun
            flat_doc_embeddings = tf.nn.dropout(flat_doc_embeddings, 0.5)

        # Reshape back to [batch_size x max_seq_len x encoding_size]
        return tf.reshape(flat_doc_embeddings, [model.batch_size, model.max_seq_len, model.embedding_size],
                          name='doc_embeddings')


def bag_encoder(model: CANTRIPModel):
    """
    Represents snapshots as a bag of clinical observations. Specifically, returns a V-length
    binary vector such that the v-th index is 1 iff the v-th observation occurs in the given snapshot
    :param model: CANTRIP model
    :return: clinical snapshot encoding
    """
    with tf.variable_scope('bow_encoder'):
        # Use the CPU cause everything will be vocabulary-length
        with tf.device("/cpu:0"):
            return tf.sparse_tensor_to_dense(get_bag_vectors(model))


def dan_encoder(obs_hidden_units, avg_hidden_units):
    """
    Represents snapshots as a modified element-wise averages of embedded clinical observations
    :param obs_hidden_units: number of hidden units in dense layers between observation embeddings and average;
           if iterable multiple dense layers will be added using the respective hidden units
    :param avg_hidden_units: number of hidden units in dense layers between average embeddings and snapshot encoding;
           if iterable multiple dense layers will be added using the respective hidden units
    :return: clinical snapshot encoding
    """

    def _dan_encoder(model: CANTRIPModel):
        with tf.variable_scope('dan_encoder'):
            embedded_observations = embedding_layer(model.observations, model.vocabulary_size, model.embedding_size)

            # Reshape to (batch * seq_len * doc_len) x embedding
            flattened_embedded_observations = tf.reshape(
                embedded_observations,
                [model.batch_size * model.max_seq_len * model.max_snapshot_size,
                 model.embedding_size]
            )
            # Add dense observation layers
            # TODO: switch back to ReLU as described in the paper
            obs_layer = flattened_embedded_observations
            for num_hidden in obs_hidden_units:
                obs_layer = tf.layers.dense(obs_layer, num_hidden, tf.nn.tanh)

            # Reshape final output by grouping observations in the same snapshot together
            obs_layer = tf.reshape(obs_layer, [model.batch_size * model.max_seq_len,
                                               model.max_snapshot_size,
                                               obs_layer.shape[-1]])

            # Divide by active number of observations rather than the padded snapshot size; requires reshaping to
            # (batch x seq_len) x 1 so we can divide by this
            flattened_snapshot_sizes = tf.reshape(model.snapshot_sizes, [model.batch_size * model.max_seq_len, 1])

            # Compute dynamic-size element-wise average
            avg_layer = tf.reduce_mean(obs_layer, axis=1) / tf.to_float(tf.maximum(flattened_snapshot_sizes, 1))

            # More fun dense layers
            # TODO: switch back to ReLU as described in the paper
            for num_hidden in avg_hidden_units:
                avg_layer = tf.layers.dense(avg_layer, num_hidden, tf.nn.tanh)

            # Final output of the model
            output = tf.layers.dense(avg_layer, model.embedding_size, tf.nn.tanh)

            # Reshape to [batch_size x seq_len x encoding_size]
            return tf.reshape(output, [model.batch_size, model.max_seq_len, model.embedding_size])

    return _dan_encoder
