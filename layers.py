# import tensorflow.compat.v1 as tf
import typing

import numpy as np
import tensorflow.compat.v1 as tf

import rnn_cell


def create_embeddings(vocab_size, embedding_size, vocab_dropout, training):
    embeddings = tf.get_variable("embeddings", [vocab_size, embedding_size])
    if vocab_dropout > 0:
        # We use vocabulary-level dropout
        # Encourage the model not to depend on specific words in the vocabulary
        embeddings = tf.keras.layers.Dropout(rate=vocab_dropout,
                                             noise_shape=[1, embedding_size])(embeddings, training=training)
    return embeddings


def embedding_layer(inputs, vocab_size, embedding_size, vocab_dropout, training):
    with tf.device('/cpu:0'):
        embeddings = create_embeddings(vocab_size, embedding_size, vocab_dropout, training)
        return tf.nn.embedding_lookup(embeddings, inputs)


def dense_to_sparse(tensor):
    shape = tensor.get_shape()
    indices = tf.where(tf.not_equal(tensor, 0))
    values = tf.gather_nd(tensor, indices)
    return tf.SparseTensor(indices, values, dense_shape=shape)


def bidirectional_rnn_layer(cell_fn, num_hidden, inputs, lengths, return_interpretable_weights=False):
    # Check if we were given a list of hidden units
    stacked = isinstance(num_hidden, typing.Sequence)

    if stacked and len(num_hidden) == 1:
        num_hidden = num_hidden[0]
        stacked = False

    # Define type of RNN/memory cell
    if stacked:
        cell_fw = tf.nn.rnn_cell.MultiRNNCell([cell_fn(num_units) for num_units in num_hidden])
        cell_bw = tf.nn.rnn_cell.MultiRNNCell([cell_fn(num_units) for num_units in num_hidden])
    else:
        cell_fw = cell_fn(num_hidden)
        cell_bw = cell_fn(num_hidden)

    # Initialize memory-state as zero
    state_fw = cell_fw.zero_state([inputs.shape[0]], tf.float32)
    state_bw = cell_bw.zero_state([inputs.shape[0]], tf.float32)

    # Run dynamic RNN; discard all outputs except final state
    # noinspection PyUnresolvedReferences
    _, states = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw, cell_bw=cell_bw,
                                                inputs=inputs, sequence_length=lengths,
                                                initial_state_fw=state_fw,
                                                initial_state_bw=state_bw,
                                                swap_memory=True)

    # The final state of a dynamic RNN, in TensorFlow, is either
    # (1) an LSTMStateTuple containing the final output and final memory state, or
    # (2) just the final state output
    outputs = []
    for state in states:
        if stacked:
            state = state[-1]

        if isinstance(state, tf.nn.rnn_cell.LSTMStateTuple):
            outputs.append(state.h)

        elif isinstance(state, typing.Sequence):
            print('Peeling output from RHN tuple')
            outputs.append(state[0])

        else:
            outputs.append(state)

    return tf.concat(outputs, axis=-1)


def rnn_layer(cell_fn, num_hidden, inputs, lengths, return_interpretable_weights=False):
    # Check if we were given a list of hidden units
    stacked = isinstance(num_hidden, typing.Sequence)

    if stacked and len(num_hidden) == 1:
        num_hidden = num_hidden[0]
        stacked = False

    # Define type of RNN/memory cell
    if stacked:
        cell = tf.nn.rnn_cell.MultiRNNCell([cell_fn(num_units) for num_units in num_hidden])
    else:
        cell = cell_fn(num_hidden)

    # Initialize memory-state as zero
    state = cell.zero_state([inputs.shape[0]], tf.float32)

    # Run dynamic RNN; discard all outputs except final state
    # noinspection PyUnresolvedReferences
    _, state = tf.nn.dynamic_rnn(cell=cell, inputs=inputs, sequence_length=lengths, initial_state=state,
                                 swap_memory=True)

    # If we have stacked memory cells pop the state off the top of the stack
    if stacked:
        state = state[-1]

    # if return_interpretable_weights:
    #     if isinstance(state, rnn_cell.InterpretableRANStateTuple):
    #         return state.c, state.w
    #     else:
    #         raise TypeError('Can only interpret instances of InterpretableRANStateTuple')

    # The final state of a dynamic RNN, in TensorFlow, is either
    # (1) an LSTMStateTuple containing the final output and final memory state, or
    # (2) just the final state output
    if isinstance(state, tf.nn.rnn_cell.LSTMStateTuple):
        return state.h
    elif isinstance(state, typing.Sequence):
        print('Peeling output from RHN tuple')
        return state[0]
    else:
        return state


def gelu(x):
    """Gaussian Error Linear Unit.
    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415
    Args:
      x: float Tensor to perform activation.
    Returns:
      `x` with the GELU activation applied.
    """
    cdf = 0.5 * (1.0 + tf.tanh(
        (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf
