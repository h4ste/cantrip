import tensorflow as tf

from tensorflow.nn.rnn_cell import LSTMStateTuple, MultiRNNCell

from src.models.rnn_cell.ran_cell import InterpretableRANStateTuple


def create_embeddings(vocab_size, embedding_size, dropout, training=False):
    embeddings = tf.get_variable("embeddings", [vocab_size, embedding_size])
    if dropout > 0:
        # We use vocabulary-level dropout
        # Encourage the model not to depend on specific words in the vocabulary
        embeddings = tf.layers.dropout(embeddings, rate=dropout, noise_shape=[1, embedding_size], training=training)
    return embeddings


def embedding_layer(inputs, vocab_size, embedding_size, dropout=0):
    with tf.device('/cpu:0'):
        embeddings = create_embeddings(vocab_size, embedding_size, dropout)
        return tf.nn.embedding_lookup(embeddings, inputs)


def dense_to_sparse(tensor):
    shape = tensor.get_shape()
    indices = tf.where(tf.not_equal(tensor, 0))
    values = tf.gather_nd(tensor, indices)
    return tf.SparseTensor(indices, values, dense_shape=shape)


def rnn_layer(cell_fn, num_hidden, inputs, lengths, return_interpretable_weights):
    # Check if we were given a list of hidden units
    stacked = isinstance(num_hidden, list)

    # Define type of RNN/memory cell
    if stacked:
        cell = MultiRNNCell([cell_fn(num_units) for num_units in num_hidden])
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

    if return_interpretable_weights:
        if isinstance(state, InterpretableRANStateTuple):
            return state.c, state.w
        else:
            raise TypeError('Can only interpret instances of InterpretableRANStateTuple')

    # The final state of a dynamic RNN, in TensorFlow, is either
    # (1) an LSTMStateTuple containing the final output and final memory state, or
    # (2) just the final state output
    if isinstance(state, LSTMStateTuple):
        return state.h
    else:
        return state
