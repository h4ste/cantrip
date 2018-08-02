import tensorflow as tf

from tensorflow.nn.rnn_cell import LSTMStateTuple, MultiRNNCell


def embedding_layer(inputs, vocab_size, embedding_size):
    with tf.device('/cpu:0'):
        embeddings = tf.get_variable("embeddings", [vocab_size, embedding_size])
        return tf.nn.embedding_lookup(embeddings, inputs)


def dense_to_sparse(tensor):
    shape = tensor.get_shape()
    indices = tf.where(tf.not_equal(tensor, 0))
    values = tf.gather_nd(tensor, indices)
    return tf.SparseTensor(indices, values, dense_shape=shape)


def rnn_layer(cell_fn, num_hidden, inputs, lengths):
    stacked = isinstance(num_hidden, list)

    # Define type of RNN/memory cell
    if stacked:
        cell = MultiRNNCell([cell_fn(num_units) for num_units in num_hidden])
    else:
        cell = cell_fn(num_hidden)

    # Initialize memory-state as zero

    # if isinstance(inputs, tf.SparseTensor):
    #     print('Getting sparse batch size for', inputs)
    #     print('Input Shape:', inputs.get_shape())
    #     batch_size = inputs.get_shape()[0]
    # else:
    #     batch_size = inputs.shape[0]

    state = cell.zero_state([inputs.shape[0]], tf.float32)

    print('RNN Cell:', cell)
    print('RNN Initial State:', state)

    # Run dynamic RNN; discard all outputs except final state
    output, state = tf.nn.dynamic_rnn(cell=cell, inputs=inputs, sequence_length=lengths, initial_state=state)

    print('RNN Final State:', state)
    print('RNN Final State Type:', type(state))

    print('RNN Output:', output)

    if stacked:
        state = state[-1]

    print('RNN Final State Last State:', state)
    print('RNN Final State Last State Type:', type(state))

    # The final state of a dynamic RNN, in TensorFlow, is either
    # (1) a tuple containing the final output and final memory state, or
    # (2) the final output


    if isinstance(state, LSTMStateTuple):
        final_output = state.h
    else:
        final_output = state

    return final_output
