import tensorflow as tf

from tensorflow.nn.rnn_cell import LSTMStateTuple, MultiRNNCell

from src.models.ran import RANStateTuple


def create_embeddings(vocab_size, embedding_size, dropout):
    embeddings = tf.get_variable("embeddings", [vocab_size, embedding_size])
    if dropout > 0:
        # We use vocabulary-level dropout
        # Encourage the model not to depend on specific words in the vocabulary
        embeddings = tf.nn.dropout(embeddings, keep_prob=1 - dropout, noise_shape=[1, embedding_size])
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


def rnn_layer(cell_fn, num_hidden, inputs, lengths, return_input_weights=False):
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
    output, state = tf.nn.dynamic_rnn(cell=cell, inputs=inputs, sequence_length=lengths, initial_state=state,
                                      swap_memory=True)

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

    if not isinstance(state, RANStateTuple) and return_input_weights:
        return ValueError('can only return input weights from RANv2 cells')

    if isinstance(state, LSTMStateTuple):
        return state.h
    elif isinstance(state, RANStateTuple):
        if return_input_weights:
            print('Weights:', state.w)
            return state.h, state.w
        else:
            return state.h
    else:
        return state
