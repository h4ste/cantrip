import tensorflow as tf
from tensorflow.nn.rnn_cell import GRUCell

from src.models.layers import rnn_layer, embedding_layer, create_embeddings


def rnn_encoder(num_hidden):
    def _rnn_encoder(model):
        with tf.variable_scope('rnn_encoder'):
            embedded_words = embedding_layer(model.words, model.vocabulary_size, model.embedding_size)

            # Reshape to (batch * seq_len) x doc_len x embedding
            flattened_embedded_words = tf.reshape(embedded_words,
                                                  [model.batch_size * model.max_seq_len,
                                                   model.max_doc_len,
                                                   model.embedding_size],
                                                  name='flat_emb_words')
            flattened_doc_lengths = tf.reshape(model.doc_lengths, [model.batch_size * model.max_seq_len],
                                               name='flat_lengths')

            # Apply RNN to all documents in all batches
            flattened_doc_embeddings = rnn_layer(cell_fn=GRUCell,
                                                 num_hidden=num_hidden,
                                                 inputs=flattened_embedded_words,
                                                 lengths=flattened_doc_lengths)

            print('RNN Output:', flattened_doc_embeddings)
            print('RNN Output Type:', type(flattened_doc_embeddings))

            # Reshape back to (batch x seq_len x num_features)
            return tf.reshape(flattened_doc_embeddings,
                              [model.batch_size, model.max_seq_len, flattened_doc_embeddings.shape[-1]],
                              name='rnn_doc_embeddings')

    return _rnn_encoder


def cnn_encoder(ngram_filters=None, num_filters=1000, dropout=0.0):
    if ngram_filters is None:
        ngram_filters = [3, 4, 5]

    def _cnn_encoder(model):
        with tf.variable_scope('cnn_encoder'):
            embedded_words = embedding_layer(model.words, model.vocabulary_size, model.embedding_size)

            # Reshape to (batch * seq_len) x doc_len x embedding
            flattened_embedded_words = tf.reshape(embedded_words,
                                                  [model.batch_size * model.max_seq_len,
                                                   model.max_doc_len,
                                                   model.embedding_size])
            outputs = []
            for n in ngram_filters:
                dropout_layer = tf.nn.dropout(flattened_embedded_words, 1. - dropout)
                conv_layer = tf.layers.conv1d(dropout_layer, num_filters,
                                              kernel_size=n,
                                              activation=tf.nn.leaky_relu,
                                              name="conv_%dgram" % n)
                pool_layer = tf.layers.max_pooling1d(conv_layer, 1, model.max_doc_len - n + 1,
                                                     name="maxpool_%dgram" % n)
                outputs.append(pool_layer)

            embeddings = tf.layers.dense(tf.concat(outputs, axis=-1), model.embedding_size, activation=tf.nn.leaky_relu)

            return tf.reshape(embeddings, [model.batch_size, model.max_seq_len, model.embedding_size])

    return _cnn_encoder


def get_bow_vector(model):
    # 1. Evaluate which entries in model.words are non-zero
    mask = tf.not_equal(model.words, 0)
    where = tf.where(mask)

    print('Mask:', mask)
    print('Where:', where)
    print('Words:', model.words)
    # 2. Get the vocabulary indices for non-zero words
    vocab_indices = tf.boolean_mask(model.words, mask)
    vocab_indices = tf.expand_dims(vocab_indices[:], axis=-1)
    vocab_indices = tf.to_int64(vocab_indices)

    # 3. Get batch and sequence indices for non-zero words
    tensor_indices = where[:, :-1]

    # Concat batch, sequence, and vocabulary indices
    indices = tf.concat([tensor_indices, vocab_indices], axis=-1)

    # Our sparse tensor will be 1 for observed words, 0, otherwise
    ones = tf.ones_like(indices[:, 0], dtype=tf.float32)

    # The dense shape will be the same as model.words, but using the entire vocabulary as the final dimension
    dense_shape = model.words.get_shape().as_list()
    dense_shape[2] = model.vocabulary_size

    print('Indices:', indices)
    print('Values:', ones)
    print('Dense Shape:', dense_shape)

    st = tf.SparseTensor(indices=indices, values=ones, dense_shape=dense_shape)
    return tf.sparse_reorder(st)


def dense_encoder(model):
    with tf.variable_scope('dense_encoder'):
        with tf.device("/cpu:0"):
            bow_vectors = get_bow_vector(model)
            model.word_embeddings = create_embeddings(model.vocabulary_size, model.embedding_size, model.dropout)
            flat_bow_vectors = tf.sparse_reshape(bow_vectors, [model.batch_size * model.max_seq_len,
                                                               model.vocabulary_size],
                                                 name='flat_bow_vectors')
            # flat_bow_vectors = tf.nn.dropout(flat_bow_vectors, 0.5)
            flat_doc_embeddings = tf.sparse_tensor_dense_matmul(flat_bow_vectors, model.word_embeddings,
                                                                name='flat_doc_embeddings')
            flat_doc_embeddings = tf.nn.dropout(flat_doc_embeddings, 0.5)

        return tf.reshape(flat_doc_embeddings, [model.batch_size, model.max_seq_len, model.embedding_size],
                          name='doc_embeddings')


def bow_encoder(model):
    with tf.variable_scope('bow_encoder'):
        with tf.device("/cpu:0"):
            return tf.sparse_tensor_to_dense(get_bow_vector(model))


def dan_encoder(word_hidden_units, doc_hidden_units):
    def _dan_encoder(model):
        with tf.variable_scope('dan_encoder'):
            embedded_words = embedding_layer(model.words, model.vocabulary_size, model.embedding_size)

            batch_size = embedded_words.shape[0]

            # Reshape to (batch * seq_len * doc_len) x embedding
            flattened_embedded_words = tf.reshape(embedded_words,
                                                  [batch_size * model.max_seq_len * model.max_doc_len,
                                                   model.embedding_size])

            flattened_doc_lengths = tf.reshape(model.doc_lengths, [batch_size * model.max_seq_len, 1])

            word_layer = flattened_embedded_words
            for num_hidden in word_hidden_units:
                word_layer = tf.layers.dense(word_layer, num_hidden, tf.nn.tanh)

            word_layer = tf.reshape(word_layer, [batch_size * model.max_seq_len,
                                                 model.max_doc_len,
                                                 word_layer.shape[-1]])

            avg_layer = tf.reduce_mean(word_layer, axis=1) / tf.to_float(tf.maximum(flattened_doc_lengths, 1))

            for num_hidden in doc_hidden_units:
                avg_layer = tf.layers.dense(avg_layer, num_hidden, tf.nn.tanh)

            output = tf.layers.dense(avg_layer, model.embedding_size, tf.nn.tanh)

            return tf.reshape(output, [batch_size, model.max_seq_len, model.embedding_size])

    return _dan_encoder
