from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
# import tensorflow.compat.v1 as tf
import tensorflow as tf

from tensorflow.python.ops.rnn_cell_impl import RNNCell


# from tensorflow import contrib as tfcontrib

class LayerNormGRUCell(RNNCell):

    def __init__(
            self, size, activation=tf.tanh, reuse=None,
            normalizer=tf.keras.layers.LayerNormalization(),
            initializer=tf.initializers.glorot_uniform()):
        super(LayerNormGRUCell, self).__init__(_reuse=reuse)
        self._size = size
        self._activation = activation
        self._normalizer = normalizer
        self._initializer = initializer

    @property
    def state_size(self):
        return self._size

    @property
    def output_size(self):
        return self._size

    def __call__(self, input_, state, scope=None):
        update, reset = tf.split(self._forward(
            'update_reset', [state, input_], 2 * self._size, tf.nn.sigmoid,
            bias_initializer=tf.constant_initializer(-1.)), 2, 1)
        candidate = self._forward(
            'candidate', [reset * state, input_], self._size, self._activation)
        state = (1 - update) * state + update * candidate
        return state, state

    def _forward(self, name, inputs, size, activation, **kwargs):
        with tf.variable_scope(name):
            return _forward(
                inputs, size, activation, normalizer=self._normalizer,
                weight_initializer=self._initializer, **kwargs)


def _forward(
        inputs, size, activation, normalizer=tf.keras.layers.LayerNormalization(),
        weight_initializer=tf.initializers.glorot_uniform(),
        bias_initializer=tf.zeros_initializer()):
    if not isinstance(inputs, (tuple, list)):
        inputs = (inputs,)
    shapes = []
    outputs = []
    # Map each input to individually normalize their outputs.
    for index, input_ in enumerate(inputs):
        shapes.append(input_.shape[1: -1].as_list())
        input_ = tf.keras.layers.Flatten()(input_)
        weight = tf.get_variable(
            'weight_{}'.format(index + 1), (int(input_.shape[1]), size),
            tf.float32, weight_initializer)
        output = tf.matmul(input_, weight)
        if normalizer:
            output = normalizer(output)
        outputs.append(output)
    output = tf.reduce_mean(outputs, 0)
    # Add bias after normalization.
    bias = tf.get_variable(
        'weight', (size,), tf.float32, bias_initializer)
    output += bias
    # Activation function.
    if activation:
        output = activation(output)
    # Restore shape dimensions that are consistent among inputs.
    min_dim = min(len(shape[1:]) for shape in shapes)
    dim_shapes = [[shape[dim] for shape in shapes] for dim in range(min_dim)]
    matching_dims = ''.join('NY'[len(set(x)) == 1] for x in dim_shapes) + 'N'
    agreement = matching_dims.index('N')
    remaining = sum(np.prod(shape[agreement:]) for shape in shapes)
    if agreement:
        batch_size = output.shape[0].value or -1
        shape = [batch_size] + shapes[:agreement] + [remaining]
        output = tf.reshape(output, shape)
    return output
