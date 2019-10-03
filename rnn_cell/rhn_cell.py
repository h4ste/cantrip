import tensorflow as tf
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops.rnn_cell import RNNCell

from .ran_cell import _linear as linear


class RHNCell(RNNCell):
    """Variational Recurrent Highway Layer
    Reference: https://arxiv.org/abs/1607.03474
    """

    def __init__(self, num_units, in_size, is_training, keep_i=0.25, keep_h=.75, depth=3, forget_bias=None,
                 normalize=False):
        self._num_units = num_units
        self._in_size = in_size
        self.is_training = is_training
        self.depth = depth
        self.forget_bias = forget_bias
        self._keep_i = keep_i
        self._keep_h = keep_h
        self._normalize = normalize

    def zero_state(self, batch_size, dtype):
        state = super().zero_state(batch_size, dtype)

        assert type(batch_size) == list

        input_shape = tensor_shape.as_shape(batch_size + [tensor_shape.as_dimension(self.input_size)])
        if self._keep_i < 1.0:
            noise_i = tf.random.uniform(shape=input_shape, dtype=dtype) / self._keep_i
        else:
            noise_i = tf.ones(shape=input_shape, dtype=dtype)

        state_shape = tensor_shape.as_shape(batch_size + [tensor_shape.as_dimension(self._num_units)])
        if self._keep_h < 1.0:
            noise_h = tf.random.uniform(shape=state_shape, dtype=dtype) / self._keep_h
        else:
            noise_h = tf.ones(shape=state_shape, dtype=dtype)

        return [state, noise_i, noise_h]

    @property
    def input_size(self):
        return self._in_size

    @property
    def output_size(self):
        return self._num_units

    @property
    def state_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        current_state = state[0]
        noise_i = state[1]
        noise_h = state[2]
        for i in range(self.depth):
            with tf.variable_scope('h_' + str(i)):
                if i == 0:
                    h = tf.tanh(linear([inputs * noise_i, current_state * noise_h], self._num_units, True,
                                       normalize=self._normalize))
                else:
                    h = tf.tanh(linear([current_state * noise_h], self._num_units, True, normalize=self._normalize))
            with tf.variable_scope('t_' + str(i)):
                if i == 0:
                    t = tf.sigmoid(
                        linear([inputs * noise_i, current_state * noise_h], self._num_units, True, self.forget_bias,
                               normalize=self._normalize))
                else:
                    t = tf.sigmoid(linear([current_state * noise_h], self._num_units, True, self.forget_bias,
                                          normalize=self._normalize))
            current_state = (h - current_state) * t + current_state

        return current_state, [current_state, noise_i, noise_h]
