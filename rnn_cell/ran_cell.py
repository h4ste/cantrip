"""Recurrent Additive Network memory cell implementations.
"""
import numpy as np

import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.framework import ops, tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.rnn_cell_impl import RNNCell, LSTMStateTuple
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"


def _linear(args,
            output_size,
            bias,
            bias_initializer=None,
            kernel_initializer=None,
            kernel_regularizer=None,
            bias_regularizer=None,
            normalize=False):
    """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
       Args:
         args: a 2D Tensor or a list of 2D, batch x n, Tensors.
         output_size: int, second dimension of W[i].
         bias: boolean, whether to add a bias term or not.
         bias_initializer: starting value to initialize the bias
           (default is all zeros).
         kernel_initializer: starting value to initialize the weight.
         kernel_regularizer: kernel regularizer
         bias_regularizer: bias regularizer
       Returns:
         A 2D Tensor with shape [batch x output_size] equal to
        sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
       Raises:
         ValueError: if some of the arguments has unspecified or wrong shape.
      """
    if args is None or (nest.is_sequence(args) and not args):
        raise ValueError("`args` must be specified")
    if not nest.is_sequence(args):
        args = [args]

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape() for a in args]
    for shape in shapes:
        if shape.ndims != 2:
            raise ValueError("linear is expecting 2D arguments: %s" % shapes)
        if shape[1].value is None:
            raise ValueError("linear expects shape[1] to be provided for shape %s, "
                             "but saw %s" % (shape, shape[1]))
        else:
            total_arg_size += shape[1].value

    dtype = [a.dtype for a in args][0]

    # Now the computation.
    scope = vs.get_variable_scope()
    with vs.variable_scope(scope) as outer_scope:
        weights = vs.get_variable(
            _WEIGHTS_VARIABLE_NAME, [total_arg_size, output_size],
            dtype=dtype,
            initializer=kernel_initializer,
            regularizer=kernel_regularizer)

        if len(args) == 1:
            res = math_ops.matmul(args[0], weights)
        else:
            res = math_ops.matmul(array_ops.concat(args, 1), weights)

        if normalize:
            res = tf.keras.layers.LayerNormalization()(res)

        # remove the layer’s bias if there is one (because it would be redundant)
        if not bias or normalize:
            return res

        with vs.variable_scope(outer_scope) as inner_scope:
            inner_scope.set_partitioner(None)
            if bias_initializer is None:
                bias_initializer = init_ops.constant_initializer(0.0, dtype=dtype)
            biases = vs.get_variable(
                _BIAS_VARIABLE_NAME, [output_size],
                dtype=dtype,
                initializer=bias_initializer,
                regularizer=bias_regularizer)

    return nn_ops.bias_add(res, biases)


# noinspection PyAbstractClass,PyMissingConstructor
class SimpleRANCell(RNNCell):
    """Recurrent Additive Networks (cf. https://arxiv.org/abs/1705.07393).

    This is an implementation of the simplified RAN cell described in Equation group (2)."""

    def __init__(self, num_units, input_size=None, normalize=False, reuse=None):
        if input_size is not None:
            logging.warn("%s: The input_size parameter is deprecated.", self)
        self._normalize = normalize
        self._num_units = num_units
        self._reuse = reuse

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        with vs.variable_scope(scope or "ran_cell", reuse=self._reuse):
            with vs.variable_scope("gates"):
                value = tf.nn.sigmoid(_linear([state, inputs], 2 * self._num_units, True, normalize=self._normalize))
                i, f = array_ops.split(value=value, num_or_size_splits=2, axis=1)

            new_c = i * inputs + f * state

        return new_c, new_c


class RANStateTuple(LSTMStateTuple):
    pass


# noinspection PyAbstractClass,PyMissingConstructor
class RANCell(RNNCell):
    """Recurrent Additive Networks (cf. https://arxiv.org/abs/1705.07393)

    This is an implementation of the standard RAN cell described in Equation group (1)."""

    def __init__(self, num_units, input_size=None, activation=math_ops.tanh, normalize=False, reuse=None):
        if input_size is not None:
            logging.warn("%s: The input_size parameter is deprecated.", self)
        self._num_units = num_units
        self._activation = activation
        self._normalize = normalize
        self._reuse = reuse

    @property
    def state_size(self):
        return RANStateTuple(self._num_units, self.output_size)

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        with vs.variable_scope(scope or "ran_cell", reuse=self._reuse) as scope:
            with vs.variable_scope("gates"):
                c, h = state
                gates = tf.nn.sigmoid(_linear([inputs, h], 2 * self._num_units, True, normalize=self._normalize,
                                              kernel_regularizer=scope.regularizer))
                i, f = array_ops.split(value=gates, num_or_size_splits=2, axis=1)

            with vs.variable_scope("content"):
                content = _linear([inputs], self._num_units, True, normalize=self._normalize,
                                  kernel_regularizer=scope.regularizer)

            new_c = i * content + f * c

            if self._activation:
                new_h = self._activation(new_c)
            else:
                new_h = c

            new_state = RANStateTuple(new_c, new_h)
            output = new_h
        return output, new_state


# noinspection PyAbstractClass,PyMissingConstructor
class VHRANCell(RNNCell):
    """Variational Highway variant of a RAN CELL

    This is an implementation of the standard RAN cell described in Equation group (1)."""

    def __init__(self, num_units, input_size, keep_i=0.25, keep_h=.75, depth=8, activation=math_ops.tanh,
                 normalize=False, reuse=None, forget_bias=None):
        self.input_size = input_size
        self._num_units = num_units
        self.forget_bias = forget_bias

        self._activation = activation
        self._normalize = normalize
        self._reuse = reuse

        self._keep_i = keep_i
        self._keep_h = keep_h
        self._depth = depth

    def zero_state(self, batch_size, dtype):
        state = super().zero_state(batch_size, dtype)

        assert type(batch_size) == list

        input_shape = tensor_shape.as_shape(batch_size + [tensor_shape.as_dimension(self.input_size)])
        if self._keep_i < 1.0:
            noise_i = tf.random.uniform(shape=input_shape, dtype=tf.float32) / self._keep_i
        else:
            noise_i = tf.ones(shape=input_shape, dtype=tf.float32)

        state_shape = tensor_shape.as_shape(batch_size + [tensor_shape.as_dimension(self._num_units)])
        if self._keep_h < 1.0:
            noise_h = tf.random.uniform(shape=state_shape, dtype=tf.float32) / self._keep_h
        else:
            noise_h = tf.ones(shape=state_shape, dtype=tf.float32)

        return [state, noise_i, noise_h]

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        current_state = state[0]
        noise_i = state[1]
        noise_h = state[2]
        with vs.variable_scope(scope or "ran_cell", reuse=self._reuse) as scope:
            for i in range(self._depth):
                with tf.variable_scope('h_' + str(i)):
                    if i == 0:
                        h = _linear([inputs * noise_i], self._num_units, True, normalize=self._normalize)
                with tf.variable_scope('t_' + str(i)):
                    if i == 0:
                        t = tf.sigmoid(_linear([inputs * noise_i, current_state * noise_h], self._num_units, True,
                                               self.forget_bias, normalize=self._normalize))
                    else:
                        t = tf.sigmoid(_linear([current_state * noise_h], self._num_units, True, self.forget_bias,
                                               normalize=self._normalize))
                current_state = (h - current_state) * t + current_state

            return current_state, [current_state, noise_i, noise_h]


class InterpretableRANStateTuple(object):
    __slots__ = ('c', '_i_list', '_f_list', 'w')

    def __init__(self, c, i_list, f_list, w_list):
        self.c = c
        self._i_list = i_list
        self._f_list = f_list
        self.w = w_list

    @classmethod
    def zero(cls, batch_size, state_size, dtype):
        return InterpretableRANStateTuple(
            c=tf.zeros([batch_size, state_size], dtype=dtype),
            i_list=[],
            f_list=[],
            w_list=[]
        )

    @classmethod
    def succeed(cls, c, i, f, prev: 'InterpretableRANStateTuple'):
        f_list = prev._f_list + [f]
        i_list = prev._i_list + [i]
        return InterpretableRANStateTuple(
            c=c,
            f_list=f_list,
            i_list=i_list,
            w_list=prev.w + InterpretableRANStateTuple._calc_weights(f=f_list, i=i_list)
        )

    @staticmethod
    def _calc_weights(i, f):
        t = len(i)
        w = np.zeros(t)
        for j in range(t):
            w[j] = i[j]
            for k in range(j + 1, t):
                w[j] *= f[k]
        return w

    @property
    def dtype(self):
        return self.c.dtype


# noinspection PyAbstractClass,PyMissingConstructor
class InterpretableSimpleRANCell(RNNCell):
    """Recurrent Additive Networks (cf. https://arxiv.org/abs/1705.07393).

    This is an implementation of the simplified RAN cell described in Equation group (2)."""

    def __init__(self, num_units, input_size=None, normalize=False, reuse=None):
        if input_size is not None:
            logging.warn("%s: The input_size parameter is deprecated.", self)
        self._normalize = normalize
        self._num_units = num_units
        self._reuse = reuse

    def zero_state(self, batch_size, dtype):
        state_size = self._num_units
        is_eager = context.executing_eagerly()
        if is_eager and hasattr(self, "_last_zero_state"):
            (last_state_size, last_batch_size, last_dtype,
             last_output) = getattr(self, "_last_zero_state")
            if (last_batch_size == batch_size and
                    last_dtype == dtype and
                    last_state_size == state_size):
                return last_output

        with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
            output = InterpretableRANStateTuple.zero(batch_size, state_size, dtype)
        if is_eager:
            # noinspection PyAttributeOutsideInit
            self._last_zero_state = (state_size, batch_size, dtype, output)
        return output

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state: InterpretableRANStateTuple, scope=None):
        with vs.variable_scope(scope or "ran_cell", reuse=self._reuse):
            with vs.variable_scope("gates"):
                c = state.c
                value = tf.nn.sigmoid(_linear([c, inputs], 2 * self._num_units, True, normalize=self._normalize))
                i, f = array_ops.split(value=value, num_or_size_splits=2, axis=1)

            new_c = i * inputs + f * c

        return InterpretableRANStateTuple.succeed(new_c, i, f, state)
