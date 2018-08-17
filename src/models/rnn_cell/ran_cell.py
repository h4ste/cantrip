"""Recurrent Additive Network memory cell implementations.
"""

import tensorflow as tf
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
            res = tf.contrib.layers.layer_norm(res)

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
        with vs.variable_scope(scope or "ran_cell", reuse=self._reuse):
            with vs.variable_scope("gates"):
                c, h = state
                gates = tf.nn.sigmoid(_linear([inputs, h], 2 * self._num_units, True, normalize=self._normalize))
                i, f = array_ops.split(value=gates, num_or_size_splits=2, axis=1)

            with vs.variable_scope("content"):
                content = _linear([inputs], self._num_units, True, normalize=self._normalize)

            new_c = i * content + f * c

            if self._activation:
                new_h = self._activation(new_c)
            else:
                new_h = c

            new_state = RANStateTuple(new_c, new_h)
            output = new_h
        return output, new_state
