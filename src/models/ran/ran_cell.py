import collections
from typing import Iterable, Tuple, Any, Iterator

import tensorflow as tf
from tensorflow.python.ops.rnn_cell_impl import RNNCell, LSTMStateTuple

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope as vs

from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"


# def _checked_scope(cell, scope, reuse=None, **kwargs):
#     if reuse is not None:
#         kwargs["reuse"] = reuse
#     with vs.variable_scope(scope, **kwargs) as checking_scope:
#         scope_name = checking_scope.name
#         if hasattr(cell, "_scope"):
#             cell_scope = cell._scope  # pylint: disable=protected-access
#             if cell_scope.name != checking_scope.name:
#                 raise ValueError(
#                     "Attempt to reuse RNNCell %s with a different variable scope than "
#                     "its first use.  First use of cell was with scope '%s', this "
#                     "attempt is with scope '%s'.  Please create a new instance of the "
#                     "cell if you would like it to use a different set of weights.  "
#                     "If before you were using: MultiRNNCell([%s(...)] * num_layers), "
#                     "change to: MultiRNNCell([%s(...) for _ in range(num_layers)]).  "
#                     "If before you were using the same cell instance as both the "
#                     "forward and reverse cell of a bidirectional RNN, simply create "
#                     "two instances (one for forward, one for reverse).  "
#                     "In May 2017, we will start transitioning this cell's behavior "
#                     "to use existing stored weights, if any, when it is called "
#                     "with scope=None (which can lead to silent model degradation, so "
#                     "this error will remain until then.)"
#                     % (cell, cell_scope.name, scope_name, type(cell).__name__,
#                        type(cell).__name__))
#         else:
#             weights_found = False
#             try:
#                 with vs.variable_scope(checking_scope, reuse=True):
#                     vs.get_variable(_WEIGHTS_VARIABLE_NAME)
#                 weights_found = True
#             except ValueError:
#                 pass
#             if weights_found and reuse is None:
#                 raise ValueError(
#                     "Attempt to have a second RNNCell use the weights of a variable "
#                     "scope that already has weights: '%s'; and the cell was not "
#                     "constructed as %s(..., reuse=True).  "
#                     "To share the weights of an RNNCell, simply "
#                     "reuse it in your second calculation, or create a new one with "
#                     "the argument reuse=True." % (scope_name, type(cell).__name__))
#
#         # Everything is OK.  Update the cell's scope and yield it.
#         cell._scope = checking_scope  # pylint: disable=protected-access
#         return checking_scope


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


_RANStateTuple = collections.namedtuple('RANStateTuple', ('h', 'c', 'w'))


class RANStateTuple(_RANStateTuple):
    __slots__ = ()


    @property
    def dtype(self):
        (c, h) = self
        if c.dtype != h.dtype:
            raise TypeError("Inconsistent internal state: %s vs %s" %
                            (str(c.dtype), str(h.dtype)))
        return c.dtype


# class Test(collections.namedtuple('a', 'b', 'c')):
#
#     def __iter__(self) -> Iterator[_T_co]:
#         return super().__iter__()


# class RANStateTuple(object):
#
#     __slots__ = ['h', 'c', 'w']
#
#     def __init__(self, h, c, w=None) -> None:
#         self.h = h,
#         self.c = c
#         self.w = w
#
#     def __iter__(self):
#         t = (self.h, self.c)
#         return t.__iter__()
#
#     @property
#     def dtype(self):
#         (c, h) = self
#         if c.dtype != h.dtype:
#             raise TypeError("Inconsistent internal state: %s vs %s" %
#                             (str(c.dtype), str(h.dtype)))
#         return c.dtype





# noinspection PyAbstractClass,PyMissingConstructor
class RANCell(RNNCell):
    """Recurrent Additive Networks (cf. https://arxiv.org/abs/1705.07393)."""

    def __init__(self, num_units, input_size=None, activation=math_ops.tanh, normalize=False, reuse=None):
        if input_size is not None:
            logging.warn("%s: The input_size parameter is deprecated.", self)
        self._num_units = num_units
        self._activation = activation
        self._normalize = normalize
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

            with vs.variable_scope("candidate"):
                c = _linear([inputs], self._num_units, True, normalize=self._normalize)

            new_c = i * c + f * state
            new_h = self._activation(c)

        return new_h, new_c


# noinspection PyAbstractClass,PyMissingConstructor
class RANCellv2(RNNCell):
    """Recurrent Additive Networks (cf. https://arxiv.org/abs/1705.07393)."""

    def __init__(self, num_units, input_size, activation=math_ops.tanh, normalize=False, reuse=None):
        self._input_size = input_size
        self._num_units = num_units
        self._activation = activation
        self._normalize = normalize
        self._reuse = reuse

    @property
    def state_size(self):
        return RANStateTuple(self._num_units, self.output_size, self._input_size)

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        with vs.variable_scope(scope or "ran_cell", reuse=self._reuse):
            with vs.variable_scope("gates"):
                c, h, w = state
                gates = tf.nn.sigmoid(_linear([inputs, h], 2 * self._num_units, True, normalize=self._normalize))
                i, f = array_ops.split(value=gates, num_or_size_splits=2, axis=1)

            with vs.variable_scope("candidate") as scope:
                content = _linear([inputs], self._num_units, True, normalize=self._normalize)
                scope.reuse_variables()
                weights = tf.get_variable('kernel')

            new_c = i * content + f * c

            if self._activation:
                new_h = self._activation(c)
            else:
                new_h = c

            component_weights = tf.matmul((new_c / content), tf.transpose(weights))
            new_state = RANStateTuple(new_c, new_h, component_weights)
            output = new_h
        return output, new_state


class RANCellv3(RNNCell):
    """Recurrent Additive Networks (cf. https://arxiv.org/abs/1705.07393)."""

    def __init__(self, input_size, reuse=None):
        self._input_size = input_size
        self._reuse = reuse

    @property
    def state_size(self):
        return RANStateTuple(self._input_size, self._input_size, self._input_size)

    @property
    def output_size(self):
        return self._input_size

    def __call__(self, inputs, state, scope=None):
        with vs.variable_scope(scope or "ran_cell", reuse=self._reuse):
            with vs.variable_scope("gates"):
                c, h, w = state
                gates = tf.nn.sigmoid(_linear([inputs, h], 2 * self._input_size, True))
                i, f = array_ops.split(value=gates, num_or_size_splits=2, axis=1)

            new_c = i * inputs + f * c
            new_h = new_c

            component_weights = new_c / inputs
            new_state = RANStateTuple(new_c, new_h, component_weights)
            output = new_h
        return output, new_state
