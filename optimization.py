import re
from typing import Union

import tensorflow.compat.v1 as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops, control_flow_ops
# noinspection PyProtectedMember
from tensorflow.python.ops.losses.losses_impl import _remove_squeezable_dimensions

import modeling


class CANTRIPOptimizer(object):

    def __init__(self, model, sparse, learning_rate):
        """
        Creates a new CANTRIPOptimizer responsible for optimizing CANTRIP. Allegedly, some day I will get around to
        looking at other optimization strategies (e.g., sequence optimization).
        :param model: a CANTRIPModel object
        :type model:  modeling.CANTRIPModel
        :param sparse: whether to use sparse softmax or not
        :type sparse: bool
        :param learning_rate: learning rate of the optimizer
        :type learning_rate: float
        """
        self.model = model

        # If sparse calculate sparse softmax directly from integer labels
        if sparse:
            self.loss = tf.losses.sparse_softmax_cross_entropy(model.labels, model.logits)

        # If not sparse, convert labels to one hots and do softmax
        else:
            y_true = tf.one_hot(model.labels, model.num_classes, name='labels_onehot')
            self.loss = tf.losses.softmax_cross_entropy(y_true, model.logits)

        # Global step used for coordinating summarizes and checkpointing
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

        # Training operation: fetch this to run a step of the adam optimizer!
        self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss, self.global_step)


class BERTOptimizer(object):

    def __init__(self, model, num_train_steps, steps_per_epoch, num_warmup_steps=None, init_lr=1e-3,
                 lr_decay=True, l2_reg=False, l1_reg=True, clip_norm=1.0, weights=1,
                 normalize_weights=False, focal_loss=False):
        """
        Creates a new CANTRIPOptimizer responsible for optimizing CANTRIP. Allegedly, some day I will get around to
        looking at other optimization strategies (e.g., sequence optimization).
        :param model: a CANTRIPModel object
        :type model: modeling.CANTRIPModel
        :param num_train_steps: number of training steps between decay steps
        :type num_train_steps: int
        :param num_warmup_steps: number of training steps before starting decay
        :type num_warmup_steps: int
        :param init_lr: initial learning rate of the optimizer
        :type init_lr: float
        """
        self.model = model

        # Global step used for coordinating summarizes and checkpointing
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False,
                                       name='global_step')  # type: Union[int, tf.Variable]

        learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)

        # Implements linear decay of learning rate
        if lr_decay:
            learning_rate = polynomial_decay(
                learning_rate,
                self.global_step,
                num_train_steps,
                cycle=True  # False
            )

        # Implements linear warm up. I.e., if global_step < num_warmup_steps, the
        # learning rate will be `global_step/num_warmup_steps * init_lr`.
        if num_warmup_steps:
            global_steps_int = tf.cast(self.global_step, tf.int32)
            warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)

            global_steps_float = tf.cast(global_steps_int, tf.float32)
            warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

            warmup_percent_done = global_steps_float / warmup_steps_float
            warmup_learning_rate = init_lr * warmup_percent_done

            is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
            learning_rate = (
                    (1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)

        if weights != 1:
            assert isinstance(weights, list)
            self.class_weights = tf.constant(weights, dtype=tf.float32, shape=[len(weights), 1])
            if normalize_weights:
                self.class_weights = self.class_weights / tf.reduce_sum(self.class_weights)
            weights = tf.nn.embedding_lookup(self.class_weights, model.labels)

        if focal_loss:
            def sparse_focal_softmax_cross_entropy(labels, logits, weights=1., alpha=1, gamma=2.0,
                                                   scope=None,
                                                   loss_collection=tf.GraphKeys.LOSSES,
                                                   reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS):
                if labels is None:
                    raise ValueError("labels must not be None.")
                if logits is None:
                    raise ValueError("logits must not be None.")
                with tf.name_scope(scope, "focal_softmax_cross_entropy_loss", (logits, labels, weights)) as scope:
                    # noinspection PyProtectedMember
                    labels, logits, weights = _remove_squeezable_dimensions(
                        labels, logits, weights, expected_rank_diff=1)
                    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                                            logits=logits,
                                                                            name='xentropy')
                    predictions = tf.nn.softmax(logits)
                    predictions_pt = tf.squeeze(
                        tf.gather(predictions,
                                  tf.expand_dims(model.labels, axis=-1),
                                  batch_dims=1)
                    )
                    # add small value to avoid 0
                    print('Losses:', losses)
                    print('Predictions:', predictions)
                    print('Predictions Pt:', predictions_pt)
                    print('Labels:', labels)
                    print('(1 - Pt)^gamma:', tf.pow(1 - predictions_pt, gamma))
                    print('Weights:', weights)
                    focal_weights = tf.pow(1 - predictions_pt, gamma) * alpha * weights

                    return tf.losses.compute_weighted_loss(
                        losses, focal_weights, scope, loss_collection, reduction=reduction)

            self.loss = sparse_focal_softmax_cross_entropy(model.labels, model.logits, weights=weights)
        else:
            self.loss = tf.losses.sparse_softmax_cross_entropy(model.labels, model.logits, weights=weights)

        tvars = tf.trainable_variables()
        grads = tf.gradients(self.loss, tvars)

        if clip_norm:
            (grads, _) = tf.clip_by_global_norm(grads, clip_norm=clip_norm)

        optimizer = AdamWeightDecayOptimizer(
            learning_rate=learning_rate,
            l1_weight=.1 / steps_per_epoch if l1_reg else 0.0,
            l2_weight=.1 / steps_per_epoch if l2_reg else 0.0,
            weight_decay_rate=.1,
            beta_1=.9,
            beta_2=0.999,
            epsilon=1e-8,
            regularize=["snapshot_encoder"],
            exclude_from_l1_reg=["LayerNorm", "layer_norm", "bias", "Bias"],
            exclude_from_l2_reg=["LayerNorm", "layer_norm", "bias", "Bias"],
            exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias", "Bias"]
        )

        train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)

        new_global_step = self.global_step + 1
        self.train_op = tf.group(train_op, [self.global_step.assign(new_global_step)])


# noinspection PyAbstractClass
class AdamWeightDecayOptimizer(tf.train.Optimizer):
    """A basic Adam optimizer that includes "correct" L2 weight decay."""

    def __init__(self,
                 learning_rate,
                 l1_weight=0.0,
                 l2_weight=0.0,
                 weight_decay_rate=0.0,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-8,
                 exclude_from_weight_decay=None,
                 exclude_from_l1_reg=None,
                 exclude_from_l2_reg=None,
                 name="AdamWeightDecayOptimizer",
                 regularize=None):
        """Constructs a AdamWeightDecayOptimizer."""
        super(AdamWeightDecayOptimizer, self).__init__(False, name)

        self.learning_rate = learning_rate
        self.l1_param = l1_weight
        self.l2_param = l2_weight
        self.weight_decay_rate = weight_decay_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.exclude_from_weight_decay = exclude_from_weight_decay
        self.exclude_from_l1_reg = exclude_from_l1_reg or exclude_from_weight_decay
        self.exclude_from_l2_reg = exclude_from_l2_reg or exclude_from_weight_decay
        self.regularize = regularize

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        """See base class."""
        assignments = []
        for (grad, param) in grads_and_vars:
            if grad is None or param is None:
                continue

            param_name = self._get_variable_name(param.name)

            if self._do_use_l2_reg(param_name):
                grad += self.l2_param * param

            if self._do_use_l1_reg(param_name):
                grad += self.l1_param * tf.sign(param)

            m = tf.get_variable(
                name=param_name + "/adam_m",
                shape=param.shape.as_list(),
                dtype=tf.float32,
                trainable=False,
                initializer=tf.zeros_initializer())
            v = tf.get_variable(
                name=param_name + "/adam_v",
                shape=param.shape.as_list(),
                dtype=tf.float32,
                trainable=False,
                initializer=tf.zeros_initializer())

            # Standard Adam update.
            next_m = (
                    tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, grad))
            next_v = (
                    tf.multiply(self.beta_2, v) + tf.multiply(1.0 - self.beta_2,
                                                              tf.square(grad)))

            update = next_m / (tf.sqrt(next_v) + self.epsilon)

            # Just adding the square of the weights to the loss function is *not*
            # the correct way of using L2 regularization/weight decay with Adam,
            # since that will interact with the m and v parameters in strange ways.
            #
            # Instead we want ot decay the weights in a manner that doesn't interact
            # with the m/v parameters. This is equivalent to adding the square
            # of the weights to the loss with plain (non-momentum) SGD.
            if self._do_use_weight_decay(param_name):
                update += self.weight_decay_rate * param

            update_with_lr = self.learning_rate * update

            next_param = param - update_with_lr

            assignments.extend(
                [param.assign(next_param),
                 m.assign(next_m),
                 v.assign(next_v)])
        return tf.group(*assignments, name=name)

    def _do_use_l2_reg(self, param_name):
        """Whether to use L2 regularization for `param_name`."""
        if not self.l2_param:
            return False
        if self.exclude_from_l2_reg:
            for r in self.exclude_from_l2_reg:
                if re.search(r, param_name) is not None:
                    return False
        if self.regularize:
            for r in self.regularize:
                if re.search(r, param_name) is not None:
                    return True
        else:
            return True

    def _do_use_l1_reg(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if not self.l1_param:
            return False
        if self.exclude_from_l1_reg:
            for r in self.exclude_from_l1_reg:
                if re.search(r, param_name) is not None:
                    return False
        if self.regularize:
            for r in self.regularize:
                if re.search(r, param_name) is not None:
                    return True
        else:
            return True

    def _do_use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if not self.weight_decay_rate:
            return False
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        if self.regularize:
            for r in self.regularize:
                if re.search(r, param_name) is not None:
                    return True
        else:
            return True

    # noinspection PyMethodMayBeStatic
    def _get_variable_name(self, param_name):
        """Get the variable name from the tensor name."""
        m = re.match("^(.*):\\d+$", param_name)
        if m is not None:
            param_name = m.group(1)
        return param_name


def polynomial_decay(learning_rate,
                     global_step,
                     decay_steps,
                     end_learning_rate=0.0001,
                     power=1.0,
                     cycle=False,
                     name=None):
    """Applies a polynomial decay to the learning rate.
      It is commonly observed that a monotonically decreasing learning rate, whose
      degree of change is carefully chosen, results in a better performing model.
      This function applies a polynomial decay function to a provided initial
      `learning_rate` to reach an `end_learning_rate` in the given `decay_steps`.
      It requires a `global_step` value to compute the decayed learning rate.  You
      can just pass a TensorFlow variable that you increment at each training step.
      The function returns a no-arg callable that outputs the decayed learning
      rate. This can be useful for changing the learning rate value across
      different invocations of optimizer functions. It is computed as:
      ```python
      global_step = min(global_step, decay_steps)
      decayed_learning_rate = (learning_rate - end_learning_rate) *
                              (1 - global_step / decay_steps) ^ (power) +
                              end_learning_rate
      ```
      If `cycle` is True then a multiple of `decay_steps` is used, the first one
      that is bigger than `global_steps`.
      ```python
      decay_steps = decay_steps * ceil(global_step / decay_steps)
      decayed_learning_rate_fn = (learning_rate - end_learning_rate) *
                              (1 - global_step / decay_steps) ^ (power) +
                              end_learning_rate
      decayed_learning_rate = decayed_learning_rate_fn()
      ```
      Example: decay from 0.1 to 0.01 in 10000 steps using sqrt (i.e. power=0.5):
      ```python
      ...
      global_step = tf.Variable(0, trainable=False)
      starter_learning_rate = 0.1
      end_learning_rate = 0.01
      decay_steps = 10000
      learning_rate_fn = tf.train.polynomial_decay(starter_learning_rate,
                                                   global_step, decay_steps,
                                                   end_learning_rate,
                                                   power=0.5)
      # Passing global_step to minimize() will increment it at each step.
      learning_step = (
          tf.train.GradientDescentOptimizer(learning_rate_fn)
          .minimize(...my loss..., global_step=global_step)
      )
      ```
      Args:
        learning_rate: A scalar `float32` or `float64` `Tensor` or a
          Python number.  The initial learning rate.
        global_step: A scalar `int32` or `int64` `Tensor` or a Python number.
          Global step to use for the decay computation.  Must not be negative.
        decay_steps: A scalar `int32` or `int64` `Tensor` or a Python number.
          Must be positive.  See the decay computation above.
        end_learning_rate: A scalar `float32` or `float64` `Tensor` or a
          Python number.  The minimal end learning rate.
        power: A scalar `float32` or `float64` `Tensor` or a
          Python number.  The power of the polynomial. Defaults to linear, 1.0.
        cycle: A boolean, whether or not it should cycle beyond decay_steps.
        name: String.  Optional name of the operation. Defaults to
          'PolynomialDecay'.
      Returns:
        A no-arg function that outputs the decayed learning rate, a scalar `Tensor`
        of the same type as `learning_rate`.
      Raises:
        ValueError: if `global_step` is not supplied.
      """
    if global_step is None:
        raise ValueError("global_step is required for polynomial_decay.")

    with ops.name_scope(
            name, "PolynomialDecay",
            [learning_rate, global_step, decay_steps, end_learning_rate, power]
    ) as name:
        learning_rate = ops.convert_to_tensor(learning_rate, name="learning_rate")
        dtype = learning_rate.dtype
        end_learning_rate = math_ops.cast(end_learning_rate, dtype)
        power = math_ops.cast(power, dtype)

        global_step_recomp = math_ops.cast(global_step, dtype)
        decay_steps_recomp = math_ops.cast(decay_steps, dtype)
        if cycle:
            # Find the first multiple of decay_steps that is bigger than
            # global_step. If global_step is zero set the multiplier to 1
            multiplier = control_flow_ops.cond(
                math_ops.equal(global_step_recomp, 0), lambda: 1.0,
                lambda: math_ops.ceil(global_step_recomp / decay_steps))
            decay_steps_recomp = math_ops.multiply(decay_steps_recomp, multiplier)
        else:
            # Make sure that the global_step used is not bigger than decay_steps.
            global_step_recomp = math_ops.minimum(global_step_recomp, decay_steps)

        p = math_ops.divide(global_step_recomp, decay_steps_recomp)
        return math_ops.add(
            math_ops.multiply(learning_rate - end_learning_rate,
                              math_ops.pow(1 - p, power)),
            end_learning_rate,
            name=name)
