import re

import tensorflow as tf

from modelling import CANTRIPModel
from learning_rate_decay import polynomial_decay

class CANTRIPOptimizer(object):

    def __init__(self, model: CANTRIPModel, sparse: bool = False, learning_rate: float = 1e-3):
        """
        Creates a new CANTRIPOptimizer responsible for optimizing CANTRIP. Allegedly, some day I will get around to
        looking at other optimization strategies (e.g., sequence optimization).
        :param model: a CANTRIPModel object
        :param sparse: whether to use sparse softmax or not (I never actually tested this)
        :param learning_rate: float, learning rate of the optimizer
        """
        self.model = model

        # If sparse calculate sparsesoftmax directly from integer labels
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

    def __init__(self, model: CANTRIPModel, num_train_steps: int, num_warmup_steps: int = None, init_lr: float = 1e-3,
                 lr_decay=True, clip_norm=1.0):
        """
        Creates a new CANTRIPOptimizer responsible for optimizing CANTRIP. Allegedly, some day I will get around to
        looking at other optimization strategies (e.g., sequence optimization).
        :param model: a CANTRIPModel object
        :param num_train_steps: number of training steps between decay steps
        :param num_warmup_steps: number of training steps before starting decay
        :param init_lr: float, initial learning rate of the optimizer
        """
        self.model = model

        # Global step used for coordinating summarizes and checkpointing
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

        learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)

        # Implements exponential decay of learning rate
        # TODO: linear decay
        if lr_decay:
            learning_rate = polynomial_decay(
                learning_rate,
                self.global_step,
                num_train_steps,
                end_learning_rate=0.0,
                power=1.0,
                cycle=False
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

        optimizer = AdamWeightDecayOptimizer(
            learning_rate=learning_rate,
            weight_decay_rate=0.1,
            beta_1=.9,
            beta_2=0.999,
            epsilon=1e-7,
            exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"]
        )

        self.loss = tf.losses.sparse_softmax_cross_entropy(model.labels, model.logits)

        tvars = tf.trainable_variables()
        grads = tf.gradients(self.loss, tvars)

        if clip_norm:
            (grads, _) = tf.clip_by_global_norm(grads, clip_norm=clip_norm)

        train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)

        new_global_step = self.global_step + 1
        self.train_op = tf.group(train_op, [self.global_step.assign(new_global_step)])


class AdamWeightDecayOptimizer(tf.train.Optimizer):
    """A basic Adam optimizer that includes "correct" L2 weight decay."""

    def __init__(self,
                 learning_rate,
                 weight_decay_rate=0.0,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-6,
                 exclude_from_weight_decay=None,
                 name="AdamWeightDecayOptimizer"):
        """Constructs a AdamWeightDecayOptimizer."""
        super(AdamWeightDecayOptimizer, self).__init__(False, name)

        self.learning_rate = learning_rate
        self.weight_decay_rate = weight_decay_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.exclude_from_weight_decay = exclude_from_weight_decay

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        """See base class."""
        assignments = []
        for (grad, param) in grads_and_vars:
            if grad is None or param is None:
                continue

            param_name = self._get_variable_name(param.name)

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

    def _do_use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if not self.weight_decay_rate:
            return False
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True

    def _get_variable_name(self, param_name):
        """Get the variable name from the tensor name."""
        m = re.match("^(.*):\\d+$", param_name)
        if m is not None:
            param_name = m.group(1)
        return param_name
