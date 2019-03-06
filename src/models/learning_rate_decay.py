from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops, control_flow_ops


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

        p = math_ops.div(global_step_recomp, decay_steps_recomp)
        return math_ops.add(
            math_ops.multiply(learning_rate - end_learning_rate,
                              math_ops.pow(1 - p, power)),
            end_learning_rate,
            name=name)
