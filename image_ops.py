from tensorflow.python.framework import ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import gen_image_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
import tensorflow as tf


def random_transformation(x,
                          padding,
                          phase_train,
                          rnd_vflip=True,
                          rnd_hflip=True,
                          rnd_transpose=True,
                          rnd_colour=False,
                          y=None,
                          d=None,
                          c=None):
  """
  Perform random crop, flip, transpose, hue, saturation, brightness, contrast.
  Args:
    x: [B, H, W, 3] Input image.
    y: [B, T, H, W] Instance segmentation.
    d: [B, H, W, 8] Instance orientation.
    c: [B, H, W, 1] Semantic segmentation.
    padding: int
    phase_train: bool
  """
  # Random image transformation layers.
  phase_train_f = tf.to_float(phase_train)
  x_shape = tf.shape(x)
  num_ex = x_shape[0]
  inp_height = x_shape[1]
  inp_width = x_shape[2]

  # Add padding
  x_pad = tf.pad(x, [[0, 0], [padding, padding], [padding, padding], [0, 0]])
  if y is not None:
    y_pad = tf.pad(y, [[0, 0], [0, 0], [padding, padding], [padding, padding]])

  if d is not None:
    assert not rnd_vflip, "Orientation mode is on, no random flips"
    assert not rnd_hflip, "Orientation mode is on, no random flips"
    assert not rnd_transpose, "Orientation mode is on, no random transpose"

  if d is not None:
    d_pad = tf.pad(d, [[0, 0], [padding, padding], [padding, padding], [0, 0]])
  if c is not None:
    c_pad = tf.pad(c, [[0, 0], [padding, padding], [padding, padding], [0, 0]])

  # Random crop
  offset = tf.random_uniform([2], dtype='int32', maxval=padding * 2)
  x_rand = tf.slice(x_pad,
                    tf.pack([0, offset[0], offset[1], 0]),
                    tf.pack([-1, inp_height, inp_width, -1]))
  if y is not None:
    y_rand = tf.slice(y_pad,
                      tf.pack([0, 0, offset[0], offset[1]]),
                      tf.pack([-1, -1, inp_height, inp_width]))
  if d is not None:
    d_rand = tf.slice(d_pad,
                      tf.pack([0, offset[0], offset[1], 0]),
                      tf.pack([-1, inp_height, inp_width, -1]))
  if c is not None:
    c_rand = tf.slice(c_pad,
                      tf.pack([0, offset[0], offset[1], 0]),
                      tf.pack([-1, inp_height, inp_width, -1]))

  # Center slices (for inference)
  x_ctr = tf.slice(x_pad, [0, padding, padding, 0],
                   tf.pack([-1, inp_height, inp_width, -1]))
  if y is not None:
    y_ctr = tf.slice(y_pad, [0, 0, padding, padding],
                     tf.pack([-1, -1, inp_height, inp_width]))
  if d is not None:
    d_ctr = tf.slice(d_pad, [0, padding, padding, 0],
                     tf.pack([-1, inp_height, inp_width, -1]))
  if c is not None:
    c_ctr = tf.slice(c_pad, [0, padding, padding, 0],
                     tf.pack([-1, inp_height, inp_width, -1]))

  if d is None:
    # Random horizontal & vertical flip & transpose
    rand_h = tf.random_uniform([1], 1.0 - float(rnd_hflip), 1.0)
    rand_v = tf.random_uniform([1], 1.0 - float(rnd_vflip), 1.0)
    mirror_x = tf.pack([1.0, rand_v[0], rand_h[0], 1.0]) < 0.5
    mirror_y = tf.pack([1.0, 1.0, rand_v[0], rand_h[0]]) < 0.5
    x_rand = tf.reverse(x_rand, mirror_x)
    if y is not None:
      y_rand = tf.reverse(y_rand, mirror_y)

    rand_t = tf.random_uniform([1], 1.0 - float(rnd_transpose), 1.0)
    do_tr = tf.cast(rand_t[0] < 0.5, 'int32')
    x_rand = tf.transpose(x_rand, tf.pack([0, 1 + do_tr, 2 - do_tr, 3]))
    if y is not None:
      y_rand = tf.transpose(y_rand, tf.pack([0, 1, 2 + do_tr, 3 - do_tr]))

  # Random hue, saturation, brightness, contrast
  if rnd_colour:
    x_rand = random_hue(x_rand, 0.1)
    x_rand = random_saturation(x_rand, 0.9, 1.1)
    x_rand = tf.image.random_brightness(x_rand, 0.1)
    x_rand = tf.image.random_contrast(x_rand, 0.9, 1.1)

  results = {}
  results['x'] = (1.0 - phase_train_f) * x_ctr + phase_train_f * x_rand
  if y is not None:
    results['y'] = (1.0 - phase_train_f) * y_ctr + phase_train_f * y_rand
  if d is not None:
    results['d'] = (1.0 - phase_train_f) * d_ctr + phase_train_f * d_rand
  if c is not None:
    results['c'] = (1.0 - phase_train_f) * c_ctr + phase_train_f * c_rand
  return results


def random_flip_left_right(image, seed=None):
  uniform_random = random_ops.random_uniform([], 0, 1.0, seed=seed)
  mirror = math_ops.less(tf.pack([1.0, 1.0, uniform_random, 1.0]), 0.5)
  return tf.reverse(image, mirror)


def random_flip_up_down(image, seed=None):
  uniform_random = random_ops.random_uniform([], 0, 1.0, seed=seed)
  mirror = math_ops.less(tf.pack([1.0, uniform_random, 1.0, 1.0]), 0.5)
  return tf.reverse(image, mirror)


def random_hue(image, max_delta, seed=None):
  """Adjust the hue of an RGB image by a random factor.
  Equivalent to `adjust_hue()` but uses a `delta` randomly
  picked in the interval `[-max_delta, max_delta]`.
  `max_delta` must be in the interval `[0, 0.5]`.
  Args:
  image: RGB image or images. Size of the last dimension must be 3.
  max_delta: float.  Maximum value for the random delta.
  seed: An operation-specific seed. It will be used in conjunction
    with the graph-level seed to determine the real seeds that will be
    used in this operation. Please see the documentation of
    set_random_seed for its interaction with the graph-level random seed.
  Returns:
  3-D float tensor of shape `[height, width, channels]`.
  Raises:
  ValueError: if `max_delta` is invalid.
  """
  if max_delta > 0.5:
    raise ValueError('max_delta must be <= 0.5.')

  if max_delta < 0:
    raise ValueError('max_delta must be non-negative.')

  delta = random_ops.random_uniform([], -max_delta, max_delta, seed=seed)
  return tf.image.adjust_hue(image, delta)


def random_saturation(image, lower, upper, seed=None):
  """Adjust the saturation of an RGB image by a random factor.
  Equivalent to `adjust_saturation()` but uses a `saturation_factor` randomly
  picked in the interval `[lower, upper]`.
  Args:
  image: RGB image or images. Size of the last dimension must be 3.
  lower: float.  Lower bound for the random saturation factor.
  upper: float.  Upper bound for the random saturation factor.
  seed: An operation-specific seed. It will be used in conjunction
    with the graph-level seed to determine the real seeds that will be
    used in this operation. Please see the documentation of
    set_random_seed for its interaction with the graph-level random seed.
  Returns:
  Adjusted image(s), same shape and DType as `image`.
  Raises:
  ValueError: if `upper <= lower` or if `lower < 0`.
  """
  if upper <= lower:
    raise ValueError('upper must be > lower.')

  if lower < 0:
    raise ValueError('lower must be non-negative.')

  # Pick a float in [lower, upper]
  saturation_factor = random_ops.random_uniform([], lower, upper, seed=seed)
  return tf.image.adjust_saturation(image, saturation_factor)
