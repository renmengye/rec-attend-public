import numpy as np
import cv2


def apply_threshold(y_out, thresh):
  """Threshold the soft output into binary map.
    Args:
      y_out: list of [T, H, W] soft output.
    Returns:
      y_out_thresh: list of [T, H, W] binary map.
    """
  return [(_y > thresh).astype('float32') for _y in y_out]


def apply_confidence(y_out, s_out):
  """Convert soft prediction to hard prediction.
  Args:
    y_out: [B, T, H, W]
    s_out: [B, T]
  Returns:
    y_out: [B, T, H, W]
    s_out: [B, T]
  """
  # Weight the prediction by its confidence score.
  s_mask = np.reshape(s_out, [-1, s_out.shape[1], 1, 1])
  y_out = y_out * s_mask
  s_out_hard = (s_out > 0.5).astype('float')
  return y_out, s_out_hard


def apply_one_label(y_out):
  """Ensure single label accross the image.

    Args:
        y_out: list of [T, H, W].

    Returns:
        y_out_proc: list of [T, H, W].
    """
  num_ex = len(y_out)
  timespan = y_out[0].shape[0]
  y_out_proc = []
  for ii in range(num_ex):
    _y = y_out[ii]
    y_out_max = np.argmax(_y, axis=0)
    _y2 = np.zeros(_y.shape)
    for jj in range(timespan):
      _y2[jj] = (y_out_max == jj).astype('float32') * _y[jj]
      pass
    y_out_proc.append(_y2)
    pass
  return y_out_proc


def morph(y_out):
  """Morphological transform.
  Args:
    y_out: list of [T, H, W]
  """
  return [morph_single(_y) for _y in y_out]


def morph_single(y_out):
  """Morphological transform.
  Args:
    y_out: [T, H, W]
  """
  y_out_morph = np.zeros(y_out.shape)
  kernel = np.ones([5, 5])
  for ch in range(y_out.shape[0]):
    y_out_morph[ch] = cv2.dilate(y_out[ch], kernel)
  return y_out_morph


def upsample(y_out, y_gt):
  """Upsample y_out into size of y_gt.
  Args:
    y_out: list of [T, H', W']
    y_gt: list of [T, H, W]
  Returns:
    y_out_resize: list of [T, H, W]
  """
  y_out_resize = []
  num_ex = len(y_gt)
  timespan = y_gt[0].shape[0]
  for ii in range(num_ex):
    size = (y_gt[ii].shape[-1], y_gt[ii].shape[-2])
    _y = np.zeros(y_gt[ii].shape, dtype='float32')
    for jj in range(timespan):
      _y[jj] = upsample_single(y_out[ii][jj], size)
    y_out_resize.append(_y)
  return y_out_resize


def upsample_single(a, size):
  """Upsample single image, with bilateral filtering.
  Args:
    a: [H', W', 3]
    size: [W, H]
  Returns:
    b: [H, W, 3]
  """
  interpolation = cv2.INTER_LINEAR
  b = cv2.resize(a, size, interpolation=interpolation)
  b = cv2.bilateralFilter(b, 5, 10, 10)
  return b


def remove_tiny(y_out, conf, threshold=200):
  """Remove tiny regions.
  Args:
    y_out: list of [T, H, W],
    conf: [B, T]
  """
  if threshold == 0:
    return y_out, conf
  y_out_removed = []
  for ii, _y in enumerate(y_out):
    _y_removed, _conf = remove_tiny_single(_y, conf[ii], threshold=threshold)
    y_out_removed.append(_y_removed)
    conf[ii] = _conf
  return y_out_removed, conf


def remove_tiny_single(y_out, conf, threshold=200):
  """Remove tiny regions.
  Args:
      y_out: [T, H, W]
  """
  y_out_size = y_out.sum(axis=1, keepdims=True).sum(axis=2, keepdims=True)
  is_not_tiny = (y_out_size > threshold).astype('float32')
  conf = conf * np.reshape(is_not_tiny, [-1])
  y_out_ret = y_out * is_not_tiny
  return y_out_ret, conf


def mask_foreground(y_out, fg):
  """
    Add foreground mask.

    Args:
        y_out: list of [T, H, W]
        fg: list of [H, W]
    """
  return [_y * _fg for _y, _fg in zip(y_out, fg)]