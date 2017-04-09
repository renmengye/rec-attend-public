from tensorflow.python.framework import ops
from utils import logger
import numpy as np
import tensorflow as tf

hungarian_module = None

log = logger.get()

# Register gradient for Hungarian algorithm.
ops.NoGradient("Hungarian")


def get_device_fn(device):
  """Choose device for different ops."""
  OPS_ON_CPU = set([
      'ResizeBilinear', 'ResizeBilinearGrad', 'Mod', 'Hungarian',
      'SparseToDense', 'Print', 'Gather', 'Reverse'
  ])
  def _device_fn(op):
    if op.type in OPS_ON_CPU:
      return "/cpu:0"
    else:
      return device
  return _device_fn


def get_identity_match(num_ex, timespan, s_gt):
  zeros = tf.zeros(tf.pack([num_ex, timespan, timespan]))
  eye = tf.expand_dims(tf.constant(np.eye(timespan), dtype='float32'), 0)
  mask_x = tf.expand_dims(s_gt, 1)
  mask_y = tf.expand_dims(s_gt, 2)
  match = zeros + eye
  match = match * mask_x * mask_y

  return match


def f_cum_min(s, d):
  """Calculates cumulative minimum.
  Args:
    s: Input matrix [B, D].
    d: Second dim.
  Returns:
    s_min: [B, D], cumulative minimum across the second dim.
  """
  s_min_list = [None] * d
  s_min_list[0] = s[:, 0:1]
  for ii in range(1, d):
    s_min_list[ii] = tf.minimum(s_min_list[ii - 1], s[:, ii:ii + 1])

  return tf.concat(1, s_min_list)


def f_cum_max(s, d):
  """Calculates cumulative maximum.
  Args:
    s: Input matrix [B, D].
    d: Second dim.
  Returns:
    s_max: [B, D], cumulative maximum across the second dim, reversed.
  """
  s_max_list = [None] * d
  s_max_list[-1] = s[:, d - 1:d]
  for ii in range(d - 2, -1, -1):
    s_max_list[ii] = tf.maximum(s_max_list[ii + 1], s[:, ii:ii + 1])

  return tf.concat(1, s_max_list)


def f_dice(a, b, timespan, pairwise=False):
  """Computes DICE score.
  Args:
    a: [B, N, H, W], or [N, H, W], or [H, W]
    b: [B, N, H, W], or [N, H, W], or [H, W]
       in pairwise mode, the second dimension can be different,
       e.g. [B, M, H, W], or [M, H, W], or [H, W]
    pairwise: whether the inputs are already aligned, outputs [B, N] or
              the inputs are orderless, outputs [B, N, M].
  """
  if pairwise:
    # N * [B, 1, M]
    y_list = [None] * timespan
    # [B, N, H, W] => [B, N, 1, H, W]
    a = tf.expand_dims(a, 2)
    # [B, N, 1, H, W] => N * [B, 1, 1, H, W]
    a_list = tf.split(1, timespan, a)
    # [B, M, H, W] => [B, 1, M, H, W]
    b = tf.expand_dims(b, 1)
    card_b = tf.reduce_sum(b + 1e-5, [3, 4])

    for ii in range(timespan):
      # [B, 1, M]
      y_list[ii] = 2 * f_inter(a_list[ii], b) / \
          (tf.reduce_sum(a_list[ii] + 1e-5, [3, 4]) + card_b)
    # N * [B, 1, M] => [B, N, M]
    return tf.concat(1, y_list)
  else:
    card_a = tf.reduce_sum(a + 1e-5, _get_reduction_indices(a))
    card_b = tf.reduce_sum(b + 1e-5, _get_reduction_indices(b))
    return 2 * f_inter(a, b) / (card_a + card_b)


def f_inter(a, b):
  """Computes intersection."""
  reduction_indices = _get_reduction_indices(a)
  return tf.reduce_sum(a * b, reduction_indices=reduction_indices)


def f_union(a, b, eps=1e-5):
  """Computes union."""
  reduction_indices = _get_reduction_indices(a)
  return tf.reduce_sum(
      a + b - (a * b) + eps, reduction_indices=reduction_indices)


def _get_reduction_indices(a):
  """Gets the list of axes to sum over."""
  dim = tf.shape(tf.shape(a))

  return tf.concat(0, [dim - 2, dim - 1])


def f_iou(a, b, timespan=None, pairwise=False):
  """
  Computes IOU score.

  Args:
    a: [B, N, H, W], or [N, H, W], or [H, W]
    b: [B, N, H, W], or [N, H, W], or [H, W]
       in pairwise mode, the second dimension can be different,
       e.g. [B, M, H, W], or [M, H, W], or [H, W]
    pairwise: whether the inputs are already aligned, outputs [B, N] or
              the inputs are orderless, outputs [B, N, M].
  Returns:
      iou: [B, N]
  """
  if pairwise:
    # N * [B, 1, M]
    y_list = [None] * timespan
    # [B, N, H, W] => [B, N, 1, H, W]
    a = tf.expand_dims(a, 2)
    # [B, N, 1, H, W] => N * [B, 1, 1, H, W]
    a_list = tf.split(1, timespan, a)
    # [B, M, H, W] => [B, 1, M, H, W]
    b = tf.expand_dims(b, 1)

    for ii in range(timespan):
      # [B, 1, M]
      y_list[ii] = f_inter(a_list[ii], b) / f_union(a_list[ii], b)

    # N * [B, 1, M] => [B, N, M]
    return tf.concat(1, y_list)
  else:
    return f_inter(a, b) / f_union(a, b)


def f_iou_pair_new(a, b):
  """
  a: [B, N, H, W]
  b: [B, N, H, W]
  """
  a = tf.tile(tf.expand_dims(a, 2), tf.pack([1, 1, tf.shape(b)[1], 1, 1]))
  b = tf.expand_dims(b, 1)
  inter = tf.reduce_sum(a * b, [3, 4])
  union = tf.reduce_sum(a + b, [3, 4])
  union = tf.maximum(union - inter, 1)
  return inter / union


def f_iou_all(a, b):
  """Computes total IOU score
  Args:
      a: Any shape
      b: Any shape
  Returns:
      iou: float
  """
  inter = tf.reduce_sum(a * b)
  union = tf.reduce_sum(a) + tf.reduce_sum(b) - inter + 1e-5
  return inter / union


def f_inter_box(top_left_a, bot_right_a, top_left_b, bot_right_b):
  """Computes intersection area with boxes.
  Args:
    top_left_a: [B, T, 2] or [B, 2]
    bot_right_a: [B, T, 2] or [B, 2]
    top_left_b: [B, T, 2] or [B, 2]
    bot_right_b: [B, T, 2] or [B, 2]
  Returns:
    area: [B, T]
  """
  top_left_max = tf.maximum(top_left_a, top_left_b)
  bot_right_min = tf.minimum(bot_right_a, bot_right_b)
  ndims = tf.shape(tf.shape(top_left_a))

  # Check if the resulting box is valid.
  overlap = tf.to_float(top_left_max < bot_right_min)
  overlap = tf.reduce_prod(overlap, ndims - 1)
  area = tf.reduce_prod(bot_right_min - top_left_max, ndims - 1)
  area = overlap * tf.abs(area)
  return area


def f_iou_box(top_left_a, bot_right_a, top_left_b, bot_right_b):
  """Compute IOU of boxes.
  Args:
    top_left_a: [B, T, 2]
    bot_right_a: [B, T, 2]
    top_left_b: [B, T, 2]
    bot_right_b: [B, T, 2]
  Returns:
    iou: [B, T] or [B]
  """
  y1A = top_left_a[:, :, 0]
  x1A = top_left_a[:, :, 1]
  y2A = bot_right_a[:, :, 0]
  x2A = bot_right_a[:, :, 1]
  y1B = top_left_b[:, :, 0]
  x1B = top_left_b[:, :, 1]
  y2B = bot_right_b[:, :, 0]
  x2B = bot_right_b[:, :, 1]

  # compute intersection
  x1_max = tf.maximum(x1A, x1B)
  y1_max = tf.maximum(y1A, y1B)
  x2_min = tf.minimum(x2A, x2B)
  y2_min = tf.minimum(y2A, y2B)

  overlap_flag = tf.to_float(x1_max < x2_min) * tf.to_float(y1_max < y2_min)
  overlap_area = overlap_flag * (x2_min - x1_max) * (y2_min - y1_max)

  # compute union
  areaA = (x2A - x1A) * (y2A - y1A)
  areaB = (x2B - x1B) * (y2B - y1B)
  union_area = areaA + areaB - overlap_area
  return tf.div(overlap_area, union_area)


def f_iou_box_old(top_left_a, bot_right_a, top_left_b, bot_right_b):
  """Computes IoU of boxes.
  Args:
    top_left_a: [B, T, 2] or [B, 2]
    bot_right_a: [B, T, 2] or [B, 2]
    top_left_b: [B, T, 2] or [B, 2]
    bot_right_b: [B, T, 2] or [B, 2]
  Returns:
    iou: [B, T]
  """
  inter_area = f_inter_box(top_left_a, bot_right_a, top_left_b, bot_right_b)
  inter_area = tf.maximum(inter_area, 1e-6)
  ndims = tf.shape(tf.shape(top_left_a))
  # area_a = tf.reduce_prod(bot_right_a - top_left_a, ndims - 1)
  # area_b = tf.reduce_prod(bot_right_b - top_left_b, ndims - 1)
  check_a = tf.reduce_prod(tf.to_float(top_left_a < bot_right_a), ndims - 1)
  area_a = check_a * tf.reduce_prod(bot_right_a - top_left_a, ndims - 1)
  check_b = tf.reduce_prod(tf.to_float(top_left_b < bot_right_b), ndims - 1)
  area_b = check_b * tf.reduce_prod(bot_right_b - top_left_b, ndims - 1)
  union_area = (area_a + area_b - inter_area + 1e-5)
  union_area = tf.maximum(union_area, 1e-5)
  iou = inter_area / union_area
  iou = tf.maximum(iou, 1e-5)
  iou = tf.minimum(iou, 1.0)
  return iou


def f_coverage(iou):
  """Coverage function proposed in [1]
  [1] N. Silberman, D. Sontag, R. Fergus. Instance segmentation of indoor
  scenes using a coverage loss. ECCV 2015.
  Args:
    iou: [B, N, N]. Pairwise IoU.
  """
  return tf.reduce_max(iou, [1])


def f_coverage_weight(y_gt):
  """Compute the normalized weight for each groundtruth instance."""
  # [B, T]
  y_gt_sum = tf.reduce_sum(y_gt, [2, 3])
  # Plus one to avoid dividing by zero.
  # The resulting weight will be zero for any zero cardinality instance.
  # [B, 1]
  y_gt_sum_sum = tf.reduce_sum(
      y_gt_sum, [1], keep_dims=True) + tf.to_float(tf.equal(y_gt_sum, 0))

  # [B, T]
  return y_gt_sum / y_gt_sum_sum


def f_weighted_coverage(iou, y_gt):
  """Weighted coverage score.
  Args:
    iou: [B, N, N]. Pairwise IoU.
    y_gt: [B, N, H, W]. Groundtruth segmentations.
  """
  cov = f_coverage(iou)
  wt = f_coverage_weight(y_gt)
  num_ex = tf.to_float(tf.shape(y_gt)[0])

  return tf.reduce_sum(cov * wt) / num_ex


def f_unweighted_coverage(iou, count):
  """Unweighted coverage score.
  Args:
    iou: [B, N, N]. Pairwise IoU.
  """
  # [B, N]
  cov = f_coverage(iou)
  num_ex = tf.to_float(tf.shape(iou)[0])
  return tf.reduce_sum(tf.reduce_sum(cov, [1]) / count) / num_ex


def f_conf_loss(s_out, match, timespan, use_cum_min=True):
  """Loss function for confidence score sequence.
  Args:
    s_out:
    match:
    use_cum_min:
  """
  s_out_shape = tf.shape(s_out)
  num_ex = tf.to_float(s_out_shape[0])
  max_num_obj = tf.to_float(s_out_shape[1])
  match_sum = tf.reduce_sum(match, reduction_indices=[2])

  # Loss for confidence scores.
  if use_cum_min:
    # [B, N]
    s_out_min = f_cum_min(s_out, timespan)
    s_out_max = f_cum_max(s_out, timespan)
    # [B, N]
    s_bce = f_bce_minmax(s_out_min, s_out_max, match_sum)
  else:
    s_bce = f_bce(s_out, match_sum)
  loss = tf.reduce_sum(s_bce) / num_ex / max_num_obj

  return loss


def f_sem_loss(s_out,
               match,
               c_gt,
               timespan,
               num_semantic_classes,
               use_cum_min=True):
  # General monotonic score loss.
  c_loss = f_conf_loss(
      1 - s_out[:, :, 0], match, timespan, use_cum_min=use_cum_min)

  # Match [B, T, T]
  # C_gt  [B, T, C] => [B, 1, T, C]
  # C_gt' [B, T, T] * [B, 1, T, C] = [B, T, T, C] => [B, T, C]
  m2 = tf.tile(tf.expand_dims(match, 3), [1, 1, 1, num_semantic_classes])
  c_gt2 = tf.reduce_sum(m2 * tf.expand_dims(c_gt, 1), [2])

  s_out_shape = tf.shape(s_out)
  num_ex = tf.to_float(s_out_shape[0])
  max_num_obj = tf.to_float(s_out_shape[1])
  s_loss = tf.reduce_sum(f_ce(s_out, c_gt2)) / num_ex / max_num_obj
  return c_loss + s_loss
  # return s_loss


def f_greedy_match(score, matched):
  """Compute greedy matching given the IOU, and matched.
  Args:
    score: [B, N] relatedness score, positive.
    matched: [B, N] binary mask
  Returns:
    match: [B, N] binary mask
  """
  score = score * (1.0 - matched)
  max_score = tf.reshape(tf.reduce_max(score, reduction_indices=[1]), [-1, 1])
  match = tf.to_float(tf.equal(score, max_score))
  match_sum = tf.reshape(tf.reduce_sum(match, reduction_indices=[1]), [-1, 1])

  return match / match_sum


def f_segm_match(iou, s_gt):
  """Matching between segmentation output and groundtruth.
  Args:
    y_out: [B, T, H, W], output segmentations
    y_gt: [B, T, H, W], groundtruth segmentations
    s_gt: [B, T], groudtruth score sequence
  """
  global hungarian_module
  if hungarian_module is None:
    mod_name = './hungarian.so'
    hungarian_module = tf.load_op_library(mod_name)
    log.info('Loaded library "{}"'.format(mod_name))

  # Mask X, [B, M] => [B, 1, M]
  mask_x = tf.expand_dims(s_gt, dim=1)
  # Mask Y, [B, M] => [B, N, 1]
  mask_y = tf.expand_dims(s_gt, dim=2)
  iou_mask = iou * mask_x * mask_y

  # Keep certain precision so that we can get optimal matching within
  # reasonable time.
  eps = 1e-5
  precision = 1e6
  iou_mask = tf.round(iou_mask * precision) / precision
  match_eps = hungarian_module.hungarian(iou_mask + eps)[0]

  # [1, N, 1, 1]
  s_gt_shape = tf.shape(s_gt)
  num_segm_out = s_gt_shape[1]
  num_segm_out_mul = tf.pack([1, num_segm_out, 1])
  # Mask the graph algorithm output.
  match = match_eps * mask_x * mask_y

  return match


def f_ce(y_out, y_gt):
  """Multiclass cross entropy."""
  eps = 1e-5
  return -y_gt * tf.log(y_out + eps)


def f_bce(y_out, y_gt):
  """Binary cross entropy."""
  eps = 1e-5
  return -y_gt * tf.log(y_out + eps) - (1 - y_gt) * tf.log(1 - y_out + eps)


def f_bce_minmax(y_out_min, y_out_max, y_gt):
  """Binary cross entropy (encourages monotonic decreasing).
  Use minimum (cumulative from start) to compare against 1.
  Use maximum (cumulative till end) to compare against 0.
  """
  eps = 1e-5
  return -y_gt * tf.log(y_out_min + eps) - (1 - y_gt
                                           ) * tf.log(1 - y_out_max + eps)


def f_match_loss(y_out, y_gt, match, timespan, loss_fn, model=None):
  """Binary cross entropy with matching.
  Args:
    y_out: [B, N, H, W] or [B, N, D]
    y_gt: [B, N, H, W] or [B, N, D]
    match: [B, N, N]
    match_count: [B]
    timespan: N
    loss_fn: 
  """
  # N * [B, 1, H, W]
  y_out_list = tf.split(1, timespan, y_out)
  # N * [B, 1, N]
  match_list = tf.split(1, timespan, match)
  err_list = [None] * timespan
  shape = tf.shape(y_out)
  num_ex = tf.to_float(shape[0])
  num_dim = tf.to_float(tf.reduce_prod(tf.to_float(shape[2:])))
  sshape = tf.size(shape)

  # [B, N, M] => [B, N]
  match_sum = tf.reduce_sum(match, reduction_indices=[2])
  # [B, N] => [B]
  match_count = tf.reduce_sum(match_sum, reduction_indices=[1])
  match_count = tf.maximum(match_count, 1)

  for ii in range(timespan):
    # [B, 1, H, W] * [B, N, H, W] => [B, N, H, W] => [B, N]
    # [B, N] * [B, N] => [B]
    # [B] => [B, 1]
    red_idx = tf.range(2, sshape)
    err_list[ii] = tf.expand_dims(
        tf.reduce_sum(
            tf.reduce_sum(loss_fn(y_out_list[ii], y_gt), red_idx) *
            tf.reshape(match_list[ii], [-1, timespan]), [1]), 1)

  # N * [B, 1] => [B, N] => [B]
  err_total = tf.reduce_sum(tf.concat(1, err_list), reduction_indices=[1])

  return tf.reduce_sum(err_total / match_count) / num_ex / num_dim


def f_count_acc(s_out, s_gt):
  """Counting accuracy.

    Args:
        s_out:
        s_gt:
    """
  num_ex = tf.to_float(tf.shape(s_out)[0])
  count_out = tf.reduce_sum(tf.to_float(s_out > 0.5), reduction_indices=[1])
  count_gt = tf.reduce_sum(s_gt, reduction_indices=[1])
  count_acc = tf.reduce_sum(tf.to_float(tf.equal(count_out, count_gt))) / num_ex

  return count_acc


def f_dic(s_out, s_gt, abs=False):
  """Difference in count.

    Args:
        s_out:
        s_gt:
    """
  num_ex = tf.to_float(tf.shape(s_out)[0])
  count_out = tf.reduce_sum(tf.to_float(s_out > 0.5), reduction_indices=[1])
  count_gt = tf.reduce_sum(s_gt, reduction_indices=[1])
  count_diff = count_out - count_gt
  if abs:
    count_diff = tf.abs(count_diff)
  count_diff = tf.reduce_sum(tf.to_float(count_diff)) / num_ex
  return count_diff


def f_huber(y_out, y_gt, threshold=1.0):
  """Huber loss. Smooth combination of L2 and L1 loss for robustness."""
  size = tf.size(y_out)
  err = y_out - y_gt
  ind = tf.to_float(err <= 1)
  squared_err = 0.5 * err * err
  l1_err = tf.abs(err) - (threshold - 0.5 * (threshold**2))
  huber = squared_err * ind + l1_err * (1 - ind)
  return huber


def f_squared_err(y_out, y_gt):
  """Mean squared error (L2) loss."""
  err = y_out - y_gt
  squared_err = 0.5 * err * err

  return squared_err


def build_skip_conn_inner(cnn_channels, h_cnn, x):
  """Build skip connection."""
  skip = [None]
  skip_ch = [0]
  for jj, layer in enumerate(h_cnn[-2::-1] + [x]):
    skip.append(layer_reshape)
    ch_idx = len(cnn_channels) - jj - 2
    skip_ch.append(cnn_channels[ch_idx])

  return skip, skip_ch


def build_skip_conn(cnn_channels, h_cnn, x, timespan):
  """Build skip connection."""
  skip = [None]
  skip_ch = [0]
  for jj, layer in enumerate(h_cnn[-2::-1] + [x]):
    ss = tf.shape(layer)
    zeros = tf.zeros(tf.pack([ss[0], timespan, ss[1], ss[2], ss[3]]))
    new_shape = tf.pack([ss[0] * timespan, ss[1], ss[2], ss[3]])
    layer_reshape = tf.reshape(tf.expand_dims(layer, 1) + zeros, new_shape)
    skip.append(layer_reshape)
    ch_idx = len(cnn_channels) - jj - 2
    skip_ch.append(cnn_channels[ch_idx])
  return skip, skip_ch


def build_skip_conn_attn(cnn_channels, h_cnn_time, x_time, timespan):
  """Build skip connection for attention based model."""
  skip = [None]
  skip_ch = [0]
  nlayers = len(h_cnn_time[0])
  timespan = len(h_cnn_time)
  for jj in range(nlayers):
    lidx = nlayers - jj - 2
    if lidx >= 0:
      ll = [h_cnn_time[tt][lidx] for tt in range(timespan)]
    else:
      ll = x_time
    layer = tf.concat(1, [tf.expand_dims(l, 1) for l in ll])
    ss = tf.shape(layer)
    layer = tf.reshape(layer, tf.pack([-1, ss[2], ss[3], ss[4]]))
    skip.append(layer)
    ch_idx = lidx + 1
    skip_ch.append(cnn_channels[ch_idx])
  return skip, skip_ch


def get_gaussian_filter(center, size, lg_var, image_size, filter_size):
  """Get Gaussian-based attention filter along one dimension
  Args:
    center: center of one dimension (mean), [B]
    delta: delta of one dimension (size), [B]
    lg_var: variance of the filter, [B]
    image_size: image size of one dimension, [B]
    filter_size: filter size of one dimension, [B]
  """
  # [1, 1, F].
  span_filter = tf.to_float(tf.reshape(tf.range(filter_size), [1, 1, -1]))

  # [B, 1, 1]
  center = tf.reshape(center, [-1, 1, 1])
  size = tf.reshape(size, [-1, 1, 1])

  # [B, 1, 1] + [B, 1, 1] * [1, F, 1] = [B, 1, F]
  # mu = center + size / filter_size * (span_filter - (filter_size - 1) / 2.0)
  mu = center + (size + 1) / filter_size * \
      (span_filter - (filter_size - 1) / 2.0)

  # [B, 1, 1]
  lg_var = tf.reshape(lg_var, [-1, 1, 1])

  # [1, L, 1]
  span = tf.to_float(
      tf.reshape(tf.range(image_size), tf.pack([1, image_size, 1])))

  # [1, L, 1] - [B, 1, F] = [B, L, F]
  filt = tf.mul(1 / tf.sqrt(tf.exp(lg_var)) / tf.sqrt(2 * np.pi),
                tf.exp(-0.5 * (span - mu) * (span - mu) / tf.exp(lg_var)))
  return filt


def extract_patch(x, f_y, f_x, nchannels, normalize=False):
  """
  Args:
      x: [B, H, W, D]
      f_y: [B, H, FH]
      f_x: [B, W, FH]
      nchannels: D
  Returns:
      patch: [B, FH, FW]
  """
  patch = [None] * nchannels
  fsize_h = tf.shape(f_y)[2]
  fsize_w = tf.shape(f_x)[2]
  hh = tf.shape(x)[1]
  ww = tf.shape(x)[2]

  for dd in range(nchannels):
    # [B, H, W]
    x_ch = tf.reshape(
        tf.slice(x, [0, 0, 0, dd], [-1, -1, -1, 1]), tf.pack([-1, hh, ww]))
    patch[dd] = tf.reshape(
        tf.batch_matmul(
            tf.batch_matmul(
                f_y, x_ch, adj_x=True), f_x),
        tf.pack([-1, fsize_h, fsize_w, 1]))

  return tf.concat(3, patch)


def get_gt_attn(y_gt,
                filter_height,
                filter_width,
                padding_ratio=0.0,
                center_shift_ratio=0.0,
                min_padding=10.0):
  """Get groundtruth attention box given segmentation."""
  top_left, bot_right, box = get_gt_box(
      y_gt,
      padding_ratio=padding_ratio,
      center_shift_ratio=center_shift_ratio,
      min_padding=min_padding)
  ctr, size = get_box_ctr_size(top_left, bot_right)
  # lg_var = tf.zeros(tf.shape(ctr)) + 1.0
  lg_var = get_normalized_var(size, filter_height, filter_width)
  lg_gamma = get_normalized_gamma(size, filter_height, filter_width)
  return ctr, size, lg_var, lg_gamma, box, top_left, bot_right


def get_gt_box(y_gt,
               padding_ratio=0.0,
               center_shift_ratio=0.0,
               min_padding=10.0):
  """Get groundtruth bounding box given segmentation.
  Current only support [B, T, H, W] as input!!!

  Args:
    y_gt: Groundtruth segmentation [B, T, H, W], or [B, H, W]

  Returns:
    top_left: Bounding box top left coordinates [B, T, 2], or [B, 2]
    bot_right: Bounding box bottom right coordinates [B, T, 2], or [B, 2]
  """
  s = tf.shape(y_gt)
  # [B, T, H, W, 2]
  idx = get_idx_map(s)
  y_gt_not_zero = tf.to_float(tf.reduce_sum(y_gt, [2, 3]) > 0)
  y_gt_not_zero = tf.expand_dims(y_gt_not_zero, 2)
  idx_min = idx + tf.expand_dims((1.0 - y_gt) * tf.to_float(s[2] * s[3]), 4)
  idx_max = idx * tf.expand_dims(y_gt, 4)
  # [B, T, 2]
  top_left = tf.reduce_min(idx_min, reduction_indices=[2, 3])
  bot_right = tf.reduce_max(idx_max, reduction_indices=[2, 3])

  # Enlarge the groundtruth box by some padding.
  size = bot_right - top_left
  top_left += center_shift_ratio * size
  top_left -= tf.maximum(padding_ratio * size, min_padding)
  bot_right += center_shift_ratio * size
  bot_right += tf.maximum(padding_ratio * size, min_padding)
  box = get_filled_box_idx(idx, top_left, bot_right)

  # If the segmentation is zero, then fix to top left corner.
  top_left *= y_gt_not_zero
  bot_right = y_gt_not_zero * bot_right + \
      (1 - y_gt_not_zero) * (2 * min_padding)

  return top_left, bot_right, box


def get_idx_map(shape):
  """Get index map for a image.
  Args:
    shape: [B, T, H, W] or [B, H, W]
  Returns:
    idx: [B, T, H, W, 2], or [B, H, W, 2]
  """
  s = shape
  ndims = tf.shape(s)
  wdim = ndims - 1
  hdim = ndims - 2
  idx_shape = tf.concat(0, [s, tf.constant([1])])
  ones_h = tf.ones(hdim - 1, dtype='int32')
  ones_w = tf.ones(wdim - 1, dtype='int32')
  h_shape = tf.concat(0, [ones_h, tf.constant([-1]), tf.constant([1, 1])])
  w_shape = tf.concat(0, [ones_w, tf.constant([-1]), tf.constant([1])])

  idx_y = tf.zeros(idx_shape, dtype='float')
  idx_x = tf.zeros(idx_shape, dtype='float')

  h = tf.slice(s, ndims - 2, [1])
  w = tf.slice(s, ndims - 1, [1])
  idx_y += tf.reshape(tf.to_float(tf.range(h[0])), h_shape)
  idx_x += tf.reshape(tf.to_float(tf.range(w[0])), w_shape)
  idx = tf.concat(ndims[0], [idx_y, idx_x])
  return idx


def get_filled_box_idx(idx, top_left, bot_right):
  """Fill a box with top left and bottom right coordinates.
  Args:
    idx: [B, T, H, W, 2] or [B, H, W, 2] or [H, W, 2]
    top_left: [B, T, 2] or [B, 2] or [2]
    bot_right: [B, T, 2] or [B, 2] or [2]
  """
  ss = tf.shape(idx)
  ndims = tf.shape(ss)
  batch = tf.slice(ss, [0], ndims - 3)
  coord_shape = tf.concat(0, [batch, tf.constant([1, 1, 2])])
  top_left = tf.reshape(top_left, coord_shape)
  bot_right = tf.reshape(bot_right, coord_shape)
  lower = tf.reduce_prod(tf.to_float(idx >= top_left), ndims - 1)
  upper = tf.reduce_prod(tf.to_float(idx <= bot_right), ndims - 1)
  box = lower * upper

  return box


def get_unnormalized_center(ctr_norm, inp_height, inp_width):
  """Get unnormalized center coordinates
  Args:
    ctr_norm: [B, T, 2] or [B, 2] or [2], normalized within range [-1, +1]
    inp_height: int, image height
    inp_width: int, image width
  Returns:
    ctr: [B, 2]
  """
  img_size = tf.to_float(tf.pack([inp_height, inp_width]))
  img_size = img_size / 2.0
  ctr = (ctr_norm + 1.0) * img_size
  return ctr


def get_normalized_center(ctr, inp_height, inp_width):
  """Get unnormalized center coordinates
  Args:
    ctr: [B, T, 2] or [B, 2] or [2]
    inp_height: int, image height
    inp_width: int, image width
  Returns:
    ctr: [B, 2], normalized within range [-1, +1]
  """
  img_size = tf.to_float(tf.pack([inp_height, inp_width]))
  img_size = img_size / 2.0
  ctr = ctr / img_size - 1
  return ctr


def get_normalized_var(size, filter_height, filter_width):
  """Get normalized variance.
  Args:
    size: [B, T, 2] or [B, 2] or [2]
    filter_height: int
    filter_width: int
  Returns:
    lg_var: [B, T, 2] or [B, 2] or [2]
  """
  filter_size = tf.to_float(tf.pack([filter_height, filter_width]))
  lg_var = tf.log(size) - tf.log(filter_size)
  return lg_var


def get_normalized_gamma(size, filter_height, filter_width):
  """Get normalized gamma.
  Args:
    size: [B, T, 2] or [B, 2] or [2]
    filter_height: int
    filter_width: int
  Returns:
    lg_gamma: [B, T] or [B] or float
  """
  rank = tf.shape(tf.shape(size))
  filter_area = filter_height * filter_width
  area = tf.reduce_prod(size, rank - 1)
  lg_gamma = tf.log(float(filter_area)) - tf.log(area)
  return lg_gamma


def get_unnormalized_size(lg_size, inp_height, inp_width):
  """Get unnormalized patch size.
  Args:
    lg_size: [B, T, 2] or [B, 2] or [2], logarithm of delta.
    inp_height: int, image height.
    inp_width: int, image width.
  Returns:
    size: [B, T, 2] or [B, 2] or [2], patch size.
  """
  size = tf.exp(lg_size)
  img_size = tf.to_float(tf.pack([inp_height, inp_width]))
  size *= img_size

  return size


def get_normalized_size(size, inp_height, inp_width):
  """Get normalized patch size.
  Args:
    patch: [B, 2], patch size.
    inp_height: int, image height.
    inp_width: int, image width.
    patch_size: int patch size.
  Returns:
    lg_delta: [B, 2], logarithm of delta.
  """
  img_size = tf.to_float(tf.pack([inp_height, inp_width]))
  lg_size = tf.log(size / img_size)
  return lg_size


def get_unnormalized_attn(ctr, lg_size, inp_height, inp_width):
  """Unnormalize the attention parameters to image size."""
  ctr = get_unnormalized_center(ctr, inp_height, inp_width)
  size = get_unnormalized_size(lg_size, inp_height, inp_width)
  return ctr, size


def get_box_coord(ctr, size, truncate=True):
  """Get box coordinates given parameters."""
  return ctr - size / 2.0, ctr + size / 2.0


def get_box_ctr_size(top_left, bot_right):
  return (top_left + bot_right) / 2.0, (bot_right - top_left)
