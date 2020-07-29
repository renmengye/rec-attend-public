from __future__ import division

try:
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  import matplotlib.patches as patches
except:
  pass
import cv2
import numpy as np


def calc_row_col(num_ex, num_items, max_items_per_row=9):
  num_rows_per_ex = int(np.ceil(num_items / max_items_per_row))
  if num_items > max_items_per_row:
    num_col = max_items_per_row
    num_row = num_rows_per_ex * num_ex
  else:
    num_row = num_ex
    num_col = num_items

  def calc(ii, jj):
    col = jj % max_items_per_row
    row = num_rows_per_ex * ii + int(jj / max_items_per_row)

    return row, col

  return num_row, num_col, calc


def set_axis_off(axarr, num_row, num_col):
  for row in range(num_row):
    for col in range(num_col):
      if num_col > 1 and num_row > 1:
        ax = axarr[row, col]
      elif num_col > 1:
        ax = axarr[col]
      elif num_row > 1:
        ax = axarr[row]
      else:
        ax = axarr
      ax.set_axis_off()
  pass


def plot_thumbnails(fname,
                    img,
                    axis,
                    max_items_per_row=9,
                    width=10,
                    height=None):
  """Plot activation map.

    Args:
        img: [B, T, H, W, 3] or [B, H, W, D]
    """
  num_ex = img.shape[0]
  if axis > 0:
    num_items = img.shape[axis]
  else:
    num_ex = 1
    num_items = img.shape[0]
  num_row, num_col, calc = calc_row_col(
      num_ex, num_items, max_items_per_row=max_items_per_row)
  if height is None:
    height = num_row
  f1, axarr = plt.subplots(num_row, num_col, figsize=(width, height))
  set_axis_off(axarr, num_row, num_col)

  for ii in range(num_ex):
    for jj in range(num_items):
      row, col = calc(ii, jj)
      if axis == 3:
        x = img[ii, :, :, jj]
      elif axis == 1:
        x = img[ii, jj]
      elif axis == 0:
        x = img[jj]
      if num_col > 1 and num_row > 1:
        ax = axarr[row, col]
      elif num_row > 1:
        ax = axarr[row]
      elif num_col > 1:
        ax = axarr[col]
      else:
        ax = axarr
      if x.shape[-1] == 3:
        x = x[:, :, [2, 1, 0]]
      ax.imshow(x)
      ax.text(
          0,
          -0.5,
          '[{:.2g}, {:.2g}]'.format(x.min(), x.max()),
          color=(0, 0, 0),
          size=8)

  plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
  plt.savefig(fname, dpi=150)
  plt.close('all')
  pass


def plot_input(fname, x, y_gt, s_gt, max_items_per_row=9):
  """Plot input, transformed input and output groundtruth sequence.
    """
  num_ex = y_gt.shape[0]
  num_items = y_gt.shape[1]
  num_row, num_col, calc = calc_row_col(
      num_ex, num_items, max_items_per_row=max_items_per_row)

  f1, axarr = plt.subplots(num_row, num_col, figsize=(20, num_row))
  set_axis_off(axarr, num_row, num_col)
  cmap = ['r', 'y', 'c', 'g', 'm']

  for ii in range(num_ex):
    _x = x[ii]
    _x = _x[:, :, [2, 1, 0]]
    # _x = x[ii, :, :, [2, 1, 0]]
    for jj in range(num_items):
      row, col = calc(ii, jj)
      axarr[row, col].imshow(_x)
      nz = y_gt[ii, jj].nonzero()
      if nz[0].size > 0:
        top_left_x = nz[1].min()
        top_left_y = nz[0].min()
        bot_right_x = nz[1].max() + 1
        bot_right_y = nz[0].max() + 1
        axarr[row, col].add_patch(
            patches.Rectangle(
                (top_left_x, top_left_y),
                bot_right_x - top_left_x,
                bot_right_y - top_left_y,
                fill=False,
                color=cmap[jj % len(cmap)]))
        axarr[row, col].add_patch(
            patches.Rectangle(
                (top_left_x, top_left_y - 25),
                25,
                25,
                fill=True,
                color=cmap[jj % len(cmap)]))
        axarr[row, col].text(
            top_left_x + 5, top_left_y - 5, '{}'.format(jj), size=5)

  plt.tight_layout(pad=2.0, w_pad=0.0, h_pad=0.0)
  plt.savefig(fname, dpi=150)
  plt.close('all')
  pass


def plot_output(fname,
                y_out,
                s_out,
                match,
                attn=None,
                c_out=None,
                max_items_per_row=9):
  """Plot some test samples.

    Args:
        fname: str, image output filename.
        y_out: [B, T, H, W, D], segmentation output of the model.
        s_out: [B, T], confidence score output of the model.
        match: [B, T, T], matching matrix.
        attn: ([B, T, 2], [B, T, 2]), top left and bottom right coordinates of
        the attention box.
    """
  num_ex = y_out.shape[0]
  num_items = y_out.shape[1]
  num_row, num_col, calc = calc_row_col(
      num_ex, num_items, max_items_per_row=max_items_per_row)

  f1, axarr = plt.subplots(num_row, num_col, figsize=(10, num_row))
  cmap = ['r', 'y', 'c', 'g', 'm']

  if attn:
    attn_top_left_y = attn[0][:, :, 0]
    attn_top_left_x = attn[0][:, :, 1]
    attn_bot_right_y = attn[1][:, :, 0]
    attn_bot_right_x = attn[1][:, :, 1]

  set_axis_off(axarr, num_row, num_col)

  for ii in range(num_ex):
    for jj in range(num_items):
      if len(y_out.shape) == 5 and y_out.shape[4] == 3:
        _x = y_out[ii, jj]
        _x = _x[:, :, [2, 1, 0]]
      else:
        _x = y_out[ii, jj]
      row, col = calc(ii, jj)
      if num_row > 1 and num_col > 1:
        ax = axarr[row, col]
      elif num_row > 1:
        ax = axarr[row]
      elif num_col > 1:
        ax = axarr[col]
      else:
        ax = axarr
      ax.imshow(_x)
      matched = match[ii, jj].nonzero()[0]
      disp_str = '{:.2f} {}'.format(s_out[ii, jj], matched)
      if c_out is not None:
        disp_str += ' [{}]'.format(np.argmax(c_out[ii, jj]))
      ax.text(0, 0, disp_str, color=(0, 0, 0), size=8)

      if attn:
        # Plot attention box.
        ax.add_patch(
            patches.Rectangle(
                (attn_top_left_x[ii, jj], attn_top_left_y[ii, jj]),
                attn_bot_right_x[ii, jj] - attn_top_left_x[ii, jj],
                attn_bot_right_y[ii, jj] - attn_top_left_y[ii, jj],
                fill=False,
                color='m'))

  plt.tight_layout(pad=2.0, w_pad=0.0, h_pad=0.0)
  plt.savefig(fname, dpi=150)
  plt.close('all')
  pass


def plot_total_instances(fname, y_out, s_out, max_items_per_row=9):
  """Plot cumulative image with different colour at each timestep.

    Args:
        y_out: [B, T, H, W]
    """
  num_ex = y_out.shape[0]
  num_items = y_out.shape[1]
  num_row, num_col, calc = calc_row_col(
      num_ex, num_items, max_items_per_row=max_items_per_row)

  f1, axarr = plt.subplots(num_row, num_col, figsize=(10, num_row))
  set_axis_off(axarr, num_row, num_col)

  cmap2 = np.array(
      [[192, 57, 43], [243, 156, 18], [26, 188, 156], [41, 128, 185],
       [142, 68, 173], [44, 62, 80], [127, 140, 141], [17, 75, 95],
       [2, 128, 144], [228, 253, 225], [69, 105, 144], [244, 91, 105],
       [91, 192, 235], [253, 231, 76], [155, 197, 61], [229, 89, 52],
       [250, 121, 33]],
      dtype='uint8')

  for ii in range(num_ex):
    total_img = np.zeros([y_out.shape[2], y_out.shape[3], 3])
    for jj in range(num_items):
      row, col = calc(ii, jj)
      if s_out[ii, jj] > 0.5:
        total_img += np.expand_dims(
            (y_out[ii, jj] > 0.5).astype('uint8'), 2) * \
            cmap2[jj % cmap2.shape[0]]
      axarr[row, col].imshow(total_img)
      total_img = np.copy(total_img)

  plt.tight_layout(pad=2.0, w_pad=0.0, h_pad=0.0)
  plt.savefig(fname, dpi=150)
  plt.close('all')
  pass


def plot_double_attention(fname, x, glimpse_map, max_items_per_row=9):
  """Plot double attention.

    Args:
        fname: str, image output filename.
        x: [B, H, W, 3], input image.
        glimpse_map: [B, T, T2, H', W']: glimpse attention map.
    """
  num_ex = x.shape[0]
  timespan = glimpse_map.shape[1]
  im_height = x.shape[1]
  im_width = x.shape[2]
  num_glimpse = glimpse_map.shape[2]
  num_items = num_glimpse
  num_row, num_col, calc = calc_row_col(
      num_ex * timespan, num_items, max_items_per_row=max_items_per_row)

  f1, axarr = plt.subplots(num_row, num_col, figsize=(10, num_row))
  set_axis_off(axarr, num_row, num_col)

  for ii in range(num_ex):
    _x = x[ii]
    _x = _x[:, :, [2, 1, 0]]
    for tt in range(timespan):
      for jj in range(num_glimpse):
        row, col = calc(ii * timespan + tt, jj)
        total_img = np.zeros([im_height, im_width, 3])
        total_img += _x * 0.5
        glimpse = glimpse_map[ii, tt, jj]
        glimpse = cv2.resize(glimpse, (im_width, im_height))
        glimpse = np.expand_dims(glimpse, 2)
        glimpse_norm = glimpse / glimpse.max() * 0.5
        total_img += glimpse_norm
        if num_col > 1 and num_row > 1:
          ax_ = axarr[row, col]
        elif num_col > 1:
          ax_ = axarr[row]
        elif num_row > 1:
          ax_ = axarr[col]
        else:
          ax_ = axarr
        ax_.imshow(total_img)
        ax_.text(
            0,
            -0.5,
            '[{:.2g}, {:.2g}]'.format(glimpse.min(), glimpse.max()),
            color=(0, 0, 0),
            size=8)

  plt.tight_layout(pad=2.0, w_pad=0.0, h_pad=0.0)
  plt.savefig(fname, dpi=150)
  plt.close('all')
  pass
