from __future__ import division

import tensorflow as tf
import image_ops_old as img
import nnlib as nn

import modellib
from utils import logger


def get_model(opt, device='/cpu:0'):
  """A fully-convolutional neural network for foreground segmentation."""
  log = logger.get()
  model = {}
  inp_depth = opt['inp_depth']
  padding = opt['padding']
  cnn_filter_size = opt['cnn_filter_size']
  cnn_depth = opt['cnn_depth']
  cnn_pool = opt['cnn_pool']
  dcnn_filter_size = opt['dcnn_filter_size']
  dcnn_depth = opt['dcnn_depth']
  dcnn_pool = opt['dcnn_pool']
  use_bn = opt['use_bn']
  wd = opt['weight_decay']
  rnd_hflip = opt['rnd_hflip']
  rnd_vflip = opt['rnd_vflip']
  rnd_transpose = opt['rnd_transpose']
  rnd_colour = opt['rnd_colour']
  base_learn_rate = opt['base_learn_rate']
  learn_rate_decay = opt['learn_rate_decay']
  steps_per_learn_rate_decay = opt['steps_per_learn_rate_decay']
  add_skip_conn = opt['add_skip_conn']

  if 'segm_loss_fn' in opt:
    segm_loss_fn = opt['segm_loss_fn']
  else:
    segm_loss_fn = 'iou'

  if 'cnn_skip_mask' in opt:
    cnn_skip_mask = opt['cnn_skip_mask']
  else:
    if 'cnn_skip' in opt:
      cnn_skip_mask = opt['cnn_skip']
    else:
      cnn_skip_mask = [add_skip_conn] * len(cnn_filter_size)

  if 'add_orientation' in opt:
    add_orientation = opt['add_orientation']
    num_orientation_classes = opt['num_orientation_classes']
  else:
    add_orientation = False

  if 'dcnn_skip_mask' in opt:
    dcnn_skip_mask = opt['dcnn_skip_mask']
  else:
    dcnn_skip_mask = cnn_skip_mask[::-1]

  if 'num_semantic_classes' in opt:
    num_semantic_classes = opt['num_semantic_classes']
  else:
    num_semantic_classes = 1

  if 'optimizer' in opt:
    optimizer = opt['optimizer']
  else:
    optimizer = 'adam'

  x = tf.placeholder('float', [None, None, None, inp_depth])
  y_gt = tf.placeholder('float', [None, None, None, num_semantic_classes])

  phase_train = tf.placeholder('bool')
  model['x'] = x
  model['y_gt'] = y_gt
  model['phase_train'] = phase_train

  if add_orientation:
    d_gt = tf.placeholder('float', [None, None, None, num_orientation_classes])
    model['d_gt'] = d_gt
  else:
    d_gt = None

  global_step = tf.Variable(0.0)
  model['global_step'] = global_step
  x_shape = tf.shape(x)
  num_ex = x_shape[0]
  inp_height = x_shape[1]
  inp_width = x_shape[2]

  if add_orientation:
    assert not rnd_hflip, "Orientation mode, rnd_hflip not supported"
    assert not rnd_vflip, "Orientation mode, rnd_vflip not supported"
    assert not rnd_transpose, "Orientation mode, rnd_transpose not supported"

  results = img.random_transformation(
      x,
      padding,
      phase_train,
      rnd_hflip=rnd_hflip,
      rnd_vflip=rnd_vflip,
      rnd_transpose=rnd_transpose,
      rnd_colour=rnd_colour,
      y=None,
      d=d_gt,
      c=y_gt)
  x = results['x']
  y_gt = results['c']
  model['x_trans'] = x
  model['y_gt_trans'] = y_gt
  if add_orientation:
    d_gt = results['d']
    model['d_gt_trans'] = d_gt

  cnn_nlayers = len(cnn_depth)
  cnn_filter_size = [3] * cnn_nlayers
  cnn_channels = [inp_depth] + cnn_depth
  cnn_act = [tf.nn.relu] * cnn_nlayers
  cnn_use_bn = [use_bn] * cnn_nlayers

  cnn = nn.cnn(cnn_filter_size,
               cnn_channels,
               cnn_pool,
               cnn_act,
               cnn_use_bn,
               phase_train=phase_train,
               wd=wd,
               model=model)
  h_cnn = cnn(x)

  dcnn_nlayers = len(dcnn_filter_size)
  dcnn_act = [tf.nn.relu] * (dcnn_nlayers - 1) + [None]
  if add_skip_conn:
    dcnn_skip_ch = [0]
    dcnn_skip = [None]
    cnn_skip_layers = []
    cnn_skip_ch = []
    h_cnn_all = [x] + h_cnn[:-1]
    cnn_channels_all = cnn_channels
    for sk, ch, h in zip(cnn_skip_mask, cnn_channels_all, h_cnn_all):
      if sk:
        cnn_skip_ch.append(ch)
        cnn_skip_layers.append(h)
    counter = len(cnn_skip_ch) - 1
    for sk in dcnn_skip_mask:
      if sk:
        dcnn_skip_ch.append(cnn_skip_ch[counter])
        dcnn_skip.append(cnn_skip_layers[counter])
        counter -= 1
      else:
        dcnn_skip_ch.append(0)
        dcnn_skip.append(None)
  else:
    dcnn_skip_ch = None
    dcnn_skip = None
  dcnn_channels = [cnn_channels[-1]] + dcnn_depth
  dcnn_use_bn = [use_bn] * (dcnn_nlayers - 1) + [False]
  dcnn = nn.dcnn(
      dcnn_filter_size,
      dcnn_channels,
      dcnn_pool,
      dcnn_act,
      dcnn_use_bn,
      skip_ch=dcnn_skip_ch,
      model=model,
      phase_train=phase_train,
      wd=wd)
  h_cnn_last = h_cnn[-1]
  h_dcnn = dcnn(h_cnn_last, skip=dcnn_skip)
  if add_orientation:
    if dcnn_channels[-1] != num_orientation_classes + num_semantic_classes:
      log.error('Expecting last channel to be {}'.format(
          num_orientation_classes + num_semantic_classes))
      raise Exception('Expecting last channel to be {}'.format(
          num_orientation_classes + num_semantic_classes))
  else:
    if dcnn_channels[-1] != num_semantic_classes:
      log.error('Expecting last channel to be 1')
      raise Exception('Expecting last channel to be 1')

  if add_orientation:
    y_out = h_dcnn[-1][:, :, :, 0:num_semantic_classes]
    d_out = h_dcnn[-1][:, :, :, num_semantic_classes:]
    d_out = tf.nn.softmax(tf.reshape(d_out, [-1, num_orientation_classes]))
    d_out = tf.reshape(
        d_out, tf.pack([-1, inp_height, inp_width, num_orientation_classes]))
    model['d_out'] = d_out
  else:
    y_out = h_dcnn[-1]
  if num_semantic_classes == 1:
    y_out = tf.sigmoid(y_out)
  else:
    y_out_s = tf.shape(y_out)
    y_out = tf.reshape(
        tf.nn.softmax(tf.reshape(y_out, [-1, num_semantic_classes])), y_out_s)
  model['y_out'] = y_out

  num_ex_f = tf.to_float(num_ex)
  inp_height_f = tf.to_float(inp_height)
  inp_width_f = tf.to_float(inp_width)
  num_pixel = num_ex_f * inp_height_f * inp_width_f

  if num_semantic_classes > 1:
    y_gt_mask = tf.reduce_max(
        y_gt[:, :, :, 1:num_semantic_classes], [3], keep_dims=True)
  else:
    y_gt_mask = y_gt
  num_pixel_ori = tf.reduce_sum(y_gt_mask)

  if num_semantic_classes == 1:
    y_out_hard = tf.to_float(y_out > 0.5)
    iou_soft = modellib.f_iou_all(y_out, y_gt)
    iou_hard = modellib.f_iou_all(y_out_hard, y_gt)
  else:
    y_out_hard = tf.reduce_max(y_out, [3], keep_dims=True)
    y_out_hard = tf.to_float(tf.equal(y_out, y_out_hard))
    iou_soft = modellib.f_iou_all(y_out[:, :, :, 1:num_semantic_classes],
                                  y_gt[:, :, :, 1:num_semantic_classes])
    iou_hard = modellib.f_iou_all(y_out_hard[:, :, :, 1:num_semantic_classes],
                                  y_gt[:, :, :, 1:num_semantic_classes])
  model['iou_soft'] = iou_soft
  model['iou_hard'] = iou_hard
  if num_semantic_classes == 1:
    segloss = tf.reduce_sum(modellib.f_bce(y_out, y_gt), [1, 2, 3])
    segloss = tf.reduce_sum(segloss) / num_pixel
  else:
    segloss = tf.reduce_sum(modellib.f_ce(y_out, y_gt), [1, 2, 3])
    segloss = tf.reduce_sum(segloss) / num_pixel

  if segm_loss_fn == 'iou':
    loss = -iou_soft
  elif segm_loss_fn == 'bce':
    loss = segloss

  model['foreground_loss'] = loss

  if add_orientation:
    ys = tf.shape(y_gt_mask)
    orientation_ce = tf.reduce_sum(
        modellib.f_ce(d_out, d_gt) * y_gt_mask, [1, 2, 3])
    orientation_ce = tf.reduce_sum(orientation_ce) / num_pixel_ori
    loss += orientation_ce
    model['orientation_ce'] = orientation_ce
    correct = tf.equal(tf.argmax(d_out, 3), tf.argmax(d_gt, 3))
    y_gt_mask = tf.squeeze(y_gt_mask, [3])
    orientation_acc = tf.reduce_sum(tf.to_float(correct) *
                                    y_gt_mask) / tf.reduce_sum(y_gt_mask)
    model['orientation_acc'] = orientation_acc

  model['loss'] = loss
  tf.add_to_collection('losses', loss)
  total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

  learn_rate = tf.train.exponential_decay(
      base_learn_rate,
      global_step,
      steps_per_learn_rate_decay,
      learn_rate_decay,
      staircase=True)
  eps = 1e-7

  if optimizer == 'adam':
    optim = tf.train.AdamOptimizer(learn_rate, epsilon=eps)
  elif optimizer == 'momentum':
    optim = tf.train.MomentumOptimizer(learn_rate, momentum=0.9)

  train_step = optim.minimize(total_loss, global_step=global_step)
  model['train_step'] = train_step
  return model


def get_save_var(model):
  results = {}
  results['step'] = model['global_step']
  for net in ['cnn', 'dcnn']:
    for ii in range(10000):
      if '{}_w_{}'.format(net, ii) not in model:
        break
      for w in ['w', 'b']:
        key = '{}_{}_{}'.format(net, w, ii)
        results['{}/layer_{}/{}'.format(net, ii, w)] = model[key]
      if net == 'cnn' or net == 'dcnn':
        for w in ['beta', 'gamma', 'ema_mean', 'ema_var']:
          key = '{}_{}_{}_{}'.format(net, ii, 0, w)
          if key in model:
            results['{}/layer_{}/bn/{}'.format(net, ii, w)] = \
                model[key]
  return results