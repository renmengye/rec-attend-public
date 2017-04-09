import h5py
import image_ops as img
import nnlib as nn
import numpy as np
import tensorflow as tf

import modellib
from utils import logger


def get_model(opt):
  """The box model"""
  log = logger.get()
  model = {}

  timespan = opt['timespan']
  inp_height = opt['inp_height']
  inp_width = opt['inp_width']
  inp_depth = opt['inp_depth']
  padding = opt['padding']
  filter_height = opt['filter_height']
  filter_width = opt['filter_width']

  ctrl_cnn_filter_size = opt['ctrl_cnn_filter_size']
  ctrl_cnn_depth = opt['ctrl_cnn_depth']
  ctrl_cnn_pool = opt['ctrl_cnn_pool']
  ctrl_rnn_hid_dim = opt['ctrl_rnn_hid_dim']

  num_ctrl_mlp_layers = opt['num_ctrl_mlp_layers']
  ctrl_mlp_dim = opt['ctrl_mlp_dim']

  attn_box_padding_ratio = opt['attn_box_padding_ratio']

  wd = opt['weight_decay']
  use_bn = opt['use_bn']
  box_loss_fn = opt['box_loss_fn']
  base_learn_rate = opt['base_learn_rate']
  learn_rate_decay = opt['learn_rate_decay']
  steps_per_learn_rate_decay = opt['steps_per_learn_rate_decay']
  pretrain_cnn = opt['pretrain_cnn']

  if 'pretrain_net' in opt:
    pretrain_net = opt['pretrain_net']
  else:
    pretrain_net = None

  if 'freeze_pretrain_cnn' in opt:
    freeze_pretrain_cnn = opt['freeze_pretrain_cnn']
  else:
    freeze_pretrain_cnn = True

  squash_ctrl_params = opt['squash_ctrl_params']
  clip_gradient = opt['clip_gradient']
  fixed_order = opt['fixed_order']
  num_ctrl_rnn_iter = opt['num_ctrl_rnn_iter']
  num_glimpse_mlp_layers = opt['num_glimpse_mlp_layers']

  if 'fixed_var' in opt:
    fixed_var = opt['fixed_var']
  else:
    fixed_var = True

  if 'use_iou_box' in opt:
    use_iou_box = opt['use_iou_box']
  else:
    use_iou_box = False

  if 'dynamic_var' in opt:
    dynamic_var = opt['dynamic_var']
  else:
    dynamic_var = False

  if 'num_semantic_classes' in opt:
    num_semantic_classes = opt['num_semantic_classes']
  else:
    num_semantic_classes = 1

  if 'add_d_out' in opt:
    add_d_out = opt['add_d_out']
    add_y_out = opt['add_y_out']
  else:
    add_d_out = False
    add_y_out = False

  rnd_hflip = opt['rnd_hflip']
  rnd_vflip = opt['rnd_vflip']
  rnd_transpose = opt['rnd_transpose']
  rnd_colour = opt['rnd_colour']

  ############################
  # Input definition
  ############################
  # Input image, [B, H, W, D]
  x = tf.placeholder(
      'float', [None, inp_height, inp_width, inp_depth], name='x')
  x_shape = tf.shape(x)
  num_ex = x_shape[0]

  # Groundtruth segmentation, [B, T, H, W]
  y_gt = tf.placeholder(
      'float', [None, timespan, inp_height, inp_width], name='y_gt')

  # Groundtruth confidence score, [B, T]
  s_gt = tf.placeholder('float', [None, timespan], name='s_gt')

  if add_d_out:
    d_in = tf.placeholder(
        'float', [None, inp_height, inp_width, 8], name='d_in')
    model['d_in'] = d_in
  if add_y_out:
    y_in = tf.placeholder(
        'float', [None, inp_height, inp_width, num_semantic_classes],
        name='y_in')
    model['y_in'] = y_in
  # Whether in training stage.
  phase_train = tf.placeholder('bool', name='phase_train')
  phase_train_f = tf.to_float(phase_train)
  model['x'] = x
  model['y_gt'] = y_gt
  model['s_gt'] = s_gt
  model['phase_train'] = phase_train

  # Global step
  global_step = tf.Variable(0.0, name='global_step')

  ###############################
  # Random input transformation
  ###############################
  # Either add both or add nothing.
  assert (add_d_out and add_y_out) or (not add_d_out and not add_y_out)
  if not add_d_out:
    results = img.random_transformation(
        x,
        padding,
        phase_train,
        rnd_hflip=rnd_hflip,
        rnd_vflip=rnd_vflip,
        rnd_transpose=rnd_transpose,
        rnd_colour=rnd_colour,
        y=y_gt)
    x, y_gt = results['x'], results['y']
  else:
    results = img.random_transformation(
        x,
        padding,
        phase_train,
        rnd_hflip=rnd_hflip,
        rnd_vflip=rnd_vflip,
        rnd_transpose=rnd_transpose,
        rnd_colour=rnd_colour,
        y=y_gt,
        d=d_in,
        c=y_in)
    x, y_gt, d_in, y_in = results['x'], results['y'], results['d'], results['c']
    model['d_in_trans'] = d_in
    model['y_in_trans'] = y_in
  model['x_trans'] = x
  model['y_gt_trans'] = y_gt

  ############################
  # Canvas: external memory
  ############################
  canvas = tf.zeros(tf.pack([num_ex, inp_height, inp_width, 1]))
  ccnn_inp_depth = inp_depth + 1
  acnn_inp_depth = inp_depth + 1
  if add_d_out:
    ccnn_inp_depth += 8
    acnn_inp_depth += 8
  if add_y_out:
    ccnn_inp_depth += num_semantic_classes
    acnn_inp_depth += num_semantic_classes

  ############################
  # Controller CNN definition
  ############################
  ccnn_filters = ctrl_cnn_filter_size
  ccnn_nlayers = len(ccnn_filters)
  ccnn_channels = [ccnn_inp_depth] + ctrl_cnn_depth
  ccnn_pool = ctrl_cnn_pool
  ccnn_act = [tf.nn.relu] * ccnn_nlayers
  ccnn_use_bn = [use_bn] * ccnn_nlayers
  pt = pretrain_net or pretrain_cnn

  if pt:
    log.info('Loading pretrained weights from {}'.format(pt))
    with h5py.File(pt, 'r') as h5f:
      pt_cnn_nlayers = 0
      # Assuming pt_cnn_nlayers is smaller than or equal to
      # ccnn_nlayers.
      for ii in range(ccnn_nlayers):
        if 'attn_cnn_w_{}'.format(ii) in h5f:
          cnn_prefix = 'attn_'
          log.info('Loading attn_cnn_w_{}'.format(ii))
          log.info('Loading attn_cnn_b_{}'.format(ii))
          pt_cnn_nlayers += 1
        elif 'cnn_w_{}'.format(ii) in h5f:
          cnn_prefix = ''
          log.info('Loading cnn_w_{}'.format(ii))
          log.info('Loading cnn_b_{}'.format(ii))
          pt_cnn_nlayers += 1
        elif 'ctrl_cnn_w_{}'.format(ii) in h5f:
          cnn_prefix = 'ctrl_'
          log.info('Loading ctrl_cnn_w_{}'.format(ii))
          log.info('Loading ctrl_cnn_b_{}'.format(ii))
          pt_cnn_nlayers += 1

      ccnn_init_w = [{
          'w': h5f['{}cnn_w_{}'.format(cnn_prefix, ii)][:],
          'b': h5f['{}cnn_b_{}'.format(cnn_prefix, ii)][:]
      } for ii in range(pt_cnn_nlayers)]
      for ii in range(pt_cnn_nlayers):
        for tt in range(timespan):
          for w in ['beta', 'gamma']:
            ccnn_init_w[ii]['{}_{}'.format(w, tt)] = h5f[
                '{}cnn_{}_{}_{}'.format(cnn_prefix, ii, tt, w)][:]
      ccnn_frozen = [freeze_pretrain_cnn] * pt_cnn_nlayers
      for ii in range(pt_cnn_nlayers, ccnn_nlayers):
        ccnn_init_w.append(None)
        ccnn_frozen.append(False)
  else:
    ccnn_init_w = None
    ccnn_frozen = None

  ccnn = nn.cnn(ccnn_filters,
                ccnn_channels,
                ccnn_pool,
                ccnn_act,
                ccnn_use_bn,
                phase_train=phase_train,
                wd=wd,
                scope='ctrl_cnn',
                model=model,
                init_weights=ccnn_init_w,
                frozen=ccnn_frozen)
  h_ccnn = [None] * timespan

  ############################
  # Controller RNN definition
  ############################
  ccnn_subsample = np.array(ccnn_pool).prod()
  crnn_h = inp_height / ccnn_subsample
  crnn_w = inp_width / ccnn_subsample
  crnn_dim = ctrl_rnn_hid_dim
  canvas_dim = inp_height * inp_width / (ccnn_subsample**2)

  glimpse_map_dim = crnn_h * crnn_w
  glimpse_feat_dim = ccnn_channels[-1]
  crnn_inp_dim = glimpse_feat_dim

  pt = pretrain_net
  if pt:
    log.info('Loading pretrained controller RNN weights from {}'.format(pt))
    h5f = h5py.File(pt, 'r')
    crnn_init_w = {}
    for w in [
        'w_xi', 'w_hi', 'b_i', 'w_xf', 'w_hf', 'b_f', 'w_xu', 'w_hu', 'b_u',
        'w_xo', 'w_ho', 'b_o'
    ]:
      key = 'ctrl_lstm_{}'.format(w)
      crnn_init_w[w] = h5f[key][:]
    crnn_frozen = None
  else:
    crnn_init_w = None
    crnn_frozen = None

  crnn_state = [None] * (timespan + 1)
  crnn_glimpse_map = [None] * timespan
  crnn_g_i = [None] * timespan
  crnn_g_f = [None] * timespan
  crnn_g_o = [None] * timespan
  h_crnn = [None] * timespan
  crnn_state[-1] = tf.zeros(tf.pack([num_ex, crnn_dim * 2]))
  crnn_cell = nn.lstm(
      crnn_inp_dim,
      crnn_dim,
      wd=wd,
      scope='ctrl_lstm',
      init_weights=crnn_init_w,
      frozen=crnn_frozen,
      model=model)

  ############################
  # Glimpse MLP definition
  ############################
  gmlp_dims = [crnn_dim] * num_glimpse_mlp_layers + [glimpse_map_dim]
  gmlp_act = [tf.nn.relu] * \
      (num_glimpse_mlp_layers - 1) + [tf.nn.softmax]
  gmlp_dropout = None

  pt = pretrain_net
  if pt:
    log.info('Loading pretrained glimpse MLP weights from {}'.format(pt))
    h5f = h5py.File(pt, 'r')
    gmlp_init_w = [{
        'w': h5f['glimpse_mlp_w_{}'.format(ii)][:],
        'b': h5f['glimpse_mlp_b_{}'.format(ii)][:]
    } for ii in range(num_glimpse_mlp_layers)]
    gmlp_frozen = None
  else:
    gmlp_init_w = None
    gmlp_frozen = None

  gmlp = nn.mlp(gmlp_dims,
                gmlp_act,
                add_bias=True,
                dropout_keep=gmlp_dropout,
                phase_train=phase_train,
                wd=wd,
                scope='glimpse_mlp',
                init_weights=gmlp_init_w,
                frozen=gmlp_frozen,
                model=model)

  ############################
  # Controller MLP definition
  ############################
  cmlp_dims = [crnn_dim] + [ctrl_mlp_dim] * \
      (num_ctrl_mlp_layers - 1) + [9]
  cmlp_act = [tf.nn.relu] * (num_ctrl_mlp_layers - 1) + [None]
  cmlp_dropout = None

  pt = pretrain_net
  if pt:
    log.info('Loading pretrained controller MLP weights from {}'.format(pt))
    h5f = h5py.File(pt, 'r')
    cmlp_init_w = [{
        'w': h5f['ctrl_mlp_w_{}'.format(ii)][:],
        'b': h5f['ctrl_mlp_b_{}'.format(ii)][:]
    } for ii in range(num_ctrl_mlp_layers)]
    cmlp_frozen = None
  else:
    cmlp_init_w = None
    cmlp_frozen = None

  cmlp = nn.mlp(cmlp_dims,
                cmlp_act,
                add_bias=True,
                dropout_keep=cmlp_dropout,
                phase_train=phase_train,
                wd=wd,
                scope='ctrl_mlp',
                init_weights=cmlp_init_w,
                frozen=cmlp_frozen,
                model=model)

  ##########################
  # Score MLP definition
  ##########################
  pt = pretrain_net
  if pt:
    log.info('Loading score mlp weights from {}'.format(pt))
    h5f = h5py.File(pt, 'r')
    smlp_init_w = [{
        'w': h5f['score_mlp_w_{}'.format(ii)][:],
        'b': h5f['score_mlp_b_{}'.format(ii)][:]
    } for ii in range(1)]
  else:
    smlp_init_w = None
  smlp = nn.mlp([crnn_dim, num_semantic_classes], [None],
                wd=wd,
                scope='score_mlp',
                init_weights=smlp_init_w,
                model=model)
  s_out = [None] * timespan

  ##########################
  # Attention box
  ##########################
  attn_ctr_norm = [None] * timespan
  attn_lg_size = [None] * timespan
  attn_lg_var = [None] * timespan
  attn_ctr = [None] * timespan
  attn_size = [None] * timespan
  attn_top_left = [None] * timespan
  attn_bot_right = [None] * timespan
  attn_box = [None] * timespan
  attn_box_lg_gamma = [None] * timespan
  attn_box_gamma = [None] * timespan
  const_ones = tf.ones(tf.pack([num_ex, filter_height, filter_width, 1]))
  attn_box_beta = tf.constant([-5.0])
  iou_soft_box = [None] * timespan

  #############################
  # Groundtruth attention box
  #############################
  attn_top_left_gt, attn_bot_right_gt, attn_box_gt = modellib.get_gt_box(
      y_gt, padding_ratio=attn_box_padding_ratio, center_shift_ratio=0.0)
  attn_ctr_gt, attn_size_gt = modellib.get_box_ctr_size(attn_top_left_gt,
                                                        attn_bot_right_gt)
  attn_ctr_norm_gt = modellib.get_normalized_center(attn_ctr_gt, inp_height,
                                                    inp_width)
  attn_lg_size_gt = modellib.get_normalized_size(attn_size_gt, inp_height,
                                                 inp_width)

  ##########################
  # Groundtruth mix
  ##########################
  grd_match_cum = tf.zeros(tf.pack([num_ex, timespan]))

  ##########################
  # Computation graph
  ##########################
  for tt in range(timespan):
    # Controller CNN
    ccnn_inp_list = [x, canvas]
    if add_d_out:
      ccnn_inp_list.append(d_in)
    if add_y_out:
      ccnn_inp_list.append(y_in)
    ccnn_inp = tf.concat(3, ccnn_inp_list)
    acnn_inp = ccnn_inp
    h_ccnn[tt] = ccnn(ccnn_inp)
    _h_ccnn = h_ccnn[tt]
    h_ccnn_last = _h_ccnn[-1]

    # Controller RNN [B, R1]
    crnn_inp = tf.reshape(h_ccnn_last, [-1, glimpse_map_dim, glimpse_feat_dim])
    crnn_state[tt] = [None] * (num_ctrl_rnn_iter + 1)
    crnn_g_i[tt] = [None] * num_ctrl_rnn_iter
    crnn_g_f[tt] = [None] * num_ctrl_rnn_iter
    crnn_g_o[tt] = [None] * num_ctrl_rnn_iter
    h_crnn[tt] = [None] * num_ctrl_rnn_iter

    crnn_state[tt][-1] = tf.zeros(tf.pack([num_ex, crnn_dim * 2]))

    crnn_glimpse_map[tt] = [None] * num_ctrl_rnn_iter
    crnn_glimpse_map[tt][0] = tf.ones(tf.pack([num_ex, glimpse_map_dim, 1
                                              ])) / glimpse_map_dim

    # Inner glimpse RNN
    for tt2 in range(num_ctrl_rnn_iter):
      crnn_glimpse = tf.reduce_sum(crnn_inp * crnn_glimpse_map[tt][tt2], [1])
      crnn_state[tt][tt2], crnn_g_i[tt][tt2], crnn_g_f[tt][tt2], \
          crnn_g_o[tt][tt2] = \
          crnn_cell(crnn_glimpse, crnn_state[tt][tt2 - 1])
      h_crnn[tt][tt2] = tf.slice(crnn_state[tt][tt2], [0, crnn_dim],
                                 [-1, crnn_dim])
      h_gmlp = gmlp(h_crnn[tt][tt2])
      if tt2 < num_ctrl_rnn_iter - 1:
        crnn_glimpse_map[tt][tt2 + 1] = tf.expand_dims(h_gmlp[-1], 2)

    ctrl_out = cmlp(h_crnn[tt][-1])[-1]

    attn_ctr_norm[tt] = tf.slice(ctrl_out, [0, 0], [-1, 2])
    attn_lg_size[tt] = tf.slice(ctrl_out, [0, 2], [-1, 2])

    # Restrict to (-1, 1), (-inf, 0)
    if squash_ctrl_params:
      attn_ctr_norm[tt] = tf.tanh(attn_ctr_norm[tt])
      attn_lg_size[tt] = -tf.nn.softplus(attn_lg_size[tt])

    attn_ctr[tt], attn_size[tt] = modellib.get_unnormalized_attn(
        attn_ctr_norm[tt], attn_lg_size[tt], inp_height, inp_width)
    attn_box_lg_gamma[tt] = tf.slice(ctrl_out, [0, 7], [-1, 1])

    if fixed_var:
      attn_lg_var[tt] = tf.zeros(tf.pack([num_ex, 2]))
    else:
      attn_lg_var[tt] = modellib.get_normalized_var(attn_size[tt],
                                                    filter_height, filter_width)
    if dynamic_var:
      attn_lg_var[tt] = tf.slice(ctrl_out, [0, 4], [-1, 2])
    attn_box_gamma[tt] = tf.reshape(
        tf.exp(attn_box_lg_gamma[tt]), [-1, 1, 1, 1])
    attn_top_left[tt], attn_bot_right[tt] = modellib.get_box_coord(
        attn_ctr[tt], attn_size[tt])

    # Initial filters (predicted)
    filter_y = modellib.get_gaussian_filter(
        attn_ctr[tt][:, 0], attn_size[tt][:, 0], attn_lg_var[tt][:, 0],
        inp_height, filter_height)
    filter_x = modellib.get_gaussian_filter(
        attn_ctr[tt][:, 1], attn_size[tt][:, 1], attn_lg_var[tt][:, 1],
        inp_width, filter_width)
    filter_y_inv = tf.transpose(filter_y, [0, 2, 1])
    filter_x_inv = tf.transpose(filter_x, [0, 2, 1])

    # Attention box
    attn_box[tt] = attn_box_gamma[tt] * modellib.extract_patch(
        const_ones, filter_y_inv, filter_x_inv, 1)
    attn_box[tt] = tf.sigmoid(attn_box[tt] + attn_box_beta)
    attn_box[tt] = tf.reshape(attn_box[tt], [-1, 1, inp_height, inp_width])

    if fixed_order:
      _y_out = tf.expand_dims(y_gt[:, tt, :, :], 3)
    else:
      if use_iou_box:
        iou_soft_box[tt] = modellib.f_iou_box(
            tf.expand_dims(attn_top_left[tt], 1),
            tf.expand_dims(attn_bot_right[tt], 1), attn_top_left_gt,
            attn_bot_right_gt)
      else:
        iou_soft_box[tt] = modellib.f_inter(
            attn_box[tt], attn_box_gt) / \
            modellib.f_union(attn_box[tt], attn_box_gt, eps=1e-5)
      grd_match = modellib.f_greedy_match(iou_soft_box[tt], grd_match_cum)
      grd_match = tf.expand_dims(tf.expand_dims(grd_match, 2), 3)
      _y_out = tf.expand_dims(tf.reduce_sum(grd_match * y_gt, 1), 3)

    # Add independent uniform noise to groundtruth.
    _noise = tf.random_uniform(
        tf.pack([num_ex, inp_height, inp_width, 1]), 0, 0.3)
    _y_out = _y_out - _y_out * _noise
    canvas = tf.stop_gradient(tf.maximum(_y_out, canvas))
    # canvas += tf.stop_gradient(_y_out)

    # Scoring network
    s_out[tt] = smlp(h_crnn[tt][-1])[-1]

    if num_semantic_classes == 1:
      s_out[tt] = tf.sigmoid(s_out[tt])
    else:
      s_out[tt] = tf.nn.softmax(s_out[tt])

  #########################
  # Model outputs
  #########################
  s_out = tf.concat(1, [tf.expand_dims(tmp, 1) for tmp in s_out])
  if num_semantic_classes == 1:
    s_out = s_out[:, :, 0]
  model['s_out'] = s_out
  attn_box = tf.concat(1, attn_box)
  model['attn_box'] = attn_box
  attn_top_left = tf.concat(1,
                            [tf.expand_dims(tmp, 1) for tmp in attn_top_left])
  attn_bot_right = tf.concat(1,
                             [tf.expand_dims(tmp, 1) for tmp in attn_bot_right])
  attn_ctr = tf.concat(1, [tf.expand_dims(tmp, 1) for tmp in attn_ctr])
  attn_size = tf.concat(1, [tf.expand_dims(tmp, 1) for tmp in attn_size])
  model['attn_top_left'] = attn_top_left
  model['attn_bot_right'] = attn_bot_right
  model['attn_ctr'] = attn_ctr
  model['attn_size'] = attn_size
  model['attn_ctr_norm_gt'] = attn_ctr_norm_gt
  model['attn_lg_size_gt'] = attn_lg_size_gt
  model['attn_top_left_gt'] = attn_top_left_gt
  model['attn_bot_right_gt'] = attn_bot_right_gt
  model['attn_box_gt'] = attn_box_gt
  attn_ctr_norm = tf.concat(1,
                            [tf.expand_dims(tmp, 1) for tmp in attn_ctr_norm])
  attn_lg_size = tf.concat(1, [tf.expand_dims(tmp, 1) for tmp in attn_lg_size])
  model['attn_ctr_norm'] = attn_ctr_norm
  model['attn_lg_size'] = attn_lg_size

  attn_params = tf.concat(2, [attn_ctr_norm, attn_lg_size])
  attn_params_gt = tf.concat(2, [attn_ctr_norm_gt, attn_lg_size_gt])

  #########################
  # Loss function
  #########################
  y_gt_shape = tf.shape(y_gt)
  num_ex_f = tf.to_float(y_gt_shape[0])
  max_num_obj = tf.to_float(y_gt_shape[1])

  ############################
  # Box loss
  ############################
  if fixed_order:
    # [B, T] for fixed order.
    iou_soft_box = modellib.f_iou(attn_box, attn_box_gt, pairwise=False)
  else:
    # [B, T, T] for matching.
    iou_soft_box = tf.concat(
        1, [tf.expand_dims(iou_soft_box[tt], 1) for tt in range(timespan)])

  identity_match = modellib.get_identity_match(num_ex, timespan, s_gt)
  if fixed_order:
    match_box = identity_match
  else:
    match_box = modellib.f_segm_match(iou_soft_box, s_gt)

  model['match_box'] = match_box
  match_sum_box = tf.reduce_sum(match_box, reduction_indices=[2])
  match_count_box = tf.reduce_sum(match_sum_box, reduction_indices=[1])
  match_count_box = tf.maximum(1.0, match_count_box)

  # [B] if fixed order, [B, T] if matching.
  if fixed_order:
    iou_soft_box_mask = iou_soft_box
  else:
    iou_soft_box_mask = tf.reduce_sum(iou_soft_box * match_box, [1])
  iou_soft_box = tf.reduce_sum(iou_soft_box_mask, [1])
  iou_soft_box = tf.reduce_sum(iou_soft_box / match_count_box) / num_ex_f

  if box_loss_fn == 'mse':
    box_loss = modellib.f_match_loss(
        attn_params,
        attn_params_gt,
        match_box,
        timespan,
        modellib.f_squared_err,
        model=model)
  elif box_loss_fn == 'huber':
    box_loss = modellib.f_match_loss(attn_params, attn_params_gt, match_box,
                                     timespan, modellib.f_huber)
  if box_loss_fn == 'iou':
    box_loss = -iou_soft_box
  elif box_loss_fn == 'wt_iou':
    box_loss = -wt_iou_soft_box
  elif box_loss_fn == 'wt_cov':
    box_loss = -modellib.f_weighted_coverage(iou_soft_box, box_map_gt)
  elif box_loss_fn == 'bce':
    box_loss = modellib.f_match_loss(box_map, box_map_gt, match_box, timespan,
                                     modellib.f_bce)
  else:
    raise Exception('Unknown box_loss_fn: {}'.format(box_loss_fn))
  model['box_loss'] = box_loss

  box_loss_coeff = tf.constant(1.0)
  model['box_loss_coeff'] = box_loss_coeff
  tf.add_to_collection('losses', box_loss_coeff * box_loss)

  ####################
  # Score loss
  ####################
  if num_semantic_classes == 1:
    conf_loss = modellib.f_conf_loss(
        s_out, match_box, timespan, use_cum_min=True)
  else:
    conf_loss = modellib.f_conf_loss(
        1 - s_out[:, :, 0], match_box, timespan, use_cum_min=True)
  model['conf_loss'] = conf_loss
  conf_loss_coeff = tf.constant(1.0)
  tf.add_to_collection('losses', conf_loss_coeff * conf_loss)

  ####################
  # Total loss
  ####################
  total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
  model['loss'] = total_loss

  ####################
  # Optimizer
  ####################
  learn_rate = tf.train.exponential_decay(
      base_learn_rate,
      global_step,
      steps_per_learn_rate_decay,
      learn_rate_decay,
      staircase=True)
  model['learn_rate'] = learn_rate
  eps = 1e-7
  optim = tf.train.AdamOptimizer(learn_rate, epsilon=eps)
  gvs = optim.compute_gradients(total_loss)
  capped_gvs = []
  for grad, var in gvs:
    if grad is not None:
      capped_gvs.append((tf.clip_by_value(grad, -1, 1), var))
    else:
      capped_gvs.append((grad, var))
  train_step = optim.apply_gradients(capped_gvs, global_step=global_step)
  model['train_step'] = train_step

  ####################
  # Glimpse
  ####################
  # T * T2 * [B, H' * W'] => [B, T, T2, H', W']
  crnn_glimpse_map = tf.concat(1, [
      tf.expand_dims(
          tf.concat(1, [
              tf.expand_dims(crnn_glimpse_map[tt][tt2], 1)
              for tt2 in range(num_ctrl_rnn_iter)
          ]), 1) for tt in range(timespan)
  ])
  crnn_glimpse_map = tf.reshape(
      crnn_glimpse_map, [-1, timespan, num_ctrl_rnn_iter, crnn_h, crnn_w])
  model['ctrl_rnn_glimpse_map'] = crnn_glimpse_map

  return model
