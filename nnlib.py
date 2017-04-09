import tensorflow as tf
import os
import utils.logger as logger


def conv2d(x, w, stride=1):
  """2-D convolution.
  Args:
    x: input tensor, [B, H, W, D]
    w: filter tensor, [F, F, In, Out]
  """
  return tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding='SAME')


def max_pool(x, ratio):
  """N x N max pooling.
  Args:
    x: input tensor, [B, H, W, D]
    ratio: N by N pooling ratio
  """
  return tf.nn.max_pool(
      x,
      ksize=[1, ratio, ratio, 1],
      strides=[1, ratio, ratio, 1],
      padding='SAME')


def avg_pool(x, ratio):
  """N x N max pooling.
  Args:
    x: input tensor, [B, H, W, D]
    ratio: N by N pooling ratio
  """
  return tf.nn.avg_pool(
      x,
      ksize=[1, ratio, ratio, 1],
      strides=[1, ratio, ratio, 1],
      padding='SAME')


def weight_variable(shape,
                    initializer=None,
                    init_val=None,
                    wd=None,
                    name=None,
                    trainable=True):
  """Initialize weights.
  Args:
    shape: shape of the weights, list of int
    wd: weight decay
  """
  log = logger.get()
  if initializer is None:
    initializer = tf.truncated_normal_initializer(stddev=0.01)
  if init_val is None:
    var = tf.Variable(initializer(shape), name=name, trainable=trainable)
  else:
    var = tf.Variable(init_val, name=name, trainable=trainable)
  if wd:
    weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


def batch_norm(x,
               n_out,
               phase_train,
               scope='bn',
               scope2='bn',
               affine=True,
               init_beta=None,
               init_gamma=None,
               frozen=False,
               model=None):
  """
  Batch normalization on convolutional maps.
  Args:
      x: input tensor, [B, H, W, D]
      n_out: integer, depth of input maps
      phase_train: boolean tf.Variable, true indicates training phase
      scope: string, variable scope
      affine: whether to affine-transform outputs
  Return:
      normed: batch-normalized maps
  """
  trainable = not frozen
  with tf.variable_scope(scope):
    if init_beta is None:
      init_beta = tf.constant(0.0, shape=[n_out])
    if init_gamma is None:
      init_gamma = tf.constant(1.0, shape=[n_out])

    beta = weight_variable(
        [n_out], init_val=init_beta, name='beta', trainable=trainable)
    gamma = weight_variable(
        [n_out], init_val=init_gamma, name='gamma', trainable=trainable)

    batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
    batch_mean.set_shape([n_out])
    batch_var.set_shape([n_out])

    phase_train_f = tf.to_float(phase_train)
    decay = 1 - 0.1 * phase_train_f
    ema = tf.train.ExponentialMovingAverage(decay=decay)

    def mean_var_with_update():
      ema_apply_op_local = ema.apply([batch_mean, batch_var])
      with tf.control_dependencies([ema_apply_op_local]):
        return tf.identity(batch_mean), tf.identity(batch_var)
      with tf.control_dependencies([ema_apply_op]):
        return tf.identity(batch_mean), tf.identity(batch_var)

    def mean_var_no_update():
      ema_mean_local, ema_var_local = ema.average(batch_mean), ema.average(
          batch_var)
      return ema_mean_local, ema_var_local

    mean, var = tf.cond(phase_train, mean_var_with_update, mean_var_no_update)
    normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    if model is not None:
      for name, param in zip(['beta', 'gamma', 'ema_mean', 'ema_var'], [
          beta, gamma, ema.average(batch_mean), ema.average(batch_var)
      ]):
        key = '{}_{}'.format(scope2, name)
        if key in model:
          raise Exception('Key exists: {}'.format(key))
        model[key] = param
  return normed


def cnn(f,
        ch,
        pool,
        act,
        use_bn,
        phase_train=None,
        wd=None,
        scope='cnn',
        model=None,
        init_weights=None,
        frozen=None,
        shared_weights=None):
  """Add CNN. N = number of layers.

    Args:
        f: filter size, list of N int
        ch: number of channels, list of (N + 1) int
        pool: pooling ratio, list of N int
        act: activation function, list of N function
        use_bn: whether to use batch normalization, list of N bool
        phase_train: whether in training phase, tf bool variable
        wd: weight decay

    Returns:
        run_cnn: a function that runs the CNN
    """
  log = logger.get()

  nlayers = len(f)
  w = [None] * nlayers
  b = [None] * nlayers
  log.info('CNN: {}'.format(scope))
  log.info('Channels: {}'.format(ch))
  log.info('Activation: {}'.format(act))
  log.info('Pool: {}'.format(pool))
  log.info('BN: {}'.format(use_bn))
  log.info('Shared weights: {}'.format(shared_weights))
  net_scope = None
  layer_scope = [None] * nlayers
  with tf.variable_scope(scope):
    for ii in range(nlayers):
      with tf.variable_scope('layer_{}'.format(ii)):
        if init_weights:
          init = tf.constant_initializer
        else:
          init = None

        if init_weights is not None and init_weights[ii] is not None:
          init_val_w = init_weights[ii]['w']
          init_val_b = init_weights[ii]['b']
        else:
          init_val_w = None
          init_val_b = None

        if frozen is not None and frozen[ii]:
          trainable = False
        else:
          trainable = True

        if shared_weights:
          w[ii] = shared_weights[ii]['w']
          b[ii] = shared_weights[ii]['b']
        else:
          w[ii] = weight_variable(
              [f[ii], f[ii], ch[ii], ch[ii + 1]],
              name='w',
              init_val=init_val_w,
              wd=wd,
              trainable=trainable)
          b[ii] = weight_variable(
              [ch[ii + 1]], init_val=init_val_b, name='b', trainable=trainable)

        log.info('Filter: {}, Trainable: {}'.format(
            [f[ii], f[ii], ch[ii], ch[ii + 1]], trainable))

        if model is not None:
          for name, param in zip(['w', 'b'], [w[ii], b[ii]]):
            key = '{}_{}_{}'.format(scope, name, ii)
            if key in model:
              raise Exception('Key exists: {}'.format(key))
            model[key] = param
  copy = [0]

  def run_cnn(x):
    """
    Run CNN on an input.
    Args:
      x: input image, [B, H, W, D]
    """
    h = [None] * nlayers
    with tf.variable_scope(scope):
      for ii in range(nlayers):
        with tf.variable_scope('layer_{}'.format(ii)):
          out_ch = ch[ii + 1]
          if ii == 0:
            prev_inp = x
          else:
            prev_inp = h[ii - 1]
          h[ii] = conv2d(prev_inp, w[ii]) + b[ii]
          if use_bn[ii]:
            if frozen is not None and frozen[ii]:
              bn_frozen = True
            else:
              bn_frozen = False
            if init_weights is not None and \
                    init_weights[ii] is not None:
              init_beta = init_weights[ii]['beta_{}'.format(copy[0])]
              init_gamma = init_weights[ii]['gamma_{}'.format(copy[0])]
            else:
              init_beta = None
              init_gamma = None
            h[ii] = batch_norm(
                h[ii],
                out_ch,
                phase_train,
                scope2='{}_{}_{}'.format(scope, ii, copy[0]),
                init_beta=init_beta,
                init_gamma=init_gamma,
                model=model)
          if act[ii] is not None:
            h[ii] = act[ii](h[ii])
          if pool[ii] > 1:
            h[ii] = max_pool(h[ii], pool[ii])
    copy[0] += 1
    return h

  return run_cnn


def dcnn(f,
         ch,
         pool,
         act,
         use_bn,
         skip_ch=None,
         phase_train=None,
         wd=None,
         scope='dcnn',
         model=None,
         init_weights=None,
         frozen=None):
  """Add DCNN. N = number of layers.
  Args:
    f: filter size, list of size N  int
    ch: number of channels, list of (N + 1) int
    pool: pooling ratio, list of N int
    act: activation function, list of N function
    use_bn: whether to use batch normalization, list of N bool
    skip_ch: skip connection, list of N int
    phase_train: whether in training phase, tf bool variable
    wd: weight decay
  Returns:
    run_dcnn: a function that runs the DCNN
  """
  log = logger.get()

  nlayers = len(f)
  w = [None] * nlayers
  b = [None] * nlayers
  bn = [None] * nlayers

  log.info('DCNN: {}'.format(scope))
  log.info('Channels: {}'.format(ch))
  log.info('Activation: {}'.format(act))
  log.info('Unpool: {}'.format(pool))
  log.info('Skip channels: {}'.format(skip_ch))
  log.info('BN: {}'.format(use_bn))

  with tf.variable_scope(scope):
    in_ch = ch[0]
    for ii in range(nlayers):
      with tf.variable_scope('layer_{}'.format(ii)):
        out_ch = ch[ii + 1]
        if skip_ch is not None:
          if skip_ch[ii] is not None:
            in_ch += skip_ch[ii]

        if init_weights is not None and init_weights[ii] is not None:
          init_val_w = init_weights[ii]['w']
          init_val_b = init_weights[ii]['b']
        else:
          init_val_w = None
          init_val_b = None

        if frozen is not None and frozen[ii]:
          trainable = False
        else:
          trainable = True

        w[ii] = weight_variable(
            [f[ii], f[ii], out_ch, in_ch],
            name='w',
            init_val=init_val_w,
            wd=wd,
            trainable=trainable)
        b[ii] = weight_variable(
            [out_ch], init_val=init_val_b, name='b', trainable=trainable)
        log.info('Filter: {}, Trainable: {}'.format(
            [f[ii], f[ii], out_ch, in_ch], trainable))

        in_ch = out_ch

        if model is not None:
          model['{}_w_{}'.format(scope, ii)] = w[ii]
          model['{}_b_{}'.format(scope, ii)] = b[ii]

  copy = [0]

  def run_dcnn(x, skip=None):
    """Run DCNN on an input.
    Args:
      x: input image, [B, H, W, D]
      skip: skip connection activation map, list of 4-D tensor
    """
    with tf.variable_scope(scope):
      h = [None] * nlayers
      out_shape = [None] * nlayers
      batch = tf.shape(x)[0:1]
      inp_size = tf.shape(x)[1:3]
      cum_pool = 1

      for ii in range(nlayers):
        with tf.variable_scope('layer_{}'.format(ii)):
          cum_pool *= pool[ii]
          out_ch = ch[ii + 1]

          if ii == 0:
            prev_inp = x
          else:
            prev_inp = h[ii - 1]

          if skip is not None:
            if skip[ii] is not None:
              if ii == 0:
                prev_inp = tf.concat(3, [prev_inp, skip[ii]])
              else:
                prev_inp = tf.concat(3, [prev_inp, skip[ii]])

          out_shape[ii] = tf.concat(
              0, [batch, inp_size * cum_pool, tf.constant([out_ch])])

          h[ii] = tf.nn.conv2d_transpose(
              prev_inp,
              w[ii],
              out_shape[ii],
              strides=[1, pool[ii], pool[ii], 1]) + b[ii]

          if use_bn[ii]:
            if frozen is not None and frozen[ii]:
              bn_frozen = True
            else:
              bn_frozen = False

            if init_weights is not None and \
                    init_weights[ii] is not None:
              init_beta = init_weights[ii]['beta_{}'.format(copy[0])]
              init_gamma = init_weights[ii]['gamma_{}'.format(copy[0])]
            else:
              init_beta = None
              init_gamma = None
            h[ii] = batch_norm(
                h[ii],
                out_ch,
                phase_train,
                scope2='{}_{}_{}'.format(scope, ii, copy[0]),
                init_beta=init_beta,
                init_gamma=init_gamma,
                model=model)
          if act[ii] is not None:
            h[ii] = act[ii](h[ii])
    copy[0] += 1
    return h

  return run_dcnn


def dropout(x, keep_prob, phase_train):
  """Add dropout layer"""
  phase_train_f = tf.to_float(phase_train)
  keep_prob = (1.0 - phase_train_f) * 1.0 + phase_train_f * keep_prob
  return tf.nn.dropout(x, keep_prob)


def mlp(dims,
        act,
        add_bias=True,
        dropout_keep=None,
        phase_train=None,
        wd=None,
        scope='mlp',
        model=None,
        init_weights=None,
        frozen=None):
  """Add MLP. N = number of layers.
  Args:
    dims: layer-wise dimensions, list of N int
    act: activation function, list of N function
    dropout_keep: keep prob of dropout, list of N float
    phase_train: whether in training phase, tf bool variable
    wd: weight decay
  """
  log = logger.get()

  nlayers = len(dims) - 1
  w = [None] * nlayers
  b = [None] * nlayers

  log.info('MLP: {}'.format(scope))
  log.info('Dimensions: {}'.format(dims))
  log.info('Activation: {}'.format(act))
  log.info('Dropout: {}'.format(dropout_keep))
  log.info('Add bias: {}'.format(add_bias))

  with tf.variable_scope(scope):
    for ii in range(nlayers):
      with tf.variable_scope('layer_{}'.format(ii)):
        nin = dims[ii]
        nout = dims[ii + 1]
        if init_weights is not None and init_weights[ii] is not None:
          init_val_w = init_weights[ii]['w']
          init_val_b = init_weights[ii]['b']
        else:
          init_val_w = None
          init_val_b = None
        if frozen is not None and frozen[ii]:
          trainable = False
        else:
          trainable = True
        w[ii] = weight_variable(
            [nin, nout],
            init_val=init_val_w,
            wd=wd,
            name='w',
            trainable=trainable)
        log.info('Weights: {} Trainable: {}'.format([nin, nout], trainable))
        if add_bias:
          b[ii] = weight_variable(
              [nout], init_val=init_val_b, name='b', trainable=trainable)
          log.info('Bias: {} Trainable: {}'.format([nout], trainable))

        if model is not None:
          model['{}_w_{}'.format(scope, ii)] = w[ii]
          if add_bias:
            model['{}_b_{}'.format(scope, ii)] = b[ii]

  def run_mlp(x):
    h = [None] * nlayers
    with tf.variable_scope(scope):
      for ii in range(nlayers):
        with tf.variable_scope('layer_{}'.format(ii)):
          if ii == 0:
            prev_inp = x
          else:
            prev_inp = h[ii - 1]
          if dropout_keep is not None:
            if dropout_keep[ii] is not None:
              prev_inp = dropout(prev_inp, dropout_keep[ii], phase_train)
          h[ii] = tf.matmul(prev_inp, w[ii])
          if add_bias:
            h[ii] += b[ii]
          if act[ii]:
            h[ii] = act[ii](h[ii])
    return h

  return run_mlp


def lstm(inp_dim,
         hid_dim,
         wd=None,
         scope='lstm',
         model=None,
         init_weights=None,
         frozen=False):
  """Adds an LSTM component.

    Args:
        inp_dim: Input data dim
        hid_dim: Hidden state dim
        wd: Weight decay
        scope: Prefix
    """
  log = logger.get()

  log.info('LSTM: {}'.format(scope))
  log.info('Input dim: {}'.format(inp_dim))
  log.info('Hidden dim: {}'.format(hid_dim))

  if init_weights is None:
    init_weights = {}
    for w in [
        'w_xi', 'w_hi', 'b_i', 'w_xf', 'w_hf', 'b_f', 'w_xu', 'w_hu', 'b_u',
        'w_xo', 'w_ho', 'b_o'
    ]:
      init_weights[w] = None

  trainable = not frozen
  log.info('Trainable: {}'.format(trainable))

  with tf.variable_scope(scope):
    # Input gate
    w_xi = weight_variable(
        [inp_dim, hid_dim],
        init_val=init_weights['w_xi'],
        wd=wd,
        name='w_xi',
        trainable=trainable)
    w_hi = weight_variable(
        [hid_dim, hid_dim],
        init_val=init_weights['w_hi'],
        wd=wd,
        name='w_hi',
        trainable=trainable)
    b_i = weight_variable(
        [hid_dim],
        init_val=init_weights['b_i'],
        initializer=tf.constant_initializer(0.0),
        name='b_i',
        trainable=trainable)

    # Forget gate
    w_xf = weight_variable(
        [inp_dim, hid_dim],
        init_val=init_weights['w_xf'],
        wd=wd,
        name='w_xf',
        trainable=trainable)
    w_hf = weight_variable(
        [hid_dim, hid_dim],
        init_val=init_weights['w_hf'],
        wd=wd,
        name='w_hf',
        trainable=trainable)
    b_f = weight_variable(
        [hid_dim],
        init_val=init_weights['b_f'],
        initializer=tf.constant_initializer(1.0),
        name='b_f',
        trainable=trainable)

    # Input activation
    w_xu = weight_variable(
        [inp_dim, hid_dim],
        init_val=init_weights['w_xu'],
        wd=wd,
        name='w_xu',
        trainable=trainable)
    w_hu = weight_variable(
        [hid_dim, hid_dim],
        init_val=init_weights['w_hu'],
        wd=wd,
        name='w_hu',
        trainable=trainable)
    b_u = weight_variable(
        [hid_dim],
        init_val=init_weights['b_u'],
        initializer=tf.constant_initializer(0.0),
        name='b_u',
        trainable=trainable)

    # Output gate
    w_xo = weight_variable(
        [inp_dim, hid_dim],
        init_val=init_weights['w_xo'],
        wd=wd,
        name='w_xo',
        trainable=trainable)
    w_ho = weight_variable(
        [hid_dim, hid_dim],
        init_val=init_weights['w_ho'],
        wd=wd,
        name='w_ho',
        trainable=trainable)
    b_o = weight_variable(
        [hid_dim],
        init_val=init_weights['b_o'],
        initializer=tf.constant_initializer(0.0),
        name='b_o',
        trainable=trainable)

    if model is not None:
      model['{}_w_xi'.format(scope)] = w_xi
      model['{}_w_hi'.format(scope)] = w_hi
      model['{}_b_i'.format(scope)] = b_i
      model['{}_w_xf'.format(scope)] = w_xf
      model['{}_w_hf'.format(scope)] = w_hf
      model['{}_b_f'.format(scope)] = b_f
      model['{}_w_xu'.format(scope)] = w_xu
      model['{}_w_hu'.format(scope)] = w_hu
      model['{}_b_u'.format(scope)] = b_u
      model['{}_w_xo'.format(scope)] = w_xo
      model['{}_w_ho'.format(scope)] = w_ho
      model['{}_b_o'.format(scope)] = b_o

      model['{}_w_x_mean'.format(scope)] = (
          tf.reduce_sum(tf.abs(w_xi)) + tf.reduce_sum(tf.abs(w_xf)) +
          tf.reduce_sum(tf.abs(w_xu)) + tf.reduce_sum(tf.abs(w_xo))
      ) / inp_dim / hid_dim / 4
      model['{}_w_h_mean'.format(scope)] = (
          tf.reduce_sum(tf.abs(w_hi)) + tf.reduce_sum(tf.abs(w_hf)) +
          tf.reduce_sum(tf.abs(w_hu)) + tf.reduce_sum(tf.abs(w_ho))
      ) / hid_dim / hid_dim / 4
      model['{}_b_mean'.format(scope)] = (
          tf.reduce_sum(tf.abs(b_i)) + tf.reduce_sum(tf.abs(b_f)) +
          tf.reduce_sum(tf.abs(b_u)) + tf.reduce_sum(tf.abs(b_o))) / hid_dim / 4

  def unroll(inp, state):
    with tf.variable_scope(scope):
      c = tf.slice(state, [0, 0], [-1, hid_dim])
      h = tf.slice(state, [0, hid_dim], [-1, hid_dim])
      g_i = tf.sigmoid(tf.matmul(inp, w_xi) + tf.matmul(h, w_hi) + b_i)
      g_f = tf.sigmoid(tf.matmul(inp, w_xf) + tf.matmul(h, w_hf) + b_f)
      g_o = tf.sigmoid(tf.matmul(inp, w_xo) + tf.matmul(h, w_ho) + b_o)
      u = tf.tanh(tf.matmul(inp, w_xu) + tf.matmul(h, w_hu) + b_u)
      c = g_f * c + g_i * u
      h = g_o * tf.tanh(c)
      state = tf.concat(1, [c, h])

    return state, g_i, g_f, g_o

  return unroll


def gru(inp_dim, hid_dim, wd=None, scope='gru'):
  """Adds a GRU component.

    Args:
        inp_dim: Input data dim
        hid_dim: Hidden state dim
        wd: Weight decay
        scope: Prefix
    """
  log = logger.get()

  log.info('GRU: {}'.format(scope))
  log.info('Input dim: {}'.format(inp_dim))
  log.info('Hidden dim: {}'.format(hid_dim))

  with tf.variable_scope(scope):
    w_xi = weight_variable([inp_dim, hid_dim], wd=wd, name='w_xi')
    w_hi = weight_variable([hid_dim, hid_dim], wd=wd, name='w_hi')
    b_i = weight_variable([hid_dim], name='b_i')

    w_xu = weight_variable([inp_dim, hid_dim], wd=wd, name='w_xu')
    w_hu = weight_variable([hid_dim, hid_dim], wd=wd, name='w_hu')
    b_u = weight_variable([hid_dim], name='b_u')

    w_xr = weight_variable([inp_dim, hid_dim], wd=wd, name='w_xr')
    w_hr = weight_variable([hid_dim, hid_dim], wd=wd, name='w_hr')
    b_r = weight_variable([hid_dim], name='b_r')

  def unroll(inp, state):
    g_i = tf.sigmoid(tf.matmul(inp, w_xi) + tf.matmul(state, w_hi) + b_i)
    g_r = tf.sigmoid(tf.matmul(inp, w_xr) + tf.matmul(state, w_hr) + b_r)
    u = tf.tanh(tf.matmul(inp, w_xu) + g_r * tf.matmul(state, w_hu) + b_u)
    state = state * (1 - g_i) + u * g_i

    return state

  return unroll
