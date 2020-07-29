#!/usr/bin/env python
import argparse
import h5py
import os
import sys
import tensorflow as tf

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from utils import logger
from utils.saver import Saver
from box_model import get_model

log = logger.get()


def read(folder):
  log.info('Reading pretrained network from {}'.format(folder))
  saver = Saver(folder)
  ckpt_info = saver.get_ckpt_info()
  model_opt = ckpt_info['model_opt']
  ckpt_fname = ckpt_info['ckpt_fname']
  model_id = ckpt_info['model_id']
  model = get_model(model_opt)
  ctrl_cnn_nlayers = len(model_opt['ctrl_cnn_filter_size'])
  ctrl_mlp_nlayers = model_opt['num_ctrl_mlp_layers']
  timespan = model_opt['timespan']
  glimpse_mlp_nlayers = model_opt['num_glimpse_mlp_layers']
  weights = {}
  with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    saver.restore(sess, ckpt_fname)
    output_list = []
    for net, nlayers in zip(
        ['ctrl_cnn', 'ctrl_mlp', 'glimpse_mlp', 'score_mlp'],
        [ctrl_cnn_nlayers, ctrl_mlp_nlayers, glimpse_mlp_nlayers, 1]):
      for ii in range(nlayers):
        for w in ['w', 'b']:
          key = '{}_{}_{}'.format(net, w, ii)
          log.info(key)
          output_list.append(key)
        if net == 'ctrl_cnn':
          for tt in range(timespan):
            for w in ['beta', 'gamma']:
              key = '{}_{}_{}_{}'.format(net, ii, tt, w)
              log.info(key)
              output_list.append(key)
    for net in ['ctrl_lstm']:
      for w in [
          'w_xi', 'w_hi', 'b_i', 'w_xf', 'w_hf', 'b_f', 'w_xu', 'w_hu', 'b_u',
          'w_xo', 'w_ho', 'b_o'
      ]:
        key = '{}_{}'.format(net, w)
        log.info(key)
        output_list.append(key)
    output_var = []
    for key in output_list:
      output_var.append(model[key])
    output_var_value = sess.run(output_var)
    for key, value in zip(output_list, output_var_value):
      weights[key] = value
      log.info(key)
      log.info(value.shape)
  return weights


def save(fname, folder):
  weights = read(folder)
  h5f = h5py.File(fname, 'w')
  for key in weights:
    h5f[key] = weights[key]
  h5f.close()
  log.info('Saved weights to {}'.format(fname))


def parse_args():
  """Parse input arguments."""
  parser = argparse.ArgumentParser(description='Read pretrained weights')
  parser.add_argument('--model_id', default=None)
  parser.add_argument('--results', default='results')
  parser.add_argument('--output', default=None)
  args = parser.parse_args()
  return args


if __name__ == '__main__':
  args = parse_args()
  exp_folder = os.path.join(args.results, args.model_id)
  if args.output is None:
    output = os.path.join(exp_folder, 'weights.h5')
  else:
    output = args.output
  save(output, exp_folder)