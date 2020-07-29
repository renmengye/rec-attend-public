#!/usr/bin/env python
"""Run model inference, output to H5."""
from __future__ import division

import cv2
import os
import h5py
import numpy as np

from cmd_args_parser import EvalArgsParser, DataArgsParser
from evaluation import OneTimeEvalBase
from experiment import EvalExperimentBase
from analysis import RenderInstanceAnalyzer
from full_model import get_model


class PackRunner(OneTimeEvalBase):

  def __init__(self, sess, model, dataset, train_opt, model_opt):
    outputs = ['y_out', 's_out']
    self.input_variables = set(['x', 'y_out', 'd_out', 'idx_map'])
    super(PackRunner, self).__init__(sess, model, dataset, train_opt, model_opt,
                                     outputs)

  def get_batch(self, idx):
    """Transform a dataset get_batch into a dictionary to feed."""
    _batch = self.dataset.get_batch(idx, variables=self.input_variables)
    return {
        'x': _batch['x'],
        'y_in': _batch['y_out'],
        'd_in': _batch['d_out'],
        'idx_map': _batch['idx_map']
    }

  def write_log(self, results):
    """Process results
    Args:
      results: y_out, s_out
    """
    inp = results['_batches'][0]
    y_out = results['y_out']
    s_out = results['s_out']
    with h5py.File(self.dataset.h5_fname, 'r+') as h5f:
      print(inp['idx_map'])
      for ii in range(y_out.shape[0]):
        idx = inp['idx_map'][ii]
        group = h5f[self.dataset.get_str_id(idx)]
        if 'instance_pred' in group:
          del group['instance_pred']
        for ins in range(y_out.shape[1]):
          y_out_arr = y_out[ii, ins]
          y_out_arr = (y_out_arr * 255).astype('uint8')
          y_out_str = cv2.imencode('.png', y_out_arr)[1]
          group['instance_pred/{:02d}'.format(ins)] = y_out_str
        if 'score_pred' in group:
          del group['score_pred']
        group['score_pred'] = s_out[ii]


class PackExperiment(EvalExperimentBase):

  def get_runner(self, split):
    return PackRunner(self.sess, self.model, self.dataset[split], self.opt,
                      self.model_opt)

  def get_model(self):
    self.model_opt['use_knob'] = False
    return get_model(self.model_opt)


if __name__ == '__main__':
  parsers = {'default': EvalArgsParser(), 'data': DataArgsParser()}
  PackExperiment.create_from_main(
      'ris_pack', parsers=parsers, description='Pack ris output').run()
