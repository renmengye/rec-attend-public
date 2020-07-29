#!/usr/bin/env python
from __future__ import division

import cv2
import os
import numpy as np

from evaluation import OneTimeEvalBase
from analysis import (create_analyzer, RenderForegroundAnalyzer,
                      RenderOrientationAnalyzer)
from experiment import EvalExperimentBase
from cmd_args_parser import EvalArgsParser, DataArgsParser
from fg_model import get_model


class FGEvalRunner(OneTimeEvalBase):

  def __init__(self,
               sess,
               model,
               dataset,
               train_opt,
               model_opt,
               output_folder,
               threshold_list=None):
    outputs = ['y_out']
    if 'd_out' in model:
      outputs.append('d_out')
      if train_opt['render_ori']:
        self.ori_render = RenderOrientationAnalyzer(
            os.path.join(output_folder, 'ori'), dataset)
      else:
        self.ori_render = None
      if train_opt['render_soft']:
        self.soft_render = RenderForegroundAnalyzer(
            os.path.join(output_folder, 'soft'), dataset)
      else:
        self.soft_render = None
      if train_opt['render_gt']:
        self.gt_render = RenderForegroundAnalyzer(
            os.path.join(output_folder, 'gt'), dataset)
      else:
        self.gt_render = None
    else:
      self.ori_render = None

    if not os.path.exists(output_folder):
      os.makedirs(output_folder)
    fname = os.path.join(output_folder, 'report.csv')
    with open(fname, 'w') as f:
      f.write('')

    if threshold_list is None:
      threshold_list = np.arange(10) * 0.1

    self.output_folder = output_folder
    self.threshold_list = threshold_list
    self.analyzer_names = ['fg_iou_all', 'bg_iou_all']
    self.analyzers = []
    # self.input_variables = set(['x', 'idx_map', 'orig_size'])

    for tt in self.threshold_list:
      _analyzers = []
      for name in self.analyzer_names:
        thresh_suffix = ' {:.2f}'.format(tt)
        thresh_folder = '{:02d}'.format(int(tt * 100))
        _analyzers.append(
            create_analyzer(
                name, display_name=name + thresh_suffix, fname=fname))
      if output_folder is not None:
        _analyzers.append(
            RenderForegroundAnalyzer(
                os.path.join(output_folder, thresh_folder), dataset))
      self.analyzers.append(_analyzers)

    super(FGEvalRunner, self).__init__(sess, model, dataset, train_opt,
                                       model_opt, outputs)

  def get_input_variables(self):
    variables = ['x', 's_gt', 'idx_map', 'orig_size']
    return set(variables)

  def get_batch(self, idx):
    """Transform a dataset get_batch into a dictionary to feed."""
    _batch = self.dataset.get_batch(idx, variables=self.input_variables)
    # return {'x': self._batch['x']}
    return _batch

  def upsample(self, y_out, size_list):
    """Upsample y_out into size of y_gt.
    Args:
      y_out: list of [H', W']
      size_list: list of [H, W]
    Returns:
      y_out_resize: list of [H, W]
    """
    y_out_resize = []
    num_ex = y_out.shape[0]
    for ii in range(num_ex):
      _sz = size_list[ii]
      _y = self.upsample_single(y_out[ii], (_sz[1], _sz[0]))
      y_out_resize.append(_y)
      pass
    return y_out_resize

  def upsample_single(self, a, size):
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

  def apply_threshold(self, y_out, thresh):
    """Threshold the soft output into binary map.
    Args:
      y_out: list of [H, W] soft output.
    Returns:
      y_out_thresh: list of [H, W] binary map.
    """
    return [(_y > thresh).astype('float32') for _y in y_out]

  def upsample_d_out(self, d_out, size):
    d_out_h = np.zeros([size[0], size[1], d_out.shape[-1]])
    for ch in range(d_out.shape[-1]):
      d_out_h[:, :, ch] = cv2.resize(d_out[:, :, ch], (size[1], size[0]))
    return d_out_h

  def write_log(self, results):
    """Process results
    Args:
      results: y_out, s_out
    """
    # inp = self._batch
    inp = results['_batches'][0]

    y_gt_h = self.dataset.get_full_size_labels(inp['idx_map'], timespan=23)
    y_gt_h = [y_gt_.sum(axis=0) for y_gt_ in y_gt_h]
    size_list = inp['orig_size']
    y_out = results['y_out']
    y_out = self.upsample(y_out, size_list)
    num_ex = len(y_gt_h)
    results_soft = {'y_out': y_out, 'y_gt': y_gt_h, 'indices': inp['idx_map']}
    if self.soft_render is not None:
      self.soft_render.stage(results_soft)
    if self.gt_render is not None:
      results_soft['y_out'] = y_gt_h
      self.gt_render.stage(results_soft)
    if self.ori_render is not None:
      d_out = results['d_out']
      d_out_h = [
          self.upsample_d_out(d_out[ii], size_list[ii]) for ii in range(num_ex)
      ]
      results_dout = {
          'd_out': d_out_h,
          'mask': y_gt_h,
          'indices': inp['idx_map']
      }
      self.ori_render.stage(results_dout)

    for tt, thres in enumerate(self.threshold_list):
      y_out_thresh = self.apply_threshold(y_out, thres)
      results_thresh = {
          'y_out': y_out_thresh,
          'y_gt': y_gt_h,
          'indices': inp['idx_map']
      }
      [aa.stage(results_thresh) for aa in self.analyzers[tt]]

  def finalize(self):
    """Finalize report"""
    for tt, thresh in enumerate(self.threshold_list):
      [aa.finalize() for aa in self.analyzers[tt]]


class FGEvalExperiment(EvalExperimentBase):

  def get_runner(self, split):
    output_folder = self.opt['output']
    if output_folder is None:
      output_folder = os.path.join(opt['restore'], 'output')
    return FGEvalRunner(self.sess, self.model, self.dataset[split], self.opt,
                        self.model_opt, output_folder,
                        self.opt['threshold_list'])

  def get_model(self):
    return get_model(self.model_opt)


class FGEvalArgsParser(EvalArgsParser):

  def add_args(self):
    self.parser.add_argument('--threshold_list', default=None)
    self.parser.add_argument('--render_ori', action='store_true')
    self.parser.add_argument('--render_soft', action='store_true')
    self.parser.add_argument('--render_gt', action='store_true')
    super(FGEvalArgsParser, self).add_args()
    pass

  def make_opt(self, args):
    opt = super(FGEvalArgsParser, self).make_opt(args)
    opt['render_ori'] = args.render_ori
    opt['render_gt'] = args.render_gt
    opt['render_soft'] = args.render_soft
    if args.threshold_list is None:
      opt['threshold_list'] = [0.3]
    else:
      opt['threshold_list'] = [
          float(tt) for tt in args.threshold_list.split(',')
      ]
    return opt


def main():
  parsers = {'default': FGEvalArgsParser(), 'data': DataArgsParser()}
  FGEvalExperiment.create_from_main(
      'fg_eval', parsers=parsers, description='Eval fg output').run()


if __name__ == '__main__':
  main()