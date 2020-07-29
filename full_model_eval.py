#!/usr/bin/env python
"""Run evaluation."""
from __future__ import division

import cv2
import numpy as np
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from utils import logger
from utils import postprocess as pp

from cmd_args_parser import DataArgsParser, EvalArgsParser
from experiment import EvalExperimentBase
from analysis import (f_iou_pairwise, create_analyzer, RenderInstanceAnalyzer,
                      RenderGroundtruthInstanceAnalyzer, CountAnalyzer)
from evaluation import OneTimeEvalBase
from full_model import get_model


class EvalRunner(OneTimeEvalBase):

  def __init__(self,
               sess,
               model,
               dataset,
               opt,
               model_opt,
               output_folder,
               threshold_list,
               analyzer_names,
               foreground_folder=None,
               render_gt=False,
               render_output=False,
               output_count=False):
    outputs = ['y_out', 's_out']
    if not os.path.exists(output_folder):
      os.makedirs(output_folder)

    if threshold_list is None:
      threshold_list = np.arange(10) * 0.1

    if analyzer_names is None:
      analyzer_names = [
          'sbd', 'wt_cov', 'unwt_cov', 'fg_dice', 'fg_iou', 'fg_iou_all',
          'bg_iou_all', 'avg_fp', 'avg_fn', 'avg_pr', 'avg_re', 'obj_pr',
          'obj_re', 'count_acc', 'count_mse', 'dic', 'dic_abs'
      ]

    self.output_folder = output_folder
    self.threshold_list = threshold_list
    self.analyzer_names = analyzer_names
    self.foreground_folder = foreground_folder
    self.analyzers = []
    self.render_gt = render_gt
    if render_gt:
      self.gt_render = RenderGroundtruthInstanceAnalyzer(
          os.path.join(output_folder, 'gt'), dataset)
    self.render_output = render_output
    self.output_count = output_count

    # Create a set of analyzers for each threshold.
    for tt in threshold_list:
      _analyzers = []
      thresh_suffix = ' {:.2f}'.format(tt)
      thresh_folder = '{:02d}'.format(int(tt * 100))
      for name in analyzer_names:
        fname = os.path.join(output_folder, '{}.csv'.format(name))
        _analyzers.append(
            create_analyzer(
                name, display_name=name + thresh_suffix, fname=fname))
      if output_folder is not None:
        if render_output:
          _analyzers.append(
              RenderInstanceAnalyzer(
                  os.path.join(output_folder, thresh_folder), dataset))
        if output_count:
          _analyzers.append(
              CountAnalyzer(
                  os.path.join(output_folder, thresh_folder, 'count.csv')))
      self.analyzers.append(_analyzers)

    super(EvalRunner, self).__init__(sess, model, dataset, opt, model_opt,
                                     outputs)

  def read_foreground(self, idx, y_gt=None):
    if self.foreground_folder is None:
      return None
    else:
      fg = []
      for ii in idx:
        fg_fname = os.path.join(self.foreground_folder,
                                self.dataset.get_fname(ii))
        fg_ = cv2.imread(fg_fname).astype('float32').max(axis=2) / 255.0
        fg.append(fg_)
    return fg

  def write_log(self, results):
    """Process results
    Args:
      results: y_out, s_out
    """
    inp = results['_batches'][0]
    y_gt_h = self.dataset.get_full_size_labels(
        inp['idx_map'], timespan=results['y_out'].shape[1])
    y_out = results['y_out']
    s_out = results['s_out']

    # Multi-class
    if len(s_out.shape) == 3:
      s_out = s_out[:, :, 0]

    y_out, s_out = pp.apply_confidence(y_out, s_out)
    fg = self.read_foreground(inp['idx_map'])
    y_out = pp.upsample(y_out, y_gt_h)
    if fg is not None:
      if not self.opt['no_morph']:
        y_out = pp.morph(y_out)
    y_out = pp.apply_one_label(y_out)
    for tt, thresh in enumerate(self.threshold_list):
      y_out_thresh = pp.apply_threshold(y_out, thresh)
      if fg is not None:
        y_out_thresh = pp.mask_foreground(y_out_thresh, fg)
        y_out_thresh, s_out = pp.remove_tiny(
            y_out_thresh, s_out, threshold=self.opt['remove_tiny'])
      iou_pairwise = [
          f_iou_pairwise(a, b) for a, b in zip(y_out_thresh, y_gt_h)
      ]
      results_thresh = {
          'y_out': y_out_thresh,
          'y_gt': y_gt_h,
          's_out': s_out,
          's_gt': inp['_s_gt'],
          'iou_pairwise': iou_pairwise,
          'indices': inp['idx_map']
      }
      # Run each analyzer.
      [aa.stage(results_thresh) for aa in self.analyzers[tt]]
    # Plot groundtruth.
    if self.render_gt:
      self.gt_render.stage(results_thresh)

  def finalize(self):
    """Finalize report"""
    for tt, thresh in enumerate(self.threshold_list):
      [aa.finalize() for aa in self.analyzers[tt]]


class EvalExperiment(EvalExperimentBase):

  def get_runner(self, split):
    if self.opt['output'] is None:
      output_folder = self.opt['restore']
    else:
      output_folder = self.opt['output']

    output_folder_prefix = 'output_'
    output_folder_split = os.path.join(output_folder,
                                       output_folder_prefix + split)

    return EvalRunner(
        self.sess,
        self.model,
        self.dataset[split],
        self.opt,
        self.model_opt,
        output_folder_split,
        self.opt['threshold_list'],
        self.opt['analyzers'],
        foreground_folder=self.opt['foreground_folder'],
        render_output=True)

  def get_model(self):
    self.model_opt['use_knob'] = False
    return get_model(self.model_opt)


class MyEvalArgsParser(EvalArgsParser):

  def add_args(self):
    self.parser.add_argument('--foreground_folder', default=None)
    self.parser.add_argument('--threshold_list', default=None)
    self.parser.add_argument('--analyzers', default=None)
    self.parser.add_argument('--test', action='store_true')
    self.parser.add_argument('--no_morph', action='store_true')
    self.parser.add_argument('--remove_tiny', default=0, type=int)
    super(MyEvalArgsParser, self).add_args()

  def make_opt(self, args):
    opt = super(MyEvalArgsParser, self).make_opt(args)
    opt['foreground_folder'] = args.foreground_folder
    opt['no_morph'] = args.no_morph
    opt['remove_tiny'] = args.remove_tiny
    if args.threshold_list is None:
      opt['threshold_list'] = [0.3]  # Usually 0.3 is good threshold.
    else:
      opt['threshold_list'] = [
          float(tt) for tt in args.threshold_list.split(',')
      ]
    if args.analyzers is None:
      if args.test:
        opt['analyzers'] = []
      else:
        opt['analyzers'] = [
            'sbd', 'wt_cov', 'unwt_cov', 'avg_fp', 'avg_fn', 'avg_pr', 'avg_re',
            'obj_pr', 'obj_re', 'count_acc', 'count_mse', 'dic', 'dic_abs'
        ]
    else:
      if args.analyzers == '':
        opt['analyzers'] = []
      else:
        opt['analyzers'] = args.analyzers.split(',')
    return opt


def main():
  parsers = {'default': MyEvalArgsParser(), 'data': DataArgsParser()}
  EvalExperiment.create_from_main(
      'eval', parsers=parsers, description='Evaluate output').run()


if __name__ == '__main__':
  main()
