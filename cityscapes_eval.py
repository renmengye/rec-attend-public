#!/usr/bin/env python
"""
Plot a ris net. Usage: python ris_eval_exp.py --help
"""
from __future__ import division

import cv2
import numpy as np
import scipy.io
import os

from utils import plot_utils as pu
from utils import postprocess as pp
from utils.batch_iter import BatchIterator
from utils.time_series_logger import TimeSeriesLogger

from cmd_args_parser import DataArgsParser, EvalArgsParser
from experiment import EvalExperimentBase
from analysis import (f_iou_pairwise, create_analyzer, RenderInstanceAnalyzer,
                      RenderCityScapesOutputAnalyzer,
                      RenderGroundtruthInstanceAnalyzer, CountAnalyzer)
from evaluation import OneTimeEvalBase


class CityscapesEvalRunner(OneTimeEvalBase):

  def __init__(self,
               sess,
               model,
               dataset,
               opt,
               model_opt,
               output_folder,
               threshold_list,
               analyzer_names,
               split,
               foreground_folder=None):
    outputs = ['y_out']
    if opt['split_id'] == -1:
      start_idx = -1
      end_idx = -1
    else:
      start_idx = opt['split_id'] * opt['num_split']
      end_idx = (opt['split_id'] + 1) * opt['num_split']

    if output_folder is not None:
      if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
      fname = None
      raise Exception('not output')

    if threshold_list is None:
      threshold_list = np.arange(10) * 0.1

    if analyzer_names is None:
      analyzer_names = [
          'sbd', 'wt_cov', 'unwt_cov', 'fg_dice', 'fg_iou', 'fg_iou_all',
          'bg_iou_all', 'avg_fp', 'avg_fn', 'avg_pr', 'avg_re', 'obj_pr',
          'obj_re', 'count_acc', 'count_mse', 'dic', 'dic_abs'
      ]

    self.output_folder = output_folder
    self.foreground_folder = foreground_folder
    self.threshold_list = threshold_list
    self.analyzer_names = analyzer_names
    self.analyzers = []
    self.split = split
    self.gt_render = RenderGroundtruthInstanceAnalyzer(
        os.path.join(output_folder, 'gt'), dataset)

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
        if dataset.get_name() == 'cityscapes':
          _analyzers.append(
              RenderCityScapesOutputAnalyzer(
                  os.path.join(output_folder, 'cityscapes'), dataset))
          sem_labels = [
              'person', 'rider', 'car', 'truck', 'bus', 'train', 'moto', 'bike'
          ]
        else:
          sem_labels = None
        _analyzers.append(
            RenderInstanceAnalyzer(
                os.path.join(output_folder, thresh_folder),
                dataset,
                semantic_labels=sem_labels))
        _analyzers.append(
            CountAnalyzer(
                os.path.join(output_folder, thresh_folder, 'count.csv')))
      self.analyzers.append(_analyzers)

    opt['batch_size'] = 1  # Set batch size to 1.
    super(CityscapesEvalRunner, self).__init__(
        sess,
        model,
        dataset,
        opt,
        model_opt,
        outputs,
        start_idx=start_idx,
        end_idx=end_idx)

  def get_input_variables(self):
    variables = [
        'x_full', 'y_gt_full', 'y_out', 'd_out', 'y_out_ins', 's_out', 's_gt',
        'idx_map'
    ]
    return set(variables)

  def _run_step(self, inp):
    return {}

  def get_batch(self, idx):
    """Transform a dataset get_batch into a dictionary to feed."""
    idx_new = self.all_idx[idx]
    _batch = self.dataset.get_batch(idx_new, variables=self.input_variables)
    batch = {}
    x = _batch['x_full']
    y_in = _batch['y_out_ins']
    fg_in = _batch['y_out']
    d_in = _batch['d_out']
    s_out = _batch['s_out']
    # [T, H, W, C]
    x = np.tile(np.expand_dims(x, 0), [y_in.shape[1], 1, 1, 1])
    fg_in = np.tile(fg_in, [y_in.shape[1], 1, 1, 1])
    d_in = np.tile(d_in, [y_in.shape[1], 1, 1, 1])
    # [T, H, W]
    y_in = y_in.reshape([-1, y_in.shape[2], y_in.shape[3]])
    batch['x'] = x
    batch['y_in'] = y_in
    batch['fg_in'] = fg_in
    batch['d_in'] = d_in
    batch['idx_map'] = _batch['idx_map']
    batch['_y_gt_full'] = _batch['y_gt_full']  # [T, H, W]
    batch['_s_gt'] = _batch['s_gt']
    batch['_s_out'] = _batch['s_out']
    return batch

  def write_log(self, results):
    """Process results
    Args:
      results: y_out, s_out
    """
    inp = results['_batches'][0]
    s_out = inp['_s_out']
    conf = s_out
    s_gt = inp['_s_gt']  # [T]
    y_gt_h = [inp['_y_gt_full']]  # [T, H, W]

    # Upsample the foreground semantic segmentation
    full_size = (y_gt_h[0].shape[1], y_gt_h[0].shape[2])

    if self.opt['lrr_seg']:
      fg_h = [self.read_foreground_lrr(inp['idx_map'][0])]
      fg_mask = [1 - fg_h[0][:, :, 0]]
    else:
      fg = inp['fg_in'][0]  # [1, H, W, C]
      fg_h = np.zeros(
          [full_size[0], full_size[1], fg.shape[2]], dtype='float32')
      for cc in range(fg_h.shape[2]):
        fg_h[:, :, cc] = cv2.resize(fg[:, :, cc], (full_size[1], full_size[0]))

      FG_THRESHOLD = 0.3
      if fg.shape[2] == 1:
        fg_mask = [(np.squeeze(fg_h, 2) > FG_THRESHOLD).astype('float32')]
      else:
        fg_mask = [(fg_h[:, :, 0] <= (1 - FG_THRESHOLD)).astype('float32')]
      fg_h = [fg_h]

    y_out = pp.upsample(np.expand_dims(inp['y_in'], 0), y_gt_h)
    y_out, conf_hard = pp.apply_confidence(y_out, conf)
    y_out = pp.apply_one_label(y_out)

    for tt, thresh in enumerate(self.threshold_list):
      y_out_thresh = pp.apply_threshold(y_out, thresh)
      y_out_thresh = pp.mask_foreground(y_out_thresh, fg_mask)

      # Remove tiny patches.
      y_out_thresh, conf = pp.remove_tiny(
          y_out_thresh, conf=conf, threshold=self.opt['remove_tiny'])
      results_thresh = {
          'y_out': y_out_thresh,
          'y_gt': y_gt_h,
          's_out': conf_hard,
          'conf': conf,
          'y_in': fg_h,
          's_gt': s_gt,
          'indices': inp['idx_map']
      }
      if not self.opt['no_iou']:
        results_thresh['iou_pairwise'] = [
            f_iou_pairwise(a, b) for a, b in zip(y_out_thresh, y_gt_h)
        ]
      [aa.stage(results_thresh) for aa in self.analyzers[tt]]
    if self.opt['render_gt']:
      self.gt_render.stage(results_thresh)

  def finalize(self):
    """Finalize report"""
    for tt, thresh in enumerate(self.threshold_list):
      [aa.finalize() for aa in self.analyzers[tt]]

  def read_foreground_lrr(self, idx):
    # 14=car, 12=person, 13=rider, 18=motorcycle, 19=bicycle, 15=truck, 16=bus,
    # 17=train
    # LRR/val/munster/munster_000051_000019_ss.mat
    sem_ids = [12, 13, 14, 15, 16, 17, 18, 19]
    if self.split.startswith('train'):
      folder = 'train'
    elif self.split.startswith('val'):
      folder = 'val'
    elif self.split.startswith('test'):
      folder = 'test'
    runname = idx.split('_')[0]
    print(idx)
    matfn = '/ais/gobi4/mren/models/LRR/{}/{}/{}_ss.mat'.format(folder, runname,
                                                                idx)
    fgraw = scipy.io.loadmat(matfn)['semanticPrediction']
    fg = np.zeros(list(fgraw.shape) + [9], dtype='float32')
    for ii in range(8):
      fg[:, :, ii + 1] = (fgraw == sem_ids[ii]).astype('float32')
    fg[:, :, 0] = 1 - fg.max(axis=-1)
    return fg


class CityscapesEvalExperiment(EvalExperimentBase):

  def get_runner(self, split):
    if self.opt['output'] is None:
      output_folder = self.opt['restore']
    else:
      output_folder = self.opt['output']

    output_folder_prefix = 'output_'
    output_folder_split = os.path.join(output_folder,
                                       output_folder_prefix + split)

    return CityscapesEvalRunner(
        self.sess, self.model, self.dataset[split], self.opt, self.model_opt,
        output_folder_split, self.opt['threshold_list'], self.opt['analyzers'],
        split, self.opt['foreground_folder'])

  def init_model(self):
    pass

  def get_model(self):
    return None


class CityscapesEvalArgsParser(EvalArgsParser):

  def add_args(self):
    self.parser.add_argument('--threshold_list', default=None)
    self.parser.add_argument('--analyzers', default=None)
    self.parser.add_argument('--test', action='store_true')
    self.parser.add_argument('--split_id', default=-1, type=int)
    self.parser.add_argument('--num_split', default=100, type=int)
    self.parser.add_argument('--remove_tiny', default=400, type=int)
    self.parser.add_argument('--foreground_folder', default=None)
    self.parser.add_argument('--no_iou', action='store_true')
    self.parser.add_argument('--render_gt', action='store_true')
    self.parser.add_argument('--lrr_seg', action='store_true')
    self.parser.add_argument(
        '--lrr_filename', default='/ais/gobi4/mren/models/LRR/{}/{}/{}_ss.mat')
    super(CityscapesEvalArgsParser, self).add_args()

  def make_opt(self, args):
    opt = super(CityscapesEvalArgsParser, self).make_opt(args)
    opt['foreground_folder'] = args.foreground_folder
    opt['split_id'] = args.split_id
    opt['num_split'] = args.num_split
    opt['remove_tiny'] = args.remove_tiny
    opt['no_iou'] = args.no_iou
    opt['render_gt'] = args.render_gt
    opt['lrr_seg'] = args.lrr_seg
    if args.threshold_list is None:
      opt['threshold_list'] = np.arange(10) * 0.1
    else:
      opt['threshold_list'] = [
          float(tt) for tt in args.threshold_list.split(',')
      ]
    if args.analyzers is None:
      if args.test:
        opt['analyzers'] = ['fg_iou', 'fg_iou_all', 'bg_iou_all']
      else:
        opt['analyzers'] = [
            'sbd', 'wt_cov', 'unwt_cov', 'fg_dice', 'fg_iou', 'fg_iou_all',
            'bg_iou_all', 'avg_fp', 'avg_fn', 'avg_pr', 'avg_re', 'obj_pr',
            'obj_re', 'count_acc', 'count_mse', 'dic', 'dic_abs'
        ]
    else:
      if args.analyzers == '':
        opt['analyzers'] = []
      else:
        opt['analyzers'] = args.analyzers.split(',')
    return opt


def main():
  parsers = {'default': CityscapesEvalArgsParser(), 'data': DataArgsParser()}
  CityscapesEvalExperiment.create_from_main(
      'ris_pp_eval', parsers=parsers, description='Eval ris pp output').run()


if __name__ == '__main__':
  main()
