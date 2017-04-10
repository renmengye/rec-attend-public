#!/usr/bin/env python
"""Train foreground segmentation network."""
from __future__ import division

import numpy as np
import os

from utils import logger
from utils import BatchIterator, ConcurrentBatchIterator
from utils import plot_utils as pu
from utils.lazy_registerer import LazyRegisterer
from utils.step_counter import StepCounter
from utils.time_series_logger import TimeSeriesLogger
from cmd_args_parser import TrainArgsParser, DataArgsParser, CmdArgsParser
from experiment import TrainingExperimentBase
from runner import RunnerBase
from fg_model import get_model


class Runner(RunnerBase):

  def __init__(self,
               sess,
               model,
               dataset,
               num_batch,
               train_opt,
               model_opt,
               outputs,
               step=StepCounter(0),
               loggers=None,
               phase_train=True,
               increment_step=False):
    self.dataset = dataset
    self.log = logger.get()
    self.loggers = loggers
    self.add_orientation = model_opt['add_orientation']
    self.num_orientation_classes = model_opt['num_orientation_classes']
    self.input_variables = self.get_input_variables()
    num_ex = dataset.get_dataset_size()
    batch_iter = BatchIterator(
        num_ex,
        batch_size=train_opt['batch_size'],
        get_fn=self.get_batch,
        cycle=True,
        progress_bar=False,
        shuffle=True,
        log_epoch=-1)
    if train_opt['prefetch']:
      batch_iter = ConcurrentBatchIterator(
          batch_iter,
          max_queue_size=train_opt['queue_size'],
          num_threads=train_opt['num_worker'],
          log_queue=-1)
    super(Runner, self).__init__(
        sess,
        model,
        batch_iter,
        outputs,
        num_batch=num_batch,
        step=step,
        phase_train=phase_train,
        increment_step=increment_step)

  def get_input_variables(self):
    variables = ['x', 'c_gt']
    if self.add_orientation:
      variables.append('d_gt')
    return set(variables)

  def get_batch(self, idx):
    """Transform a dataset get_batch into a dictionary to feed."""
    batch_ = self.dataset.get_batch(idx, variables=self.input_variables)
    batch = {}
    batch['x'] = batch_['x']
    batch['y_gt'] = batch_['c_gt']
    if 'sem_weights' in batch_ and 'sem_weights' in self.model:
      batch['sem_weights'] = batch_['sem_weights']
    if 'ori_weights' in batch_ and 'ori_weights' in self.model:
      batch['ori_weights'] = batch_['ori_weights']
    if self.add_orientation:
      batch['d_gt'] = batch_['d_gt']
    return batch


class Trainer(Runner):

  def __init__(self,
               sess,
               model,
               dataset,
               train_opt,
               model_opt,
               step=StepCounter(0),
               loggers=None,
               steps_per_log=10):
    outputs = ['loss', 'train_step']
    num_batch = steps_per_log
    super(Trainer, self).__init__(
        sess,
        model,
        dataset,
        num_batch,
        train_opt,
        model_opt,
        outputs,
        step=step,
        loggers=loggers,
        phase_train=True,
        increment_step=True)

  def write_log(self, results):
    self.log.info('{:d} loss {:.4f} t {:.2f}ms'.format(self.step.get(), results[
        'loss'], results['step_time']))
    self.loggers['loss'].add(self.step.get(), [results['loss'], ''])
    self.loggers['step_time'].add(self.step.get(), results['step_time'])


class Evaluator(Runner):

  def __init__(self,
               sess,
               model,
               dataset,
               train_opt,
               model_opt,
               step=StepCounter(0),
               num_batch=10,
               loggers=None,
               phase_train=True):
    outputs = ['iou_soft', 'iou_hard', 'foreground_loss', 'loss']
    if model_opt['add_orientation']:
      outputs.extend(['orientation_ce', 'orientation_acc'])
    super(Evaluator, self).__init__(
        sess,
        model,
        dataset,
        num_batch,
        train_opt,
        model_opt,
        outputs,
        step=step,
        loggers=loggers,
        phase_train=phase_train,
        increment_step=False)

  def write_log(self, results):
    if self.loggers is not None:
      self.log.info('{:d} loss {:.4f} t {:.2f}ms'.format(self.step.get(
      ), results['loss'], results['step_time']))
      if 'loss' in self.loggers:
        if self.phase_train:
          line = [results['loss'], '']
        else:
          line = ['', results['loss']]
        self.loggers['loss'].add(self.step.get(), line)
      if 'iou' in self.loggers:
        if self.phase_train:
          line = [results['iou_soft'], '', results['iou_hard'], '']
        else:
          line = ['', results['iou_soft'], '', results['iou_hard']]
        self.loggers['iou'].add(self.step.get(), line)
      if 'foreground_loss' in self.loggers:
        if self.phase_train:
          line = [results['foreground_loss'], '']
        else:
          line = ['', results['foreground_loss']]
        self.loggers['foreground_loss'].add(self.step.get(), line)
      if self.add_orientation:
        if 'orientation_ce' in self.loggers:
          if self.phase_train:
            line = [results['orientation_ce'], '']
          else:
            line = ['', results['orientation_ce']]
          self.loggers['orientation_ce'].add(self.step.get(), line)
        if 'orientation_acc' in self.loggers:
          if self.phase_train:
            line = [results['orientation_acc'], '']
          else:
            line = ['', results['orientation_acc']]
          self.loggers['orientation_acc'].add(self.step.get(), line)


class Plotter(Runner):

  def __init__(self,
               sess,
               model,
               dataset,
               train_opt,
               model_opt,
               logs_folder,
               step=StepCounter(0),
               split='train',
               phase_train=False):
    outputs = ['x_trans', 'y_gt_trans', 'y_out']
    if model_opt['add_orientation']:
      outputs.extend(['d_out', 'd_gt_trans'])
    num_batch = 1
    self.split = split
    self.logs_folder = logs_folder
    self.ori_color_wheel = np.array(
        [[255, 17, 0], [255, 137, 0], [230, 255, 0], [34, 255, 0],
         [0, 255, 213], [0, 154, 255], [9, 0, 255], [255, 0, 255]],
        dtype='uint8')
    self.sem_color_wheel = np.array(
        [[0, 0, 0], [255, 17, 0], [255, 137, 0], [230, 255, 0], [34, 255, 0],
         [0, 255, 213], [0, 154, 255], [9, 0, 255], [255, 0, 255]],
        dtype='uint8')
    loggers = self.get_loggers(model_opt['add_orientation'], split)
    super(Plotter, self).__init__(
        sess,
        model,
        dataset,
        num_batch,
        train_opt,
        model_opt,
        outputs,
        step=step,
        loggers=loggers,
        phase_train=phase_train,
        increment_step=False)

  def get_loggers(self, add_orientation, split):
    loggers = {}
    labels = ['input', 'gt_segmentation', 'output_segmentation']
    if add_orientation:
      labels.extend(['gt_orientation', 'output_orientation'])
    for name in labels:
      key = '{}_{}'.format(name, split)
      loggers[name] = LazyRegisterer(
          os.path.join(self.logs_folder, '{}.png'.format(key)), 'image',
          'Samples {} {}'.format(name, split))
    return loggers

  def check_register(self):
    if not self.loggers[self.loggers.keys()[0]].is_registered():
      for name in self.loggers.iterkeys():
        self.loggers[name].register()

  @staticmethod
  def get_max_items_per_row(inp_height, inp_width):
    if inp_height == inp_width:
      return 8
    else:
      return 4

  def write_log(self, results):
    x = results['x_trans']
    y_gt = results['y_gt_trans']
    y_out = results['y_out']
    max_items = self.get_max_items_per_row(x.shape[1], x.shape[2])
    if self.loggers is not None:
      if 'input' in self.loggers:
        pu.plot_thumbnails(
            self.loggers['input'].get_fname(),
            results['x_trans'],
            axis=0,
            max_items_per_row=max_items)
      if 'gt_segmentation' in self.loggers:
        if y_gt.shape[3] == 1:
          plot_img = np.squeeze(y_gt, axis=3)
          y_gt_mask = y_gt
        else:
          y_gt_mask = y_gt[:, :, :, 1:].max(axis=3, keepdims=True)
          plot_img = self.build_orientation_img(y_gt, None,
                                                self.sem_color_wheel)
        pu.plot_thumbnails(
            self.loggers['gt_segmentation'].get_fname(),
            plot_img,
            axis=0,
            max_items_per_row=max_items)
      if 'output_segmentation' in self.loggers:
        # Single class segmentation
        if y_gt.shape[3] == 1:
          plot_img = np.squeeze(y_out, 3)
        else:
          plot_img = self.build_orientation_img(y_out, None,
                                                self.sem_color_wheel)
        pu.plot_thumbnails(
            self.loggers['output_segmentation'].get_fname(),
            plot_img,
            axis=0,
            max_items_per_row=max_items)
      if self.add_orientation:
        d_gt = results['d_gt_trans']
        d_out = results['d_out']
        if 'gt_orientation' in self.loggers:
          img = self.build_orientation_img(d_gt, y_gt_mask,
                                           self.ori_color_wheel)
          pu.plot_thumbnails(
              self.loggers['gt_orientation'].get_fname(),
              img,
              axis=0,
              max_items_per_row=max_items)
        if 'output_orientation' in self.loggers:
          img = self.build_orientation_img(d_out, y_gt_mask,
                                           self.ori_color_wheel)
          pu.plot_thumbnails(
              self.loggers['output_orientation'].get_fname(),
              img,
              axis=0,
              max_items_per_row=max_items)
    self.check_register()
    self.batch_iter.reset()

  def build_orientation_img(self, d, y, cw):
    d2 = np.expand_dims(d, 4)
    did = np.argmax(d, -1)
    c2 = cw[did.reshape([-1])].reshape(d.shape[0], d.shape[1], d.shape[2], 3)
    if y is not None:
      img = (c2 * y).astype('uint8')
    else:
      img = (c2).astype('uint8')
    return img


class FGExperiment(TrainingExperimentBase):

  def get_ts_loggers(self):
    model_opt = self.model_opt
    loggers = {}
    restore_step = self.step.get()
    loggers['loss'] = TimeSeriesLogger(
        os.path.join(self.logs_folder, 'loss.csv'), ['train', 'valid'],
        name='Loss',
        buffer_size=1,
        restore_step=restore_step)
    loggers['iou'] = TimeSeriesLogger(
        os.path.join(self.logs_folder, 'iou.csv'),
        ['train soft', 'valid soft', 'train hard', 'valid hard'],
        name='IoU',
        buffer_size=1,
        restore_step=restore_step)
    loggers['foreground_loss'] = TimeSeriesLogger(
        os.path.join(self.logs_folder, 'foreground_loss.csv'),
        ['train', 'valid'],
        name='Foreground loss',
        buffer_size=1,
        restore_step=restore_step)
    if model_opt['add_orientation']:
      loggers['orientation_ce'] = TimeSeriesLogger(
          os.path.join(self.logs_folder, 'orientation_ce.csv'),
          ['train', 'valid'],
          name='Orientation CE',
          buffer_size=1,
          restore_step=restore_step)
      loggers['orientation_acc'] = TimeSeriesLogger(
          os.path.join(self.logs_folder, 'orientation_acc.csv'),
          ['train', 'valid'],
          name='Orientation accuracy',
          buffer_size=1,
          restore_step=restore_step)
    loggers['step_time'] = TimeSeriesLogger(
        os.path.join(self.logs_folder, 'step_time.csv'),
        'step time (ms)',
        name='Step time',
        buffer_size=1,
        restore_step=restore_step)
    return loggers

  def get_runner_trainval(self):
    return Evaluator(
        self.sess,
        self.model,
        self.dataset['train'],
        self.opt,
        self.model_opt,
        step=self.step,
        loggers=self.loggers,
        phase_train=True)

  def get_runner_train(self):
    return Trainer(
        self.sess,
        self.model,
        self.dataset['train'],
        self.opt,
        self.model_opt,
        step=self.step,
        loggers=self.loggers)

  def get_runner_valid(self):
    return Evaluator(
        self.sess,
        self.model,
        self.dataset['valid'],
        self.opt,
        self.model_opt,
        step=self.step,
        loggers=self.loggers,
        phase_train=False)

  def get_runner_plot_train(self):
    return Plotter(
        self.sess,
        self.model,
        self.dataset['train'],
        self.opt,
        self.model_opt,
        step=self.step,
        logs_folder=self.logs_folder,
        split='train',
        phase_train=True)

  def get_runner_plot_valid(self):
    return Plotter(
        self.sess,
        self.model,
        self.dataset['valid'],
        self.opt,
        self.model_opt,
        step=self.step,
        logs_folder=self.logs_folder,
        split='valid',
        phase_train=False)

  def get_model(self):
    return get_model(self.model_opt)


class FGModelArgsParser(CmdArgsParser):

  def add_args(self):
    self.parser.add_argument('--cnn_filter_size', default='3,3,3,3,3,3,3,3,3,3')
    self.parser.add_argument(
        '--cnn_depth', default='8,8,16,16,32,32,64,64,128,128')
    self.parser.add_argument('--cnn_pool', default='1,2,1,2,1,2,1,2,1,2')
    self.parser.add_argument(
        '--dcnn_filter_size', default='3,3,3,3,3,3,3,3,3,3,3')
    self.parser.add_argument(
        '--dcnn_depth', default='128,128,64,64,32,32,16,16,8,8,1')
    self.parser.add_argument('--dcnn_pool', default='2,1,2,1,2,1,2,1,2,1,1')
    self.parser.add_argument('--add_skip_conn', action='store_true')
    # From the input image all the way to the last second layer of CNN.
    self.parser.add_argument('--cnn_skip_mask', default='1,0,0,0,0,0,1,0,1,0')
    self.parser.add_argument('--dcnn_skip_mask', default='0,1,0,1,0,0,0,0,0,1')
    self.parser.add_argument('--segm_loss_fn', default='iou')
    self.parser.add_argument('--add_orientation', action='store_true')
    self.parser.add_argument('--num_orientation_classes', default=8, type=int)
    self.parser.add_argument('--num_semantic_classes', default=1, type=int)
    self.parser.add_argument('--base_learn_rate', default=1e-3, type=float)
    self.parser.add_argument('--learn_rate_decay', default=0.96, type=float)
    self.parser.add_argument(
        '--steps_per_learn_rate_decay', default=5000, type=int)
    self.parser.add_argument('--rnd_colour', action='store_true')
    self.parser.add_argument('--padding', default=16, type=int)
    self.parser.add_argument('--optimizer', default='adam')

  def make_opt(self, args):
    cnn_fsize_list = args.cnn_filter_size.split(',')
    cnn_fsize_list = [int(fsize) for fsize in cnn_fsize_list]
    cnn_depth_list = args.cnn_depth.split(',')
    cnn_depth_list = [int(depth) for depth in cnn_depth_list]
    cnn_pool_list = args.cnn_pool.split(',')
    cnn_pool_list = [int(pool) for pool in cnn_pool_list]
    cnn_skip_mask_list = args.cnn_skip_mask.split(',')
    cnn_skip_mask_list = [bool(sk == '1') for sk in cnn_skip_mask_list]

    dcnn_fsize_list = args.dcnn_filter_size.split(',')
    dcnn_fsize_list = [int(fsize) for fsize in dcnn_fsize_list]
    dcnn_depth_list = args.dcnn_depth.split(',')
    dcnn_depth_list = [int(depth) for depth in dcnn_depth_list]
    dcnn_pool_list = args.dcnn_pool.split(',')
    dcnn_pool_list = [int(pool) for pool in dcnn_pool_list]
    dcnn_skip_mask_list = args.dcnn_skip_mask.split(',')
    dcnn_skip_mask_list = [bool(sk == '1') for sk in dcnn_skip_mask_list]

    inp_height, inp_width, timespan = self.get_inp_dim(args.dataset)

    model_opt = {
        'inp_height': inp_height,
        'inp_width': inp_width,
        'inp_depth': 3,
        'padding': args.padding,
        'cnn_filter_size': [3] * len(cnn_depth_list),
        'cnn_depth': cnn_depth_list,
        'cnn_pool': cnn_pool_list,
        'cnn_skip_mask': cnn_skip_mask_list,
        'dcnn_filter_size': [3] * len(dcnn_depth_list),
        'dcnn_depth': dcnn_depth_list,
        'dcnn_pool': dcnn_pool_list,
        'dcnn_skip_mask': dcnn_skip_mask_list,
        'weight_decay': 5e-5,
        'use_bn': True,
        'segm_loss_fn': args.segm_loss_fn,
        'rnd_hflip': False,
        'rnd_vflip': False,
        'rnd_transpose': False,
        'rnd_colour': args.rnd_colour,
        'add_skip_conn': args.add_skip_conn,
        'base_learn_rate': args.base_learn_rate,
        'learn_rate_decay': args.learn_rate_decay,
        'steps_per_learn_rate_decay': args.steps_per_learn_rate_decay,
        'add_orientation': args.add_orientation,
        'num_orientation_classes': args.num_orientation_classes,
        'num_semantic_classes': args.num_semantic_classes,
        'optimizer': args.optimizer
    }
    return model_opt


if __name__ == '__main__':
  parsers = {
      'default': TrainArgsParser(),
      'data': DataArgsParser(),
      'model': FGModelArgsParser()
  }
  FGExperiment.create_from_main(
      'fg_model', parsers=parsers, description='training').run()