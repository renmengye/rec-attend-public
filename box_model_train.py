#!/usr/bin/env python
"""Train a box net."""
from __future__ import division

import matplotlib
matplotlib.use("Agg")
import numpy as np
import os
from utils import logger
from utils import plot_utils as pu
from utils.batch_iter import BatchIterator
from utils.lazy_registerer import LazyRegisterer
from utils.step_counter import StepCounter
from utils.time_series_logger import TimeSeriesLogger

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from cmd_args_parser import TrainArgsParser, DataArgsParser, CmdArgsParser
from experiment import TrainingExperimentBase
from runner import RunnerBase
# from box_model_old import get_model
from box_model import get_model

import tensorflow as tf

log = logger.get()


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
    self.loggers = loggers
    self.log = logger.get()
    self.model_opt = model_opt
    self.train_opt = train_opt
    self.input_variables = self.get_input_variables()
    num_ex = dataset.get_dataset_size()
    batch_iter = BatchIterator(
        num_ex,
        batch_size=train_opt['batch_size'],
        get_fn=self.get_batch,
        cycle=True,
        shuffle=True,
        log_epoch=-1)
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
    variables = ['x', 'y_gt', 's_gt']
    if 'add_d_out' in self.model_opt:
      if self.model_opt['add_d_out']:
        variables.append('d_out')
    if 'add_y_out' in self.model_opt:
      if self.model_opt['add_y_out']:
        variables.append('y_out')
    return variables

  def get_batch(self, idx):
    """Transform a dataset get_batch into a dictionary to feed."""
    batch_ = self.dataset.get_batch(idx, variables=self.input_variables)
    batch = {}
    batch['x'] = batch_['x']
    batch['y_gt'] = batch_['y_gt']
    if 'add_d_out' in self.model_opt:
      if self.model_opt['add_d_out']:
        batch['d_in'] = batch_['d_out']
    if 'add_y_out' in self.model_opt:
      if self.model_opt['add_y_out']:
        batch['y_in'] = batch_['y_out']
    batch['s_gt'] = batch_['s_gt']
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
    outputs = ['loss', 'box_loss', 'conf_loss']
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
      if 'box_loss' in self.loggers:
        if self.phase_train:
          line = [results['box_loss'], '']
        else:
          line = ['', results['box_loss']]
        self.loggers['box_loss'].add(self.step.get(), line)
      if 'box_iou' in self.loggers:
        if self.phase_train:
          line = [results['box_iou'], '']
        else:
          line = ['', results['box_iou']]
        self.loggers['box_iou'].add(self.step.get(), line)
      if 'conf_loss' in self.loggers:
        if self.phase_train:
          line = [results['conf_loss'], '']
        else:
          line = ['', results['conf_loss']]
        self.loggers['conf_loss'].add(self.step.get(), line)


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
    outputs = [
        'x_trans', 'y_gt_trans', 'attn_top_left', 'attn_bot_right',
        'attn_top_left_gt', 'attn_bot_right_gt', 'match_box', 's_out',
        'ctrl_rnn_glimpse_map'
    ]
    num_batch = 1
    self.split = split
    self.logs_folder = logs_folder
    self.model_opt = model_opt
    loggers = self.get_loggers()
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

  def get_loggers(self):
    loggers = {}
    labels = ['input', 'output', 'attn']
    for key in labels:
      loggers[key] = LazyRegisterer(
          os.path.join(self.logs_folder, '{}_{}.png'.format(key, self.split)),
          'image', 'Samples {} {}'.format(key, self.split))
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
      return 5

  def write_log(self, results):
    batch = results['_batches'][0]
    x = results['x_trans']
    y = results['y_gt_trans']
    s = batch['s_gt']
    max_items = self.get_max_items_per_row(x.shape[1], x.shape[2])
    x_tile = np.expand_dims(x, 1)
    x_tile = np.tile(x_tile, [1, y.shape[1], 1, 1, 1])
    if self.loggers is not None:
      if 'input' in self.loggers:
        pu.plot_output(
            self.loggers['input'].get_fname(),
            y_out=x_tile,
            s_out=s,
            match=results['match_box'],
            attn=(results['attn_top_left_gt'], results['attn_bot_right_gt']),
            max_items_per_row=max_items)
      if 'output' in self.loggers:
        if self.model_opt['num_semantic_classes'] == 1:
          s_out = results['s_out']
        else:
          # Complement of the background class.
          s_out = 1 - results['s_out'][:, :, 0]
        pu.plot_output(
            self.loggers['output'].get_fname(),
            y_out=x_tile,
            s_out=s_out,
            match=results['match_box'],
            attn=(results['attn_top_left'], results['attn_bot_right']),
            max_items_per_row=max_items)
      # if 'attn' in self.loggers:
      #   pu.plot_double_attention(
      #       self.loggers['attn'].get_fname(),
      #       x,
      #       results['ctrl_rnn_glimpse_map'],
      #       max_items_per_row=max_items)
    self.check_register()
    self.batch_iter.reset()


class BoxExperiment(TrainingExperimentBase):

  def get_ts_loggers(self):
    loggers = {}
    restore_step = self.step.get()
    loggers['loss'] = TimeSeriesLogger(
        os.path.join(self.logs_folder, 'loss.csv'), ['train', 'valid'],
        name='Loss',
        buffer_size=1,
        restore_step=restore_step)
    loggers['box_loss'] = TimeSeriesLogger(
        os.path.join(self.logs_folder, 'box_loss.csv'), ['train', 'valid'],
        name='Box Loss',
        buffer_size=1,
        restore_step=restore_step)
    loggers['conf_loss'] = TimeSeriesLogger(
        os.path.join(self.logs_folder, 'conf_loss.csv'), ['train', 'valid'],
        name='Confidence Loss',
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


class BoxModelArgsParser(CmdArgsParser):

  def add_args(self):
    self.parser.add_argument('--padding', default=16, type=int)
    self.parser.add_argument('--filter_height', default=48, type=int)
    self.parser.add_argument('--filter_width', default=48, type=int)
    self.parser.add_argument(
        '--ctrl_cnn_filter_size', default='3,3,3,3,3,3,3,3')
    self.parser.add_argument('--ctrl_cnn_depth', default='4,4,8,8,16,16,32,64')
    self.parser.add_argument('--ctrl_cnn_pool', default='1,2,1,2,1,2,2,2')
    self.parser.add_argument('--box_loss_fn', default='iou')
    self.parser.add_argument('--fixed_order', action='store_true')
    self.parser.add_argument('--pretrain_cnn', default=None)
    self.parser.add_argument('--pretrain_net', default=None)
    self.parser.add_argument('--freeze_pretrain_cnn', action='store_true')
    self.parser.add_argument('--ctrl_rnn_hid_dim', default=256, type=int)
    self.parser.add_argument('--num_ctrl_mlp_layers', default=2, type=int)
    self.parser.add_argument('--ctrl_mlp_dim', default=256, type=int)
    self.parser.add_argument('--base_learn_rate', default=0.001, type=float)
    self.parser.add_argument('--learn_rate_decay', default=0.96, type=float)
    self.parser.add_argument('--clip_gradient', default=1.0, type=float)
    self.parser.add_argument(
        '--steps_per_learn_rate_decay', default=5000, type=int)
    self.parser.add_argument('--squash_ctrl_params', action='store_true')
    self.parser.add_argument('--num_ctrl_rnn_iter', default=5, type=int)
    self.parser.add_argument('--num_glimpse_mlp_layers', default=2, type=int)
    self.parser.add_argument('--fixed_var', action='store_true')
    self.parser.add_argument('--dynamic_var', action='store_true')
    self.parser.add_argument('--add_d_out', action='store_true')
    self.parser.add_argument('--add_y_out', action='store_true')
    self.parser.add_argument('--use_iou_box', action='store_true')
    self.parser.add_argument('--num_semantic_classes', default=1, type=int)

  def make_opt(self, args):
    ccnn_fsize_list = args.ctrl_cnn_filter_size.split(',')
    ccnn_fsize_list = [int(fsize) for fsize in ccnn_fsize_list]
    ccnn_depth_list = args.ctrl_cnn_depth.split(',')
    ccnn_depth_list = [int(depth) for depth in ccnn_depth_list]
    ccnn_pool_list = args.ctrl_cnn_pool.split(',')
    ccnn_pool_list = [int(pool) for pool in ccnn_pool_list]

    inp_height, inp_width, timespan = self.get_inp_dim(args.dataset)
    rnd_hflip, rnd_vflip, rnd_transpose, rnd_colour = \
        self.get_inp_transform(args.dataset)

    if args.dataset == 'synth_shape':
      timespan = args.max_num_objects + 1

    model_opt = {
        'timespan': timespan,
        'inp_height': inp_height,
        'inp_width': inp_width,
        'inp_depth': 3,
        'padding': args.padding,
        'filter_height': args.filter_height,
        'filter_width': args.filter_width,
        'ctrl_cnn_filter_size': ccnn_fsize_list,
        'ctrl_cnn_depth': ccnn_depth_list,
        'ctrl_cnn_pool': ccnn_pool_list,
        'ctrl_rnn_hid_dim': args.ctrl_rnn_hid_dim,
        'num_ctrl_mlp_layers': args.num_ctrl_mlp_layers,
        'ctrl_mlp_dim': args.ctrl_mlp_dim,
        'attn_box_padding_ratio': 0.2,
        'weight_decay': 5e-5,
        'use_bn': True,
        'box_loss_fn': args.box_loss_fn,
        'base_learn_rate': args.base_learn_rate,
        'learn_rate_decay': args.learn_rate_decay,
        'steps_per_learn_rate_decay': args.steps_per_learn_rate_decay,
        'pretrain_cnn': args.pretrain_cnn,
        'pretrain_net': args.pretrain_net,
        'freeze_pretrain_cnn': args.freeze_pretrain_cnn,
        'squash_ctrl_params': args.squash_ctrl_params,
        'clip_gradient': args.clip_gradient,
        'fixed_order': args.fixed_order,
        'ctrl_rnn_inp_struct': "attn",
        'num_ctrl_rnn_iter': args.num_ctrl_rnn_iter,
        'num_glimpse_mlp_layers': args.num_glimpse_mlp_layers,
        'fixed_var': args.fixed_var,
        'use_iou_box': args.use_iou_box,
        'dynamic_var': args.dynamic_var,
        'add_d_out': args.add_d_out,
        'add_y_out': args.add_y_out,
        'rnd_hflip': rnd_hflip,
        'rnd_vflip': rnd_vflip,
        'rnd_transpose': rnd_transpose,
        'rnd_colour': rnd_colour,
        'num_semantic_classes': args.num_semantic_classes
    }
    return model_opt


if __name__ == '__main__':
  parsers = {
      'default': TrainArgsParser(),
      'data': DataArgsParser(),
      'model': BoxModelArgsParser()
  }
  BoxExperiment.create_from_main(
      'box_model', parsers=parsers, description='training').run()