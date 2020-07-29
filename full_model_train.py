#!/usr/bin/env python
"""Train full model."""
from __future__ import division

import numpy as np
import os
import tensorflow as tf

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from utils import logger
from utils import plot_utils as pu
from utils import BatchIterator, ConcurrentBatchIterator
from utils.lazy_registerer import LazyRegisterer
from utils.step_counter import StepCounter
from utils.time_series_logger import TimeSeriesLogger

from cmd_args_parser import TrainArgsParser, DataArgsParser, CmdArgsParser
from experiment import TrainingExperimentBase
from runner import RunnerBase
from full_model import get_model


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
    variables = ['x', 'y_gt', 's_gt']
    if self.model_opt['num_semantic_classes'] > 1:
      variables.append('c_gt_idx')
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
    if self.model_opt['num_semantic_classes'] > 1:
      batch['c_gt'] = batch_['c_gt_idx']
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
    self.log = logger.get()
    if model_opt['finetune']:
      self.log.warning('Finetuning')
      sess.run(tf.assign(model['global_step'], 0))
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
    outputs = [
        'loss', 'conf_loss', 'segm_loss', 'count_acc', 'dic', 'dic_abs',
        'learn_rate', 'box_loss', 'gt_knob_prob_box', 'gt_knob_prob_segm'
    ]
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

  def write_key(self, results, key, rkey=None):
    if rkey is None:
      rkey = key
    if key in self.loggers:
      if self.phase_train:
        line = [results[rkey], '']
      else:
        line = ['', results[rkey]]
      self.loggers[key].add(self.step.get(), line)

  def write_log(self, results):
    if self.loggers is not None:
      self.log.info('{:d} loss {:.4f} t {:.2f}ms'.format(self.step.get(
      ), results['loss'], results['step_time']))
      self.write_key(results, 'loss')
      self.write_key(results, 'segm_loss')
      self.write_key(results, 'box_loss')
      self.write_key(results, 'conf_loss')
      self.write_key(results, 'count_acc')
      self.write_key(results, 'dic')
      self.write_key(results, 'dic_abs')

      if self.phase_train:
        self.write_key(results, 'learn_rate')
        self.loggers['gt_knob'].add(self.step.get(), [
            results['gt_knob_prob_box'], results['gt_knob_prob_segm']
        ])


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
        'x_trans', 'y_gt_trans', 'y_out', 's_out', 'match', 'attn_top_left',
        'attn_bot_right', 'match_box', 's_out', 'x_patch',
        'ctrl_rnn_glimpse_map'
    ]
    num_batch = 1
    phase_train = phase_train
    self.split = split
    self.logs_folder = logs_folder
    self.model_opt = model_opt
    loggers = self.get_loggers()
    self.color_wheel = np.array(
        [[255, 17, 0], [255, 137, 0], [230, 255, 0], [34, 255, 0],
         [0, 255, 213], [0, 154, 255], [9, 0, 255], [255, 0, 255]],
        dtype='uint8')
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
    labels = ['input', 'output', 'total', 'box', 'patch', 'attn']
    if self.model_opt['add_d_out']:
      labels.append('d_in')
    if self.model_opt['add_y_out']:
      labels.append('y_in')
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
      return 4

  def write_log(self, results):
    batch = results['_batches'][0]
    x = batch['x']
    y = batch['y_gt']
    s = batch['s_gt']
    max_items = self.get_max_items_per_row(x.shape[1], x.shape[2])
    x_tile = np.expand_dims(x, 1)
    x_tile = np.tile(x_tile, [1, y.shape[1], 1, 1, 1])
    conf_out = results['s_out']
    if self.loggers is not None:
      if 'input' in self.loggers:
        pu.plot_input(
            self.loggers['input'].get_fname(),
            x=results['x_trans'],
            y_gt=results['y_gt_trans'],
            s_gt=s,
            max_items_per_row=max_items)
      if 'output' in self.loggers:
        pu.plot_output(
            self.loggers['output'].get_fname(),
            y_out=results['y_out'],
            s_out=conf_out,
            match=results['match'],
            attn=(results['attn_top_left'], results['attn_bot_right']),
            max_items_per_row=max_items)
      if 'total' in self.loggers:
        pu.plot_total_instances(
            self.loggers['total'].get_fname(),
            y_out=results['y_out'],
            s_out=conf_out,
            max_items_per_row=max_items)
      if 'box' in self.loggers:
        pu.plot_output(
            self.loggers['box'].get_fname(),
            y_out=x_tile,
            s_out=conf_out,
            match=results['match_box'],
            attn=(results['attn_top_left'], results['attn_bot_right']),
            max_items_per_row=max_items)
      if 'patch' in self.loggers:
        pu.plot_thumbnails(
            self.loggers['patch'].get_fname(),
            results['x_patch'][:, :, :, :, :3],
            axis=1,
            max_items_per_row=8)
      if 'attn' in self.loggers:
        attn = results['ctrl_rnn_glimpse_map'][:, :, :max_items]
        pu.plot_double_attention(
            self.loggers['attn'].get_fname(),
            results['x_trans'],
            attn,
            max_items_per_row=max_items)
      if 'd_in' in self.loggers:
        img = self.build_orientation_img(batch['d_in'], batch['y_gt'])
        img = np.expand_dims(img, 1)
        pu.plot_thumbnails(self.loggers['d_in'].get_fname(), img, axis=1)

      if 'y_in' in self.loggers:
        pu.plot_thumbnails(
            self.loggers['y_in'].get_fname(), batch['y_in'], axis=3)
    self.check_register()
    self.batch_iter.reset()

  def build_orientation_img(self, d, y):
    y = np.expand_dims(y.sum(axis=1), 3)
    d2 = np.expand_dims(d, 4)
    cw = self.color_wheel

    did = np.argmax(d, -1)
    c2 = cw[did.reshape([-1])].reshape(y.shape[0], y.shape[1], y.shape[2], 3)
    img = (c2 * y).astype('uint8')
    return img


class FullExperiment(TrainingExperimentBase):

  def get_ts_loggers(self):
    loggers = {}
    restore_step = self.step.get()
    loggers['loss'] = TimeSeriesLogger(
        os.path.join(self.logs_folder, 'loss.csv'), ['train', 'valid'],
        name='Loss',
        buffer_size=1,
        restore_step=restore_step)
    loggers['conf_loss'] = TimeSeriesLogger(
        os.path.join(self.logs_folder, 'conf_loss.csv'), ['train', 'valid'],
        name='Confidence Loss',
        buffer_size=1,
        restore_step=restore_step)
    loggers['segm_loss'] = TimeSeriesLogger(
        os.path.join(self.logs_folder, 'segm_loss.csv'), ['train', 'valid'],
        name='Segmentation Loss',
        buffer_size=1,
        restore_step=restore_step)
    loggers['dic'] = TimeSeriesLogger(
        os.path.join(self.logs_folder, 'dic.csv'), ['train', 'valid'],
        name='DiC',
        buffer_size=1,
        restore_step=restore_step)
    loggers['dic_abs'] = TimeSeriesLogger(
        os.path.join(self.logs_folder, 'dic_abs.csv'), ['train', 'valid'],
        name='|DiC|',
        buffer_size=1,
        restore_step=restore_step)
    loggers['learn_rate'] = TimeSeriesLogger(
        os.path.join(self.logs_folder, 'learn_rate.csv'),
        'learning rate',
        name='Learning rate',
        buffer_size=1,
        restore_step=restore_step)
    loggers['count_acc'] = TimeSeriesLogger(
        os.path.join(self.logs_folder, 'count_acc.csv'), ['train', 'valid'],
        name='Count acc',
        buffer_size=1,
        restore_step=restore_step)
    loggers['step_time'] = TimeSeriesLogger(
        os.path.join(self.logs_folder, 'step_time.csv'),
        'step time (ms)',
        name='Step time',
        buffer_size=1,
        restore_step=restore_step)
    loggers['box_loss'] = TimeSeriesLogger(
        os.path.join(self.logs_folder, 'box_loss.csv'), ['train', 'valid'],
        name='Box Loss',
        buffer_size=1,
        restore_step=restore_step)
    loggers['gt_knob'] = TimeSeriesLogger(
        os.path.join(self.logs_folder, 'gt_knob.csv'), ['box', 'segmentation'],
        name='GT mix',
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
    """Model router."""
    if self.model_opt['finetune']:
      self.model_opt['base_learn_rate'] = self.new_model_opt['base_learn_rate']
      self.model_opt['knob_box_offset'] = self.new_model_opt['knob_box_offset']
      self.model_opt['knob_segm_offset'] = self.new_model_opt[
          'knob_segm_offset']
      self.model_opt['use_knob'] = self.new_model_opt['use_knob']
      self.model_opt['steps_per_knob_decay'] = self.new_model_opt[
          'steps_per_knob_decay']
    return get_model(self.model_opt)


class ModelArgsParser(CmdArgsParser):

  def add_args(self):
    # Original model options
    self.parser.add_argument('--cnn_filter_size', default='3,3,3,3,3')
    self.parser.add_argument('--cnn_depth', default='4,8,8,12,16')
    self.parser.add_argument('--cnn_pool', default='2,2,2,2,2')
    self.parser.add_argument('--dcnn_filter_size', default='3,3,3,3,3,3')
    self.parser.add_argument('--dcnn_depth', default='8,6,4,4,2,1')
    self.parser.add_argument('--dcnn_pool', default='2,2,2,2,2,1')
    self.parser.add_argument('--rnn_type', default='lstm')
    self.parser.add_argument('--conv_lstm_filter_size', default=3, type=int)
    self.parser.add_argument('--conv_lstm_hid_depth', default=12, type=int)
    self.parser.add_argument('--rnn_hid_dim', default=256, type=int)
    self.parser.add_argument('--score_maxpool', default=1, type=int)
    self.parser.add_argument('--num_mlp_layers', default=2, type=int)
    self.parser.add_argument('--mlp_depth', default=6, type=int)
    self.parser.add_argument('--use_deconv', action='store_true')
    self.parser.add_argument('--score_use_core', action='store_true')

    # Shared options
    self.parser.add_argument('--padding', default=16, type=int)
    self.parser.add_argument('--weight_decay', default=5e-5, type=float)
    self.parser.add_argument('--base_learn_rate', default=0.001, type=float)
    self.parser.add_argument('--learn_rate_decay', default=0.96, type=float)
    self.parser.add_argument(
        '--steps_per_learn_rate_decay', default=5000, type=int)
    self.parser.add_argument('--loss_mix_ratio', default=1.0, type=float)
    self.parser.add_argument('--segm_loss_fn', default='iou')
    self.parser.add_argument('--mlp_dropout', default=None, type=float)
    self.parser.add_argument('--fixed_order', action='store_true')
    self.parser.add_argument('--add_skip_conn', action='store_true')

    # Attention-based model options
    self.parser.add_argument('--filter_height', default=48, type=int)
    self.parser.add_argument('--filter_width', default=48, type=int)
    self.parser.add_argument('--ctrl_cnn_filter_size', default='3,3,3,3,3')
    self.parser.add_argument('--ctrl_cnn_depth', default='4,8,16,16,32')
    self.parser.add_argument('--ctrl_cnn_pool', default='2,2,2,2,2')
    self.parser.add_argument('--attn_cnn_filter_size', default='3,3,3')
    self.parser.add_argument('--attn_cnn_depth', default='4,8,16')
    self.parser.add_argument('--attn_cnn_pool', default='2,2,2')
    self.parser.add_argument('--attn_dcnn_filter_size', default='3,3,3,3')
    self.parser.add_argument('--attn_dcnn_depth', default='16,8,4,1')
    self.parser.add_argument('--attn_dcnn_pool', default='2,2,2,1')
    self.parser.add_argument('--attn_cnn_skip', default='1,1,1')
    self.parser.add_argument('--ctrl_rnn_hid_dim', default=256, type=int)
    self.parser.add_argument('--num_ctrl_mlp_layers', default=1, type=int)
    self.parser.add_argument('--ctrl_mlp_dim', default=256, type=int)
    self.parser.add_argument('--box_loss_fn', default='iou')
    self.parser.add_argument(
        '--attn_box_padding_ratio', default=0.2, type=float)
    self.parser.add_argument('--use_knob', action='store_true')
    self.parser.add_argument('--knob_decay', default=0.9, type=float)
    self.parser.add_argument('--steps_per_knob_decay', default=300, type=int)
    self.parser.add_argument('--knob_base', default=1.0, type=float)
    self.parser.add_argument('--knob_box_offset', default=300, type=int)
    self.parser.add_argument('--knob_segm_offset', default=500, type=int)
    self.parser.add_argument('--knob_use_timescale', action='store_true')
    self.parser.add_argument('--gt_box_ctr_noise', default=0.05, type=float)
    self.parser.add_argument('--gt_box_pad_noise', default=0.1, type=float)
    self.parser.add_argument('--gt_segm_noise', default=0.3, type=float)
    self.parser.add_argument('--clip_gradient', default=1.0, type=float)
    self.parser.add_argument('--squash_ctrl_params', action='store_true')
    self.parser.add_argument('--fixed_gamma', action='store_true')
    self.parser.add_argument('--pretrain_ctrl_net', default=None)
    self.parser.add_argument('--pretrain_attn_net', default=None)
    self.parser.add_argument('--pretrain_net', default=None)
    self.parser.add_argument('--freeze_ctrl_cnn', action='store_true')
    self.parser.add_argument('--freeze_ctrl_rnn', action='store_true')
    self.parser.add_argument('--freeze_ctrl_mlp', action='store_true')
    self.parser.add_argument('--freeze_attn_net', action='store_true')
    self.parser.add_argument('--num_ctrl_rnn_iter', default=5, type=int)
    self.parser.add_argument('--num_glimpse_mlp_layers', default=2, type=int)
    self.parser.add_argument('--stop_canvas_grad', action='store_true')
    self.parser.add_argument('--fixed_var', action='store_true')
    self.parser.add_argument('--dynamic_var', action='store_true')
    self.parser.add_argument('--use_iou_box', action='store_true')
    self.parser.add_argument('--disable_overwrite', action='store_true')
    self.parser.add_argument('--add_d_out', action='store_true')
    self.parser.add_argument('--add_y_out', action='store_true')
    self.parser.add_argument('--num_semantic_classes', default=1, type=int)

    self.parser.add_argument('--ctrl_add_inp', action='store_true')
    self.parser.add_argument('--ctrl_add_canvas', action='store_true')
    self.parser.add_argument('--ctrl_add_d_out', action='store_true')
    self.parser.add_argument('--ctrl_add_y_out', action='store_true')
    self.parser.add_argument('--attn_add_inp', action='store_true')
    self.parser.add_argument('--attn_add_canvas', action='store_true')
    self.parser.add_argument('--attn_add_d_out', action='store_true')
    self.parser.add_argument('--attn_add_y_out', action='store_true')

    self.parser.add_argument('--finetune', action='store_true')

  def make_opt(self, args):
    """Convert command-line arguments into model opt dict."""
    inp_height, inp_width, timespan = self.get_inp_dim(args.dataset)
    rnd_hflip, rnd_vflip, rnd_transpose, rnd_colour = \
        self.get_inp_transform(args.dataset)

    ccnn_fsize_list = args.ctrl_cnn_filter_size.split(',')
    ccnn_fsize_list = [int(fsize) for fsize in ccnn_fsize_list]
    ccnn_depth_list = args.ctrl_cnn_depth.split(',')
    ccnn_depth_list = [int(depth) for depth in ccnn_depth_list]
    ccnn_pool_list = args.ctrl_cnn_pool.split(',')
    ccnn_pool_list = [int(pool) for pool in ccnn_pool_list]

    acnn_fsize_list = args.attn_cnn_filter_size.split(',')
    acnn_fsize_list = [int(fsize) for fsize in acnn_fsize_list]
    acnn_depth_list = args.attn_cnn_depth.split(',')
    acnn_depth_list = [int(depth) for depth in acnn_depth_list]
    acnn_pool_list = args.attn_cnn_pool.split(',')
    acnn_pool_list = [int(pool) for pool in acnn_pool_list]
    acnn_skip_list = args.attn_cnn_skip.split(',')
    acnn_skip_list = [bool(skip == '1') for skip in acnn_skip_list]

    attn_dcnn_fsize_list = args.attn_dcnn_filter_size.split(',')
    attn_dcnn_fsize_list = [int(fsize) for fsize in attn_dcnn_fsize_list]
    attn_dcnn_depth_list = args.attn_dcnn_depth.split(',')
    attn_dcnn_depth_list = [int(depth) for depth in attn_dcnn_depth_list]
    attn_dcnn_pool_list = args.attn_dcnn_pool.split(',')
    attn_dcnn_pool_list = [int(pool) for pool in attn_dcnn_pool_list]

    model_opt = {
        'inp_height': inp_height,
        'inp_width': inp_width,
        'inp_depth': 3,
        'padding': args.padding,
        'filter_height': args.filter_height,
        'filter_width': args.filter_width,
        'timespan': timespan,
        'ctrl_cnn_filter_size': ccnn_fsize_list,
        'ctrl_cnn_depth': ccnn_depth_list,
        'ctrl_cnn_pool': ccnn_pool_list,
        'ctrl_rnn_hid_dim': args.ctrl_rnn_hid_dim,
        'attn_cnn_filter_size': acnn_fsize_list,
        'attn_cnn_depth': acnn_depth_list,
        'attn_cnn_pool': acnn_pool_list,
        'attn_dcnn_filter_size': attn_dcnn_fsize_list,
        'attn_dcnn_depth': attn_dcnn_depth_list,
        'attn_dcnn_pool': attn_dcnn_pool_list,
        'attn_cnn_skip': acnn_skip_list,
        'num_ctrl_mlp_layers': args.num_ctrl_mlp_layers,
        'ctrl_mlp_dim': args.ctrl_mlp_dim,
        'mlp_dropout': args.mlp_dropout,
        'weight_decay': args.weight_decay,
        'base_learn_rate': args.base_learn_rate,
        'learn_rate_decay': args.learn_rate_decay,
        'steps_per_learn_rate_decay': args.steps_per_learn_rate_decay,
        'loss_mix_ratio': args.loss_mix_ratio,
        'segm_loss_fn': args.segm_loss_fn,
        'box_loss_fn': args.box_loss_fn,
        'use_bn': True,
        'attn_box_padding_ratio': args.attn_box_padding_ratio,
        'use_knob': args.use_knob,
        'knob_decay': args.knob_decay,
        'knob_base': args.knob_base,
        'steps_per_knob_decay': args.steps_per_knob_decay,
        'knob_box_offset': args.knob_box_offset,
        'knob_segm_offset': args.knob_segm_offset,
        'knob_use_timescale': args.knob_use_timescale,
        'gt_box_ctr_noise': args.gt_box_ctr_noise,
        'gt_box_pad_noise': args.gt_box_pad_noise,
        'gt_segm_noise': args.gt_segm_noise,
        'squash_ctrl_params': args.squash_ctrl_params,
        'clip_gradient': args.clip_gradient,
        'fixed_order': args.fixed_order,
        'fixed_gamma': args.fixed_gamma,
        'fixed_var': args.fixed_var,
        'dynamic_var': args.dynamic_var,
        'num_ctrl_rnn_iter': args.num_ctrl_rnn_iter,
        'num_glimpse_mlp_layers': args.num_glimpse_mlp_layers,
        'pretrain_ctrl_net': args.pretrain_ctrl_net,
        'pretrain_attn_net': args.pretrain_attn_net,
        'pretrain_net': args.pretrain_net,
        'freeze_ctrl_cnn': args.freeze_ctrl_cnn,
        'freeze_ctrl_rnn': args.freeze_ctrl_rnn,
        'freeze_ctrl_mlp': args.freeze_ctrl_mlp,
        'freeze_attn_net': args.freeze_attn_net,
        'stop_canvas_grad': args.stop_canvas_grad,
        'use_iou_box': args.use_iou_box,
        'add_skip_conn': args.add_skip_conn,
        'attn_cnn_skip': args.attn_cnn_skip,
        'disable_overwrite': args.disable_overwrite,
        'add_d_out': args.add_d_out,
        'add_y_out': args.add_y_out,
        'num_semantic_classes': args.num_semantic_classes,
        'ctrl_add_inp': args.ctrl_add_inp,
        'ctrl_add_canvas': args.ctrl_add_canvas,
        'ctrl_add_d_out': args.ctrl_add_d_out,
        'ctrl_add_y_out': args.ctrl_add_y_out,
        'attn_add_inp': args.attn_add_inp,
        'attn_add_canvas': args.attn_add_canvas,
        'attn_add_d_out': args.attn_add_d_out,
        'attn_add_y_out': args.attn_add_y_out,
        'rnd_hflip': False,
        'rnd_vflip': False,
        'rnd_transpose': False,
        'rnd_colour': rnd_colour,
        'finetune': args.finetune
    }
    return model_opt


if __name__ == '__main__':
  parsers = {
      'default': TrainArgsParser(),
      'data': DataArgsParser(),
      'model': ModelArgsParser()
  }
  FullExperiment.create_from_main(
      'full_model', parsers=parsers, description='training').run()