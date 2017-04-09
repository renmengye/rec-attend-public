from __future__ import division

import cv2
import numpy as np
import os
import time

from utils import logger
from utils import BatchIterator, ConcurrentBatchIterator

from runner import RunnerBase
import analysis

log = logger.get()


class OneTimeEvalBase(RunnerBase):

  def __init__(self,
               sess,
               model,
               dataset,
               opt,
               model_opt,
               outputs,
               start_idx=-1,
               end_idx=-1):
    self.dataset = dataset
    self.log = logger.get()
    self.model_opt = model_opt
    self.opt = opt
    self.input_variables = self.get_input_variables()
    if start_idx != -1 and end_idx != -1:
      if start_idx < 0 or end_idx < 0:
        self.log.fatal('Indices must be non-negative.')
      elif start_idx >= end_idx:
        self.log.fatal('End index must be greater than start index.')
      num_ex = end_idx - start_idx
      if end_idx > dataset.get_dataset_size():
        self.log.warning('End index exceeds dataset size.')
        end_idx = dataset.get_dataset_size()
        num_ex = end_idx - start_idx
      self.log.info('Running partial dataset: start {} end {}'.format(start_idx,
                                                                      end_idx))
      self.all_idx = np.arange(start_idx, end_idx)
    else:
      self.log.info('Running through entire dataset.')
      num_ex = dataset.get_dataset_size()
      self.all_idx = np.arange(num_ex)
    if num_ex == -1:
      num_ex = dataset.get_dataset_size()
    batch_iter = BatchIterator(
        num_ex,
        batch_size=opt['batch_size'],
        get_fn=self.get_batch,
        cycle=False,
        shuffle=False)
    if opt['prefetch']:
      batch_iter = ConcurrentBatchIterator(
          batch_iter,
          max_queue_size=opt['queue_size'],
          num_threads=opt['num_worker'],
          log_queue=-1)
    super(OneTimeEvalBase, self).__init__(
        sess,
        model,
        batch_iter,
        outputs,
        num_batch=1,
        phase_train=False,
        increment_step=False)
    pass

  def get_input_variables(self):
    variables = ['x', 's_gt', 'idx_map']
    if 'add_d_out' in self.model_opt:
      if self.model_opt['add_d_out']:
        variables.append('d_out')
    if 'add_y_out' in self.model_opt:
      if self.model_opt['add_y_out']:
        variables.append('y_out')
    return set(variables)

  def get_batch(self, idx):
    """Transform a dataset get_batch into a dictionary to feed."""
    idx_new = self.all_idx[idx]
    _batch = self.dataset.get_batch(idx_new, variables=self.input_variables)
    batch = {}
    batch['x'] = _batch['x']
    if 'add_d_out' in self.model_opt:
      if self.model_opt['add_d_out']:
        batch['d_in'] = _batch['d_out']
    if 'add_y_out' in self.model_opt:
      if self.model_opt['add_y_out']:
        batch['y_in'] = _batch['y_out']
    batch['idx_map'] = _batch['idx_map']
    batch['_s_gt'] = _batch['s_gt']
    return batch