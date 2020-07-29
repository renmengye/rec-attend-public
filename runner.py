import numpy as np
import time

from utils import logger
from utils.step_counter import StepCounter

import tensorflow as tf


class EmptyRunner(object):

  def run_step(self):
    pass

  def finalize(self):
    pass


class RunnerBase(EmptyRunner):

  def __init__(self,
               sess,
               model,
               batch_iter,
               outputs,
               num_batch=1,
               step=StepCounter(0),
               phase_train=True,
               increment_step=False):
    self._sess = sess
    self._model = model
    self._batch_iter = batch_iter
    self._num_batch = num_batch
    self._phase_train = phase_train
    self._step = step
    self._outputs = outputs
    self._current_batch = {}
    self._log = logger.get()
    self._increment_step = increment_step
    pass

  @staticmethod
  def _check_nan(var):
    # Check NaN.
    if np.isnan(var):
      self._log.error('NaN occurred.')
      raise Exception('NaN')

  def _run_step(self, inp):
    """Train step"""
    bat_sz_total = 0
    results = {}
    feed_dict = self.get_feed_dict(inp)
    start_time = time.time()
    r = self.run_model(inp)
    step_time = (time.time() - start_time) * 1000
    r['step_time'] = step_time
    if self._increment_step:
      self.step.increment()
    return r

  def run_step(self):
    bat_sz_total = 0
    results = {}
    # Initialize values.
    for key in self.outputs:
      results[key] = 0.0
    results['step_time'] = 0.0
    results['_batches'] = []

    # Run each batch.
    for bb in range(self.num_batch):
      try:
        inp = self.batch_iter.next()
      except StopIteration:
        return False
      _results = self._run_step(inp)
      bat_sz = inp[inp.keys()[0]].shape[0]
      bat_sz_total += bat_sz
      for key in _results.iterkeys():
        if _results[key] is not None:
          results[key] += _results[key] * bat_sz
      results['_batches'].append(inp)
    # Average out all batches.
    for key in results.iterkeys():
      if not key.startswith('_'):
        results[key] = results[key] / bat_sz_total
    self.write_log(results)
    return True

  def get_feed_dict(self, inp):
    feed_dict = {self.model['phase_train']: self.phase_train}
    for key in inp.iterkeys():
      if key in self.model:
        feed_dict[self.model[key]] = inp[key]
    return feed_dict

  def run_model(self, inp):
    feed_dict = self.get_feed_dict(inp)
    symbol_list = [self.model[r] for r in self.outputs]
    results = self.sess.run(symbol_list, feed_dict=feed_dict)
    results_dict = {}
    for rr, name in zip(results, self.outputs):
      results_dict[name] = rr
    return results_dict

  def write_log(self, results):
    pass

  @property
  def outputs(self):
    return self._outputs

  @property
  def model(self):
    return self._model

  @property
  def phase_train(self):
    return self._phase_train

  @property
  def num_batch(self):
    return self._num_batch

  @property
  def sess(self):
    return self._sess

  @property
  def batch_iter(self):
    return self._batch_iter

  @property
  def current_batch(self):
    return self._current_batch

  @property
  def step(self):
    return self._step
