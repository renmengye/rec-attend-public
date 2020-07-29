#!/usr/bin/env python
import cv2
import h5py

from evaluation import OneTimeEvalBase
from experiment import EvalExperimentBase
from cmd_args_parser import EvalArgsParser, DataArgsParser

from fg_model import get_model


class FGPackRunner(OneTimeEvalBase):

  def __init__(self, sess, model, dataset, train_opt, model_opt):
    outputs = ['y_out', 'd_out']
    self.input_variables = set(['x', 'idx_map'])
    super(FGPackRunner, self).__init__(sess, model, dataset, train_opt,
                                       model_opt, outputs)

  def get_batch(self, idx):
    """Transform a dataset get_batch into a dictionary to feed."""
    self._batch = self.dataset.get_batch(idx, variables=self.input_variables)
    return {'x': self._batch['x']}

  def write_log(self, results):
    """Process results
    Args:
      results: y_out, s_out
    """
    inp = self._batch
    y_out = results['y_out']
    d_out = results['d_out']
    with h5py.File(self.dataset.h5_fname, 'r+') as h5f:
      for ii in range(y_out.shape[0]):
        idx = inp['idx_map'][ii]
        group = h5f[self.dataset.get_str_id(idx)]
        if 'foreground_pred' in group:
          del group['foreground_pred']
        if 'orientation_pred' in group:
          del group['orientation_pred']
        for cl in range(y_out.shape[3]):
          y_out_arr = y_out[ii, :, :, cl]
          y_out_arr = (y_out_arr * 255).astype('uint8')
          y_out_str = cv2.imencode('.png', y_out_arr)[1]
          group['foreground_pred/{:02d}'.format(cl)] = y_out_str
        for ch in range(d_out.shape[3]):
          d_out_arr = d_out[ii, :, :, ch]
          d_out_arr = (d_out_arr * 255).astype('uint8')
          d_out_str = cv2.imencode('.png', d_out_arr)[1]
          group['orientation_pred/{:02d}'.format(ch)] = d_out_str


class FGPackExperiment(EvalExperimentBase):

  def get_runner(self, split):
    return FGPackRunner(self.sess, self.model, self.dataset[split], self.opt,
                        self.model_opt)

  def get_model(self):
    return get_model(self.model_opt)


def main():
  parsers = {'default': EvalArgsParser(), 'data': DataArgsParser()}
  FGPackExperiment.create_from_main(
      'fg_pack', parsers=parsers, description='Pack fg output').run()


if __name__ == '__main__':
  main()
