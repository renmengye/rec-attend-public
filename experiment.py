"""Basic experiment training/evaluation flow."""
from __future__ import division

import os
import sys

from data_api import data_provider
from assign_model_id import get_id
from utils import logger
from utils.log_manager import LogManager
from utils.saver import Saver
from utils.step_counter import StepCounter
from cmd_args_parser import CmdArgsBase
from tqdm import tqdm

import tensorflow as tf


class ExperimentBase(object):

  def __init__(self, name, opt, data_opt=None, model_opt=None, seed=1234):
    # Restore previously saved checkpoints.
    self.opt = opt
    self.name = name
    self.new_model_opt = None
    if self.opt['restore']:
      self.restore_options(opt, data_opt)
      if model_opt is not None:
        if 'finetune' in model_opt and model_opt['finetune']:
          self.model_opt['finetune'] = model_opt['finetune']
          self.new_model_opt = model_opt
          self.step.reset()
          self.model_id = self.get_model_id()
          self.exp_folder = os.path.join(self.opt['results'], self.model_id)
          self.saver = Saver(
              self.exp_folder, model_opt=self.model_opt, data_opt=self.data_opt)
      self.exp_folder = opt['restore']
    else:
      if self.opt['model_id']:
        self.model_id = self.opt['model_id']
      else:
        self.model_id = self.get_model_id()
      if model_opt is None or data_opt is None:
        raise Exception('You need to specify model options and data options')
      self.model_opt = model_opt
      self.data_opt = data_opt
      self.step = StepCounter()
      self.exp_folder = os.path.join(self.opt['results'], self.model_id)
      self.saver = Saver(
          self.exp_folder, model_opt=self.model_opt, data_opt=self.data_opt)

    self.init_cmd_logger()

    self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    # Log arguments
    self.log.log_args()

    # Train loop options
    self.log.info('Building model')
    self.model = self.get_model()

    # Load dataset
    self.log.info('Loading dataset')
    self.dataset_name = self.data_opt['dataset']
    self.dataset = self.get_dataset()

    self.init_model()
    self.init_logs()

  def restore_options(self, opt, data_opt):
    self.saver = Saver(opt['restore'])
    self.ckpt_info = self.saver.get_ckpt_info()
    self.model_opt = self.ckpt_info['model_opt']
    if data_opt is None:
      self.data_opt = self.ckpt_info['data_opt']
    else:
      self.data_opt = data_opt
    self.ckpt_fname = self.ckpt_info['ckpt_fname']
    self.step = StepCounter(self.ckpt_info['step'])
    self.model_id = self.ckpt_info['model_id']
    pass

  def init_model(self):
    # Restore/intialize weights
    if self.opt['restore']:
      self.saver.restore(self.sess, self.ckpt_fname)
    else:
      self.sess.run(tf.global_variables_initializer(),
                    feed_dict={self.model["phase_train"]: True})

  def init_cmd_logger(self):
    self.log = logger.get()

  def init_logs(self):
    pass

  def get_model_id(self):
    return '{}-{}'.format(self.name, get_id())

  @classmethod
  def create_from_main(cls, name, parsers=None, description=None):
    cmd = CmdArgsBase(description)
    opt = None
    if parsers is not None:
      for key in parsers.iterkeys():
        cmd.add_parser(key, parsers[key])
      opt = cmd.get_opt('default')
      if 'data' in parsers:
        data_opt = cmd.get_opt('data')
      else:
        data_opt = None
      if 'model' in parsers:
        model_opt = cmd.get_opt('model')
      else:
        model_opt = None
    return cls(name, opt, data_opt=data_opt, model_opt=model_opt)

  def get_model(self):
    raise NotImplemented()

  def get_dataset(self):
    raise NotImplemented()

  def run(self):
    raise NotImplemented()


class EvalExperimentBase(ExperimentBase):

  def __init__(self, name, opt, model_opt=None, data_opt=None):
    super(EvalExperimentBase, self).__init__(
        name, opt, model_opt=model_opt, data_opt=data_opt)

  def get_dataset(self):
    dataset = {}
    for ss in self.opt['split']:
      dataset[ss] = data_provider.get(self.dataset_name,
                                      self.data_opt,
                                      split=ss)
    return dataset

  def get_runner(self, split):
    raise NotImplemented()

  def run(self):
    runners = {}
    for ss in self.opt['split']:
      runners[ss] = self.get_runner(ss)
      self.log.info('Running split {}'.format(ss))
      while runners[ss].run_step():
        pass
      runners[ss].finalize()
    self.sess.close()


class TrainingExperimentBase(ExperimentBase):

  def init_logs(self):
    # Create time series loggers
    if self.opt['logs'] is not None:
      self.log_manager = LogManager(self.logs_folder)
      self.loggers = self.get_ts_loggers()
      self.register_raw_logs()
      self.log_url = 'http://{}/deep-dashboard?id={}'.format(
          self.opt['localhost'], self.model_id)
      self.log.info('Visualization can be viewed at: {}'.format(self.log_url))
    else:
      self.loggers = None

  def init_cmd_logger(self):
    # Logger
    if self.opt['logs'] is not None:
      self.logs_folder = self.opt['logs']
      self.logs_folder = os.path.join(self.logs_folder, self.model_id)
      self.log = logger.get(os.path.join(self.logs_folder, 'raw'))
    else:
      self.log = logger.get()

  def register_raw_logs(self):
    self.log_manager.register(self.log.filename, 'plain', 'Raw logs')
    cmd_fname = os.path.join(self.log_manager.folder, 'cmd.log')
    with open(cmd_fname, 'w') as f:
      f.write(' '.join(sys.argv))
    self.log_manager.register(cmd_fname, 'plain', 'Command-line arguments')
    model_opt_fname = os.path.join(self.log_manager.folder, 'model_opt.yaml')
    self.saver.save_opt(model_opt_fname, self.model_opt)
    self.log_manager.register(model_opt_fname, 'plain', 'Model hyperparameters')

  def get_dataset(self):
    dataset = {}
    dataset['train'] = data_provider.get(self.dataset_name,
                                         self.data_opt,
                                         split='train',
                                         h5_fname=self.opt['h5_fname_train'])
    dataset['valid'] = data_provider.get(self.dataset_name,
                                         self.data_opt,
                                         split='valid',
                                         h5_fname=self.opt['h5_fname_valid'])
    return dataset

  def get_ts_loggers(self):
    return {}

  def get_runner_trainval(self):
    return EmptyTrainer()

  def get_runner_train(self):
    return EmptyTrainer()

  def get_runner_valid(self):
    return EmptyTrainer()

  def get_runner_plot_train(self):
    return EmptyTrainer()

  def get_runner_plot_valid(self):
    return EmptyTrainer()

  def run(self):
    runner_trainval = self.get_runner_trainval()
    runner_train = self.get_runner_train()
    runner_plot_train = self.get_runner_plot_train()
    if self.opt['has_valid']:
      runner_valid = self.get_runner_valid()
      runner_plot_valid = self.get_runner_plot_valid()

    nstart = self.step.get()
    # Progress bar.
    it = tqdm(range(nstart, self.opt['num_steps']), desc=self.model_id)
    step_prev = self.step.get()
    while self.step.get() < self.opt['num_steps']:
      it.update(self.step.get() - step_prev)
      step_prev = self.step.get()

      # Plot samples
      if self.step.get() % self.opt['steps_per_plot'] == 0:
        self.log.info('Plot train samples')
        runner_plot_train.run_step()
        self.log.info('Plot valid samples')
        runner_plot_valid.run_step()

      # Train step
      runner_train.run_step()

      # Run validation stats
      if self.opt['has_valid']:
        if self.step.get() % self.opt['steps_per_valid'] == 0:
          print("\n# ------------------------------------------------------ #\n",
                self.step.get(), self.opt['steps_per_valid'], self.step.get() % self.opt['steps_per_valid'] == 0,
                "\n# ------------------------------------------------------ #\n")
          self.log.info('Running validation')
          runner_valid.run_step()

      # Train stats
      if self.step.get() % self.opt['steps_per_trainval'] == 0:
        print("\n# ------------------------------------------------------ #\n",
              self.step.get(), self.opt['steps_per_trainval'], self.step.get() % self.opt['steps_per_trainval'] == 0,
              "\n# ------------------------------------------------------ #\n")
        self.log.info('Running train validation')
        runner_trainval.run_step()

      # Save model
      if self.step.get() % self.opt['steps_per_ckpt'] == 0:
        if self.opt['save_ckpt']:
          print("\n# ------------------------------------------------------ #\n",
                self.step.get(), self.opt['steps_per_ckpt'], self.step.get() % self.opt['steps_per_ckpt'] == 0,
                "\n# ------------------------------------------------------ #\n")
          self.log.info('Saving checkpoint')
          self.saver.save(self.sess, global_step=self.step.get())
        else:
          self.log.warning('Saving is turned off. Use -save_ckpt flag to save.')

    it.close()
    runner_train.finalize()
    runner_valid.finalize()
    runner_trainval.finalize()
    runner_plot_train.finalize()
    runner_plot_valid.finalize()
    self.sess.close()

    for self.logger in self.loggers.itervalues():
      self.logger.close()
