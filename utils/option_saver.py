import logger
import os
import yaml


class OptionSaver(object):

  def __init__(self, folder, name):
    if not os.path.exists(folder):
      os.makedirs(folder)
    self.folder = folder
    self.log = logger.get()
    self.fname = os.path.join(folder, name + '.yaml')

  def read(self):
    if os.path.exists(self.fname):
      with open(self.fname) as f_opt:
        opt = yaml.load(f_opt)
        return opt
    else:
      raise Exception('File not found: {}'.format(self.fname))

  def save(self, opt):
    with open(self.fname, 'w') as f:
      yaml.dump(opt, f, default_flow_style=False)
