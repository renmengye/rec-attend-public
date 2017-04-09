import fnmatch
import logger
import os
import yaml
import tensorflow as tf

kModelOptFilename = 'model_opt.yaml'
kDatasetOptFilename = 'dataset_opt.yaml'
kMaxToKeep = 2


class Saver():

  def __init__(self, folder, model_opt=None, data_opt=None):
    if not os.path.exists(folder):
      os.makedirs(folder)
    self.folder = folder
    self.log = logger.get()
    self.tf_saver = None
    if model_opt is not None:
      self.save_opt(os.path.join(folder, kModelOptFilename), model_opt)
    if data_opt is not None:
      self.save_opt(os.path.join(folder, kDatasetOptFilename), data_opt)

  def save(self, sess, global_step=None):
    """Save checkpoint."""
    if self.tf_saver is None:
      self.tf_saver = tf.train.Saver(tf.all_variables(), max_to_keep=kMaxToKeep)
    ckpt_path = os.path.join(self.folder, 'model.ckpt')
    self.log.info('Saving checkpoint to {}'.format(ckpt_path))
    self.tf_saver.save(sess, ckpt_path, global_step=global_step)

  def save_opt(self, fname, opt):
    with open(fname, 'w') as f:
      yaml.dump(opt, f, default_flow_style=False)

  def get_latest_ckpt(self):
    """Get the latest checkpoint filename in a folder."""
    ckpt_fname_pattern = os.path.join(self.folder, 'model.ckpt-*')
    ckpt_fname_list = []
    for fname in os.listdir(self.folder):
      fullname = os.path.join(self.folder, fname)
      if fnmatch.fnmatch(fullname, ckpt_fname_pattern):
        if not fullname.endswith('.meta'):
          ckpt_fname_list.append(fullname)
    if len(ckpt_fname_list) == 0:
      raise Exception('No checkpoint file found.')
    ckpt_fname_step = [
        int(fn.split('-')[-1].split('.')[0]) for fn in ckpt_fname_list
    ]
    latest_step = max(ckpt_fname_step)
    latest_ckpt = os.path.join(self.folder, 'model.ckpt-{}'.format(latest_step))
    latest_graph = os.path.join(self.folder,
                                'model.ckpt-{}.meta'.format(latest_step))
    return (latest_ckpt, latest_graph, latest_step)

  def get_ckpt_info(self):
    """Get info of the latest checkpoint."""
    if not os.path.exists(self.folder):
      raise Exception('Folder "{}" does not exist'.format(self.folder))
    model_id = os.path.basename(self.folder.rstrip('/'))
    self.log.info('Restoring from {}'.format(self.folder))
    model_opt_fname = os.path.join(self.folder, kModelOptFilename)
    data_opt_fname = os.path.join(self.folder, kDatasetOptFilename)
    if os.path.exists(model_opt_fname):
      with open(model_opt_fname) as f_opt:
        model_opt = yaml.load(f_opt)
    else:
      model_opt = None
    self.log.info('Model options: {}'.format(model_opt))
    if os.path.exists(data_opt_fname):
      with open(data_opt_fname) as f_opt:
        data_opt = yaml.load(f_opt)
    else:
      data_opt = None
    ckpt_fname, graph_fname, latest_step = self.get_latest_ckpt()
    self.log.info('Restoring at step {}'.format(latest_step))
    return {
        'ckpt_fname': ckpt_fname,
        'graph_fname': graph_fname,
        'model_opt': model_opt,
        'data_opt': data_opt,
        'step': latest_step,
        'model_id': model_id
    }

  def restore(self, sess, ckpt_fname=None):
    """Restore the checkpoint file."""
    if ckpt_fname is None:
      ckpt_fname = self.get_latest_ckpt()[0]
    if self.tf_saver is None:
      self.tf_saver = tf.train.Saver(tf.all_variables())
    self.tf_saver.restore(sess, ckpt_fname)
