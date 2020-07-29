import argparse
import os

from utils import logger


class CmdArgsParser(object):

  def set_parser(self, parser):
    self.parser = parser

  def add_args(self):
    pass

  def make_opt(self, args):
    pass

  def get_inp_dim(self, dataset):
    kSynthShapeInpHeight = 224
    kSynthShapeInpWidth = 224
    kCvpppInpHeight = 224
    kCvpppInpWidth = 224

    # kCvpppNumObj = 20
    kCvpppNumObj = 80

    kKittiInpHeight = 128
    kKittiInpWidth = 448
    kKittiNumObj = 19
    kMscocoInpHeight = 224
    kMscocoInpWidth = 224
    kMscocoPersonNumObj = 22
    kMscocoZebraNumObj = 14
    kCityScapesInpHeight = 256
    kCityScapesInpWidth = 512
    if dataset == 'synth_shape':
      timespan = None
      inp_height = kSynthShapeInpHeight
      inp_width = kSynthShapeInpWidth
    elif dataset == 'kitti':
      timespan = kKittiNumObj + 1
      inp_height = kKittiInpHeight
      inp_width = kKittiInpWidth
    elif dataset == 'kitti_flow':
      timespan = kKittiNumObj + 1
      inp_height = kKittiInpHeight
      inp_width = kKittiInpWidth
    elif dataset == 'cvppp':
      timespan = kCvpppNumObj + 1
      inp_height = kCvpppInpHeight
      inp_width = kCvpppInpWidth
    elif dataset == 'mscoco_person':
      timespan = kMscocoPersonNumObj + 1
      inp_height = kMscocoInpHeight
      inp_width = kMscocoInpWidth
    elif dataset == 'mscoco_zebra':
      timespan = kMscocoZebraNumObj + 1
      inp_height = kMscocoInpHeight
      inp_width = kMscocoInpWidth
    elif dataset == 'cityscapes':
      timespan = 20
      inp_height = kCityScapesInpHeight
      inp_width = kCityScapesInpWidth
    else:
      raise Exception('Unknown dataset "{}"'.format(dataset))
    return inp_height, inp_width, timespan

  def get_inp_transform(self, dataset):
    if dataset == 'cvppp':
      rnd_hflip = True
      rnd_vflip = True
      rnd_transpose = True
      rnd_colour = False
    elif dataset == 'kitti':
      rnd_hflip = False
      rnd_vflip = False
      rnd_transpose = False
      rnd_colour = False
    elif dataset == 'mscoco_person' or dataset == 'mscoco_zebra':
      rnd_hflip = False
      rnd_vflip = False
      rnd_transpose = False
      rnd_colour = False
    elif dataset == 'cityscapes':
      rnd_hflip = False
      rnd_vflip = False
      rnd_transpose = False
      rnd_colour = False
    else:
      raise Exception('Unknown dataset "{}"'.format(dataset))
    return rnd_hflip, rnd_vflip, rnd_transpose, rnd_colour


class TrainArgsParser(CmdArgsParser):

  def add_args(self):
    self.parser.add_argument('--model_id', default=None)
    self.parser.add_argument('--num_steps', default=500000, type=int)
    self.parser.add_argument('--steps_per_ckpt', default=1000, type=int)

    # self.parser.add_argument('--steps_per_valid', default=50, type=int)
    self.parser.add_argument('--steps_per_valid', default=10000, type=int)

    # self.parser.add_argument('--steps_per_trainval', default=50, type=int)
    self.parser.add_argument('--steps_per_trainval', default=10000, type=int)

    # self.parser.add_argument('--steps_per_plot', default=500, type=int)
    self.parser.add_argument('--steps_per_plot', default=9999999999, type=int)

    self.parser.add_argument('--steps_per_log', default=10, type=int)
    self.parser.add_argument('--batch_size', default=32, type=int)
    self.parser.add_argument('--results', default='results')
    self.parser.add_argument('--logs', default='logs')
    self.parser.add_argument('--localhost', default='localhost')
    self.parser.add_argument('--restore', default=None)
    self.parser.add_argument('--num_samples_plot', default=5, type=int)
    self.parser.add_argument('--save_ckpt', action='store_true')
    self.parser.add_argument('--no_valid', action='store_true')

    # self.parser.add_argument('--num_batch_valid', default=10, type=int)
    self.parser.add_argument('--num_batch_valid', default=1, type=int)

    self.parser.add_argument('--h5_fname_train', default=None)
    self.parser.add_argument('--h5_fname_valid', default=None)
    self.parser.add_argument('--prefetch', action='store_true')
    self.parser.add_argument('--queue_size', default=50, type=int)
    self.parser.add_argument('--num_worker', default=4, type=int)

  def make_opt(self, args):
    return {
        'model_id': args.model_id,
        'batch_size': args.batch_size,
        'num_steps': args.num_steps,
        'steps_per_ckpt': args.steps_per_ckpt,
        'steps_per_valid': args.steps_per_valid,
        'steps_per_trainval': args.steps_per_trainval,
        'steps_per_plot': args.steps_per_plot,
        'steps_per_log': args.steps_per_log,
        'has_valid': not args.no_valid,
        'results': args.results,
        'restore': args.restore,
        'save_ckpt': args.save_ckpt,
        'logs': args.logs,
        'localhost': args.localhost,
        'num_batch_valid': args.num_batch_valid,
        'h5_fname_train': args.h5_fname_train,
        'h5_fname_valid': args.h5_fname_valid,
        'prefetch': args.prefetch,
        'queue_size': args.queue_size,
        'num_worker': args.num_worker
    }


class EvalArgsParser(CmdArgsParser):

  def add_args(self):
    self.parser.add_argument('--model_id', default=None)
    self.parser.add_argument('--batch_size', default=32, type=int)
    self.parser.add_argument('--results', default='./results')
    self.parser.add_argument('--output', default=None)
    self.parser.add_argument('--split', default='valid')
    self.parser.add_argument('--prefetch', action='store_true')
    self.parser.add_argument('--queue_size', default=50, type=int)
    self.parser.add_argument('--num_worker', default=4, type=int)

  def make_opt(self, args):
    if args.model_id is None:
      raise Exception('You must provide model ID')
    return {
        'model_id': args.model_id,
        'batch_size': args.batch_size,
        'results': args.results,
        'output': args.output,
        'restore': os.path.join(args.results, args.model_id),
        'split': args.split.split(','),
        'prefetch': args.prefetch,
        'queue_size': args.queue_size,
        'num_worker': args.num_worker
    }


class DataArgsParser(CmdArgsParser):

  def add_args(self):
    self.parser.add_argument('--dataset', default='cvppp')
    self.parser.add_argument('--dataset_folder', default=None)

  def make_opt(self, args):
    inp_height, inp_width, timespan = self.get_inp_dim(args.dataset)
    if args.dataset == 'cvppp':
      data_opt = {
          'folder': args.dataset_folder,
          'height': inp_height,
          'width': inp_width,
          'timespan': timespan
      }
    elif args.dataset == 'kitti' or args.dataset == 'kitti_flow':
      data_opt = {
          'folder': args.dataset_folder,
          'height': inp_height,
          'width': inp_width,
          'timespan': timespan
      }
    elif args.dataset == 'mscoco_person' or args.dataset == 'mscoco_zebra':
      data_opt = {
          'folder': args.dataset_folder,
          'height': inp_height,
          'width': inp_width,
          'timespan': timespan
      }
    elif args.dataset == 'cityscapes':
      data_opt = {
          'folder': args.dataset_folder,
          'height': inp_height,
          'width': inp_width,
          'timespan': timespan
      }
    data_opt['dataset'] = args.dataset
    return data_opt


class CmdArgsBase(object):

  def __init__(self, description):
    self.log = logger.get()
    self.description = description
    self.parsers = {}
    self.args_parser = argparse.ArgumentParser(description=description)
    self.args = None

  def add_parser(self, name, parser):
    if name in self.parsers:
      raise Exception('Parser {} already exists'.format(name))
    self.parsers[name] = parser
    parser.set_parser(self.args_parser)

  def add_args(self):
    for parser in self.parsers.itervalues():
      parser.add_args()

  def parse(self):
    self.args = self.args_parser.parse_args()

  def get_opt(self, name):
    if self.args is None:
      self.add_args()
      self.parse()
    if name not in self.parsers:
      raise Exception('Parser {} not found'.format(name))
    return self.parsers[name].make_opt(self.args)
