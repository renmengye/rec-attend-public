import os
from kitti import KITTI
from cvppp import CVPPP
#from mscoco_ins import MSCOCOIns
from cityscapes import Cityscapes

kDefaultCvpppTrainFolder = 'data/cvppp/A1'
kDefaultCvpppTestFolder = 'data/cvppp_test/A1'
kDefaultKittiFolder = 'data/kitti'
kDefaultMscocoPersonFolder = 'data/mscoco_person'
kDefaultMscocoZebraFolder = 'data/mscoco_zebra'
kDefaultCityscapesFolder = 'data/cityscapes'


def get(dataset_name, data_opt, split='train', h5_fname=None):
  """Get dataset for instance segmentation."""
  opt = data_opt
  if dataset_name == 'cvppp':
    dataset_folder = opt['folder']
    if dataset_folder is None:
      if split != 'test':
        dataset_folder = kDefaultCvpppTrainFolder
      else:
        dataset_folder = kDefaultCvpppTestFolder
    if h5_fname is None:
      h5_fname = os.path.join(dataset_folder, '{}_{}x{}.h5'.format(
          split, opt['height'], opt['width']))
    return CVPPP(h5_fname)
  elif dataset_name == 'kitti':
    dataset_folder = opt['folder']
    if dataset_folder is None:
      dataset_folder = kDefaultKittiFolder
    opt['timespan'] = 20
    opt['num_examples'] = -1
    if h5_fname is None:
      h5_fname = os.path.join(dataset_folder, '{}_{}x{}.h5'.format(
          split, opt['height'], opt['width']))
    return KITTI(h5_fname)
  elif dataset_name == 'mscoco_person':
    dataset_folder = opt['folder']
    if dataset_folder is None:
      dataset_folder = kDefaultMscocoPersonFolder
    if h5_fname is None:
      h5_fname = os.path.join(
          dataset_folder,
          '{}_{}x{}.h5'.format(split, opt['height'], opt['width']))
    return MSCOCOIns(h5_fname, cat='person')
  elif dataset_name == 'mscoco_zebra':
    dataset_folder = opt['folder']
    if dataset_folder is None:
      dataset_folder = kDefaultMscocoZebraFolder
    if h5_fname is None:
      h5_fname = os.path.join(
          dataset_folder,
          '{}_{}x{}.h5'.format(split, opt['height'], opt['width']))
    return MSCOCOIns(h5_fname, cat='zebra')
  elif dataset_name == 'cityscapes':
    dataset_folder = opt['folder']
    if dataset_folder is None:
      dataset_folder = kDefaultCityscapesFolder
    if h5_fname is None:
      h5_fname = os.path.join(
          dataset_folder,
          '{}_{}x{}.h5'.format(split, opt['height'], opt['width']))
    return Cityscapes(h5_fname)
  else:
    raise Exception('Unknown dataset name')
