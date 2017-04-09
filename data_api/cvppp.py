from __future__ import division

import cv2
import numpy as np
import os
import re
import sep_labels

from ins_seg_dataset import InsSegDataset
from ins_seg_assembler import InsSegAssembler
from tqdm import tqdm


class CVPPPAssembler(InsSegAssembler):

  def __init__(self, folder, opt, split=None, output_fname=None):
    self.folder = folder
    self.split = split
    self.label_regex = re.compile('plant(?P<imgid>[0-9]{3})_label.png')
    self.image_regex = re.compile('plant(?P<imgid>[0-9]{3})_rgb.png')
    self.fg_regex = re.compile('plant(?P<imgid>[0-9]{3})_fg.png')
    if output_fname is None:
      output_fname = os.path.join(folder, '{}_{}x{}.h5'.format(
          split, opt['height'], opt['width']))
    super(CVPPPAssembler, self).__init__(opt, output_fname)

  def read_ids(self):
    split_ids = None
    if self.split is not None or self.split != 'all':
      id_fname = os.path.join(self.folder, '{}.txt'.format(self.split))
      if not os.path.exists(id_fname):
        self.write_split()
      with open(id_fname) as f:
        split_ids = [int(ll.strip('\n')) for ll in f.readlines()]
    return split_ids

  def get_str_id(self, idx):
    return 'plant{:03d}'.format(idx)

  def get_image(self, img_id):
    img_id_str = self.get_str_id(img_id)
    fname = '{}_rgb.png'.format(img_id_str)
    img_fname = os.path.join(self.folder, fname)
    if not os.path.exists(img_fname):
      raise Exception('Image file not exists: {}'.format(img_fname))
    img = cv2.imread(img_fname)
    return img

  def get_segmentations(self, img_id):
    img_id_str = self.get_str_id(img_id)
    fname = '{}_label.png'.format(img_id_str)
    gt_fname = os.path.join(self.folder, fname)
    if not os.path.exists(gt_fname):
      self.log.warning('GT file not found: {}'.format(gt_fname))
      gt_fname = os.path.join(self.folder, '{}_fg.png'.format(img_id_str))
      if not os.path.exists(gt_fname):
        raise Exception('GT file not exists: {}'.format(gt_fname))
    gt_img = cv2.imread(gt_fname)
    segm, colors = sep_labels.get_separate_labels(gt_img)
    sem_segm = [np.zeros(segm[0].shape)]
    for ss in segm:
      sem_segm[0] = np.maximum(ss, sem_segm[0])
    return segm, sem_segm, [0] * len(segm)

  def write_split(self):
    random = np.random.RandomState(2)
    self.log.info('Reading images from {}'.format(self.folder))
    file_list = os.listdir(self.folder)

    image_id_dict = {}
    for fname in tqdm(file_list):
      match = self.image_regex.search(fname)
      if match:
        imgid = int(match.group('imgid'))
        image_id_dict[imgid] = True

    image_ids = np.array(image_id_dict.keys())
    num_train = np.ceil(image_ids.size * 0.8)  # 103 for 128
    idx = np.arange(len(image_ids))
    random.shuffle(idx)
    train_ids = image_ids[idx[:num_train]]
    valid_ids = image_ids[idx[num_train:]]
    self.log.info('Number of images: {}'.format(len(idx)))

    self.log.info('Train indices: {}'.format(idx[:num_train]))
    self.log.info('Valid indices: {}'.format(idx[num_train:]))
    self.log.info('Train image ids: {}'.format(train_ids))
    self.log.info('Valid image ids: {}'.format(valid_ids))

    with open(os.path.join(self.folder, 'train.txt'), 'w') as f:
      for ii in train_ids:
        f.write('{}\n'.format(ii))

    with open(os.path.join(self.folder, 'valid.txt'), 'w') as f:
      for ii in valid_ids:
        f.write('{}\n'.format(ii))

    with open(os.path.join(self.folder, 'all.txt'), 'w') as f:
      for ii in train_ids:
        f.write('{}\n'.format(ii))
      for ii in valid_ids:
        f.write('{}\n'.format(ii))


class CVPPP(InsSegDataset):

  def __init__(self, h5_fname, folder=None):
    self.folder = folder
    super(CVPPP, self).__init__(h5_fname)
    pass

  def get_fname(self, idx, fg=False):
    str_id = self.get_str_id(idx)
    if fg:
      return '{}_fg.png'.format(str_id)
    else:
      return '{}_label.png'.format(str_id)

  def get_str_id(self, idx):
    return 'plant{:03d}'.format(idx)

  def get_default_timespan(self):
    """Maximum number of objects."""
    return 21
