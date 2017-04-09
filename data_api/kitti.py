from __future__ import division

import sys
import os
import cv2
import numpy as np
import sep_labels

from ins_seg_dataset import InsSegDataset
from ins_seg_assembler import InsSegAssembler


class KITTIAssembler(InsSegAssembler):

  def __init__(self, folder, opt, split='train', output_fname=None):
    self.folder = folder
    self.split = split
    if output_fname is None:
      output_fname = os.path.join(self.folder, '{}_{}x{}.h5'.format(
          self.split, opt['height'], opt['width']))

    self.gt_folder = os.path.join(self.folder, 'gt')
    if self.split == 'valid_man' or self.split == 'test_man':
      self.gt_folder = os.path.join(self.folder, 'gt_man')
    self.image_folder = os.path.join(self.folder, 'images')
    super(KITTIAssembler, self).__init__(opt, output_fname)

  def read_ids(self):

    def _get_id_fname(split):
      return os.path.join(self.folder, '{}.txt'.format(split))

    def _read(fname):
      with open(fname) as f:
        return [int(ii.strip('\n')) for ii in f]

    return _read(_get_id_fname(self.split))

  def get_str_id(self, img_id):
    return '{:06d}'.format(img_id)

  def get_image(self, img_id):
    img_id_str = self.get_str_id(img_id)
    fname = '{}.png'.format(img_id_str)
    img_fname = os.path.join(self.image_folder, fname)
    if not os.path.exists(img_fname):
      raise Exception('Image file not exists: {}'.format(img_fname))
    img = cv2.imread(img_fname)
    return img

  def get_segmentations(self, img_id):
    img_id_str = self.get_str_id(img_id)
    fname = '{}.png'.format(img_id_str)
    gt_fname = os.path.join(self.gt_folder, fname)
    if not os.path.exists(gt_fname):
      raise Exception('GT file not exists: {} or '.format(gt_fname))
    gt_img = cv2.imread(gt_fname)
    segm, colors = sep_labels.get_separate_labels(gt_img)
    if len(segm) > 0:
      sem_segm = [np.zeros(segm[0].shape)]
      for ss in segm:
        sem_segm[0] = np.maximum(ss, sem_segm[0])
    else:
      sem_segm = []
    return segm, sem_segm, [0] * len(segm)


class KITTI(InsSegDataset):

  def get_fname(self, idx):
    return '{:06d}.png'.format(idx)

  def get_str_id(self, idx):
    return '{:06d}'.format(idx)

  def get_default_timespan(self):
    """Maximum number of objects."""
    return 20
