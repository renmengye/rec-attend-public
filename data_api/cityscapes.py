from __future__ import division

import sys
import os
sys.path.append(
    os.path.normpath(
        os.path.join(
            os.path.dirname(__file__), 'cityscapes_scripts', 'helpers')))
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(__file__), '..')))

from utils import logger

import cv2
import numpy as np
import sep_labels
from labels import labels, name2label

from ins_seg_dataset import InsSegDataset
from ins_seg_assembler import InsSegAssembler

log = logger.get()


class CityscapesAssembler(InsSegAssembler):

  def __init__(self,
               folder,
               opt,
               split='train',
               output_fname=None,
               coarse_label=False):
    self.folder = folder
    self.split = split
    if output_fname is None:
      output_fname = os.path.join(self.folder, '{}_{}x{}.h5'.format(
          self.split, opt['height'], opt['width']))

    if self.split == 'valid':
      splitname = 'val'
    else:
      splitname = self.split
    if self.split == 'train_extra':
      img_folder = 'leftImg8bit_trainextra/leftImg8bit'
    else:
      img_folder = 'leftImg8bit'
    if coarse_label:
      gt_folder = 'gtCoarse'
    else:
      gt_folder = 'gtFine'
    self.gt_subfolder = gt_folder
    self.gt_folder = os.path.join(self.folder, gt_folder, splitname)
    self.image_folder = os.path.join(self.folder, img_folder, splitname)
    self.id_to_label = {}
    for label in labels:
      self.id_to_label[label.id] = label
    super(CityscapesAssembler, self).__init__(opt, output_fname)

  def _read_ids(self):
    self.log.info(self.h5_fname)
    with h5py.File(self.h5_fname, 'r') as h5f:
      idx = np.array(
          filter(lambda x: x != 'index_map', [str(kk) for kk in h5f.keys()]))
    return idx

  def read_ids(self):
    runs = os.listdir(self.image_folder)
    image_ids = []
    for run in runs:
      folder = os.path.join(self.image_folder, run)
      image_ids.extend(
          [ll.split('_leftImg8bit.png')[0] for ll in os.listdir(folder)])
    log.info('Number of images: {}'.format(len(image_ids)))
    return image_ids

  def get_str_id(self, img_id):
    return img_id

  def get_image(self, img_id):
    img_id_str = self.get_str_id(img_id)
    run_name = img_id_str.split('_')[0]
    img_fname = os.path.join(self.image_folder, run_name,
                             img_id_str + '_leftImg8bit.png')
    if not os.path.exists(img_fname):
      raise Exception('Image file not exists: {}'.format(img_fname))
    img = cv2.imread(img_fname)
    return img

  def get_segmentations(self, img_id):
    img_id_str = self.get_str_id(img_id)
    run_name = img_id_str.split('_')[0]
    gt_fname = os.path.join(
        self.gt_folder, run_name,
        img_id_str + '_{}_instanceIds.png'.format(self.gt_subfolder))
    gt_img = cv2.imread(gt_fname, -1)
    if gt_img is None:
      log.warning('GT image does not exist: "{}"'.format(gt_fname))
      # raise Exception('Image file not exist: "{}"'.format(gt_fname))
      segm = []
      colors = []
    else:
      # log.warning('GT image does exist: "{}"'.format(gt_fname))
      segm, colors = sep_labels.get_separate_labels(gt_img)
    sem_segm = [None] * 8
    segm_final = []
    segm_sem_cls = []
    for ss, cc in zip(segm, colors):
      if cc > 1000:
        sem_cls = int(np.floor(cc / 1000))
        ins_id = int(cc) % 1000
        label = self.id_to_label[sem_cls]
        if label.trainId > 0:
          sem_cls_train = label.trainId
          segm_final.append(ss)
          if sem_segm[sem_cls_train - 1] is None:
            sem_segm[sem_cls_train - 1] = np.zeros(ss.shape)
          sem_segm[sem_cls_train - 1] = np.maximum(sem_segm[sem_cls_train - 1],
                                                   ss)
          segm_sem_cls.append(sem_cls_train - 1)
    return segm_final, sem_segm, segm_sem_cls


class Cityscapes(InsSegDataset):

  def get_name(self):
    return 'cityscapes'

  def get_fname(self, idx):
    return idx + '_gtFine_instanceIds.png'

  def get_str_id(self, idx):
    return idx

  def get_num_semantic_classes(self):
    return 8

  def get_default_timespan(self):
    """Maximum number of objects."""
    return 20

  def get_batch(self, idx, timespan=None, variables=None):
    batch = super(Cityscapes, self).get_batch(
        idx, timespan=timespan, variables=variables)
    sem_weights = np.ones(len(idx))
    ori_weights = np.ones(len(idx))
    if 'source' in batch:
      for kk, ii in enumerate(idx):
        src = batch['source'][kk]
        if src == 'train_extra':
          sem_weights[kk] = 0.1  # Weights for coarse segmentation.
          ori_weights[kk] = 0.0  # No instance info in coarse.
        elif src == 'train' or src == 'valid':
          sem_weights[kk] = 1.0
          ori_weights[kk] = 1.0
        else:
          raise Exception('Unknown data source "{}"'.format(src))
    batch['sem_weights'] = sem_weights
    batch['ori_weights'] = ori_weights
    return batch