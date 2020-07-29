from __future__ import division

import cv2
import h5py
import numpy as np
import orientation as orient

from utils import logger
from tqdm import tqdm


class InsSegAssembler(object):

  def __init__(self, opt, output_fname):
    self.opt = opt
    self.log = logger.get()
    self.output_fname = output_fname

    self.log.info('Output h5 dataset: {}'.format(self.output_fname))
    self.log.info('Reading image IDs')
    self.img_ids = self.read_ids()

    # Shuffle sequence.
    random = np.random.RandomState(2)
    shuffle = np.arange(len(self.img_ids))
    random.shuffle(shuffle)
    self.img_ids = [
        self.img_ids[shuffle[idx]] for idx in range(len(self.img_ids))
    ]

  def read_ids(self):
    raise Exception('Not implemented')

  def get_str_id(self, img_id):
    raise Exception('Not implemented')

  def get_image(self, img_id):
    raise Exception('Not implemented')

  def get_segmentations(self, img_id):
    """
        Returns a tuple:
            T * [H, W] instance segmentation,
            C * [H, W] semantic segmentation
        """
    raise Exception('Not implemented')

  def save_inp_image(self, img, group):
    img_str = cv2.imencode('.png', img)[1]
    self.save('input', img_str, group)

  def save_full_image(self, img, group):
    img_str = cv2.imencode('.png', img)[1]
    self.save('input_full_size', img_str, group)

  def save_seg(self, seg_id, seg, group):
    seg_str = cv2.imencode('.png', seg)[1]
    key = 'label_segmentation/{:02d}'.format(seg_id)
    self.save(key, seg_str, group)

  def save_ori(self, ori, group):
    ori_str = cv2.imencode('.png', ori)[1]
    self.save('orientation', ori_str, group)

  def save_full_seg(self, seg_id, seg, group):
    seg_str = cv2.imencode('.png', seg)[1]
    key = 'label_segmentation_full_size/{:02d}'.format(seg_id)
    self.save(key, seg_str, group)

  def save_sem_seg(self, cls_id, seg, group):
    seg_str = cv2.imencode('.png', seg)[1]
    key = 'label_semantic_segmentation/{:02d}'.format(cls_id)
    self.save(key, seg_str, group)

  def save_full_sem_seg(self, cls_id, seg, group):
    seg_str = cv2.imencode('.png', seg)[1]
    key = 'label_semantic_segmentation_full_size/{:02d}'.format(cls_id)
    self.save(key, seg_str, group)

  def save(self, key, data, group):
    if key in group:
      del group[key]
    group[key] = data

  def assemble(self):
    inp_height = self.opt['height']
    inp_width = self.opt['width']
    inp_shape = (inp_width, inp_height)
    img_ids = self.img_ids
    num_ex = len(img_ids)
    self.log.info('Reading {} images'.format(num_ex))
    idx_map = []

    max_num_obj = 0
    self.log.info('Writing to {}'.format(self.output_fname))
    with h5py.File(self.output_fname, 'a') as h5f:
      for idx in tqdm(range(num_ex)):
        img_id = img_ids[idx]
        img_id_str = self.get_str_id(img_id)

        if img_id_str not in h5f:
          img_group = h5f.create_group(img_id_str)
        else:
          img_group = h5f[img_id_str]

        idx_map.append(img_id)
        img = self.get_image(img_id)

        orig_size = img.shape[:2]
        self.save('orig_size', np.array(orig_size), img_group)

        self.save_full_image(img, img_group)
        segm, sem_segm, segm_sem_cls = self.get_segmentations(img_id)

        # Standard size input image
        img = cv2.resize(img, inp_shape, interpolation=cv2.INTER_CUBIC)

        self.save_inp_image(img, img_group)

        # Save instance segmentation.
        max_num_obj = max(max_num_obj, len(segm))
        if len(segm) > 0:
          all_segs = []
          for jj, ss in enumerate(segm):
            seg = cv2.resize(ss, inp_shape, interpolation=cv2.INTER_NEAREST)
            # Standard size segmentation for training
            self.save_seg(jj, seg, img_group)

            seg_r = seg.reshape([1, 1, inp_height, inp_width])
            all_segs.append(seg_r)

            # Full size segmentation for evaluation
            self.save_full_seg(jj, ss, img_group)

          # Standard size orientation map
          all_segs = np.concatenate(all_segs, axis=1)
          ori = orient.get_orientation(all_segs, encoding='class')
          ori = np.squeeze(ori)
          self.save_ori(ori, img_group)

        # Save semantic segmentation.
        if len(sem_segm) > 0:
          for jj, ss in enumerate(sem_segm):
            if ss is not None:
              seg = cv2.resize(ss, inp_shape, interpolation=cv2.INTER_NEAREST)
              self.save_sem_seg(jj, seg, img_group)
              self.save_full_sem_seg(jj, ss, img_group)

        # Save semantic class info for each instance.
        self.save('instance_semantic_classes', np.array(segm_sem_cls),
                  img_group)

      # Indices
      self.save('index_map', np.array(idx_map), h5f)
    self.log.info('Maximum number of objects: {}'.format(max_num_obj))
