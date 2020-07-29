from __future__ import division

# import sys
# sys.path.insert(0, '../')

from utils import logger
import cv2
import numpy as np
import h5py


class InsSegDataset(object):

  def __init__(self, h5_fname):
    self.log = logger.get()
    self.h5_fname = h5_fname
    self.log.info('Reading image IDs')
    self.img_ids = self._read_ids()
    pass

  def _read_ids(self):
    self.log.info(self.h5_fname)
    with h5py.File(self.h5_fname, 'r') as h5f:
      idx = h5f['index_map'][:]
    return idx

  def get_name(self):
    return 'unknown'

  def get_str_id(self, idx):
    return str(idx)

  def get_dataset_size(self):
    """Get number of examples."""
    return len(self.img_ids)

  def get_default_timespan(self):
    raise Exception('Not implemented')

  def get_num_semantic_classes(self):
    return 1

  def get_full_size_labels(self, img_ids, timespan=None):
    """Get full sized labels."""
    if timespan is None:
      timespan = self.get_default_timespan()
    with h5py.File(self.h5_fname, 'r') as h5f:
      num_ex = len(img_ids)
      y_full = []
      for kk, ii in enumerate(img_ids):
        key = self.get_str_id(ii)
        data_group = h5f[key]
        if 'label_segmentation_full_size' in data_group:
          y_gt_group = data_group['label_segmentation_full_size']
          num_obj = len(y_gt_group.keys())
          y_full_kk = None
          for jj in range(min(num_obj, timespan)):
            y_full_jj_str = y_gt_group['{:02d}'.format(jj)][:]
            y_full_jj = cv2.imdecode(
                y_full_jj_str, cv2.CV_LOAD_IMAGE_GRAYSCALE).astype('float32')
            if y_full_kk is None:
              y_full_kk = np.zeros(
                  [timespan, y_full_jj.shape[0], y_full_jj.shape[1]])
            y_full_kk[jj] = y_full_jj
          y_full.append(y_full_kk)
        else:
          y_full.append(np.zeros([timespan] + list(data_group['orig_size'][:])))
    return y_full

  def get_batch(self, idx, timespan=None, variables=None):
    """Get a mini-batch."""
    if timespan is None:
      timespan = self.get_default_timespan()
    if variables is None:
      variables = set(
          ['x', 'y_gt', 'y_out', 'c_gt', 'd_gt', 'd_out', 's_gt', 'idx_map'])

    with h5py.File(self.h5_fname, 'r') as h5f:
      img_ids = self.img_ids[idx]
      key = self.get_str_id(img_ids[0])
      num_ex = len(idx)
      created_arr = False
      results = {}
      for kk, ii in enumerate(img_ids):
        key = self.get_str_id(ii)
        data_group = h5f[key]
        x_str = data_group['input'][:]
        # x = cv2.imdecode(x_str, cv2.CV_LOAD_IMAGE_COLOR)
        x = cv2.imdecode(x_str, -1)
        height = x.shape[0]
        width = x.shape[1]
        depth = x.shape[2]
        num_ori_classes = 8
        num_sem_classes = self.get_num_semantic_classes()
        area_sort = None
        if num_sem_classes == 1:
          nc = 1
        else:
          nc = num_sem_classes + 1  # Including background
        # self.log.error(('Num semantic classes', num_sem_classes,
        # self))

        if not created_arr:
          if 'source' in data_group:
            results['source'] = []
          if 'x' in variables:
            results['x'] = np.zeros(
                [num_ex, height, width, depth], dtype='float32')
          if 'y_gt' in variables:
            results['y_gt'] = np.zeros(
                [num_ex, timespan, height, width], dtype='float32')
          if 'x_full' in variables:
            if len(idx) > 1:
              raise Exception(('x_full can be only provided in '
                               'batch_size=1 mode.'))
            results['x_full'] = None
          if 'y_gt_full' in variables:
            if len(idx) > 1:
              raise Exception(('y_gt_full can be only provided in '
                               'batch_size=1 mode.'))
            results['y_gt_full'] = None
          if 'y_out_ins' in variables:
            results['y_out_ins'] = np.zeros(
                [num_ex, timespan, height, width], dtype='float32')
          if 'c_gt' in variables:
            results['c_gt'] = np.zeros(
                [num_ex, height, width, nc], dtype='float32')
          if 'c_gt_idx' in variables:
            results['c_gt_idx'] = np.zeros(
                [num_ex, timespan, nc], dtype='float32')
          if 'd_gt' in variables:
            results['d_gt'] = np.zeros(
                [num_ex, height, width, num_ori_classes], dtype='float32')
          if 'y_out' in variables:
            results['y_out'] = np.zeros(
                [num_ex, height, width, nc], dtype='float32')
          if 'd_out' in variables:
            results['d_out'] = np.zeros(
                [num_ex, height, width, num_ori_classes], dtype='float32')
          if 's_out' in variables:
            results['s_out'] = np.zeros([num_ex, timespan], dtype='float32')
          if 's_gt' in variables:
            results['s_gt'] = np.zeros([num_ex, timespan], dtype='float32')
          if 'orig_size' in variables:
            results['orig_size'] = np.zeros([num_ex, 2], dtype='int32')
          created_arr = True

        if 'x' in variables:
          results['x'][kk] = x.astype('float32') / 255

        if 'x_full' in variables:
          if 'input_full_size' in data_group:
            x_full_group = data_group['input_full_size']
            x_full_str = x_full_group[:]
            x_full = cv2.imdecode(x_full_str, -1).astype('float32') / 255
            results['x_full'] = x_full

        if 'y_gt' in variables:
          if 'label_segmentation' in data_group:
            y_gt_group = data_group['label_segmentation']
            num_obj = len(y_gt_group.keys())
            # if num_obj > timespan:
            _y_gt = []
            # If we cannot fit in all the objects,
            # Sort instances such that the largest will be fed.
            for jj in range(num_obj):
              y_gt_str = y_gt_group['{:02d}'.format(jj)][:]
              _y_gt.append(cv2.imdecode(y_gt_str, -1).astype('float32'))
            area = np.array([yy.sum() for yy in _y_gt])
            area_sort = np.argsort(area)[::-1]
            for jj in range(min(num_obj, timespan)):
              results['y_gt'][kk, jj] = _y_gt[area_sort[jj]]

        if 'y_gt_full' in variables:
          if 'label_segmentation_full_size' in data_group:
            y_gt_full_group = data_group['label_segmentation_full_size']
            num_obj = len(y_gt_full_group.keys())
            _y_gt_full = []
            for jj in range(num_obj):
              y_gt_str = y_gt_full_group['{:02d}'.format(jj)][:]
              _y_gt_full.append(cv2.imdecode(y_gt_str, -1).astype('float32'))
            area = np.array([yy.sum() for yy in _y_gt_full])
            area_sort_full = np.argsort(area)[::-1]
            results['y_gt_full'] = np.zeros(
                [timespan, _y_gt_full[0].shape[0], _y_gt_full[0].shape[1]])
            for jj in range(min(num_obj, timespan)):
              results['y_gt_full'][jj] = _y_gt_full[area_sort_full[jj]]
          else:
            if 'orig_size' in data_group:
              results['y_gt_full'] = \
                  np.zeros([timespan] +
                           list(data_group['orig_size'][:]))
            else:
              results['y_gt_full'] = \
                  np.zeros(
                      [timespan] +
                  list(data_group['input_full_size'].shape))

        if 'y_out_ins' in variables:
          if 'instance_pred' in data_group:
            y_out_ins_group = data_group['instance_pred']
            num_obj = len(y_out_ins_group.keys())
            # if num_obj > timespan:
            # _y_out_ins = []
            # If we cannot fit in all the objects,
            # Sort instances such that the largest will be fed.
            for jj in range(num_obj):
              _y_out_jj_str = y_out_ins_group['{:02d}'.format(jj)][:]
              _y_out_jj = cv2.imdecode(_y_out_jj_str,
                                       -1).astype('float32') / 255
              # _y_out_ins.append(_y_out_jj)
              results['y_out_ins'][kk, jj] = _y_out_jj
          else:
            raise Exception('Key not found: {}'.format('instance_pred'))

        if 'c_gt' in variables:
          if 'label_semantic_segmentation' in data_group:
            c_gt_group = data_group['label_semantic_segmentation']
            if num_sem_classes > 1:
              for jj in range(num_sem_classes):
                if num_sem_classes == 1:
                  cid = jj
                else:
                  cid = jj + 1  # Including background
                cstr = '{:02d}'.format(jj)
                if cstr in c_gt_group:
                  c_gt_str = c_gt_group[cstr][:]
                  results['c_gt'][kk, :, :, cid] = cv2.imdecode(
                      c_gt_str, -1).astype('float32')
              # Background class, everything else.
              results['c_gt'][kk, :, :, 0] = 1 - \
                  results['c_gt'][kk].max(axis=2)
            else:
              c_gt_str = c_gt_group['00'][:]
              results['c_gt'][kk, :, :, 0] = cv2.imdecode(c_gt_str,
                                                          -1).astype('float32')
          # else:
          #     raise Exception('Key not found: {}'.format(
          #         'label_semantic_segmentation'))

        if 'c_gt_idx' in variables:
          if 'instance_semantic_classes' in data_group:
            c_gt_idx = data_group['instance_semantic_classes'][:]
            num_obj = len(c_gt_idx)
            if num_obj > 0:
              c_gt_idx = c_gt_idx[area_sort]

            for jj in range(min(num_obj, timespan)):
              results['c_gt_idx'][kk, :jj, c_gt_idx[jj] + 1] = 1.0
            if num_obj < timespan:
              for jj in range(num_obj, timespan):
                results['c_gt_idx'][kk, :jj, 0] = 1.0
          # else:
          #     raise Exception('Key not found: {}'.format(
          #         'instance_semantic_classes'))

        if 'd_gt' in variables:
          if 'orientation' in data_group:
            d_gt_str = data_group['orientation'][:]
            d_gt_ = cv2.imdecode(d_gt_str, -1).astype('float32')
            for oo in range(num_ori_classes):
              results['d_gt'][kk, :, :, oo] = (d_gt_ == oo).astype('float32')
          # else:
          #     raise Exception('Key not found: {}'.format(
          #         'orientation'))

        if 's_gt' in variables:
          if 'label_segmentation' in data_group:
            y_gt_group = data_group['label_segmentation']
            num_obj = len(y_gt_group.keys())
            results['s_gt'][kk, :min(num_obj, timespan)] = 1.0

        if 'd_out' in variables:
          for oo in range(num_ori_classes):
            d_out_str = data_group['orientation_pred/{:02d}'.format(oo)][:]
            d_out_arr = cv2.imdecode(d_out_str, -1)
            d_out_arr = d_out_arr.astype('float32') / 255
            results['d_out'][kk, :, :, oo] = d_out_arr

        if 'y_out' in variables:
          for cc in range(nc):
            if nc == 1:
              # Backward compatibility.
              if 'foreground_pred/{:02d}'.format(cc) not in data_group:
                y_out_str = data_group['foreground_pred'][:]
              else:
                y_out_str = data_group['foreground_pred/{:02d}'.format(cc)][:]
            else:
              y_out_str = data_group['foreground_pred/{:02d}'.format(cc)][:]
            y_out_arr = cv2.imdecode(y_out_str, -1)
            y_out_arr = y_out_arr.astype('float32') / 255
            results['y_out'][kk, :, :, cc] = y_out_arr

        if 's_out' in variables:
          _s = data_group['score_pred'][:]
          results['s_out'][kk] = _s

        # For combined datasets, the source of the data example.
        if 'source' in data_group:
          results['source'].append(data_group['source'][0])

        if 'orig_size' in variables:
          results['orig_size'][kk] = data_group['orig_size'][:]

      if 'idx_map' in variables:
        results['idx_map'] = img_ids

    return results
