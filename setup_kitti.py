#!/usr/bin/env python
import os
from data_api.kitti import KITTIAssembler


def main():
  opt = {'height': 224, 'width': 224}
  folder = 'data/kitti'
  for split, output in zip(['valid_man', 'test_man', 'train'],
                           ['valid', 'test', 'train']):
    dataset = KITTIAssembler(
        folder,
        opt=opt,
        split=split,
        output_fname=os.path.join(folder, '{}_{}x{}.h5'.format(
            output, opt['height'], opt['width']))).assemble()


if __name__ == '__main__':
  main()