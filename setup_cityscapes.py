#!/usr/bin/env python
import os
from data_api.cityscapes import CityscapesAssembler


def main():
  folder = 'data/cityscapes'
  opt = {'height': 256, 'width': 512}
  for split in ['train', 'valid', 'test']:
    a = CityscapesAssembler(
        folder=folder,
        opt=opt,
        split=split,
        coarse_label=False,
        output_fname=os.path.join(folder, '{}_{}x{}.h5'.format(
            split, opt['height'], opt['width'])))
    a.assemble()


if __name__ == '__main__':
  main()