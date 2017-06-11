#!/usr/bin/env python
import os
from data_api.cvppp import CVPPPAssembler


def main():
  train_folder = 'data/cvppp'
  test_folder = 'data/cvppp_test'
  opt = {'height': 224, 'width': 224}
  for subset in ['A1']:
    for split in ['train', 'valid', 'all']:
      CVPPPAssembler(
          os.path.join(train_folder, subset), opt, split=split).assemble()
  for subset in ['A1']:
    for split in ['test']:
      CVPPPAssembler(
          os.path.join(test_folder, subset), opt, split=split).assemble()


if __name__ == '__main__':
  main()