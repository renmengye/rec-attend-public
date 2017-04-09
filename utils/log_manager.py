import os
from utils import logger

log = logger.get()


class LogManager(object):

  def __init__(self, folder):
    self.folder = folder
    if not os.path.exists(self.folder):
      os.makedirs(self.folder)

    self.catalog = os.path.join(folder, 'catalog')
    self.filelist = []

    if not os.path.exists(self.catalog):
      with open(self.catalog, 'w') as f:
        f.write('filename,type,name\n')
    else:
      with open(self.catalog, 'r') as f:
        lines = f.readlines()
        lines = lines[1:]
        for line in lines:
          parts = line.strip('\n').split(',')
          self.filelist.append((parts[0], parts[1], parts[2]))

  def register(self, filename, typ, name):
    basename = os.path.basename(filename)
    for pair in self.filelist:
      if pair[0] == basename:
        log.info('File {} is already registered.'.format(basename))
        return

    with open(self.catalog, 'a') as f:
      f.write('{},{},{}\n'.format(basename, typ, name))
    self.filelist.append((basename, typ, name))
