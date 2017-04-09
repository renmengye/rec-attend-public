from log_manager import LogManager

import datetime
from utils import logger
import os

registry = {}


def get(name):
  return registry[name]


def register(filename, labels, name, buffer_size=1, restore_step=0):
  if name not in registry:
    registry[name] = TimeSeriesLogger(
        filename,
        labels,
        name,
        buffer_size=buffer_size,
        restore_step=restore_step)
  return registry[name]


# Maybe rewrite this time-series logger to have a mapping between the file
# and the memory data structure.


class TimeSeriesLogger(object):
  """Log time series data to CSV file."""

  def __init__(self, filename, labels, name, buffer_size=1, restore_step=0):
    """
        Args:
            label: list of string
            name: string
            buffer_size: int
        """
    self.filename = filename
    self.folder = os.path.dirname(filename)
    self.written_catalog = False
    self.log = logger.get()

    if type(labels) != list:
      labels = [labels]
    if name is None:
      self.name = labels[0]
    else:
      self.name = name

    self.labels = labels
    self.buffer_size = buffer_size
    self.buffer = []
    self.label_table = {}
    for ll, label in enumerate(labels):
      self.label_table[label] = ll
    self.log.info('Time series data "{}" log to "{}"'.format(labels, filename))
    self._has_init = False

  def init(self, restore_step=0):
    """Initialize logger position."""
    if not self._has_init:
      self.log.info('"{}" initialized to position {}'.format(self.filename,
                                                             restore_step))
      if restore_step > 0:
        if not os.path.exists(self.filename):
          self.log.error('Cannot restore from file: {}'.format(self.filename))
          self.buffer.append('step,time,{}\n'.format(','.join(self.labels)))
        else:
          with open(self.filename, 'r') as f:
            lines = f.readlines()
            for line in lines:
              parts = line.split(',')
              if parts[0].isdigit():
                step = int(parts[0])
                if step > restore_step:
                  break
              self.buffer.append(line)
          t = datetime.datetime.now()
          os.rename(self.filename, self.filename +
                    '.{:04d}{:02d}{:02d}-{:02d}{:02d}{:02d}.bak'.format(
                        t.year, t.month, t.day, t.hour, t.minute, t.second))
          self.written_catalog = True
      else:
        self.buffer = []
        if not os.path.exists(self.filename):
          self.buffer.append('step,time,{}\n'.format(','.join(self.labels)))
      self.flush()
      self._has_init = True

  def add(self, step, values):
    """Add an entry.

        Args:
            step: int
            values: list of numbers
        """
    if not self._has_init:
      self.init()
    t = datetime.datetime.utcnow()
    if type(values) != list:
      values = [values]
    self.buffer.append('{:d},{},{}\n'.format(
        step, t.isoformat(), ','.join([str(v) for v in values])))
    if len(self.buffer) >= self.buffer_size:
      self.flush()

  def add_one(self, step, value, label=None):
    """Add one cell.

        Args:
            step: int
            value: value
            order: location
        """
    values = [''] * len(self.labels)
    if label is None:
      order = 0
    else:
      order = self.label_table[label]
    values[order] = value
    return self.add(step, values)

  def flush(self):
    """Write the buffer to file."""
    if not self.written_catalog:
      LogManager(self.folder).register(self.filename, 'csv', self.name)
      self.written_catalog = True

    if not os.path.exists(self.filename):
      mode = 'w'
    else:
      mode = 'a'
    with open(self.filename, mode) as f:
      f.write(''.join(self.buffer))
    self.buffer = []

  def close(self):
    """Flush the rest."""
    self.flush()
