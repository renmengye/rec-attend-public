#!/usr/bin/env python
from __future__ import print_function
import threading
import datetime
import sys

mutex = threading.Lock()


def get_id():
  mutex.acquire()
  time_obj = datetime.datetime.now()
  model_id = '{:04d}{:02d}{:02d}{:02d}{:02d}{:02d}'.format(
      time_obj.year, time_obj.month, time_obj.day, time_obj.hour,
      time_obj.minute, time_obj.second)
  mutex.release()
  return model_id


if __name__ == '__main__':
  print(get_id())
