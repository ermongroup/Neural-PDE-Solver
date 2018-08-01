from collections import OrderedDict
import numpy as np
import os
import time


class AverageMeter(object):
  def __init__(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val*n
    self.count += n
    self.avg = self.sum/self.count


class Statistics(object):
  def __init__(self, ckpt_path=None, name='history'):
    self.meters = OrderedDict()
    self.history = OrderedDict()
    self.ckpt_path = ckpt_path
    self.name = name
    self.tm = time.time()

  def update(self, n, ordered_dict):
    log = []
    for key in ordered_dict:
      if key not in self.meters:
        self.meters.update({key: AverageMeter()})
      self.meters[key].update(ordered_dict[key], n)
      log.append('{key}={var.val:.4f}, avg {key}={var.avg:.4f}'.format(key=key, var=self.meters[key]))

    return log

  def summarize(self, reset=True):
    log = []
    for key in self.meters:
      log.append('{key}={var:.4f}'.format(key=key, var=self.meters[key].avg))
    log.append('Elapsed time: {}s'.format(time.time()-self.tm))

    if reset:
      self.reset()

    return log

  def reset(self):
    for key in self.meters:
      if key in self.history:
        self.history[key].append(self.meters[key].avg)
      else:
        self.history.update({key: [self.meters[key].avg]})

    self.meters = OrderedDict()
    self.tm = time.time()

  def load(self):
    self.history = np.load(os.path.join(self.ckpt_path, '{}.npy'.format(self.name))).item()

  def save(self):
    np.save(os.path.join(self.ckpt_path, '{}.npy'.format(self.name)), self.history)
