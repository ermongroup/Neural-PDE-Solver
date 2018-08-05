import logging
import os
import sys
from tensorboardX import SummaryWriter

from .misc import *

class Logger:
  def __init__(self, ckpt_path, name='debug'):
    self.logger = logging.getLogger()
    self.logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(message)s', datefmt=blue('[%Y-%m-%d,%H:%M:%S]'))

    fh = logging.FileHandler(os.path.join(ckpt_path, '{}.log'.format(name)), 'w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    self.logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    self.logger.addHandler(ch)

  def print(self, log):
    if isinstance(log, list):
      self.logger.info('\n - '.join(log))
    else:
      self.logger.info(log)


class Visualizer:
  def __init__(self, tb_path):
    self.tb_path = tb_path

    if os.path.exists(tb_path):
      if prompt_yes_no('{} already exists. Proceed?'.format(tb_path)):
        os.system('rm -r {}'.format(tb_path))
      else:
        exit(0)

    self.writer = SummaryWriter(tb_path)

  def add_scalar(self, scalar_dict, global_step=None):
    for tag, scalar in scalar_dict.items():
      if isinstance(scalar, dict):
        self.writer.add_scalars(tag, scalar, global_step)
      elif isinstance(scalar, list) or isinstance(scalar, np.ndarray):
        continue
      else:
        self.writer.add_scalar(tag, scalar, global_step)

  def add_images(self, image_dict, global_step=None):
    for tag, image in image_dict.items():
      self.writer.add_image(tag, image, global_step)

  def add_text(self, tag, text, global_step=None):
    self.writer.add_text(tag, text, global_step)
