import numpy as np
import os
import random
import torch

import args
from .logger import *
from .statistics import Statistics


def build(is_train, tb_dir=None, logging=True):
  opt, log = args.TrainArgs().parse() if is_train else args.TestArgs().parse()

  #os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus
  os.makedirs(opt.ckpt_path, exist_ok=True)

  # Set seed
  torch.manual_seed(666)
  torch.cuda.manual_seed_all(666)
  np.random.seed(666)
  random.seed(666)

  # Add geometry to path name
  if tb_dir is not None:
    tb_path = os.path.join(opt.ckpt_path, '{}_{}'.format(tb_dir, opt.geometry))
    if opt.poisson:
      tb_path = tb_path + '_poisson'
    vis = Visualizer(tb_path)
  else:
    vis = None

  if logging:
    logger_name = '{}_{}'.format(opt.split, opt.geometry)
    if opt.poisson:
      logger_name = logger_name + '_poisson'
    logger = Logger(opt.ckpt_path, logger_name)
    logger.print(log)
  else:
    logger = None

  stats = Statistics(opt.ckpt_path)

  return opt, logger, stats, vis
