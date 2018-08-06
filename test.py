import glob
import numpy as np
import os
import torch

import data
import utils
from models.heat_model import HeatModel


def evaluate(opt, model, data_loader):
  model.setup(is_train=False)
  for step, data in enumerate(data_loader):
    bc, final, x = data['bc'], data['final'], data['x']
    errors, fd_errors = model.evaluate(x, final, bc)
    # TODO: calculate evaluation metric
    e = errors[0].numpy()
    print(np.argmin(e))

def main():
  opt, logger, stats, vis = utils.build(is_train=False, tb_dir=None)
  data_loader = data.get_data_loader(opt)
  model = HeatModel(opt)

  for epoch in opt.which_epochs:
    if epoch < 0:
      # Pick last epoch
      checkpoints = glob.glob(os.path.join(opt.ckpt_path, 'net_*.pth'))
      epochs = [int(path[:-4].split('_')[-1]) for path in checkpoints]
      epoch = sorted(epochs)[-1]

    model.load(opt.ckpt_path, epoch)
    logger.print('Checkpoint loaded from {}, epoch {}'.format(opt.ckpt_path, epoch))
    evaluate(opt, model, data_loader)

if __name__ == '__main__':
  main()
