import glob
import numpy as np
import os
import torch

import data
import utils
from models.heat_model import HeatModel


def evaluate(opt, model, data_loader, vis=None):
  model.setup(is_train=False)
  for step, data in enumerate(data_loader):
    bc, final, x = data['bc'], data['final'], data['x']
    errors, fd_errors = model.evaluate(x, final, bc, opt.n_evaluation_steps)
    images = model.plot_error_curves(errors, fd_errors)
    if vis is not None:
      vis.add_image({'errors': images}, step)
    if (step + 1) % opt.log_every == 0:
      print('Step {}'.format(step))

def main():
  opt, logger, stats, vis = utils.build(is_train=False, tb_dir='tb_val')
  data_loader = data.get_data_loader(opt)
  # Load model opt
  model_opt = np.load(os.path.join(opt.ckpt_path, 'opt.npy')).item()
  model_opt.is_train = False
  model = HeatModel(model_opt)
  logger.print('Loading data from {}'.format(opt.dset_path))

  for epoch in opt.which_epochs:
    if epoch < 0:
      # Pick last epoch
      checkpoints = glob.glob(os.path.join(opt.ckpt_path, 'net_*.pth'))
      assert len(checkpoints) > 0
      epochs = [int(path[:-4].split('_')[-1]) for path in checkpoints]
      epoch = sorted(epochs)[-1]

    model.load(opt.ckpt_path, epoch)
    logger.print('Checkpoint loaded from {}, epoch {}'.format(opt.ckpt_path, epoch))
    evaluate(opt, model, data_loader, vis)

if __name__ == '__main__':
  main()
