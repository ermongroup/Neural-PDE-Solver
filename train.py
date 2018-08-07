import copy
import numpy as np
import os

from data import get_data_loader, get_random_data_loader
import utils
from models.heat_model import HeatModel

def main():
  opt, logger, stats, vis = utils.build(is_train=True, tb_dir='tb_train')
  data_loader = get_data_loader(opt)
# Validation
  val_opt = copy.deepcopy(opt)
  val_opt.is_train = False
  val_opt.batch_size = 4
  val_loader = get_random_data_loader(val_opt)

  model = HeatModel(opt)

  for epoch in range(opt.start_epoch, opt.n_epochs):
    model.setup(is_train=True)
    for step, data in enumerate(data_loader):
      bc, final, x = data['bc'], data['final'], data['x']
      loss_dict = model.train(x, final, bc)
      if (step + 1) % opt.log_every == 0:
        print('Epoch {}, step {}'.format(epoch, step))
        vis.add_scalar(loss_dict, epoch * len(data_loader) + step)

    logger.print(['[Summary] Epoch {}/{}:'.format(epoch, opt.n_epochs - 1)])
    if opt.evaluate_every > 0 and (epoch + 1) % opt.evaluate_every == 0:
      model.setup(is_train=False)
      # Randomly sample test data
      data = next(iter(val_loader))
      bc, final, x = data['bc'], data['final'], data['x']
      errors, fd_errors = model.evaluate(x, final, bc, opt.n_evaluation_steps)
      # Plot error curve
      images = model.plot_error_curves(errors, fd_errors)
      vis.add_image({'errors': images}, epoch)

    if (epoch + 1) % opt.save_every == 0 or epoch == opt.n_epochs - 1:
      model.save(opt.ckpt_path, epoch + 1)

if __name__ == '__main__':
  main()
