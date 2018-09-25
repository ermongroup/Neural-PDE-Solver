import copy
import numpy as np
import os

from data import get_data_loader
import utils
from models.heat_model import HeatModel
from test import evaluate

def main():
  opt, logger, stats, vis = utils.build(is_train=True, tb_dir='tb_train')
  np.save(os.path.join(opt.ckpt_path, 'opt.npy'), opt)
  data_loader = get_data_loader(opt)
  print('####### Data loaded #########')
  # Validation
  val_opt = copy.deepcopy(opt)
  val_opt.is_train = False
  val_opt.data_limit = 20
  val_loader = get_data_loader(val_opt)

  model = HeatModel(opt)

  for epoch in range(opt.start_epoch, opt.n_epochs):
    model.setup(is_train=True)
    for step, data in enumerate(data_loader):
      bc, final, x = data['bc'], data['final'], data['x']
      f = None if 'f' not in data else data['f']
      x = utils.initialize(x, bc, opt.initialization)
      loss_dict = model.train(x, final, bc, f)
      if (step + 1) % opt.log_every == 0:
        print('Epoch {}, step {}'.format(epoch, step))
        vis.add_scalar(loss_dict, epoch * len(data_loader) + step)

    logger.print(['[Summary] Epoch {}/{}:'.format(epoch, opt.n_epochs - 1)])

    # Evaluate
    if opt.evaluate_every > 0 and (epoch + 1) % opt.evaluate_every == 0:
      model.setup(is_train=False)
      # Find eigenvalues
      if opt.iterator != 'cg' and opt.iterator != 'unet':
        w, _ = utils.calculate_eigenvalues(model, image_size=67)
        w = sorted(np.abs(w))
        eigenvalues = {'first': w[-2], 'second': w[-3], 'third': w[-4]}
        vis.add_scalar({'eigenvalues': eigenvalues}, epoch)
        logger.print('Eigenvalues: {:.2f}, {:.3f}, {:.3f}, {:.3f}'\
                      .format(w[-1], w[-2], w[-3], w[-4]))

      # Evaluate entire val set
      results, images = evaluate(opt, model, val_loader, logger)
      vis.add_image({'errors': images['error_curves'][0]}, epoch + 1)
      vis.add_scalar({'steps': {'Jacobi': results['Jacobi'], 'model': results['model']},
                      'ratio': results['ratio']}, epoch + 1)

    if (epoch + 1) % opt.save_every == 0 or epoch == opt.n_epochs - 1:
      model.save(opt.ckpt_path, epoch + 1)

if __name__ == '__main__':
  main()
