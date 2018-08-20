import copy
import numpy as np
import os

from data import get_data_loader
import utils
from models.heat_model import HeatModel

def main():
  opt, logger, stats, vis = utils.build(is_train=True, tb_dir='tb_train')
  np.save(os.path.join(opt.ckpt_path, 'opt.npy'), opt)
  data_loader = get_data_loader(opt)
  # Validation
  val_opt = copy.deepcopy(opt)
  val_opt.is_train = False
  val_opt.data_limit = 40
  val_loader = get_data_loader(val_opt)

  model = HeatModel(opt)
  metric = utils.Metrics(scale=model.n_operations, error_threshold=0.05)

  for epoch in range(opt.start_epoch, opt.n_epochs):
    model.setup(is_train=True)
    for step, data in enumerate(data_loader):
      bc, final, x = data['bc'], data['final'], data['x']
      loss_dict = model.train(x, final, bc)
      if (step + 1) % opt.log_every == 0:
        print('Epoch {}, step {}'.format(epoch, step))
        vis.add_scalar(loss_dict, epoch * len(data_loader) + step)

    logger.print(['[Summary] Epoch {}/{}:'.format(epoch, opt.n_epochs - 1)])

    # Evaluate
    if opt.evaluate_every > 0 and (epoch + 1) % opt.evaluate_every == 0:
      model.setup(is_train=False)
      # Find eigenvalues
      w, _ = utils.calculate_eigenvalues(model)
      w = sorted(np.abs(w))
      eigenvalues = {'first': w[-2], 'second': w[-3], 'third': w[-4]}
      vis.add_scalar({'eigenvalues': eigenvalues}, epoch)
      logger.print('Eigenvalues: {:.2f}, {:.3f}, {:.3f}, {:.3f}'\
                    .format(w[-1], w[-2], w[-3], w[-4]))

      for step, data in enumerate(val_loader):
        bc, final, x = data['bc'], data['final'], data['x']
        error_dict = model.evaluate(x, final, bc, opt.n_evaluation_steps)
        if step == 0:
          # Plot the first 4 error curves
          images = utils.plot_error_curves(error_dict, num=4)
          vis.add_image({'errors': images}, epoch)
        metric.update(error_dict)
      results = metric.get_results()
      vis.add_scalar({'steps': {'Jacobi': results['Jacobi'], 'model': results['model']},
                      'ratio': results['ratio']}, epoch)
      for key in results:
        logger.print('{}: {}'.format(key, results[key]))
      metric.reset()

    if (epoch + 1) % opt.save_every == 0 or epoch == opt.n_epochs - 1:
      model.save(opt.ckpt_path, epoch + 1)

if __name__ == '__main__':
  main()
