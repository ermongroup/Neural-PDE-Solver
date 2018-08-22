import glob
import numpy as np
import os
import torch

import data
import utils
from models.heat_model import HeatModel

def evaluate(opt, model, data_loader, logger, limit=None):
  '''
  Loop through the dataset and calculate evaluation metrics.
  '''
  metric = utils.Metrics(scale=model.n_operations, error_threshold=0.05)
  images = []

  for step, data in enumerate(data_loader):
    bc, final, x = data['bc'], data['final'], data['x']
    if opt.initialization != 'random':
      # Test time: do not change data if 'random'
      x = utils.initialize(x, bc, opt.initialization)
    error_dict = model.evaluate(x, final, bc, opt.n_evaluation_steps, opt.switch_to_fd)
    # Update metric
    metric.update(error_dict)

    if step % opt.log_every == 0:
      img = utils.plot_error_curves(error_dict, num=4)
      images.append(img)
    if (step + 1) % opt.log_every == 0:
      print('Step {}'.format(step + 1))

    if limit is not None and (step + 1) == limit:
      break

  # Get results
  results = metric.get_results()
  for key in results:
    logger.print('{}: {}'.format(key, results[key]))
  metric.reset()

  return results, images

def check_eigenvalues(opt, model, logger, vis):
  '''
  Construct update matrix and calculate eigenvalues.
  Compare with finite difference. Plot eigenvalues.
  '''
  image_size = 15
  w, v = utils.calculate_eigenvalues(model, image_size)
  np.save(os.path.join(opt.ckpt_path, 'eigenvalues.npy'), w)
  np.save(os.path.join(opt.ckpt_path, 'eigenvectors.npy'), v)
  logger.print('Absolute eigenvalues:\n{}\n'.format(sorted(np.abs(w))))
  w = sorted(np.abs(w))

  # Finite difference
  A, B = utils.construct_matrix(np.zeros((1, 4)), image_size, utils.fd_step)
  w_fd, v_fd = np.linalg.eig(B)
  w_fd = sorted(np.abs(w_fd))
  print('Finite difference eigenvalues:\n{}\n'.format(w_fd))
  if vis is not None:
    img = utils.plot([{'y': w_fd, 'label': 'Jacobi eigenvalues'},
                      {'y': w, 'label': 'model eigenvalues'}],
                     config={'title': 'Eigenvalues'})
    vis.add_image({'eigenvalues': img})

def test(opt, model, data_loader, logger, vis=None):
  '''
  Calculate eigenvalues and runtime comparison with finite difference.
  '''
  model.setup(is_train=False)

  check_eigenvalues(opt, model, logger, vis)

  # Print model parameters
  state_dict = model.iterator.state_dict()
  for key in state_dict.keys():
    logger.print('{}\n{}'.format(key, state_dict[key]))

  results, images = evaluate(opt, model, data_loader, logger)
  if vis is not None:
    for i, img in enumerate(images):
      vis.add_image({'errors': img}, i)

def main():
  opt, logger, stats, vis = utils.build(is_train=False, tb_dir='tb_val')
  # Load model opt
  model_opt = np.load(os.path.join(opt.ckpt_path, 'opt.npy')).item()
  model_opt.is_train = False
  model = HeatModel(model_opt)
  logger.print('Loading data from {}'.format(opt.dset_path))

  # In case I forget
  opt.initialization = model_opt.initialization
  data_loader = data.get_data_loader(opt)

  for epoch in opt.which_epochs:
    if epoch < 0:
      # Pick last epoch
      checkpoints = glob.glob(os.path.join(opt.ckpt_path, 'net_*.pth'))
      assert len(checkpoints) > 0
      epochs = [int(path[:-4].split('_')[-1]) for path in checkpoints]
      epoch = sorted(epochs)[-1]

    model.load(opt.ckpt_path, epoch)
    logger.print('Checkpoint loaded from {}, epoch {}'.format(opt.ckpt_path, epoch))
    test(opt, model, data_loader, logger, vis)

if __name__ == '__main__':
  main()
