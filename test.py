import glob
import numpy as np
import os
import torch

import data
import utils
from models.heat_model import HeatModel

def evaluate(opt, model, data_loader, logger, error_threshold=0.05, limit=None):
  '''
  Loop through the dataset and calculate evaluation metrics.
  '''
  if model.compare_model is not None:
    logger.print('Comparison: {} ({}), {} ({})'.format(\
                     model.iterator.name(), model.iterator.n_operations,
                     model.compare_model.name(), model.compare_model.n_operations))

  metric = utils.Metrics(scale=model.operations_ratio, error_threshold=error_threshold)
  images = {'error_curves': [], 'results': []}

  for step, data in enumerate(data_loader):
    bc, gt, x = data['bc'], data['final'], data['x']
    f = None if 'f' not in data else data['f']
    if opt.initialization != 'random':
      # Test time: do not change data if 'random'
      x = utils.initialize(x, bc, opt.initialization)
    results, x = model.evaluate(x, gt, bc, f, opt.n_evaluation_steps)
    # Update metric
    metric.update(results)

    if step % opt.log_every == 0:
      img = utils.plot_error_curves(results, num=4)
      images['error_curves'].append(img)
      img = utils.plot_results({'x': x, 'gt': gt})
      images['results'].append(img)
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
  if opt.iterator == 'cg':
    return

  image_size = 15
  w, v = utils.calculate_eigenvalues(model, image_size)
  np.save(os.path.join(opt.ckpt_path, 'eigenvalues.npy'), w)
  np.save(os.path.join(opt.ckpt_path, 'eigenvectors.npy'), v)
  logger.print('Absolute eigenvalues:\n{}\n'.format(sorted(np.abs(w))))
  w = sorted(np.abs(w))

  # Finite difference
  if model.compare_model is not None:
    A, B = utils.construct_matrix(np.zeros((1, 4)), image_size, model.compare_model.iter_step)
    w_fd, v_fd = np.linalg.eig(B)
    w_fd = sorted(np.abs(w_fd))
    print('Finite difference eigenvalues:\n{}\n'.format(w_fd))
    if vis is not None:
      img = utils.plot_curves([{'y': w_fd, 'label': 'Jacobi eigenvalues'},
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

  if opt.geometry == 'square':
    # random initialization
    results, images = evaluate(opt, model, data_loader, logger)
    if vis is not None:
      for i, img in enumerate(images['error_curves']):
        vis.add_image({'errors_{}_init'.format(opt.initialization): img}, i)

  # avg initialization
  opt.initialization = 'avg'
  results, images = evaluate(opt, model, data_loader, logger)
  if vis is not None:
    for i, img in enumerate(images['error_curves']):
      vis.add_image({'errors_avg_init': img}, i)
    for i, img in enumerate(images['results']):
      vis.add_image({'results': img}, i)

def main():
  opt, logger, stats, vis = utils.build(is_train=False, tb_dir='tb_val')
  # Load model opt
  model_opt = np.load(os.path.join(opt.ckpt_path, 'opt.npy')).item()
  model_opt.is_train = False
  # Change geometry to the testing one
  model_opt.geometry = opt.geometry
  model = HeatModel(model_opt)
  logger.print('Loading data from {}'.format(opt.dset_path))

  # For convenience
  opt.initialization = model_opt.initialization
  opt.iterator = model_opt.iterator
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
