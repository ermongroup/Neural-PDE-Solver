import glob
import numpy as np
import os

import utils
from models.heat_model import HeatModel

def EHET_EH_ET(H, T, e):
  '''
  rho(EHET - EH + ET).
  '''
  E = np.diag(e)
  ET = E.dot(T)
  EH = E.dot(H)
  A = EH.dot(ET) - EH + ET
  r = utils.spectral_radius(A)
  return r

def test_specific(H, T, image_size):
  '''
  Specific geometries>
  '''
  pad = 2
  e_square = np.zeros((image_size, image_size))
  e_square[pad:-pad, pad:-pad] = 1

  e2 = np.ones((image_size - 2 * pad, image_size - 2 * pad))
  e2[0, 1] = 0
  e2[1, 0] = 0
  e1 = e_square.copy()
  e1[pad:-pad, pad:-pad] = e2
  print(e1)
  r = EHET_EH_ET(H, T, e1.flatten())
  print(r)
  print('')

  e2 = np.ones((image_size - 2 * pad, image_size - 2 * pad))
  e2[0, 0] = 0
  e2[1, 1] = 0
  e2[0, 2] = 0
  e1 = e_square.copy()
  e1[pad:-pad, pad:-pad] = e2
  print(e1)
  r = EHET_EH_ET(H, T, e1.flatten())
  print(r)
  print('')

  e2 = np.ones((image_size - 2 * pad, image_size - 2 * pad))
  e2[0, 1] = 0
  e2[1, 0] = 0
  e2[1, 2] = 0
  e2[2, 1] = 0
  e1 = e_square.copy()
  e1[pad:-pad, pad:-pad] = e2
  print(e1)
  r = EHET_EH_ET(H, T, e1.flatten())
  print(r)
  print('')

  e2 = np.ones((image_size - 2 * pad, image_size - 2 * pad))
  e2[1, 2] = 0
  e2[2, 1] = 0
  e2[3, 2] = 0
  e2[2, 3] = 0
  e1 = e_square.copy()
  e1[pad:-pad, pad:-pad] = e2
  print(e1)
  r = EHET_EH_ET(H, T, e1.flatten())
  print(r)
  print('')

  e2 = np.ones((image_size - 2 * pad, image_size - 2 * pad))
  e2[1, 2] = 0
  e2[2, 1] = 0
  e2[3, 2] = 0
  e2[2, 3] = 0
  e2[4, 3] = 0
  e2[3, 4] = 0
  e1 = e_square.copy()
  e1[pad:-pad, pad:-pad] = e2
  print(e1)
  r = EHET_EH_ET(H, T, e1.flatten())
  print(r)
  print('')

def spectral(opt, model):
  '''
  Construct update matrix and calculate eigenvalues.
  Compare with finite difference. Plot eigenvalues.
  '''
  image_size = 10
  # Remove activation and bc_mask
  activation = model.get_activation()
  model.change_activation('none')
  T = utils.construct_matrix_wraparound(image_size, utils.fd_step)
  H = utils.construct_matrix_wraparound(image_size, model.H)
  np.save('tmp/H.npy', H)
  np.save('tmp/T.npy', T)

  r = utils.spectral_radius(H.dot(T) + T - H)
  print('rho(HT + T - H):', r)

  # Different E's
  size = image_size * image_size
  e = np.ones(size)

  print('Square geometry')
  pad = 2
  e_square = np.zeros((image_size, image_size))
  e_square[pad:-pad, pad:-pad] = 1
  r = EHET_EH_ET(H, T, e_square.flatten())
  assert r < 1

  test_specific(H, T, image_size)
  return

  for n_zeros in range(1, 11):
    print('\n###############################\n{} zeros\n'.format(n_zeros))
    for i in range(1000):
      e2 = np.ones((image_size - 2 * pad) ** 2)
      indices = np.random.choice((image_size - 2 * pad) ** 2, n_zeros, replace=False)
      e2[indices] = 0
      e1 = e_square.copy()
      e1[pad:-pad, pad:-pad] = e2.reshape(image_size - 2 * pad, image_size - 2 * pad)
      r = EHET_EH_ET(H, T, e1.flatten())
      if r >= 1:
        print('************ Rho > 1 ************')
        print(e1.reshape((image_size, image_size)))
        print('*********************************')

def main():
  opt, logger, stats, vis = utils.build(is_train=False, tb_dir=None, logging=False)
  # Load model opt
  model_opt = np.load(os.path.join(opt.ckpt_path, 'opt.npy')).item()
  model_opt.is_train = False
  # Change geometry to the testing one
  model_opt.geometry = opt.geometry
  model = HeatModel(model_opt)

  # For convenience
  opt.initialization = model_opt.initialization
  opt.iterator = model_opt.iterator

  for epoch in opt.which_epochs:
    if epoch < 0:
      # Pick last epoch
      checkpoints = glob.glob(os.path.join(opt.ckpt_path, 'net_*.pth'))
      assert len(checkpoints) > 0
      epochs = [int(path[:-4].split('_')[-1]) for path in checkpoints]
      epoch = sorted(epochs)[-1]

    model.load(opt.ckpt_path, epoch)
    print('Checkpoint loaded from {}, epoch {}'.format(opt.ckpt_path, epoch))
    spectral(opt, model)

if __name__ == '__main__':
  main()
