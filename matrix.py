import numpy as np
import os
import torch

def fd_1d():
  print('1D finite difference')
  size = 8
  x = np.zeros((size, size))
  x[np.arange(1, size), np.arange(size - 1)] = 0.5
  x[np.arange(size - 1), np.arange(1, size)] = 0.5
  print(x)
  w, v = np.linalg.eig(x)
  print('Eigenvalues:', w)
  print('')

  # Add boundary
  left, right = 3, 5 # random values
  y = np.zeros((size + 1, size + 1))
  y[:-1, :-1] = x
  y[0, -1] = 0.5 * left
  y[-2, -1] = 0.5 * right
  y[-1, -1] = 1
  print(y)
  w, v = np.linalg.eig(y)
  print('Eigenvalues:\n', w)
  print('Last eigenvector:\n', v[:, -1])
  print('Solution:\n', v[:, -1] / v[-1, -1])
  print('Ground truth\n', np.linspace(left, right, num=(size + 2)))

#import utils

def construct_matrix(bc, image_size):
  x = set_boundary(np.zeros((image_size, image_size)), bc)
  y = utils.fd_step(x, bc)

def fd_2d():
  data_dir = os.path.join(os.environ['HOME'], 'slowbro/PDE/heat/16x16')
  # Take first instance
  bc = np.load(os.path.join(data_dir, 'bc.npy'))[0] / 100
  frames = np.load(os.path.join(data_dir, 'frames', '0000.npy')) / 100
  idx = 0
  bc = bc[idx]
  gt = frames[idx, -1]
  x = set_boundary(np.zeros_like(gt), bc)
  construct_matrix()

if __name__ == '__main__':
  fd_1d()
