import numpy as np
import os
import torch

def fd_1d():
  print('1D finite difference')
  size = 8
  # loss matrix
  loss = np.eye(size) * -1
  loss[np.arange(1, size), np.arange(size - 1)] = 0.5
  loss[np.arange(size - 1), np.arange(1, size)] = 0.5
  beta = 1.0
  x = np.eye(size) + beta * loss
  print(x)
  w, v = np.linalg.eig(x)
  print('Eigenvalues:\n', sorted(w))
  print('')
  print('SA^-1:\n{}'.format(x.dot(np.linalg.inv(loss))))

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


def fd_2d():
  import utils

  data_dir = os.path.join(os.environ['HOME'], 'slowbro/PDE/heat/16x16')
  # Take first instance
  bc = np.load(os.path.join(data_dir, 'bc.npy'))[0] / 100
  frames = np.load(os.path.join(data_dir, 'frames', '0000.npy')) / 100
  idx = 3 # random
  bc = bc[idx]
  gt = frames[idx, -1]
  A, B = utils.construct_matrix(bc, 16, utils.fd_step)
  A, B = A.numpy(), B.numpy()

  w, v = np.linalg.eig(A)
  print('A')
  print('Eigenvalues:\n', sorted(np.abs(w)))

  w, v = np.linalg.eig(B)
  print('B')
  print('Eigenvalues:\n', sorted(np.abs(w)))
  v = np.real(v)
  y = v[:, -1] / v[-1, -1]
  diff = y[:-1] - gt[1:-1, 1:-1].flatten()
  if np.all(diff <  1e-5):
    print('Solution correct!')


if __name__ == '__main__':
  fd_1d()
  #fd_2d()
