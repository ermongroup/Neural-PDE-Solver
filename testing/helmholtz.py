import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import utils

image_size = 65
k = 10 / (image_size ** 2)

update_kernel = np.array([[0, 1, 0],
                          [1, 0 + k, 1],
                          [0, 1, 0]]) * 0.25

loss_kernel = np.array([[0, 1, 0],
                        [1, -4 + k, 1],
                        [0, 1, 0]]) * 0.25
update_kernel = torch.Tensor(update_kernel)
loss_kernel = torch.Tensor(loss_kernel)
if torch.cuda.is_available():
  update_kernel = update_kernel.cuda()
  loss_kernel = loss_kernel.cuda()

def step(x, bc, f):
  y = F.conv2d(x.unsqueeze(1), update_kernel.view(1, 1, 3, 3), padding=1).view_as(x)
  x = utils.set_boundary(y, bc)
  return x

def test():
  bc = np.array([0, 0.2, 0.6, 0.85])
  x = np.random.rand(1, image_size, image_size)
  x = torch.Tensor(x)
  bc = torch.Tensor(bc).view(1, 4)
  x = utils.set_boundary(x, bc)
  for i in range(2000):
    print(i)
    x = step(x, bc, None)

  x = x.numpy().squeeze(0)
  plt.imshow(x)
  plt.colorbar()
  plt.show()

def eigenvalues():
  test_bc = np.zeros((1, 4))
  A, B = utils.construct_matrix(test_bc, image_size, step)
  w, v = np.linalg.eig(A)
  w = sorted(np.abs(w))
  print(w)

if __name__ == '__main__':
  eigenvalues()
  #test()
