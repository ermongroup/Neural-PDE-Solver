import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

update_kernel = np.array([[0, 1, 0],
                          [1, 0, 1],
                          [0, 1, 0]]) * 0.25

loss_kernel = np.array([[0, -1, 0],
                        [-1, 4, -1],
                        [0, -1, 0]])

update_kernel = torch.Tensor(update_kernel)
loss_kernel = torch.Tensor(loss_kernel)

def set_boundary(x, boundary_conditions):
  x1, x2, x3, x4 = boundary_conditions
  x[0, :] = x1
  x[-1, :] = x2
  x[:, 0] = x3
  x[:, -1] = x4
  return x

def fd_step(x, boundary_conditions):
  x = set_boundary(x, boundary_conditions)
  # Convolution
  y = F.conv2d(x.view(1, 1, *x.size()), update_kernel.view(1, 1, 3, 3))
  y = y.view(*y.size()[-2:])
  # Add boundaries back
  y = F.pad(y, (1, 1, 1, 1))
  y = set_boundary(y, boundary_conditions)
  return y

def calc_loss(x):
  l = F.conv2d(x.view(1, 1, *x.size()), loss_kernel.view(1, 1, 3, 3))
  loss = torch.sum(l ** 2)
  return loss

def main():
  size = 32
  max_temp = 100
  x = np.random.rand(size, size) * max_temp
  bc = np.random.rand(4) * max_temp
  x = torch.Tensor(x)
  bc = torch.Tensor(bc)

  diff = []
  for i in range(500):
    y = fd_step(x, bc)
    diff = torch.mean(torch.abs(y - x))
    loss = calc_loss(y)
    print(diff.item(), loss.item())
    x = y

  x = x.numpy()
  img = plt.imshow(x)
  plt.show()

if __name__ == '__main__':
  main()
