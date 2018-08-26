import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

import utils

def gaussian(image_size=63):
  width = image_size // 2
  x_mean = np.random.randint(-width // 2, width // 2 + 1)
  y_mean = np.random.randint(-width // 2, width // 2 + 1)
  print(x_mean, y_mean)
  x = np.linspace(-width, width, num=image_size, endpoint=True) + x_mean
  y = np.linspace(-width, width, num=image_size, endpoint=True) + y_mean
  # Randomly determine size of kernel
  s = np.random.uniform(2, 3)
  sd = width / s
  x = np.exp(-(x ** 2) / 2 / (sd ** 2))
  y = np.exp(-(y ** 2) / 2 / (sd ** 2))
  kernel = x[:, np.newaxis] * y[np.newaxis, :]
  print(kernel.shape)

#  kernel = (kernel * 255).astype(np.uint8)
#  img = Image.fromarray(kernel)
#  img.show()
  return kernel

def test_heat():
  image_size = 63
  scale = np.random.uniform(300, 400) / (image_size ** 2)
  f = - gaussian(image_size) * scale
  f = torch.Tensor(f).unsqueeze(0)
  f = utils.pad_boundary(f, torch.zeros(1, 4))

  bc = torch.Tensor(np.random.rand(1, 4) * 80)
  x = torch.zeros(1, image_size + 2, image_size + 2)
  x = utils.set_boundary(x, bc)
  x = utils.initialize(x, bc, 'avg')

  y = x.clone()
  for i in range(2000):
    y = utils.fd_step(y, bc)

  z = x.clone()
  for i in range(4000):
    z = utils.fd_step(z, bc) - f

  # Au = 0
  A = utils.loss_kernel.view(1, 1, 3, 3)
  r = F.conv2d(y.unsqueeze(1), A).squeeze(1)
  error = torch.abs(r).max().item()
  print(error)

  # Au = f
  A = utils.loss_kernel.view(1, 1, 3, 3)
  r = F.conv2d(z.unsqueeze(1), A).squeeze(1) - f[:, 1:-1, 1:-1]
  error = torch.abs(r).max().item()
  print(error)

  y = (y / 100).numpy().squeeze(0)
  z = (z / 100).numpy().squeeze(0)

  plt.imshow(y)
  plt.colorbar()
  plt.show()

  plt.imshow(z)
  plt.colorbar()
  plt.show()

if __name__ == '__main__':
  np.random.seed(666)
  for i in range(5):
    test_heat()
