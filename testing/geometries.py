import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import utils

def plot(x):
  plt.imshow(x)
  plt.colorbar()
  plt.show()

def get_geometry():
  image_size = 255
  c = (image_size + 1) // 2

  r1 = (image_size - 7) // 2
  r2 = int(r1 / 4)

  x = np.zeros((image_size, image_size)).astype(np.uint8)

  # First circle
  x = cv2.circle(x, (c, c), r1, 255, -1) / 255
  x = 1 - x

  y = np.zeros((image_size, image_size)).astype(np.uint8)
  center = np.random.randint(-r2 * 1, r2 * 1 + 1, size=2) + c
  center = tuple(center.astype(int))
  y = cv2.circle(y, center, r2, 255, -1) / 255

  mask = x + y
  assert np.all(np.logical_or(mask == 0, mask == 1))

  # values
  outer = cv2.circle(np.zeros_like(x), (c, c), r1 + 2, 255, -1) / 255
  outer_shell = outer - (1 - x)

  half = (image_size + 1) // 2
  v1 = np.random.uniform(0.1, 0.3)
  v2 = np.random.uniform(0.4, 0.6)
  outer_shell[:half, :] = outer_shell[:half, :] * v1
  outer_shell[half:, :] = outer_shell[half:, :] * v2

  values = outer_shell + y
  assert np.all(values * (1 - mask) == 0)
  values = values + (1 - mask) * ((v1 + v2 + 1) / 3)
  assert not np.any(values > 1)
  assert not np.any(mask > 1)

  plot(values)
  plot(mask)

  return values, mask

def test_geometry(geometry, image_size):
  print('########### Test {} ##########\n'.format(geometry))
#  x, mask = get_geometry()
  x, bc_values, bc_mask = utils.get_geometry(geometry, image_size, 1, 1)
  x = x.squeeze(0)
  bc_values = bc_values.squeeze(0)
  bc_mask = bc_mask.squeeze(0)
  print(bc_mask)

  plot(bc_values)
  plot(bc_mask)
  plot(x)

  image_size = x.shape[0]
  x = torch.Tensor(x).view(1, 1, image_size, image_size)
  bc_values = torch.Tensor(bc_values).view_as(x)
  bc_mask = torch.Tensor(bc_mask).view_as(x)

  print(x.size())

  for i in range(1000):
    x = F.conv2d(x, utils.update_kernel.view(1, 1, 3, 3), padding=1)
    x = x * (1 - bc_mask) + bc_values
    r = F.conv2d(x, utils.loss_kernel.view(1, 1, 3, 3), padding=1)
    r = r * (1 - bc_mask)
    error = torch.sum(r ** 2).squeeze()
    print(error.cpu().numpy())

  x = x.squeeze().cpu().numpy()
  plot(x)

def test_subsampling(geometry, image_size):
  print('########### Test subsampling ##########\n')
  x, bc_values, bc_mask = utils.get_geometry(geometry, image_size, 1, 1)

  bc_values = torch.Tensor(bc_values)
  bc_mask = torch.Tensor(bc_mask)
  plot(bc_values.squeeze(0).numpy())
  plot(bc_mask.squeeze(0).numpy())

  n_layers = 3
  for i in range(n_layers):
    bc_values = utils.subsample(bc_values)
    bc_mask = utils.subsample(bc_mask)

    mask = bc_mask.squeeze(0).numpy()
    assert np.all(np.logical_or(np.isclose(mask, 0), np.isclose(mask, 1)))

    bc_values = bc_values * bc_mask
    values = bc_values.squeeze(0).numpy()
    assert np.all(values < 1.00001)

    plot(values)
    plot(mask)

if __name__ == '__main__':
  np.random.seed(666)
  geometry = 'Lshape'
  test_geometry(geometry, 257)
  test_subsampling(geometry, 65)
