import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import utils

def main():
  size = 128
  max_temp = 100

  batch_size = 4
  x = np.random.rand(batch_size, size, size) * max_temp
#  bc = np.random.rand(batch_size, 4) * max_temp
  bc = np.ones((batch_size, 4)) * max_temp
  x = torch.Tensor(x)
  bc = torch.Tensor(bc)

  error_threshold = 0.001 * size * size
  y = utils.fd_iter(x, bc, error_threshold)

  for i in range(batch_size):
    img = y[i].numpy()
    img = plt.imshow(img)
    plt.show()

if __name__ == '__main__':
  main()
