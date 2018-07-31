import numpy as np
import torch
import torch.nn.functional as F

# Kernels
update_kernel = np.array([[0, 1, 0],
                          [1, 0, 1],
                          [0, 1, 0]]) * 0.25

loss_kernel = np.array([[0, -1, 0],
                        [-1, 4, -1],
                        [0, -1, 0]])

update_kernel = torch.Tensor(update_kernel)
loss_kernel = torch.Tensor(loss_kernel)
if torch.cuda.is_available():
  update_kernel = update_kernel.cuda()
  loss_kernel = loss_kernel.cuda()

def set_boundary(x, bc):
  '''
  x: batch_size x H x W
  bc: batch_size x 4 x 4
  '''
  x[:, 0, :] = bc[:, 0:1]
  x[:, -1, :] = bc[:, 1:2]
  x[:, :, 0] = bc[:, 2:3]
  x[:, :, -1] = bc[:, 3:4]
  return x

def fd_step(x, bc):
  '''
  One update of finite difference method.
  '''
  x = set_boundary(x, bc)
  # Convolution
  y = F.conv2d(x.unsqueeze(1), update_kernel.view(1, 1, 3, 3))
  # Add boundaries back
  y = F.pad(y, (1, 1, 1, 1)).view_as(x)
  y = set_boundary(y, bc)
  return y

def calc_error(x):
  '''
  Use loss kernel to calculate error.
  '''
  l = F.conv2d(x.unsqueeze(1), loss_kernel.view(1, 1, 3, 3))
  error = (l ** 2).sum(dim=2).sum(dim=2)
  return error

def fd_iter(x, bc, error_threshold, max_iters=100000):
  '''
  Run finite different iterations.
  x: torch tensor of size (batch_size x H x W)
  error_threshold is the sum of pixels.
  '''
  for i in range(max_iters):
    x = fd_step(x, bc)
    # Calculate error
    error = calc_error(x)
    largest_error = error.max().item() # largest error in the batch
    if (i + 1) % 100 == 0:
      print('Iter {}: largest error {}'.format(i+1, largest_error))
    if largest_error < error_threshold:
      break
  return x
