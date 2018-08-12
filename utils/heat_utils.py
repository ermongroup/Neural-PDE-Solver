import numpy as np
import torch
import torch.nn.functional as F
import torchvision

from .misc import plot

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
  bc: batch_size x 4
  '''
  x[:, 0, :] = bc[:, 0:1]
  x[:, -1, :] = bc[:, 1:2]
  x[:, :, 0] = bc[:, 2:3]
  x[:, :, -1] = bc[:, 3:4]
  return x

def pad_boundary(x, bc):
  '''
  Pad and set boundary.
  x: batch_size x H x W
  '''
  y = F.pad(x.unsqueeze(1), (1, 1, 1, 1)).squeeze(1)
  y = set_boundary(y, bc)
  return y

def fd_step(x, bc):
  '''
  One update of finite difference method.
  x: torch tensor of size (batch_size x H x W)
  '''
  x = set_boundary(x, bc)
  # Convolution
  y = F.conv2d(x.unsqueeze(1), update_kernel.view(1, 1, 3, 3))
  # Add boundaries back
  y = F.pad(y, (1, 1, 1, 1)).view_as(x)
  y = set_boundary(y, bc)
  return y

def fd_error(x, aggregate='max'):
  '''
  Use loss kernel to calculate absolute error.
  '''
  l = F.conv2d(x.unsqueeze(1), loss_kernel.view(1, 1, 3, 3))
  l = l.view(l.size(0), -1)
  if aggregate == 'max':
    error = torch.abs(l).max(dim=1)[0]
  elif aggregate == 'mean':
    error = torch.abs(l).mean(dim=1)
  else:
    raise NotImplementedError
  return error

def l2_error(x, gt):
  '''
  Calculate L2 error.
  x, gt: (H x W) or (batch_size x H x W)
  return a scalar loss.
  '''
  diff = (x - gt) ** 2
  error = diff.mean(dim=-1).mean(dim=-1)
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
    error = fd_error(x)
    largest_error = error.max().item() # largest error in the batch
    if (i + 1) % 100 == 0:
      print('Iter {}: largest error {}'.format(i+1, largest_error))
    if largest_error < error_threshold:
      break
  return x

def plot_error_curves(results):
  '''
  Plot model and fd error curves.
  results: dictionary of torch Tensors with size (batch_size x n_steps),
           returned by model.evaluate().
  Return images: torch Tensor, size (3 x H x (W * batch_size))
  '''
  W, H = 640, 480
  images = []
  batch_size = results['fd errors'].size(0)
  for i in range(batch_size):
    curves = []
    for label in results.keys():
      curves.append({'y': results[label][i], 'label': label})
    img = plot(curves, config={'title': 'iterations', 'image_size': (W, H)})
    img = img.transpose((2, 0, 1)) / 255 # 3 x H x W
    images.append(torch.Tensor(img))
  images = torchvision.utils.make_grid(images, nrow=batch_size)
  return images

def construct_matrix(bc, image_size, iter_func):
  '''
  Construct the update matrix given the iterator function.
  x' = Ax + b
  y' = By, where y = [- x -, 1]
  iter_func: function, iter_func(x, b).
             x must be (1 x image_size x image_size).
  Return matrix A and B (torch Tensors).
  '''
  bc = torch.Tensor(bc).view(1, 4)

  # Find bias
  x = torch.zeros(1, image_size, image_size)
  if torch.cuda.is_available():
    x = x.cuda()
    bc = bc.cuda()
  x = set_boundary(x, bc)
  y = iter_func(x, bc).detach()
  bias = y[0, 1:-1, 1:-1].cpu().view(-1)
  # columns
  columns = []
  for i in range(image_size - 2):
    for j in range(image_size - 2):
      y = x.clone()
      y[0, i + 1, j + 1] = 1
      y = iter_func(y, bc).detach()
      c = y[0, 1:-1, 1:-1].cpu().view(-1) - bias
      columns.append(c)
  A = torch.stack(columns, dim=1)
  length = (image_size - 2) ** 2
  assert A.size(0) == length

  # Add bias and last row
  bias = torch.cat([bias, torch.Tensor([1])]).view(-1, 1)
  B = torch.cat([A, torch.zeros(1, length)], dim=0)
  B = torch.cat([B, bias], dim=1)
  assert B.size(0) == length + 1
  return A, B
