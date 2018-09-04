import numpy as np
import torch
import torch.nn.functional as F
import torchvision

from .misc import plot_curves, plot_data

# Kernels
update_kernel = np.array([[0, 1, 0],
                          [1, 0, 1],
                          [0, 1, 0]]) * 0.25

loss_kernel = np.array([[0, 1, 0],
                        [1, -4, 1],
                        [0, 1, 0]]) * 0.25

restriction_kernel = np.array([[0, 1, 0],
                               [1, 4, 1],
                               [0, 1, 0]]) / 8.0

update_kernel = torch.Tensor(update_kernel)
loss_kernel = torch.Tensor(loss_kernel)
restriction_kernel = torch.Tensor(restriction_kernel)
if torch.cuda.is_available():
  update_kernel = update_kernel.cuda()
  loss_kernel = loss_kernel.cuda()
  restriction_kernel = restriction_kernel.cuda()

def is_bc_mask(x, bc):
  '''
  If bc is a mask, then it should be (batch_size x 2 x image_size x image_size).
  Otherwise it should be (batch_size x 4).
  '''
  _, image_size, _ = x.shape
  batch_size = bc.shape[0]
  if bc.shape == (batch_size, 2, image_size, image_size):
    return True
  elif bc.shape == (batch_size, 4):
    return False
  else:
    print(x.shape, bc.shape)
    raise Exception

def set_boundary(x, bc):
  '''
  x: batch_size x H x W
  bc: (batch_size x 4) or (batch_size x 2 x image_size x image_size)
  '''
  if is_bc_mask(x, bc):
    # bc and bc_mask
    bc_values = bc[:, 0]
    bc_mask = bc[:, 1]
    x = x * (1 - bc_mask) + bc_values
  else:
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

def initialize(x, bc, initialization):
  '''
  Initialize data x.
  x: batch_size x H x W
  bc: (batch_size x 4) or (batch_size x 2 x H x W)
  '''
  batch_size, image_size, _ = x.size()
  if is_bc_mask(x, bc):
    if initialization == 'random':
      x = torch.rand_like(x)
      x = set_boundary(x, bc)
  else:
    # Initialize inner part
    if initialization == 'zero':
      x[:, 1:-1, 1:-1] = 0
    elif initialization == 'random':
      x[:, 1:-1, 1:-1] = torch.rand(batch_size, image_size - 2, image_size - 2)
    elif initialization == 'avg':
      x[:, 1:-1, 1:-1] = torch.mean(bc, dim=1).view(batch_size, 1, 1)
    else:
      raise NotImplementedError
  return x

def fd_step(x, bc, f):
  '''
  One update of Jacobi iterative method.
  x: torch tensor of size (batch_size x H x W)
  '''
  # Convolution
  y = F.conv2d(x.unsqueeze(1), update_kernel.view(1, 1, 3, 3), padding=1).view_as(x)
  if f is not None:
    y = y - f
  y = set_boundary(y, bc)
  return y

def fd_error(x, bc, f, aggregate='max'):
  '''
  Use loss kernel to calculate absolute error.
  l = Au - f.
  '''
  if is_bc_mask(x, bc):
    # bc mask
    l = F.conv2d(x.unsqueeze(1), loss_kernel.view(1, 1, 3, 3), padding=1).squeeze(1)
    if f is not None:
      l = l - f
    l = l * (1 - bc[:, 1])
  else:
    # Original bc
    l = F.conv2d(x.unsqueeze(1), loss_kernel.view(1, 1, 3, 3)).squeeze(1)
    if f is not None:
      l = l - f[:, 1:-1, 1:-1]

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
  Run Jacobi iterations.
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

def calculate_errors(x, bc, f, gt, iter_func, n_steps, starting_error):
  '''
  Run iterations and calculate errors, relative to starting_error.
  '''
  batch_size = bc.size(0)
  errors = [torch.ones(batch_size, 1)]
  x = x.detach()
  for i in range(n_steps):
    x = iter_func(x, bc, f).detach()
    e = l2_error(x, gt).cpu() / starting_error # Normalize by starting_error
    if (e < 0.002).all().item():
      # Pad with zeros when error is close to 0
      zeros = torch.zeros(batch_size, n_steps - i)
      errors.append(zeros)
      break
    errors.append(e.unsqueeze(1))
  errors = torch.cat(errors, dim=1)
  return errors, x

def plot_error_curves(results, num=None):
  '''
  Plot model and Jacobi error curves.
  results: dictionary of torch Tensors, returned by model.evaluate().
  Return images: torch Tensor, size (3 x H x (W * num))
  '''
  W, H = 640, 480
  images = []
  if num is None:
    num = results['model errors'].size(0)
  for i in range(num):
    curves = []
    for label in results.keys():
      curves.append({'y': results[label][i], 'label': label})
    img = plot_curves(curves, config={'title': 'Error curves', 'image_size': (W, H),
                                      'xlabel': 'iterations'})
    img = img.transpose((2, 0, 1)) / 255 # 3 x H x W
    images.append(torch.Tensor(img))
  images = torchvision.utils.make_grid(images, nrow=num)
  return images

def plot_results(results):
  '''
  Plot x and gt.
  results: dictionary of torch Tensors with size (batch_size x n_steps),
  '''
  x_img = plot_data(results['x'][0], 'x')
  gt_img = plot_data(results['gt'][0], 'Ground truth')
  x_img = torch.Tensor(x_img.transpose((2, 0, 1)) / 255) # 3 x H x W
  gt_img = torch.Tensor(gt_img.transpose((2, 0, 1)) / 255) # 3 x H x W
  images = torchvision.utils.make_grid([x_img, gt_img], nrow=2)
  return images

def construct_matrix(bc, image_size, iter_func):
  '''
  Construct the update matrix given the iterator function.
  x' = Ax + b
  y' = By, where y = [- x -, 1]
  iter_func: function, iter_func(x, b, f).
             x must be (1 x image_size x image_size).
  Return matrix A and B (torch Tensors).
  '''
  bc = torch.Tensor(bc).view(1, 4)

  # Find bias
  x = torch.zeros(1, image_size + 2, image_size + 2)
  if torch.cuda.is_available():
    x = x.cuda()
    bc = bc.cuda()
  x = set_boundary(x, bc)
  y = iter_func(x, bc, None).detach()
  bias = y[0, 1:-1, 1:-1].cpu().view(-1)
  # columns
  columns = []
  for i in range(image_size):
    for j in range(image_size):
      y = x.clone()
      y[0, i + 1, j + 1] = 1
      y = iter_func(y, bc, None).detach()
      c = y[0, 1:-1, 1:-1].cpu().view(-1) - bias
      columns.append(c)
  A = torch.stack(columns, dim=1)
  length = image_size ** 2
  assert A.size(0) == length

  # Add bias and last row
  bias = torch.cat([bias, torch.Tensor([1])]).view(-1, 1)
  B = torch.cat([A, torch.zeros(1, length)], dim=0)
  B = torch.cat([B, bias], dim=1)
  assert B.size(0) == length + 1
  return A, B

def calculate_eigenvalues(model, image_size=16):
  '''
  Construct update matrix of the model, and calculate eigenvalues and eigenvectors.
  Return eigenvalues w and eigenvectors v.
  '''
  # Remove activation and bc_mask
  activation = model.get_activation()
  is_bc_mask = model.iterator.is_bc_mask
  model.change_activation('none')
  model.iterator.is_bc_mask = False
  # Any bc works, won't change eigenvalues
  test_bc = np.zeros((1, 4))
  A, B = construct_matrix(test_bc, image_size, model.iter_step)
  w, v = np.linalg.eig(B)
  # Change back to original setting
  model.change_activation(activation)
  model.iterator.is_bc_mask = is_bc_mask
  return w, v

def restriction(x, bc):
  '''
  Weighted restriction using restriction_kernel.
  Size of x must be odd.
  Return: downsampled x of size (N - 1) / 2 + 1. Ex. 33 -> 17
  '''
  x = x[:, 1:-1, 1:-1].unsqueeze(1)
  y = F.conv2d(x, restriction_kernel.view(1, 1, 3, 3), stride=2).squeeze(1)
  y = pad_boundary(y, bc)
  return y

def interpolation(x, bc):
  '''
  Size of x must be odd.
  Return: upsampled x of size N * 2 - 1. Ex. 17 -> 33
  '''
  _, image_size, _ = x.size()
  new_size = image_size * 2 - 1
  # align_corners True to preserve boundaries
  y = F.interpolate(x.unsqueeze(1), size=new_size, mode='bilinear', align_corners=True)
  y = set_boundary(y.squeeze(1), bc)
  return y

def subsample(x):
  '''
  Bilinear subsampling.
  Return: downsampled x of size (N - 1) / 2 + 1. Ex. 33 -> 17
  '''
  _, image_size, _ = x.size()
  new_size = (image_size - 1) // 2 + 1
  y = F.interpolate(x.unsqueeze(1), size=new_size, mode='bilinear', align_corners=True)
  return y.squeeze(1)
