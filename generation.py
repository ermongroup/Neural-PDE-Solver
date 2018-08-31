import argparse
import numpy as np
import os
import torch
import torch.nn.functional as F

import utils
from models.iterators import MultigridIterator

parser = argparse.ArgumentParser()

parser.add_argument('--save_dir', type=str,
                    default=os.path.join(os.environ['HOME'], 'slowbro/PDE/heat'))
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--save_every', type=int, default=1)
parser.add_argument('--n_frames', type=int, default=1)
parser.add_argument('--n_runs', type=int, default=1000)
# data
parser.add_argument('--image_size', type=int, default=17)
parser.add_argument('--max_temp', type=int, default=100)
parser.add_argument('--heat_source', type=int, default=0)
parser.add_argument('--geometry', type=str, default='square',
                    choices=['square', 'cylinders', 'Lshape'])

np.random.seed(666)


def get_solution(x, bc, f):
  '''
  Iterate until error is below a threshold.
  '''
  frames = [x]

  error_threshold = 0.001
  max_iters = 20000
  # Iterate with Jacobi until ground truth
  for i in range(max_iters):
    x = utils.fd_step(x, bc, f)
    error = utils.fd_error(x, bc, f)
    if (i + 1) % 100 == 0:
      largest_error = error.max().item() # largest error in the batch
      print('Iter {}: largest error {}'.format(i + 1, largest_error))
      if largest_error < error_threshold:
        break
  # Add ground truth to frames
  y = x.cpu().numpy()
  frames.append(y)
  frames = np.stack(frames, axis=1) # batch_size x (n_frames + 1) x image_size x image_size
  return frames

def get_heat_source(image_size, batch_size):
  heat_src = []
  for i in range(batch_size):
    scale = np.random.uniform(300, 400) / (image_size ** 2)
    f = - utils.gaussian(image_size) * scale
    f = torch.Tensor(f).unsqueeze(0) # (1 x image_size x image_size)
    f = utils.pad_boundary(f, torch.zeros(1, 4))
    heat_src.append(f)
  heat_src = torch.cat(heat_src, dim=0)
  return heat_src

def generate_square(opt):
  '''
  Generate data with square boundary conditions.
  '''
  opt.save_dir = os.path.join(opt.save_dir, '{}x{}'.format(opt.image_size, opt.image_size))
  frame_dir = os.path.join(opt.save_dir, 'frames')
  os.makedirs(frame_dir, exist_ok=True)

  # shape: n_runs x batch_size x 4
  boundary_conditions = np.random.rand(opt.n_runs, opt.batch_size, 4) * opt.max_temp
  np.save(os.path.join(opt.save_dir, 'bc.npy'), boundary_conditions)
  # Save opt
  np.save(os.path.join(opt.save_dir, 'opt.npy'), opt)

  for run in range(opt.n_runs):
    x = np.random.rand(opt.batch_size, opt.image_size, opt.image_size) * opt.max_temp
    bc = boundary_conditions[run]
    x = utils.set_boundary(x, bc)
    if opt.heat_source:
      f = get_heat_source(opt.image_size - 2, opt.batch_size)
    else:
      f = None

    x = torch.Tensor(x)
    bc = torch.Tensor(bc)
    if torch.cuda.is_available():
      x = x.cuda()
      bc = bc.cuda()

    # Initialize with average of boundary conditions
    x[:, 1:-1, 1:-1] = bc.mean(dim=1).view(-1, 1, 1)

    # Use Multigrid model to initialize
    # TODO: Multigrid does not support Au = f yet.
    if f is None:
      model = MultigridIterator(4, 4, 4)
      for i in range(50):
        x = model.iter_step(x, bc)
      error = utils.fd_error(x)
      largest_error = error.max().item()
      print('largest error {}'.format(largest_error))

    # Find solution
    frames = get_solution(x, bc, f)
    assert frames.shape[1] == opt.n_frames + 1
    np.save(os.path.join(frame_dir, '{:04d}.npy'.format(run)), frames)
    print('Run {} saved.'.format(run))

def generate_geometry(opt):
  '''
  Generate data with geometry.
  '''
  opt.save_dir = os.path.join(opt.save_dir, opt.geometry,
                              '{}x{}'.format(opt.image_size, opt.image_size))
  frame_dir = os.path.join(opt.save_dir, 'frames')
  os.makedirs(frame_dir, exist_ok=True)
  # Save opt
  np.save(os.path.join(opt.save_dir, 'opt.npy'), opt)

  for run in range(opt.n_runs):
    x, bc_values, bc_mask = utils.get_geometry(opt.geometry, opt.image_size,
                                               opt.batch_size, opt.max_temp)
    bc = np.stack([bc_values, bc_mask], axis=1) # batch_size x 2 x image_size x image_size
    x = torch.Tensor(x)
    bc = torch.Tensor(bc)
    f = None
    if torch.cuda.is_available():
      x = x.cuda()
      bc = bc.cuda()

    # Find solution
    frames = get_solution(x, bc, f)
    assert frames.shape[1] == 2
    # Add bc and mask
    data = np.concatenate([frames, bc_values[:, np.newaxis, :, :],
                           bc_mask[:, np.newaxis, :, :]], axis=1)
    np.save(os.path.join(frame_dir, '{:04d}.npy'.format(run)), data)
    print('Run {} saved.'.format(run))

if __name__ == '__main__':
  opt = parser.parse_args()
  if opt.geometry == 'square':
    assert False
    generate_square(opt)
  else:
    generate_geometry(opt)
