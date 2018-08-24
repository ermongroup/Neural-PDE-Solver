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
parser.add_argument('--image_size', type=int, default=16)
parser.add_argument('--max_temp', type=int, default=100)

np.random.seed(666)

def main(opt):
  opt.save_dir = os.path.join(opt.save_dir, '{}x{}'.format(opt.image_size, opt.image_size))
  frame_dir = os.path.join(opt.save_dir, 'frames')
  os.makedirs(frame_dir, exist_ok=True)

  # shape: n_runs x batch_size x 4
  boundary_conditions = np.random.rand(opt.n_runs, opt.batch_size, 4) * opt.max_temp
  np.save(os.path.join(opt.save_dir, 'bc.npy'), boundary_conditions)
  # Save opt
  np.save(os.path.join(opt.save_dir, 'opt.npy'), opt)

  for run in range(opt.n_runs):
    x = np.random.rand(opt.batch_size, opt.image_size + 2, opt.image_size + 2) * opt.max_temp
    bc = boundary_conditions[run]
    x = utils.set_boundary(x, bc)
    frames = [x]

    x = torch.Tensor(x)
    bc = torch.Tensor(bc)
    if torch.cuda.is_available():
      x = x.cuda()
      bc = bc.cuda()

    # Get n_frames frames
    for i in range(opt.n_frames - 1):
      for j in range(opt.save_every):
        x = utils.fd_step(x, bc)
      y = x.cpu().numpy()
      frames.append(y)

    # Initialize with average of boundary conditions
    x[:, 1:-1, 1:-1] = bc.mean(dim=1).view(-1, 1, 1)

    # Use Multigrid model to initialize
    model = MultigridIterator(4, 4, 4)
    for i in range(50):
      x = model.iter_step(x, bc)
    error = utils.fd_error(x)
    largest_error = error.max().item()
    print('largest error {}'.format(largest_error))

    error_threshold = 0.001
    max_iters = 20000
    # Iterate with Jacobi until ground truth
    for i in range(max_iters):
      x = utils.fd_step(x, bc)
      error = utils.fd_error(x)
      if (i + 1) % 100 == 0:
        largest_error = error.max().item() # largest error in the batch
        print('Iter {}: largest error {}'.format(opt.n_frames * opt.save_every + i + 1, largest_error))
        if largest_error < error_threshold:
          break
    # Add ground truth to frames
    y = x.cpu().numpy()
    frames.append(y)
    assert len(frames) == opt.n_frames + 1
    frames = np.stack(frames, axis=1) # batch_size x (n_frames + 1) x image_size x image_size
    np.save(os.path.join(frame_dir, '{:04d}.npy'.format(run)), frames)
    print('Run {} saved.'.format(run))


if __name__ == '__main__':
  opt = parser.parse_args()
  main(opt)
