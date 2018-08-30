import numpy as np
import os

def test(data_dir):
  bc = np.load(os.path.join(data_dir, 'bc.npy'))
  print(bc.shape)
  indices = np.random.randint(len(bc), size=10)
  for i in indices:
    frames = np.load(os.path.join(data_dir, 'frames', '{:04d}.npy'.format(i)))
    print(frames.shape)
    batch_size, length, _, _ = frames.shape
    j = np.random.randint(batch_size)
    assert np.allclose(frames[j, :, 0, 1:-1], bc[i][j][0])
    assert np.allclose(frames[j, :, -1, 1:-1], bc[i][j][1])
    assert np.allclose(frames[j, :, :, 0], bc[i][j][2])
    assert np.allclose(frames[j, :, :, -1], bc[i][j][3])

def test_geometry(data_dir, geometry, image_size):
  data_dir = os.path.join(data_dir, geometry, '{}x{}'.format(image_size, image_size))
  for i in range(50):
    frames = np.load(os.path.join(data_dir, 'frames', '{:04d}.npy'.format(i)))
    batch_size = frames.shape[0]
    assert frames.shape == (batch_size, 4, image_size, image_size), frames.shape
    data = frames
    data[:, :3] /= 100
    x, gt, bc_values, bc_mask = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
    assert np.all(x < 1.00001), x
    assert np.all(gt < 1.00001), gt
    assert np.all(np.logical_or(np.isclose(bc_mask, 0), np.isclose(bc_mask, 1)))

if __name__ == '__main__':
  data_dir = os.path.join(os.environ['HOME'], 'slowbro/PDE/heat/17x17')
  test(data_dir)

  data_dir = os.path.join(os.environ['HOME'], 'slowbro/PDE/heat/')
  test_geometry(data_dir, 'cylinders', 65)
