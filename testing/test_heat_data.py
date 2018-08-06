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


if __name__ == '__main__':
  data_dir = os.path.join(os.environ['HOME'], 'slowbro/PDE/heat/16x16')
  test(data_dir)
