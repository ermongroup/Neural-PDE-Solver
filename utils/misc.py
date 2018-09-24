import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

def to_numpy(array):
  '''
  array: numpy array or torch Tensor
  '''
  if isinstance(array, np.ndarray):
    return array
  elif isinstance(array, list):
    return np.array(array)
  elif isinstance(array, torch.Tensor):
    return array.cpu().numpy()

def prompt_yes_no(question):
  i = input(question + ' [y/n]: ')
  if len(i) > 0 and (i[0] == 'y' or i[0] == 'Y'):
    return True
  else:
    return False

def blue(string):
  return '\033[94m'+string+'\033[0m'

def yellow(string):
  return '\033[93m'+string+'\033[0m'

def red(string):
  return '\033[91m'+string+'\033[0m'

def plot_curves(data_list, config):
  '''
  data_list: A list of dictionaries.
  config: plot configurations.
  return: numpy array of image, size (H x W x 3)
  '''
  fig = plt.figure()
  for data in data_list:
    y = to_numpy(data['y'])
    x = np.arange(len(y))
    plt.plot(x, y, label=data['label'])
  plt.title(config['title'])
  if 'xlabel' in config:
    plt.xlabel(config['xlabel'])
  plt.ylim(ymin=0)
  plt.legend()

  # Plot image to numpy array
  fig.canvas.draw()
  plt.close()
  w, h = fig.canvas.get_width_height()
  data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
  data = data.reshape((h, w, 3))
  img = Image.fromarray(data)
  if 'image_size' in config:
    img = img.resize(config['image_size'])
  img = np.array(img) # H x W x 3
  return img

def plot_data(x, title=''):
  '''
  x: H x W.
  '''
  x = to_numpy(x)
  fig = plt.figure()
  plt.imshow(x)
  plt.colorbar()
  plt.title(title)
  fig.canvas.draw()
  plt.close()
  w, h = fig.canvas.get_width_height()
  data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
  data = data.reshape((h, w, 3))
  return data

def gaussian(image_size):
  '''
  Return a Gaussian with random size and location.
  Maximum value is 1.
  '''
  width = image_size // 2
  # Random location
  x_mean = np.random.randint(-width // 2, width // 2 + 1)
  y_mean = np.random.randint(-width // 2, width // 2 + 1)
  x = np.linspace(-width, width, num=image_size, endpoint=True) + x_mean
  y = np.linspace(-width, width, num=image_size, endpoint=True) + y_mean
  # Random size
  s = np.random.uniform(2, 3)
  sd = width / s
  x = np.exp(-(x ** 2) / 2 / (sd ** 2))
  y = np.exp(-(y ** 2) / 2 / (sd ** 2))
  kernel = x[:, np.newaxis] * y[np.newaxis, :]
  return kernel

def dot_product(x, y):
  '''
  Batch dot product.
  '''
  x = x.view(x.size(0), -1)
  y = y.view(y.size(0), -1)
  return torch.sum(x * y, dim=1)

def spectral_radius(A):
  '''
  Spectral radius: largest absolute eigenvalue.
  '''
  w, v = np.linalg.eig(A)
  w = sorted(np.abs(w))
  return w[-1]
