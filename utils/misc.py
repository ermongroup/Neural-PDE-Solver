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

def plot(data_list, config):
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
