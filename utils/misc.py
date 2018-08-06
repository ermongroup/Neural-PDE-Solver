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

def sample_batch(dataset, batch_size):
  '''
  dataset: torch.utils.data.Dataset
  '''
  indices = np.random.randint(len(dataset), size=batch_size)
  for idx in indices:
    data = dataset.__getitem__(idx)

def plot(data_list, config):
  '''
  data_list: A list of dictionaries.
  config: plot configurations.
  return: numpy array of image, size (H x W x 3)
  '''
  for data in data_list:
    y = to_numpy(data['y'])
    x = np.arange(len(y))
    plt.plot(x, y, label=data['label'])
  plt.xlabel(config['title'])
  if 'ylim' in config:
    plt.ylim(config['ylim'])
  plt.legend()
  plt.savefig('tmp.png')
  plt.close()
  # Read in tmp.png
  img = Image.open('tmp.png')
  if 'image_size' in config:
    img = img.resize(config['image_size'])
  img = np.array(img) # H x W x 4
  return img[:, :, :3]
