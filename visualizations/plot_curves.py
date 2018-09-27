import os
import matplotlib.pyplot as plt
import numpy as np

method_lst = ['baseline', 'ours']

# Defining plot style
def scale_rgb(rgb):
    return np.array(rgb)/255.0

METHODS = {
           'baseline': {'label': 'FEniCS',
                        'color': scale_rgb([255,127,0]),
                        'linestyle': '--',
                        'marker': '^',
                        'markersize': 8, 'mew': 2},
           'ours': {'label': 'Ours',
                   'color': scale_rgb([77,175,74]),
                   'linestyle': '-', 'zorder': 10,
                   'marker': 'o', #'mec': scale_rgb([50,69,29]),
                   'markersize': 8, 'mew': 2}
           }
f1_data = {}

geometry = 'square'
#geometry = 'Lshape'
#geometry = 'cylinders'

if geometry == 'square':
  f1_data['baseline'] = np.array([17.170, 74.683, 307.176])
  f1_data['ours'] = np.array([13.757, 66.134, 299.918])
  filename = 'fenics_square.pdf'
  ylabel = 'Time (s)'
  title = 'Square'
elif geometry == 'Lshape':
  f1_data['baseline'] = np.array([22.482, 107.379, 538.060])
  f1_data['ours'] = np.array([14.445, 66.711, 266.962])
  filename = 'fenics_Lshape.pdf'
  ylabel = 'Time (s)'
  title = 'L-shape'
elif geometry == 'cylinders':
  f1_data['baseline'] = np.array([21.833, 101.468, 517.846])
  f1_data['ours'] = np.array([12.866, 60.030, 280.200])
  filename = 'fenics_cylinders.pdf'
  ylabel = 'Time (s)'
  title = 'Cylinders'


fn_size = 20
plt.figure(num=None, figsize=(8, 6))

x = np.arange(3)
xs = np.array([256, 512, 1024])

for method_key in method_lst:
    method = METHODS[method_key]
    f1 = f1_data[method_key]
    plt.plot(x, f1, linewidth=3, **method)

plt.ylabel(ylabel, fontsize=fn_size)
plt.xlabel('Dimension', fontsize=fn_size)
plt.title(title, fontsize=24)
plt.grid(True)
# plt.xlim([10**1, 10**4])

ax = plt.gca()
#ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylim(10, 600)
plt.xticks(x, xs)
#plt.xticks([256, 512, 1024])
plt.setp(plt.axes().get_xticklabels(), fontsize=fn_size)
plt.setp(plt.axes().get_yticklabels(), fontsize=fn_size)

plt.legend([METHODS[mid]['label'] for mid in method_lst], 
           ncol=1, prop={'size': fn_size-4}, loc='best')

plt.savefig(filename, bbox_inches='tight')
plt.show()
