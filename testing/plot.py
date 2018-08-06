import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

x = np.arange(10)
a = x ** 2
b = x ** 2.1
fig = plt.figure()
plt.plot(x, a, label='a')
plt.plot(x, b, label='b')
plt.xlabel('iterations')
plt.legend()
plt.title('Test')
#plt.savefig('test.png')

# Save to numpy array
fig.canvas.draw()
w, h = fig.canvas.get_width_height()
data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
data = data.reshape((h, w, -1))
img = Image.fromarray(data)
print(img.size)
img.save('tmp.png')

plt.close()
