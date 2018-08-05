import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

x = np.arange(10)
a = x ** 2
b = x ** 2.1
plt.plot(x, a, label='a')
plt.plot(x, b, label='b')
plt.xlabel('iterations')
plt.legend()
plt.title('Test')
plt.savefig('test.png')
plt.close()
