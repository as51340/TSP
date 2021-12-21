import numpy as np
import matplotlib.pyplot as plt

arr = [27742.77,29701.98,28604.08,28457.96,30616.53,27154.49,28239.57,27353.48,29062.39,28081.50]
plt.plot(np.arange(1,11,1), arr, 'x')
plt.axhline(y=28501.47, color='b', linestyle='--')
plt.xlabel("samples", fontsize=20)
plt.ylabel("objective value", fontsize=20)
plt.show()

