import matplotlib.pyplot as plt
import numpy as np

sigma1 = 0.1
sigma2 = 0.2 - sigma1
N1 = 20
N2 = 200 - N1

data = np.loadtxt(f"data/N1_{N1}_N2_{N2}_sig1_{sigma1}_sig2_{sigma2}.txt")

plt.plot(data[0], data[7], color="red", label="$left~brush$")
plt.plot(data[0], data[8], color="black", label="$right~brush$")
plt.legend()
plt.show()
