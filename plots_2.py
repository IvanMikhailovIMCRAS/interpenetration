import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from interpenetration import MidPoint, pressure

data_N = np.loadtxt(f"data/N1_{500}_N2_{500}_sig1_0.1_sig2_0.1.txt")
MP = MidPoint(500, 0.1, 500, 0.1)
phi_mt_N = []
for k in data_N[0]:
    z, phi = MP.calc(k)
    phi_mt_N.append(phi)


plt.plot(
    data_N[0][::2],
    phi_mt_N[::2],
    linestyle="-",
    color="black",
    label=f"analytical data",
)
plt.scatter(
    data_N[0],
    data_N[2],
    label=f"modelling data",
    color="red",
)
plt.xlabel(r"$D$")
plt.ylabel(r"$\varphi_{m}$")
plt.legend(loc="upper right")
plt.title("$N_1 = N_2 = 500,~\sigma=0.1,~\chi=0.0$")
plt.savefig(f"phi_m.png", dpi=150, bbox_inches="tight")
plt.clf()
# # ---------------------------------------------------------------------------------------------------------------#
