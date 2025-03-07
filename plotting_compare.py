import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from interpenetration import MidPoint, pressure

N_total = 200
sigma_total = 0.2
colors = [
    "#5c1f7a",
    "#d62728",
    "#2ca02c",
    "#ff7f0e",
    "#1f77b4",
]
cmap = mcolors.LinearSegmentedColormap.from_list("contrast_colors", colors)

# # Mean phi on D
N1_values = [100, 80, 60, 40, 20]
sigma1_values = [0.1, 0.07, 0.05, 0.03, 0.01]
p_len = len(N1_values)
s_len = len(sigma1_values)

for i, N1 in enumerate(N1_values):
    N2 = N_total - N1

    data_N = np.loadtxt(f"data/N1_{N1}_N2_{N2}_sig1_0.1_sig2_0.1.txt")
    D_N = data_N[0]
    phi_mean_N = data_N[2]

    MP = MidPoint(N1, 0.1, N2, 0.1)
    phi_mt_N = []
    for k in D_N:
        z, phi = MP.calc(k)
        phi_mt_N.append(phi)

    color = cmap(i / (p_len - 1))
    if N1 != 100:
        plt.plot(
            D_N,
            phi_mean_N,
            linestyle="-",
            color=color,
            label=f"$N_1 = {N1},~N_2 = {N2}$",
        )
        plt.scatter(
            D_N[::2],
            phi_mt_N[::2],
            color=color,
        )

    else:
        plt.plot(
            D_N,
            phi_mean_N,
            linestyle="-.",
            linewidth="2",
            color=color,
            label=f"$N_1 = {N1},~N_2 = {N2}$",
        )
        plt.scatter(
            D_N[::2],
            phi_mt_N[::2],
            color=color,
        )

plt.xlabel(r"$D$")
plt.ylabel(r"$\varphi_{m}$")
plt.legend(loc="upper right")
plt.title("$N_1 + N_2 = 200,~\sigma=0.1,~\chi=0.0$")
plt.savefig(f"phi_m_sig_const.png", dpi=150, bbox_inches="tight")
plt.clf()
# # ---------------------------------------------------------------------------------------------------------------#
for j, sigma1 in enumerate(sigma1_values):
    sigma2 = round(sigma_total - sigma1, 2)

    data_sigma = np.loadtxt(
        f"data/N1_200_N2_200_sig1_{sigma1}_sig2_{round(sigma2, 2)}.txt"
    )
    D_sigma = data_sigma[0]
    phi_mean_sigma = data_sigma[2]

    MP = MidPoint(200, sigma1, 200, sigma2)
    phi_mt_sigma = []
    for k in D_sigma:
        z, phi = MP.calc(k)
        phi_mt_sigma.append(phi)

    color = cmap(j / (s_len - 1))
    if sigma1 != 0.1:
        plt.plot(
            D_sigma,
            phi_mean_sigma,
            linestyle="-",
            color=color,
            label=f"$\sigma_1={round(sigma1, 2)},~\sigma_2={round(sigma2, 2)}$",
        )
        plt.scatter(
            D_sigma[::2],
            phi_mt_sigma[::2],
            color=color,
        )
    else:
        plt.plot(
            D_sigma,
            phi_mean_sigma,
            linestyle="-.",
            linewidth="2",
            color=color,
            label=f"$\sigma_1={round(sigma1, 2)},~\sigma_2={round(sigma2, 2)}$",
        )
        plt.scatter(
            D_sigma[::2],
            phi_mt_sigma[::2],
            color=color,
        )
plt.xlabel(r"$D$")
plt.ylabel(r"$\varphi_{m}$")
plt.legend(loc="upper right")
plt.title("$N_1=N_2=200,~\sigma_1 + \sigma_2 = 0.2,~\chi=0.0$")
plt.savefig(f"phi_m_N_const.png", dpi=150, bbox_inches="tight")
plt.clf()

# # Mean z on D
for i, N1 in enumerate(N1_values):
    N2 = N_total - N1

    data_N = np.loadtxt(f"data/N1_{N1}_N2_{N2}_sig1_0.1_sig2_0.1.txt")
    D_N = data_N[0]
    z_mean_N = data_N[1]

    MP = MidPoint(N1, 0.1, N2, 0.1)
    z_mt_N = []
    for k in D_N:
        z, phi = MP.calc(k)
        z_mt_N.append(z)

    color = cmap(i / (p_len - 1))
    if N1 != 100:
        plt.plot(
            D_N,
            z_mean_N,
            linestyle="-",
            color=color,
            label=f"$N_1 = {N1},~N_2 = {N2}$",
        )
        plt.scatter(
            D_N[::2],
            z_mt_N[::2],
            color=color,
        )
    else:
        plt.plot(
            D_N,
            z_mean_N,
            linestyle="-.",
            linewidth="2",
            color=color,
            label=f"$N_1 = {N1},~N_2 = {N2}$",
        )
        plt.scatter(
            D_N[::2],
            z_mt_N[::2],
            color=color,
        )
plt.loglog()
plt.xlabel(r"$D$")
plt.ylabel(r"$z_{m}$")
plt.legend(loc="lower right")
plt.title("$N_1 + N_2 = 200,~\sigma=0.1,~\chi=0.0$")
plt.savefig(f"z_m_sig_const.png", dpi=150, bbox_inches="tight")
plt.clf()
# ---------------------------------------------------------------------------------------------------------------#
for j, sigma1 in enumerate(sigma1_values):
    sigma2 = round(sigma_total - sigma1, 2)

    data_sigma = np.loadtxt(
        f"data/N1_200_N2_200_sig1_{sigma1}_sig2_{round(sigma2, 2)}.txt"
    )
    D_sigma = data_sigma[0]
    z_mean_sigma = data_sigma[1]

    MP = MidPoint(200, sigma1, 200, sigma2)
    z_mt_sigma = []
    for k in D_sigma:
        z, phi = MP.calc(k)
        z_mt_sigma.append(z)

    color = cmap(j / (s_len - 1))
    if sigma1 != 0.1:
        plt.plot(
            D_sigma,
            z_mean_sigma,
            linestyle="-",
            color=color,
            label=f"$\sigma_1={round(sigma1, 2)},~\sigma_2={round(sigma2, 2)}$",
        )
        plt.scatter(
            D_sigma[::2],
            z_mt_sigma[::2],
            color=color,
        )
    else:
        plt.plot(
            D_sigma,
            z_mean_sigma,
            linestyle="-.",
            linewidth="2",
            color=color,
            label=f"$\sigma_1={round(sigma1, 2)},~\sigma_2={round(sigma2, 2)}$",
        )
        plt.scatter(
            D_sigma[::2],
            z_mt_sigma[::2],
            color=color,
        )
plt.loglog()
plt.xlabel(r"$D$")
plt.ylabel(r"$z_{m}$")
plt.legend(loc="lower right")
plt.title("$N=200,~\sigma_1 + \sigma_2 = 0.2,~\chi=0.0$")
plt.savefig(f"z_m_N_const.png", dpi=150, bbox_inches="tight")
plt.clf()

# # Pressure
for i, N1 in enumerate(N1_values):
    N2 = N_total - N1

    data_N = np.loadtxt(f"data/N1_{N1}_N2_{N2}_sig1_0.1_sig2_0.1.txt")
    phi_m_N = data_N[2]
    Pi_N = -np.diff(data_N[-2]) / np.diff(data_N[0])
    phi_m_N = 0.5 * (phi_m_N[1:] + phi_m_N[:-1])

    MP = MidPoint(N1, 0.1, N2, 0.1)
    Pi_t_N = []
    phi_mt_N = []
    for k in D_N:
        z, phi = MP.calc(k)
        Pi_t_N.append(pressure(phi))
        phi_mt_N.append(phi)

    color = cmap(i / (p_len - 1))
    if N1 != 100:
        plt.plot(
            phi_m_N,
            Pi_N,
            linestyle="-",
            color=color,
            label=f"$N_1 = {N1},~N_2 = {N2}$",
        )
        plt.scatter(
            phi_mt_N[::2],
            Pi_t_N[::2],
            color=color,
        )
    else:
        plt.plot(
            phi_m_N,
            Pi_N,
            linestyle="-.",
            linewidth="2",
            color=color,
            label=f"$N_1 = {N1},~N_2 = {N2}$",
        )
        plt.scatter(
            phi_mt_N[::2],
            Pi_t_N[::2],
            color=color,
        )
for j, sigma1 in enumerate(sigma1_values):
    sigma2 = round(sigma_total - sigma1, 2)

    data_sigma = np.loadtxt(
        f"data/N1_200_N2_200_sig1_{sigma1}_sig2_{round(sigma2, 2)}.txt"
    )
    phi_m_sigma = data_sigma[2]
    Pi_sigma = -np.diff(data_sigma[-2]) / np.diff(data_sigma[0])
    phi_m_sigma = 0.5 * (phi_m_sigma[1:] + phi_m_sigma[:-1])

    MP = MidPoint(200, sigma1, 200, sigma2)

    Pi_t_sigma = []
    phi_mt_sigma = []
    for k in D_sigma:
        z, phi = MP.calc(k)
        Pi_t_sigma.append(pressure(phi))
        phi_mt_sigma.append(phi)

    color = cmap(j / (s_len - 1))
    if sigma1 != 0.1:
        plt.plot(
            phi_m_sigma,
            Pi_sigma,
            linestyle="--",
            color=color,
            label=f"$\sigma_1={round(sigma1, 2)},~\sigma_2={round(sigma2, 2)}$",
        )
        plt.scatter(
            phi_mt_sigma[::2],
            Pi_t_sigma[::2],
            color=color,
        )
    else:
        plt.plot(
            phi_m_sigma,
            Pi_sigma,
            linestyle="-.",
            linewidth="2",
            color=color,
            label=f"$\sigma_1={round(sigma1, 2)},~\sigma_2={round(sigma2, 2)}$",
        )
        plt.scatter(
            phi_mt_sigma[::2],
            Pi_t_sigma[::2],
            color=color,
        )
plt.loglog()
plt.ylim(1e-4, 2e0)
plt.xlim(6e-3, 1e0)
plt.xlabel(r"$\varphi_m$")
plt.ylabel(r"$\Pi$")
plt.legend(loc="upper left")
plt.savefig(f"Pi.png", dpi=150, bbox_inches="tight")
plt.clf()
