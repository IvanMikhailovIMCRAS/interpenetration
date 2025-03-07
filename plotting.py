import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

N_total = 1000
sigma_total = 0.2
colors = [
    "#5c1f7a",
    "#d62728",
    "#2ca02c",
    "#ff7f0e",
    "#1f77b4",
]
cmap = mcolors.LinearSegmentedColormap.from_list("contrast_colors", colors)

# #Sigma
N1_values = [500, 400, 300, 200]
sigma1_values = [0.1, 0.07, 0.05, 0.03, 0.01]
p_len = len(N1_values)
s_len = len(sigma1_values)

for i, N1 in enumerate(N1_values):
    N2 = N_total - N1

    data_N = np.loadtxt(f"data/N1_{N1}_N2_{N2}_sig1_0.01_sig2_0.01.txt")
    D_N = data_N[0]
    Sigma1_N = data_N[5]
    Sigma2_N = data_N[6]

    color = cmap(i / (p_len - 1))
    if N1 != 500:
        plt.plot(
            D_N,
            Sigma1_N,
            linestyle="-",
            color=color,
            label=f"$Left~brush~with~N_1 = {N1}$",
        )
        plt.plot(
            D_N,
            Sigma2_N,
            linestyle="--",
            color=color,
            label=f"$Right~brush~with~N_2 = {N2}$",
        )
    else:
        plt.plot(
            D_N,
            Sigma1_N,
            linestyle="-.",
            linewidth=2.5,
            color=color,
            label=f"$Left~brush~with~N_1 = {N1}$",
        )
        plt.plot(
            D_N,
            Sigma2_N,
            linestyle="-.",
            linewidth=2.5,
            color=color,
            label=f"$Right~brush~with~N_2 = {N2}$",
        )

plt.xlabel(r"$D$")
plt.ylabel(r"$\Sigma$")
plt.legend(loc="upper right")
plt.title("$N_1 + N_2 = 1000,~\sigma=0.01,~\chi=0.0$")
plt.savefig(f"Sigma_sig_const.png", dpi=150, bbox_inches="tight")
plt.clf()
# # ---------------------------------------------------------------------------------------------------------------#
for j, sigma1 in enumerate(sigma1_values):
    sigma2 = round(sigma_total - sigma1, 2)

    data_sigma = np.loadtxt(f"data/N1_500_N2_500_sig1_{sigma1}_sig2_{sigma2}.txt")
    D_sigma = data_sigma[0]
    Sigma1_s = data_sigma[5]
    Sigma2_s = data_sigma[6]

    color = cmap(j / (s_len - 1))
    if sigma1 != 0.1:
        plt.plot(
            D_sigma,
            Sigma1_s,
            linestyle="-",
            color=color,
            label=f"$Left~brush~with~\sigma_1 = {sigma1}$",
        )
        plt.plot(
            D_sigma,
            Sigma2_s,
            linestyle="--",
            color=color,
            label=f"$Right~brush~with~\sigma_2 = {sigma2}$",
        )
    else:
        plt.plot(
            D_sigma,
            Sigma1_s,
            linestyle="-.",
            linewidth=2.5,
            color=color,
            label=f"$Left~brush~with~\sigma_1 = {sigma1}$",
        )
        plt.plot(
            D_sigma,
            Sigma2_s,
            linestyle="-.",
            linewidth=2.5,
            color=color,
            label=f"$Right~brush~with~\sigma_2 = {sigma2}$",
        )
plt.xlabel(r"$D$")
plt.ylabel(r"$\Sigma$")
plt.legend(loc="upper right")
plt.title("$N_1 = N_2 = 500,~\sigma_1 + \sigma_2 = 0.2,~\chi=0.0$")
plt.savefig(f"Sigma_N_const.png", dpi=150, bbox_inches="tight")
plt.clf()

# # Gamma
for i, N1 in enumerate(N1_values):
    N2 = N_total - N1

    data_N = np.loadtxt(f"data/N1_{N1}_N2_{N2}_sig1_0.01_sig2_0.01.txt")
    D_N = data_N[0]
    Gamma1_N = data_N[3]
    Gamma2_N = data_N[4]

    color = cmap(i / (p_len - 1))
    if N1 != 500:
        plt.plot(
            D_N,
            Gamma1_N,
            linestyle="-",
            color=color,
            label=f"$Left~brush~with~N_1 = {N1}$",
        )
        plt.plot(
            D_N,
            Gamma2_N,
            linestyle="--",
            color=color,
            label=f"$Right~brush~with~N_2 = {N2}$",
        )
    else:
        plt.plot(
            D_N,
            Gamma1_N,
            linestyle="-.",
            linewidth=2.5,
            color=color,
            label=f"$Left~brush~with~N_1 = {N1}$",
        )
        plt.plot(
            D_N,
            Gamma2_N,
            linestyle="-.",
            linewidth=2.5,
            color=color,
            label=f"$Right~brush~with~N_2 = {N2}$",
        )
plt.xlabel(r"$D$")
plt.ylabel(r"$\Gamma$")
plt.legend(loc="upper right")
plt.title("$N_1 + N_2 = 1000,~\sigma=0.01,~\chi=0.0$")
plt.savefig(f"Gamma_sig_const.png", dpi=150, bbox_inches="tight")
plt.clf()
# # ---------------------------------------------------------------------------------------------------------------#
for j, sigma1 in enumerate(sigma1_values):
    sigma2 = round(sigma_total - sigma1, 2)

    data_sigma = np.loadtxt(f"data/N1_500_N2_500_sig1_{sigma1}_sig2_{sigma2}.txt")
    D_sigma = data_sigma[0]
    Gamma1_s = data_sigma[3]
    Gamma2_s = data_sigma[4]

    color = cmap(j / (s_len - 1))
    if sigma1 != 0.1:
        plt.plot(
            D_sigma,
            Gamma1_s,
            linestyle="-",
            color=color,
            label=f"$Left~brush~with~\sigma_1 = {sigma1}$",
        )
        plt.plot(
            D_sigma,
            Gamma2_s,
            linestyle="--",
            color=color,
            label=f"$Right~brush~with~\sigma_2 = {sigma2}$",
        )
    else:
        plt.plot(
            D_sigma,
            Gamma1_s,
            linestyle="-.",
            linewidth=2.5,
            color=color,
            label=f"$Left~brush~with~\sigma_1 = {sigma1}$",
        )
        plt.plot(
            D_sigma,
            Gamma2_s,
            linestyle="-.",
            linewidth=2.5,
            color=color,
            label=f"$Right~brush~with~\sigma_2 = {sigma2}$",
        )
plt.xlabel(r"$D$")
plt.ylabel(r"$\Gamma$")
plt.legend(loc="upper right")
plt.title("$N_1 = N_2 = 500,~\sigma_1 + \sigma_2 = 0.2,~\chi=0.0$")
plt.savefig(f"Gamma_N_const.png", dpi=150, bbox_inches="tight")
plt.clf()

# # Delta
for i, N1 in enumerate(N1_values):
    N2 = N_total - N1

    data_N = np.loadtxt(f"data/N1_{N1}_N2_{N2}_sig1_0.01_sig2_0.01.txt")
    D_N = data_N[0]
    delta1_N = data_N[9]
    delta2_N = data_N[10]

    color = cmap(i / (p_len - 1))
    if N1 != 500:
        plt.plot(
            D_N,
            delta1_N,
            linestyle="-",
            color=color,
            label=f"$Left~brush~with~N_1 = {N1}$",
        )
        plt.plot(
            D_N,
            delta2_N,
            linestyle="--",
            color=color,
            label=f"$Right~brush~with~N_2 = {N2}$",
        )
    else:
        plt.plot(
            D_N,
            delta1_N,
            linestyle="-.",
            linewidth=2.5,
            color=color,
            label=f"$Left~brush~with~N_1 = {N1}$",
        )
        plt.plot(
            D_N,
            delta2_N,
            linestyle="-.",
            linewidth=2.5,
            color=color,
            label=f"$Right~brush~with~N_2 = {N2}$",
        )

plt.xlabel(r"$D$")
plt.ylabel(r"$\delta$")
plt.legend(loc="upper right")
plt.title("$N_1 + N_2 = 1000,~\sigma=0.01,~\chi=0.0$")
plt.savefig(f"delta_sig_const.png", dpi=150, bbox_inches="tight")
plt.clf()
# # ---------------------------------------------------------------------------------------------------------------#
for j, sigma1 in enumerate(sigma1_values):
    sigma2 = round(sigma_total - sigma1, 2)

    data_sigma = np.loadtxt(f"data/N1_500_N2_500_sig1_{sigma1}_sig2_{sigma2}.txt")
    D_sigma = data_sigma[0]
    delta1_s = data_sigma[9]
    delta2_s = data_sigma[10]

    color = cmap(j / (s_len - 1))
    if sigma1 != 0.1:
        plt.plot(
            D_sigma,
            delta1_s,
            linestyle="-",
            color=color,
            label=f"$Left~brush~with~\sigma_1 = {sigma1}$",
        )
        plt.plot(
            D_sigma,
            delta2_s,
            linestyle="--",
            color=color,
            label=f"$Right~brush~with~\sigma_2 = {sigma2}$",
        )
    else:
        plt.plot(
            D_sigma[::5],
            delta1_s[::5],
            "-.",
            color=color,
            linewidth=2.5,
            label=f"$Left~brush~with~\sigma_1 = {sigma1}$",
        )
        plt.plot(
            D_sigma[::5],
            delta2_s[::5],
            "-.",
            color=color,
            linewidth=2.5,
            label=f"$Right~brush~with~\sigma_2 = {sigma2}$",
        )
plt.ylim(0.0, 5.0)
plt.xlabel(r"$D$")
plt.ylabel(r"$\delta$")
plt.legend(loc="lower left")
plt.title("$N_1 = N_2 = 500,~\sigma_1 + \sigma_2 = 0.2,~\chi=0.0$")
plt.savefig(f"delta_N_const.png", dpi=150, bbox_inches="tight")
plt.clf()

# # Sigma om Pi
for i, N1 in enumerate(N1_values):
    N2 = N_total - N1

    data_N = np.loadtxt(f"data/N1_{N1}_N2_{N2}_sig1_0.01_sig2_0.01.txt")
    phi_M_N = data_N[2]
    delta1_N = data_N[5]
    delta2_N = data_N[6]

    color = cmap(i / (p_len - 1))
    if N1 != 500:
        plt.plot(
            data_N[-2],
            delta1_N,
            linestyle="-",
            color=color,
            label=f"$Left~brush~with~N_1 = {N1}$",
        )
        plt.plot(
            data_N[-2],
            delta2_N,
            linestyle="--",
            color=color,
            label=f"$Right~brush~with~N_2 = {N2}$",
        )
    else:
        plt.plot(
            data_N[-2],
            delta1_N,
            linestyle="-.",
            linewidth=2.5,
            color=color,
            label=f"$Left~brush~with~N_1 = {N1}$",
        )
        plt.plot(
            data_N[-2],
            delta2_N,
            linestyle="-.",
            linewidth=2.5,
            color=color,
            label=f"$Right~brush~with~N_2 = {N2}$",
        )

plt.xlabel(r"$\Pi$")
plt.ylabel(r"$\Sigma$")
plt.title("$N_1 + N_2 = 1000,~\sigma=0.01,~\chi=0.0$")
plt.savefig(f"Sigma_Pi_sig_const.png", dpi=150, bbox_inches="tight")
plt.clf()
# # ---------------------------------------------------------------------------------------------------------------#
for j, sigma1 in enumerate(sigma1_values):
    sigma2 = round(sigma_total - sigma1, 2)

    data_sigma = np.loadtxt(f"data/N1_500_N2_500_sig1_{sigma1}_sig2_{sigma2}.txt")
    Pi_sig = data_sigma[-2]
    delta1_s = data_sigma[5]
    delta2_s = data_sigma[6]

    color = cmap(j / (s_len - 1))
    if sigma1 != 0.1:
        plt.plot(
            Pi_sig,
            delta1_s,
            linestyle="-",
            color=color,
            label=f"$Left~brush~with~\sigma_1 = {sigma1}$",
        )
        plt.plot(
            Pi_sig,
            delta2_s,
            linestyle="--",
            color=color,
            label=f"$Right~brush~with~\sigma_2 = {sigma2}$",
        )
    else:
        plt.plot(
            Pi_sig,
            delta1_s,
            linestyle="-.",
            linewidth=2.5,
            color=color,
            label=f"$Left~brush~with~\sigma_1 = {sigma1}$",
        )
        plt.plot(
            Pi_sig,
            delta2_s,
            linestyle="-.",
            linewidth=2.5,
            color=color,
            label=f"$Right~brush~with~\sigma_2 = {sigma2}$",
        )

plt.xlabel(r"$\Pi$")
plt.ylabel(r"$\Sigma$")
plt.title("$N_1 = N_2 = 500,~\sigma_1 + \sigma_2 = 0.2,~\chi=0.0$")
plt.savefig(f"Sigma__Pi_N_const.png", dpi=150, bbox_inches="tight")
plt.clf()

# # Gamma on Pi
for i, N1 in enumerate(N1_values):
    N2 = N_total - N1

    data_N = np.loadtxt(f"data/N1_{N1}_N2_{N2}_sig1_0.01_sig2_0.01.txt")
    phi_M_N = data_N[2]
    delta1_N = data_N[3]
    delta2_N = data_N[4]

    color = cmap(i / (p_len - 1))
    if N1 != 500:
        plt.plot(
            data_N[-2],
            delta1_N,
            linestyle="-",
            color=color,
            label=f"$Left~brush~with~N_1 = {N1}$",
        )
        plt.plot(
            data_N[-2],
            delta2_N,
            linestyle="--",
            color=color,
            label=f"$Right~brush~with~N_2 = {N2}$",
        )
    else:
        plt.plot(
            data_N[-2],
            delta1_N,
            linestyle="-.",
            linewidth=2.5,
            color=color,
            label=f"$Left~brush~with~N_1 = {N1}$",
        )
        plt.plot(
            data_N[-2],
            delta2_N,
            linestyle="-.",
            linewidth=2.5,
            color=color,
            label=f"$Right~brush~with~N_2 = {N2}$",
        )

plt.xlabel(r"$\Pi$")
plt.ylabel(r"$\Gamma$")
plt.title("$N_1 + N_2 = 1000,~\sigma=0.01,~\chi=0.0$")
plt.savefig(f"Gamma_Pi_sig_const.png", dpi=150, bbox_inches="tight")
plt.clf()
# # ---------------------------------------------------------------------------------------------------------------#
for j, sigma1 in enumerate(sigma1_values):
    sigma2 = round(sigma_total - sigma1, 2)

    data_sigma = np.loadtxt(f"data/N1_500_N2_500_sig1_{sigma1}_sig2_{sigma2}.txt")
    Pi_sig = data_sigma[-2]
    delta1_s = data_sigma[3]
    delta2_s = data_sigma[4]

    color = cmap(j / (s_len - 1))
    if sigma1 != 0.1:
        plt.plot(
            Pi_sig,
            delta1_s,
            linestyle="-",
            color=color,
            label=f"$Left~brush~with~\sigma_1 = {sigma1}$",
        )
        plt.plot(
            Pi_sig,
            delta2_s,
            linestyle="--",
            color=color,
            label=f"$Right~brush~with~\sigma_2 = {sigma2}$",
        )
    else:
        plt.plot(
            Pi_sig,
            delta1_s,
            linestyle="-.",
            linewidth=2.5,
            color=color,
            label=f"$Left~brush~with~\sigma_1 = {sigma1}$",
        )
        plt.plot(
            Pi_sig,
            delta2_s,
            linestyle="-.",
            linewidth=2.5,
            color=color,
            label=f"$Right~brush~with~\sigma_2 = {sigma2}$",
        )

plt.xlabel(r"$\Pi$")
plt.ylabel(r"$\Gamma$")
plt.title("$N_1 = N_2 = 500,~\sigma_1 + \sigma_2 = 0.2,~\chi=0.0$")
plt.savefig(f"Gamma__Pi_N_const.png", dpi=150, bbox_inches="tight")
plt.clf()

# # Delta on PI
for i, N1 in enumerate(N1_values):
    N2 = N_total - N1

    data_N = np.loadtxt(f"data/N1_{N1}_N2_{N2}_sig1_0.01_sig2_0.01.txt")
    phi_M_N = data_N[2]
    delta1_N = data_N[9]
    delta2_N = data_N[10]

    color = cmap(i / (p_len - 1))
    if N1 != 500:
        plt.plot(
            data_N[-2],
            delta1_N,
            linestyle="-",
            color=color,
            label=f"$Left~brush~with~N_1 = {N1}$",
        )
        plt.plot(
            data_N[-2],
            delta2_N,
            linestyle="--",
            color=color,
            label=f"$Right~brush~with~N_2 = {N2}$",
        )
    else:
        plt.plot(
            data_N[-2],
            delta1_N,
            linestyle="-.",
            linewidth=2.5,
            color=color,
            label=f"$Left~brush~with~N_1 = {N1}$",
        )
        plt.plot(
            data_N[-2],
            delta2_N,
            linestyle="-.",
            linewidth=2.5,
            color=color,
            label=f"$Right~brush~with~N_2 = {N2}$",
        )
plt.ylim(3.0, 11.0)
plt.xlabel(r"$\Pi$")
plt.ylabel(r"$\delta$")
plt.title("$N_1 + N_2 = 1000,~\sigma=0.01,~\chi=0.0$")
plt.savefig(f"delta_Pi_sig_const.png", dpi=150, bbox_inches="tight")
plt.clf()
# # ---------------------------------------------------------------------------------------------------------------#
for j, sigma1 in enumerate(sigma1_values):
    sigma2 = round(sigma_total - sigma1, 2)

    data_sigma = np.loadtxt(f"data/N1_500_N2_500_sig1_{sigma1}_sig2_{sigma2}.txt")
    Pi_sig = data_sigma[-2]
    delta1_s = data_sigma[9]
    delta2_s = data_sigma[10]

    color = cmap(j / (s_len - 1))
    if sigma1 != 0.1:
        plt.plot(
            Pi_sig,
            delta1_s,
            linestyle="-",
            color=color,
            label=f"$Left~brush~with~\sigma_1 = {sigma1}$",
        )
        plt.plot(
            Pi_sig,
            delta2_s,
            linestyle="--",
            color=color,
            label=f"$Right~brush~with~\sigma_2 = {sigma2}$",
        )
    else:
        plt.plot(
            Pi_sig,
            delta1_s,
            linestyle="-.",
            linewidth=2.5,
            color=color,
            label=f"$Left~brush~with~\sigma_1 = {sigma1}$",
        )
        plt.plot(
            Pi_sig,
            delta2_s,
            linestyle="-.",
            linewidth=2.5,
            color=color,
            label=f"$Right~brush~with~\sigma_2 = {sigma2}$",
        )
plt.ylim(3.0, 5.0)
plt.xlabel(r"$\Pi$")
plt.ylabel(r"$\delta$")
plt.title("$N_1 = N_2 = 500,~\sigma_1 + \sigma_2 = 0.2,~\chi=0.0$")
plt.savefig(f"delta__Pi_N_const.png", dpi=150, bbox_inches="tight")
plt.clf()

# # FF
for i, N1 in enumerate(N1_values):
    N2 = N_total - N1

    data_N = np.loadtxt(f"data/N1_{N1}_N2_{N2}_sig1_0.01_sig2_0.01.txt")

    fr_N = data_N[-1]

    color = cmap(i / (p_len - 1))
    if N1 != 500:
        plt.plot(
            data_N[-2],
            fr_N,
            linestyle="-",
            color=color,
            label=f"$N_1 = {N1},~N_2 = {N2}$",
        )
    else:
        plt.plot(
            data_N[-2],
            fr_N,
            linestyle="-.",
            linewidth=2.5,
            color=color,
            label=f"$N_1 = {N1},~N_2 = {N2}$",
        )
plt.xlabel(r"$\Pi$")
plt.ylabel(r"$F_{fr}$")
plt.legend(loc="upper left")
plt.title("$N_1 + N_2 = 1000,~\sigma=0.01,~\chi=0.0$")
plt.savefig(f"fr_Pi_sig_const.png", dpi=150, bbox_inches="tight")
plt.clf()
# # ---------------------------------------------------------------------------------------------------------------#
for j, sigma1 in enumerate(sigma1_values):
    sigma2 = round(sigma_total - sigma1, 2)

    data_sigma = np.loadtxt(f"data/N1_500_N2_500_sig1_{sigma1}_sig2_{sigma2}.txt")

    fr_s = data_sigma[-1]

    color = cmap(j / (s_len - 1))
    if sigma1 != 0.1:
        plt.plot(
            data_sigma[-2],
            fr_s,
            linestyle="-",
            color=color,
            label=f"$\sigma_1 = {sigma1},~\sigma_2={sigma2}$",
        )
    else:
        plt.plot(
            data_sigma[-2],
            fr_s,
            linestyle="-.",
            linewidth=2.5,
            color=color,
            label=f"$\sigma_1 = {sigma1},~\sigma_2={sigma2}$",
        )
plt.xlabel(r"$\Pi$")
plt.ylabel(r"$F_{fr}$")
plt.legend(loc="upper left")
plt.title("$N_1 = N_2 = 500,~\sigma_1 + \sigma_2 = 0.2,~\chi=0.0$")
plt.savefig(f"fr_Pi_N_const.png", dpi=150, bbox_inches="tight")
plt.clf()
