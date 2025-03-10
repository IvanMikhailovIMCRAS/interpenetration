import numpy as np
import matplotlib.pyplot as plt
from interpenetration import BrushModel, MidPoint


if __name__=="__main__":
    N = 200
    sigma = np.arange(0.005, 1.0, 0.005)
    H_gauss = []
    H_bcc = []
    H_u = []
    phi_0 = []
    for sig in sigma:
        MP = MidPoint(N, sig, N, sig)
        H_u.append(MP.H1)
        BM = BrushModel(N, sig, 1, "gauss")
        H_gauss.append(BM.H)
        BM = BrushModel(N, sig, 1, "bcc")
        H_bcc.append(BM.H)
        phi_0.append(BM.phi(0))
    
    H_gauss = np.array(H_gauss)
    H_bcc = np.array(H_bcc)
    H_u = np.array(H_u)
    phi_0 = np.array(phi_0)
    
    plt.plot(sigma, H_u/N, color="black", label="$ \\varphi(z) = 1-\\exp(-3/8\\cdot(\\pi / N)^2)(H^2-z^2)$")
    plt.plot(sigma, H_gauss/N, color="red",  label="$\\varphi(z) = 3/8\\cdot(\\pi /N)^2(H^2-z^2)$")
    plt.plot(sigma, H_bcc/N, color="green", label="$\\varphi(z) = 1 - [\\cos(\\pi H / (2N)) ~/~ \\cos(\\pi z / (2N))]^3$")
    plt.xscale('log')
    plt.xlabel("$\sigma$")
    plt.axhline(0.4, color = "gray", linestyle = "--")
    plt.axvline(0.136, color = "gray", linestyle = "--")
    plt.ylabel("$H/N$")
    plt.ylim(0.0, 1.0)
    plt.xlim(6e-3, 1.0)
    plt.legend()
    plt.savefig("H_on_sig.png",dpi=150, bbox_inches="tight")
    plt.clf()

    plt.plot(sigma, 1-np.exp(-3*np.pi**2/8 * (H_u/N)**2), color="black", label="$ \\varphi(z) = 1-\\exp(-3/8\\cdot(\\pi / N)^2)(H^2-z^2)$")
    plt.plot(sigma, 3*np.pi**2/8 * (H_gauss/N)**2, color="red", label="$\\varphi(z) = 3/8\\cdot(\\pi /N)^2(H^2-z^2)$")
    plt.plot(sigma, phi_0, color="green", label="$\\varphi(z) = 1 - [\\cos(\\pi H / (2N)) ~/~ \\cos(\\pi z / (2N))]^3$")
    plt.xlabel("$\sigma$")
    plt.ylabel("$\\varphi(0)$")
    plt.axhline(0.4, color = "gray", linestyle = "--")
    plt.axvline(0.105, color = "gray", linestyle = "--")    
    plt.ylim(0.0, 1.02)
    plt.xlim(0.0, 1.0)
    plt.legend()
    plt.savefig("phi_on_sig.png",dpi=150, bbox_inches="tight")
    # plt.show()
    plt.clf()
