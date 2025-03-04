import matplotlib.pyplot as plt

from interpenetration import PenetrationAnalisys, frame_model

if __name__ == "__main__":
    sigma1 = 0.1
    sigma2 = 0.2 - sigma1
    N1 = 20
    N2 = 200 - N1
    D = 50
    frame = frame_model(N1, N2, sigma1, sigma2, D)

    z = frame.profile["layer"][1:-1] - 0.5
    phi1 = frame.profile["pol1"][1:-1]
    phi2 = frame.profile["pol2"][1:-1]
    PA = PenetrationAnalisys(z, phi1, phi2)

    # plt.plot(z, phi1,  'o', color="red")
    # plt.plot(z, phi2, 'o', color="black")
    plt.scatter(PA.z_m, PA.phi_m / 2, color="green")

    plt.plot(PA.z, PA.phi1(PA.z), color="red")
    plt.plot(PA.z, PA.phi2(PA.z), color="black")
    plt.plot(PA.z, PA.rho1(PA.z), color="red", linestyle="--")
    plt.plot(PA.z, PA.rho2(PA.z), color="black", linestyle="--")
    plt.show()

    # print(PA.L1)
    # print(PA.L2)

    # print(PA.Sigma1)
    # print(PA.Sigma2)

    # print(PA.Gamma1)
    # print(PA.Gamma2)

    # print(PA.Pressure)
    # print(PA.friction_force)

    print(PA.delta1)
    print(PA.delta2)
