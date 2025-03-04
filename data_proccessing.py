import msgpack
import numpy as np

from interpenetration import PenetrationAnalisys


# Custom decoder for numpy.ndarray
def ndarray_decoder(obj):
    if "__ndarray__" in obj:
        return np.frombuffer(obj["data"], dtype=obj["dtype"]).reshape(obj["shape"])
    return obj


if __name__ == "__main__":
    sigma1 = 0.1
    sigma2 = 0.2 - sigma1
    N1 = 20
    N2 = 200 - N1

    data = []
    file_path = f"data/N1_{N1}_N2_{N2}_sig1_{sigma1}_sig2_{sigma2}.msgpack"

    with open(file_path, "rb") as f:
        unpacker = msgpack.Unpacker(f, object_hook=ndarray_decoder, raw=False)
        for obj in unpacker:
            z = obj["z"]
            phi1 = obj["phi1"]
            phi2 = obj["phi2"]
            PA = PenetrationAnalisys(z, phi1, phi2)

            data.append(
                [
                    PA.D,
                    PA.Gamma1,
                    PA.Gamma2,
                    PA.Sigma1,
                    PA.Sigma2,
                    PA.L1,
                    PA.L2,
                    PA.delta1,
                    PA.delta2,
                    PA.Pressure,
                    PA.friction_force,
                ]
            )
    np.savetxt(
        f"data/N1_{N1}_N2_{N2}_sig1_{sigma1}_sig2_{sigma2}.txt",
        np.array(data).T,
    )
