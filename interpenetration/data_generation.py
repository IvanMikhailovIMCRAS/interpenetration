import os

import msgpack
import numpy as np
from model import frame_model


def ndarray_encoder(obj):
    if isinstance(obj, np.ndarray):
        return {
            "__ndarray__": True,
            "data": obj.tobytes(),
            "dtype": str(obj.dtype),
            "shape": obj.shape,
        }
    return obj


if __name__ == "__main__":

    sigma1 = 0.1
    sigma2 = 0.2 - sigma1
    N1 = 20
    N2 = 200 - N1

    data_block = {key: None for key in ["D", "F", "z", "phi1", "phi2", "g1", "g2"]}

    file_path = f"data/N1_{N1}_N2_{N2}_sig1_{sigma1}_sig2_{sigma2}.msgpack"
    if os.path.exists(file_path):
        raise FileExistsError(f"file {file_path} is already exist!")

    for D in [45, 60, 100, 20]:
        try:
            frame = frame_model(N1, N2, sigma1, sigma2, D)
            data_block["D"] = D
            data_block["F"] = frame.stats["sys : name : free energy"]
            data_block["z"] = frame.profile["layer"] - 0.5
            data_block["phi1"] = frame.profile["pol1"]
            data_block["phi2"] = frame.profile["pol2"]
            data_block["g1"] = frame.profile["E1"]
            data_block["g2"] = frame.profile["E2"]
        except:
            print(f"With D={D} the program crashes")
        with open(file_path, "ab") as f:
            f.write(
                msgpack.packb(data_block, default=ndarray_encoder, use_bin_type=True)
            )
