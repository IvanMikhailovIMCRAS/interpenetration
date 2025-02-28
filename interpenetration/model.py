from sfbox_api import Composition, Frame, Lat, Mol, Mon, Sys


def frame_model(N1: int, N2: int, sigma1: float, sigma2: float, D: int) -> Frame:

    comp1 = f"(X1)1(A){N1-2}(E1)1"
    comp2 = f"(X2)1(A){N2-2}(E2)1"
    theta1 = sigma1 * N1
    theta2 = sigma2 * N2
    lat = Lat(
        **{
            "n_layers": D,
            "geometry": "flat",
            "lowerbound": "surface",
            "upperbound": "surface",
        }
    )
    mons = [
        Mon(**{"name": "X1", "freedom": "pinned", "pinned_range": "1"}),
        Mon(**{"name": "A", "freedom": "free"}),
        Mon(**{"name": "X2", "freedom": "pinned", "pinned_range": str(D)}),
        Mon(**{"name": "E1", "freedom": "free"}),
        Mon(**{"name": "E2", "freedom": "free"}),
        Mon(**{"name": "W", "freedom": "free"}),
    ]
    mols = [
        Mol(**{"name": "Water", "composition": "(W)1", "freedom": "solvent"}),
        Mol(
            **{
                "name": "pol1",
                "composition": comp1,
                "freedom": "restricted",
                "theta": theta1,
            }
        ),
        Mol(
            **{
                "name": "pol2",
                "composition": comp2,
                "freedom": "restricted",
                "theta": theta2,
            }
        ),
    ]
    sys = Sys()

    frame = Frame(lat, sys, mols, mons)
    frame.run()

    return frame


if __name__ == "__main__":
    pass
