from typing import Callable, Tuple
import numpy as np


def func_zeros(
    a: float, b: float, func: Callable[[float], float], delta: float = 1e-7
) -> float:
    """The root(zero)-finding of f(x) an interval (a, b) by the bisection method

    Args:
        a (float): left border
        b (float): right border
        func (Callable): target function f(x)
        delta (float, optional): the finding accuracy of x0 (f(x0)=0). Defaults to 1e-7.

    Raises:
        ValueError: Ends have the same sign

    Returns:
        float: x0, which is the solution f(x0) = 0
    """
    while abs(b - a) > delta:
        y1 = func(a)
        y2 = func(0.5 * (a + b))
        y3 = func(b)
        if y1 * y3 > 0:
            raise ValueError(
                f"Ends have the same sign \\ func({a})={func(a)} and func({b})={func(b)}"
            )
        if y1 * y2 < 0:
            b = 0.5 * (a + b)
        else:
            a = 0.5 * (a + b)
    return (a + b) * 0.5


class MidPoint:
    """z_mean and phi_mean midpoint searching"""

    def __init__(self, N1: int, sigma1: float, N2: int, sigma2: float) -> None:
        """class initialization

        Args:
            N1 (int): polymerization degree of left chains
            sigma1 (float): grafting density of left chains
            N2 (int): polymerization degree of right chains
            sigma2 (float): grafting density of right chains
        """
        self.N1 = N1
        self.sigma1 = sigma1
        self.N2 = N2
        self.sigma2 = sigma2
        self.K1 = 3.0 / 8.0 * np.pi**2 / N1**2
        self.K2 = 3.0 / 8.0 * np.pi**2 / N2**2
        self.D = 0.0
        self.H1 = (np.pi/2)**(-2/3)*N1*sigma1**(1/3)
        self.H2 = (np.pi/2)**(-2/3)*N2*sigma2**(1/3)
    
    def target_func(self, z: float) -> float:
        """Intent function for z_mean finding

        Args:
            z (float): coordinate where u1(z)=u2(z)

        Returns:
            float: target result (expect 0.0)
        """
        return (
            z
            * (
                1.0
                - (1.0 - self.N2 * self.sigma2 / (self.D - z))
                * np.exp(-self.K1 * z**2 + self.K2 * (self.D - z) ** 2)
            )
            - self.N1 * self.sigma1
        )

    def calc(self, D: float) -> Tuple[float, float]:
        """z_mean and phi_mean calculation

        Args:
            D (float): distance between grafting surfaces

        Returns:
            Tuple[float, float]: z_mean and phi_mean values
        """
        self.D = D
        z_m = func_zeros(0.0, D - 1e-7, self.target_func)

        if D > self.H1+self.H2:
            phi_m = 0.0
        elif D <= self.N1*self.sigma1+self.N1*self.sigma1:
            phi_m = 1.0
        else:
            phi_m = 1.0 - (1.0 - self.N1 * self.sigma1 / z_m) * np.exp(
                self.K1 * z_m**2
            )
        return (z_m, phi_m)


if __name__ == "__main__":
    m_p = MidPoint(180, 0.1, 20, 0.1)

    print(m_p.calc(1))
