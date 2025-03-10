from typing import Callable, List, Tuple

import numpy as np
from scipy.integrate import quad
from scipy.optimize import fsolve, minimize

MINVAL = 1e-7


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


def pressure(phi_m: float) -> float:
    """Calculation of disjoining pressure

    Args:
        phi_m (float): value of midpoint volume fraction at equilibrium point of the system

    Returns:
        float: pressure value
    """
    return -phi_m - np.log(1 - phi_m)


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
        self.H1 = fsolve(
            lambda H: quad(
                lambda z: (1.0 - np.exp(-1.5 * (np.pi / 2 / N1) ** 2 * (H**2 - z**2))),
                0.0,
                H,
            )[0]
            - N1 * sigma1,
            2.0 * N1 * sigma1,
        )[0]
        self.H2 = fsolve(
            lambda H: quad(
                lambda z: (1.0 - np.exp(-1.5 * (np.pi / 2 / N2) ** 2 * (H**2 - z**2))),
                0.0,
                H,
            )[0]
            - N2 * sigma2,
            2.0 * N2 * sigma2,
        )[0]

    # def _target_func(self, z: float) -> float:
    #     """Intent function for z_mean finding

    #     Args:
    #         z (float): coordinate where u1(z)=u2(z)

    #     Returns:
    #         float: target result (expect 0.0)
    #     """
    #     return (
    #         z
    #         * (
    #             1.0
    #             - (1.0 - self.N2 * self.sigma2 / (self.D - z))
    #             * np.exp(-self.K1 * z**2 + self.K2 * (self.D - z) ** 2)
    #         )
    #         - self.N1 * self.sigma1
    #     )

    def calc(self, D: float) -> Tuple[float, float]:
        """z_mean and phi_mean calculation

        Args:
            D (float): distance between grafting surfaces

        Returns:
            Tuple[float, float]: z_mean and phi_mean values
        """
        self.D = D 
        # delta_u2 = self.optimize()[2]
        # print(delta_u1)
        # print(delta_u2)

        if D > (self.H1 + self.H2):
            phi_m = 0.0
            z_m = D/2
        elif D <= (self.N1 * self.sigma1 + self.N2 * self.sigma2):
            raise ValueError("D < N1*sigma1 + N2*sigma2")
        else:
            try:
                self.initial_guess = np.array([D*self.N1*self.sigma1/(self.N1*self.sigma1+self.N2*self.sigma2), 0.2, 0.2])
                z_m = self.optimize()[0]
            except:
                self.initial_guess = np.array([D*self.N1*self.sigma1/(self.N1*self.sigma1+self.N2*self.sigma2), 1.0, 1.0])
                z_m = self.optimize()[0]
                
            delta_u1 = self.optimize()[1]
            phi_m = 1.0 - np.exp(- self.u1(z_m)-delta_u1)
        return (z_m, phi_m)
    

            

    def u1(self, z: np.ndarray) -> np.ndarray:
        return 1.5 * (np.pi / 2 / self.N1) ** 2 * (self.H1**2 - z**2)

    def u2(self, z: np.ndarray) -> np.ndarray:
        return 1.5 * (np.pi / 2 / self.N2) ** 2 * (self.H2**2 - (self.D - z) ** 2)

    def _target_func_z_m(self, params: List[float]) -> float:
        z_m = params[0]
        delta_u1 = params[1]
        delta_u2 = params[2]
        
        if delta_u1 < 0.0:
            delta_u1== 0.0
        if delta_u2 < 0.0:
            delta_u2== 0.0
            
        func1 = (self.u1(z_m)+delta_u1 - self.u2(z_m)-delta_u2) ** 2
        func2 = (
            quad(lambda z: 1-np.exp(-self.u1(z) - delta_u1), 0.0, z_m)[0]
            - self.N1 * self.sigma1
        ) ** 2
    
        func3 = (
            quad(lambda z: 1-np.exp(-self.u2(z) - delta_u2), z_m, self.D)[0]
            - self.N2 * self.sigma2
        ) ** 2
   
        return func1 + func2 + func3

    def optimize(self) -> Tuple[float, float, float]:
        return minimize(self._target_func_z_m, self.initial_guess, method="CG", options={"eps":np.array([1e-7, 1e-7, 1e-7])}).x


if __name__ == "__main__":
    m_p = MidPoint(20, 0.1, 180, 0.1)

    print(m_p.H1 + m_p.H2)
    z, phi = m_p.calc(81)

    print(z, phi)
