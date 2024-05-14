from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import newton
from cmath import *

def main():
    TE = True

    nf = 2.935755333756289
    ns = 1.444
    nc = 1
    wl = 1.55 * (10 ** -6)
    h = 0.22 * (10 ** -6)


    k0 = 2 * pi / wl
    beta = lambda kappa: sqrt((k0 * nf) ** 2 - (kappa) ** 2)
    gammac = lambda kappa: sqrt(beta(kappa) ** 2 - (k0 ** 2 * nc ** 2))
    gammas = lambda kappa: sqrt(beta(kappa) ** 2 - (k0 ** 2 * ns ** 2))


    if TE:
        numerator = lambda kappa: gammac(kappa) + gammas(kappa)
        denom = lambda kappa: kappa * (1 - (gammac(kappa) * gammas(kappa) / (kappa ** 2)))
    else:
        numerator = lambda kappa: kappa * (nf ** 2 / ns ** 2 * gammas(kappa) + nf ** 2 / nc ** 2 * gammac(kappa))
        denom = lambda kappa: kappa ** 2 - (nf ** 4 / (nc ** 2 * ns ** 2) * gammac(kappa) * gammas(kappa))


    # rhs = lambda kappa : (gammac(kappa) + gammas(kappa)) / (kappa * (1 - (gammac(kappa)*gammas(kappa) / kappa**2)))
    rhs = lambda kappa: (numerator(kappa) / denom(kappa))
    lhs = lambda kappa: np.tan(kappa * h)
    xAxis = np.arange(1, 40000000, 100, dtype=complex)


    yPlt2 = np.array([rhs(i) for i in xAxis], dtype=complex)
    # for i in range(0, yPlt2.size):
    #     if np.real(yPlt2[i]) >= 0:
    #         yPlt2[i] = np.nan
    yPlt2[:-1][np.diff(yPlt2) >= 0] = np.nan

    yPlt1 = np.tan(xAxis * h)
    yPlt1[:-1][np.diff(yPlt1) < 0] = np.nan
    plt.xlim(0, max(xAxis))
    plt.ylim(-40, 10)
    plt.plot(xAxis, yPlt1)
    plt.plot(xAxis, yPlt2)

    plt.title("TE Charactaristic Equation for Porous Silicon Waveguide")
    plt.xlabel("Kappa")
    plt.show()
    modeNum = float(input("Mode Number"))
    intersection = newton(lambda kappa: np.real(rhs(kappa))-np.real(lhs(kappa)), ((modeNum+1/2 + 0.01) * pi / h))
    print("k0 : " + str(k0))
    print("kappa at intersection : " + str(intersection))
    print("gamma s at intersection: " + str(gammas(intersection)))
    print("gamma c at intersection: " + str(gammac(intersection)))
    # print("beta upperbound: " + str(k0 * nf))
    # print("beta lowerbound: " + str(k0 * ns))
    print("beta at intersection: " + str(beta(intersection)))
    print("effective index : " + str(beta(intersection)/k0))
    if np.real(k0*ns) <= np.real(beta(intersection)) <= np.real(k0*nf):
        print("GUIDED MODE!")
    else:
        print("NOT A GUIDED MODE!")

    plt.close()

    def ey(x):
        D = 2
        if x > h:
            return D * (cos(intersection*h) + (gammas(intersection)/intersection * sin(intersection * h))) * exp(-1 * gammac(intersection) * (x - h))
        elif 0 <= x <= h:
            return D * (cos(intersection*x) + (gammas(intersection)/intersection * sin(intersection * x)))
        else:
            return D * exp(gammas(intersection) * x)

    x = np.arange((h/2) -3*h, (h/2)+3*h, 6*h/10000)
    y = np.array([ey(elem) for elem in x])

    plt.xlim(-1 * h, 2 * h)
    plt.ylim(-5, 15)
    plt.plot(x, y)

    plt.title("Mode Profile for TE 0")
    plt.xlabel("x (m)")
    plt.ylabel("Amplitude of Electric Field")
    # plt.savefig('HW2Problem4cPlot.png', bbox_inches='tight', dpi=600)
    plt.show()


if __name__ == "__main__":
    main()