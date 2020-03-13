import numpy
import sympy
import matplotlib_helper.figure as figure
import gaussian_discriminant_analysis.gaussian_discriminant_analysis as gda


def get_data(path):
    data = numpy.loadtxt(path)
    return data


def gaussian_function(mu, sigma, x1, x2):
    sigma = sigma.I
    sigma_ = sigma.tolist()
    mu_ = mu.tolist()[0]
    m1, m2 = mu_[0], mu_[1]
    s11, s12, s21, s22 = sigma_[0][0], sigma_[0][1], sigma_[1][0], sigma_[1][1]
    z1 = x1 - m1
    z2 = x2 - m2
    ele = numpy.exp((z1 * (z1 * s11 + z2 * s21) + z2 * (z1 * s12 + z2 * s22)) * (-0.5))
    mole = 2 * numpy.pi * numpy.power(numpy.linalg.det(sigma), 1 / 2)
    return ele / mole


def solve_line_equations():
    x1 = sympy.Symbol('x1')
    x2 = sympy.Symbol('x2')
    s11 = sympy.Symbol('s11')
    s12 = sympy.Symbol('s12')
    s21 = sympy.Symbol('s21')
    s22 = sympy.Symbol('s22')
    m01 = sympy.Symbol('m01')
    m02 = sympy.Symbol('m02')
    m11 = sympy.Symbol('m11')
    m12 = sympy.Symbol('m12')
    result = sympy.solve([(x1 - m01) * ((x1 - m01) * s11 + (x2 - m02) * s21) + (x2 - m02) * (
            (x1 - m01) * s12 + (x2 - m02) * s22) - (x1 - m11) * ((x1 - m11) * s11 + (x2 - m12) * s21) - (x2 - m12) * (
                                  (x1 - m11) * s12 + (x2 - m12) * s22)], [x2])
    print(result)


def compute_line(mu0, mu1, sigma, x1):
    sigma = sigma.I
    sigma_ = sigma.tolist()
    mu0_ = mu0.tolist()[0]
    mu1_ = mu1.tolist()[0]
    m01, m02 = mu0_[0], mu0_[1]
    m11, m12 = mu1_[0], mu1_[1]
    s11, s12, s21, s22 = sigma_[0][0], sigma_[0][1], sigma_[1][0], sigma_[1][1]
    temp1 = m12 * s12 + m12 * s21 + 2 * m11 * s11 - 2 * m01 * s11 - m02 * s12 - m02 * s21
    temp2 = m01 ** 2 * s11 + m02 ** 2 * s22 - m11 ** 2 * s11 - m12 ** 2 * s22 + m01 * m02 * s12 + m01 * m02 * s21 - m11 * m12 * s12 - m11 * m12 * s21
    temp3 = m01 * s12 + m01 * s21 + 2 * m02 * s22 - 2 * m12 * s22 - m11 * s12 - m11 * s21
    x2 = (temp1 * x1 + temp2) / temp3
    return x2


def action():
    x = get_data("data_files/ex4x.dat")
    y = get_data("data_files/ex4y.dat")
    x = numpy.mat(x)
    y = numpy.mat(y).T
    phi, mu0, mu1, sigma = gda.compute_parameters(x, y)
    print("The parameter of bernoulli distribution:",phi)
    print("The mean of gaussian distribution given y=0:")
    print(mu0)
    print("The mean of gaussian distribution given y=1:")
    print(mu1)
    print("The variance of gaussian distribution:")
    print(sigma)
    figure.pic_6(mu0, mu1, sigma, x, y, pro_func=gaussian_function, line_func=compute_line)
