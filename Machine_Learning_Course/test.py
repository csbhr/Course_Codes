import project7.p7 as p7
import naive_bayes.naive_bayes_classifier as nbc
import naive_bayes.naive_bayes_MBEM as nbMBEM
import numpy

x = [[1, 0, 2, 3],
     [1, 2, 3],
     [4, 1, 2],
     [0],
     [4, 2, 1, 3]]

# phis0 = [0.3, 0.5, 0.7, 0.2, 0.6, 0.4]
# phis1 = [0.4, 0.2, 0.4, 0.6, 0.3, 0.7]
# phi = 0.3

y = [[1],
     [0],
     [1],
     [1],
     [0]]

# print(nbMBEM.probability_y(phi, y))

# print(nbMBEM.probability_x_y0(phis, x))
# print(nbMBEM.probability_x_y1(phis, x))

# print(nbMBEM.h(phi, phis0, phis1, x))

print(nbMBEM.compute_parameters(x, y, 5))
