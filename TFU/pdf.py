import numpy
from scipy.stats import norm
import pylab as plt
from fastkde import fastKDE

import config

edges = numpy.array([[0,0],[0,config.reader_height],[config.reader_width,0], [config.reader_width,config.reader_height]])

points = numpy.array([[149.0, 850.710091], [149.0,  827.278607], [149.0, 815.564041], [149.0, 803.848299], [149.0, 792.132557], [149.0, 756.986507], [149.0, 733.556199], [149.0, 710.125891], [149.0, 686.695583], [149.0, 663.264099], [149.0, 639.833791], [149.0, 616.403483], [149.0, 592.972   ]])
points = numpy.vstack((points, edges))
xrow = points.T[1]
yrow = points.T[0]

pdf, (v1, v2) = fastKDE.pdf(xrow, yrow)
plt.contour(v1,v2, pdf *10000)
plt.show()

density = norm(xrow).pdf(xrow)
#plt.plot(xrow, numpy.full_like(xrow, -0.1), '|k', markeredgewidth=1)
#plt.axis([-4, 8, -.2, 5])
