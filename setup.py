from distutils.core import setup
from Cython.Build import cythonize

setup(name='point_cloud_utilities_cy', ext_modules=cythonize("utilities/point_cloud_utilities_cy.pyx"))
setup(name='cutils', ext_modules=cythonize("thirdparty/DECOR-GAN/cutils.pyx"))