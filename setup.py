from setuptools import setup

setup(name='dca',
      version='1.0',
      description='python implementation of mfDCA, adapted from the dca.m script from http://dca.rice.edu/portal/dca/.',
      author='Jonathan Martin',
      author_email='jonathan.martin3@utdallas.edu',
      license='MIT',
      packages=['dca'],
      zip_safe=False,
      install_requires=[
          'biopython',
          'matplotlib',
          'numpy',
          'numba',
          'scipy'
      ]
      )
