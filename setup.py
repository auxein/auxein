from setuptools import setup
from setuptools import find_packages

long_description = '''
Auxein is a Python framework for evolutionary algorithms.
'''

setup(name='Auxein',
      version='0.0.3',
      description='Evolutionary algorithms for everyone.',
      long_description=long_description,
      author='Davide Palmisano',
      author_email='dpalmisano@gmail.com',
      url='https://github.com/auxein/auxein',
      download_url='https://github.com/auxein/auxein/releases',
      license='Apache 2',
      install_requires=['numpy>=1.15.4'],
      packages=find_packages()
)