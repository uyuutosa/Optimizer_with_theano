#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
import os

from setuptools import setup, find_packages

try:
    with open('README.rst') as f:
        readme = f.read()
except IOError:
    readme = ''

def _requires_from_file(filename):
    return open(filename).read().splitlines()

# version
here = os.path.dirname(os.path.abspath(__file__))
version = next((line.split('=')[1].strip().replace("'", '')
                for line in open(os.path.join(here,
                                              'Optimizer_with_theano',
                                              '__init__.py'))
                if line.startswith('__version__ = ')),
               '0.0.dev0')

setup(
    name="Optimizer_with_theano",
    packages=["Optimizer_with_theano"],
    version="0.1.13",
    #version=version,
    url='https://github.com/uyuutosa/Optimizer_with_theano',
    author='uyuutosa',
    author_email='sayu819@gmail.com',
    maintainer='uyuutosa',
    maintainer_email='sayu819@gmail.com',
    description='Package Dependency: Validates package requirements',
    long_description=readme,
    install_requires=["theano"],
    #install_requires=_requires_from_file('requirements.txt'),
    license="MIT",
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'License :: OSI Approved :: MIT License',
    ],
    entry_points="""
      # -*- Entry points: -*-
      [console_scripts]
      pkgdep = Optimizer_with_theano.scripts.command:main
    """,
    include_package_data=True
)
