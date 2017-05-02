#!/usr/bin/env python

import os
import setuptools

setuptools.setup(
    name='sequence_qe',
    version='0.1dev',
    description='Sequence models for word level quality estimation',
    long_description=open(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), 'README.md')).read(),
    license='BSD 3-clause',
    classifiers=['Development Status :: 3 - Alpha',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: BSD License',
                 'Operating System :: OS Independent',
                 'Topic :: Scientific/Engineering'],
    packages=['sequence_qe'],
)
