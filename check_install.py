#!/bin/env python3

print('\nRun Python installation test for graph ConvNets')

import os
import sys
major, minor = sys.version_info.major, sys.version_info.minor

if ( (major is not 3) or (minor is not 6) ):
    raise Exception('Code developed for PyTorch with Python 3.6. You have Python {}.{}'.format(major,minor))

try:
    import numpy 
    print('Recommended version of numpy is {}. You have {}.'.format('1.13.1',numpy.__version__))
    import jupyter
    print('Recommended version of jupyter is {}. You have {}.'.format('1.0.0',jupyter.__version__))
    import torch
    print('Recommended version of pytorch is {}. You have {}.'.format('0.2.0_1',torch.__version__))
    import tensorflow
    print('Recommended version of tensorflow is {}. You have {}.'.format('0.11.0rc0',tensorflow.__version__))
    import scipy
    print('Recommended version of scipy is {}. You have {}.'.format('0.19.1',scipy.__version__))
    import sklearn
    print('Recommended version of sklearn is {}. You have {}.'.format('0.19.0',sklearn.__version__))


except:
    print('A package is missing or the version of the package.')
    print('Install the package below.\n')
    raise

print('Successful installation of Python {}.{} and '
      'most of the packages required to run the code.\n'.format(major, minor))
