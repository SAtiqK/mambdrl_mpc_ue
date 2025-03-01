#!/usr/bin/env python

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

setup_args = generate_distutils_setup (
    name='unreal_training',  # Replace 'your_package_name' with the actual name of your ROS package
    version='0.0.0',  # Set the appropriate version number
    packages=['unreal_training'],  # Replace 'your_package_name' with the actual name of your ROS package
    package_dir={'': 'src'},  # Specify the package directory as 'src'
)

setup(**setup_args)
