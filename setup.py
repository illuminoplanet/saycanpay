import os
from setuptools import find_packages, setup

setup(
    name="saycanpay",
    packages=[
        package for package in find_packages() if package.startswith("saycanpay")
    ],
)