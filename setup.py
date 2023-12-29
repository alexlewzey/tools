#!/usr/bin/env python

"""The setup script."""

from setuptools import find_packages, setup


setup(
    entry_points={
        "console_scripts": [
            "clmac=clmac.app:run",
        ],
    },
    packages=find_packages(),
    url="https://github.com/alexlewzey/clmac"
)
