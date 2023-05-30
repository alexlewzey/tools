#!/usr/bin/env python

"""The setup script."""

from setuptools import find_packages, setup

with open("README.md") as readme_file:
    readme = readme_file.read()


setup(
    entry_points={
        "console_scripts": [
            "clmac=clmac.app",
        ],
    },
    packages=find_packages(include=["clmac", "clmac.*"]),
    url="https://github.com/alexlewzey/clmac",
)
