#!/usr/bin/env python

"""The setup script."""

from setuptools import find_packages, setup

with open("README.md") as readme_file:
    readme = readme_file.read()

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    entry_points={
        "console_scripts": [
            "clmac=clmac.app:run",
        ],
    },
    packages=find_packages(include=["clmac", "clmac.*"]),
    url="https://github.com/alexlewzey/clmac",
    install_requires=requirements,
)
