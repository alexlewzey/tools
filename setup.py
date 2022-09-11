#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = [
    'pyautogui',
    'pyperclip',
    'measurement',
    'pynput',
    'schedule',
    'Pillow',
    'PyPDF2',
    'hurry.filesize',
    'opencv-python',
    'pytesseract',
    'numpy',
    'requests',
    'selenium',
    'webdriver-manager',
    'lxml',
    'Click',
    'pyyaml',
    'tabulate',
    'gTTS',
    'pyttsx3',
]

setup(
    author="Alexander Lewzey",
    author_email='a.lewzey@hotmail.co.uk',
    python_requires='>=3.5',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="A collection of scripts and macro to carry out various different task (mainly automating repetative "
                "tasks)",
    entry_points={
        'console_scripts': [
            'clmac=clmac.cli:cli',
        ],
    },
    install_requires=requirements,
    long_description=readme,
    include_package_data=True,
    keywords='clmac',
    name='clmac',
    packages=find_packages(include=['clmac', 'clmac.*']),
    url='https://github.com/alexlewzey/clmac',
    version='0.1.0',
    zip_safe=False,
)
