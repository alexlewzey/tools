#!/bin/bash
if [ -e ./env/bin/activate ]; then source ./env/bin/activate; else python3 -m venv env && source ./env/bin/activate; fi
pip install -e .
printf "\n\n# Command to run clmac keyboard event listener" >>~/.zprofile
printf "\nalias cmr=\"cd $(pwd) && venv && clmac kel run\"" >>~/.zprofile